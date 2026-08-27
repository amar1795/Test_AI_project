from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from src.schema import ContractError, validate_llm_resolution


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class OpenAICompatibleClient:
    """Small adapter for chat-completions-compatible JSON endpoints."""

    endpoint: str
    api_key: str
    model: str
    timeout: float = 60.0

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]


def parse_resolution_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ContractError("LLM response did not contain a JSON object")
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise ContractError(f"LLM response contained invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError("LLM response JSON must be an object")
    return validate_llm_resolution(value)


def build_prompt(
    mention: dict[str, Any],
    document: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    sentence_index = mention.get("sent_id")
    sentences = document.get("sentences", [])
    context = []
    if sentence_index is not None:
        context = [
            sentence["text"] for sentence in sentences
            if sentence_index - 1 <= sentence["id"] <= sentence_index + 1
        ]
    candidate_payload = [
        {
            "entity_id": candidate["entity_id"],
            "canonical": candidate["canonical"],
            "label": candidate["label"],
            "aliases": candidate.get("aliases", []),
        }
        for candidate in candidates[:3]
    ]
    return (
        "Resolve the entity mention using only the supplied context and candidates. "
        "Choosing none is valid and expected when evidence is insufficient. Return exactly one JSON object "
        "with entity_id (a candidate id or null), confidence (0 to 1), and reasoning (a short string).\n"
        f"Mention: {json.dumps(mention['text'], ensure_ascii=False)}\n"
        f"Context: {json.dumps(context, ensure_ascii=False)}\n"
        f"Candidates: {json.dumps(candidate_payload, ensure_ascii=False)}"
    )


def disambiguate(
    client: LLMClient,
    mention: dict[str, Any],
    document: dict[str, Any],
    candidates: list[dict[str, Any]],
    attempts: int = 3,
    sleep=time.sleep,
) -> dict[str, Any] | None:
    prompt = build_prompt(mention, document, candidates)
    for attempt in range(attempts):
        try:
            result = parse_resolution_response(client.complete(prompt))
            allowed = {candidate["entity_id"] for candidate in candidates[:3]}
            if result["entity_id"] is not None and result["entity_id"] not in allowed:
                raise ContractError("LLM returned an entity outside the candidate list")
            return result
        except Exception:
            if attempt + 1 >= attempts:
                return None
            prompt += "\nYour previous response was invalid. Return only the required JSON object with all fields."
            sleep(2 ** attempt)
    return None
