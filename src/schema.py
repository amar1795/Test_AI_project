"""Pydantic data contract plus dependency-free boundary validation.

The pipeline validates plain dictionaries so its deterministic keyword mode can run
before dependencies are installed. When Pydantic is present, the same boundaries
are additionally validated by the models below.
"""

from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field

    PYDANTIC_AVAILABLE = True

    class StrictModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class Sentence(StrictModel):
        id: int = Field(ge=0)
        start: int = Field(ge=0)
        end: int = Field(ge=0)
        text: str

    class Mention(StrictModel):
        id: str
        text: str
        start: int = Field(ge=0)
        end: int = Field(ge=0)
        label: str
        source: str
        confidence: float = Field(ge=0.0, le=1.0)
        sent_id: int | None = None

    class ResolvedEntity(StrictModel):
        entity_id: str
        canonical: str
        label: str
        mention_ids: list[str]
        resolution_method: str
        confidence: float = Field(ge=0.0, le=1.0)

    class Document(StrictModel):
        doc_id: str
        url: str
        title: str
        published_at: str
        text: str
        sentences: list[Sentence] = []
        mentions: list[Mention] = []
        entities: list[ResolvedEntity] = []
        pipeline_version: str
        config: dict[str, Any] = {}

    class LLMResolution(StrictModel):
        entity_id: str | None
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

except ImportError:  # pragma: no cover - exercised in dependency-free integration tests
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]

    class Sentence:  # type: ignore[no-redef]
        pass

    class Mention:  # type: ignore[no-redef]
        pass

    class ResolvedEntity:  # type: ignore[no-redef]
        pass

    class Document:  # type: ignore[no-redef]
        pass

    class LLMResolution:  # type: ignore[no-redef]
        pass


LABELS = {
    "PERSON", "ORG", "GPE", "FACILITY", "GROUP", "GOVERNMENT_AGENCY",
    "MILITARY_UNIT", "WEAPONS_SYSTEM", "VESSEL",
}


class ContractError(ValueError):
    """Raised when a pipeline stage violates the data contract."""


def _require(value: dict[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ContractError(f"{context} is missing required fields: {', '.join(missing)}")


def validate_document(document: dict[str, Any], stage: str = "unknown") -> dict[str, Any]:
    _require(
        document,
        ("doc_id", "url", "title", "published_at", "text", "sentences", "mentions", "entities", "pipeline_version"),
        f"document at stage {stage}",
    )
    text = document["text"]
    if not isinstance(text, str):
        raise ContractError("document text must be a string")

    for sentence in document["sentences"]:
        _require(sentence, ("id", "start", "end", "text"), "sentence")
        start, end = sentence["start"], sentence["end"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text)):
            raise ContractError(f"invalid sentence offsets: {start}:{end}")
        if text[start:end] != sentence["text"]:
            raise ContractError(f"sentence {sentence['id']} does not match its source offsets")

    sentence_ids = {sentence["id"] for sentence in document["sentences"]}
    for mention in document["mentions"]:
        _require(mention, ("id", "text", "start", "end", "label", "source", "confidence", "sent_id"), "mention")
        start, end = mention["start"], mention["end"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            raise ContractError(f"invalid mention offsets: {start}:{end}")
        if text[start:end] != mention["text"]:
            raise ContractError(f"mention {mention['id']} does not match text[{start}:{end}]")
        if mention["sent_id"] is not None and mention["sent_id"] not in sentence_ids:
            raise ContractError(f"mention {mention['id']} points to an unknown sentence")
        confidence = mention["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ContractError(f"mention {mention['id']} has invalid confidence")

    if PYDANTIC_AVAILABLE:
        Document.model_validate(document)
    return document


def validate_llm_resolution(value: dict[str, Any]) -> dict[str, Any]:
    _require(value, ("entity_id", "confidence", "reasoning"), "LLM resolution")
    if value["entity_id"] is not None and not isinstance(value["entity_id"], str):
        raise ContractError("LLM entity_id must be a string or null")
    if not isinstance(value["reasoning"], str):
        raise ContractError("LLM reasoning must be a string")
    confidence = value["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ContractError("LLM confidence must be between 0 and 1")
    if PYDANTIC_AVAILABLE:
        LLMResolution.model_validate(value)
    return value

