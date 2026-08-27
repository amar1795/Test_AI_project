from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedMention:
    display: str
    canonical: str
    key: str
    initials_key: str


def _strip_prefix(value: str, prefixes: Iterable[str]) -> str:
    for prefix in sorted(prefixes, key=len, reverse=True):
        match = re.match(rf"^{re.escape(prefix)}(?:\s+|$)", value, flags=re.IGNORECASE)
        if match:
            return value[match.end():].strip()
    return value


def _strip_suffix(value: str, suffixes: Iterable[str]) -> str:
    for suffix in sorted(suffixes, key=len, reverse=True):
        match = re.search(rf"(?:,?\s+){re.escape(suffix)}\.?$", value, flags=re.IGNORECASE)
        if match:
            return value[:match.start()].strip(" ,")
    return value


def _initials_key(value: str) -> str:
    tokens = re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)
    if len(tokens) < 2:
        return " ".join(tokens)
    return " ".join([*(token[0] for token in tokens[:-1]), tokens[-1]])


def _acronym_key(value: str) -> str:
    tokens = re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)
    if len(tokens) <= 1:
        return "".join(tokens)
    return "".join(token[0] for token in tokens)


def normalize_mention(
    text: str,
    label: str,
    honorifics: Iterable[str] = (),
    corporate_suffixes: Iterable[str] = (),
) -> NormalizedMention:
    display = text
    value = unicodedata.normalize("NFKC", text)
    value = ZERO_WIDTH.sub("", value)
    value = WHITESPACE.sub(" ", value).strip()
    if label == "PERSON":
        value = _strip_prefix(value, honorifics)
    elif label == "ORG":
        previous = None
        while previous != value:
            previous = value
            value = _strip_suffix(value, corporate_suffixes)
    key = value.casefold().strip(".,;: ")
    return NormalizedMention(
        display=display,
        canonical=value.strip(".,;: "),
        key=key,
        initials_key=(
            _initials_key(value) if label == "PERSON"
            else _acronym_key(value) if label == "ORG"
            else key
        ),
    )
