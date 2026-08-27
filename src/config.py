from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    overrides = {
        "llm_model": os.getenv("LLM_MODEL"),
        "llm_base_url": os.getenv("LLM_BASE_URL"),
    }
    for key, value in overrides.items():
        if value:
            config[key] = value
    return config
