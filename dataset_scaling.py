"""
Dataset scaling helpers for PharmaDrishti.
"""

from __future__ import annotations

import json
from pathlib import Path

from generate_personas import generate_personas, save_personas


def ensure_persona_dataset(
    target_count: int,
    output_path: str | Path = "indian_healthcare_personas.json",
) -> Path:
    path = Path(output_path)
    current_count = 0

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            current_count = len(json.load(handle))

    if current_count >= target_count:
        return path

    personas = generate_personas(n=target_count)
    save_personas(personas, filepath=str(path))
    return path
