"""Shared helpers for the render stages: sizes, stable seeds, resume state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import Config
from ..project import Project

# (image w,h)  (video w,h)  -- all multiples of 32
SIZES = {
    "16:9": ((1920, 1088), (1280, 704)),
    "9:16": ((1088, 1920), (704, 1280)),
    "1:1": ((1472, 1472), (960, 960)),
    "21:9": ((2048, 864), (1344, 576)),
}
OUTPUT = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1440, 1440), "21:9": (2560, 1080)}


def image_size(aspect: str, cfg: Config) -> tuple[int, int]:
    w, h = SIZES[aspect][0]
    if aspect == "16:9":  # honour config overrides for the default format
        return cfg.image.width, cfg.image.height
    return w, h


def video_size(aspect: str, cfg: Config) -> tuple[int, int]:
    w, h = SIZES[aspect][1]
    if aspect == "16:9":
        return cfg.video.width, cfg.video.height
    return w, h


def stable_seed(base: int, key: str) -> int:
    return base + int(hashlib.sha256(key.encode()).hexdigest()[:6], 16)


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class State:
    """artifact path -> digest of the inputs that produced it. Lets every stage resume."""

    def __init__(self, project: Project):
        self.path = project.state_file
        self.data: dict[str, str] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    def fresh(self, artifact: Path, key: Any) -> bool:
        return artifact.exists() and self.data.get(str(artifact)) == digest(key)

    def record(self, artifact: Path, key: Any) -> None:
        self.data[str(artifact)] = digest(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))
