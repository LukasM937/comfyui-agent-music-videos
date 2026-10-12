"""Runtime configuration.

Layering (later wins): built-in defaults -> config/default.toml -> ~/.config/mvstudio/config.toml
-> <project>/project.toml -> MVS_* environment variables for a few common knobs.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ComfyCfg(BaseModel):
    url: str = "http://127.0.0.1:8188"
    # Directory of the ComfyUI checkout (needed by setup/doctor, not by the HTTP client).
    dir: Path = Path.home() / "ComfyUI"
    poll_interval_s: float = 2.0
    job_timeout_s: float = 3600.0

    @field_validator("dir", mode="after")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()


class LlmCfg(BaseModel):
    """OpenAI-compatible endpoint (llama.cpp `llama-server`)."""

    url: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3.8-27b"
    # llama-server settings used by `mvs llm up`
    gguf: Path = Path.home() / "models" / "llm" / "Qwen3.8-27B-UD-Q4_K_M.gguf"
    mmproj: Path | None = Path.home() / "models" / "llm" / "mmproj-F16.gguf"
    ctx: int = 32768
    n_gpu_layers: int = 999
    binary: str = "llama-server"
    # Qwen3.x thinking is useful for the critic, wasteful for schema-constrained generation.
    thinking: bool = False
    temperature: float = 0.7
    timeout_s: float = 900.0
    # Stop llama-server again after a stage if *we* started it, so ComfyUI gets the whole GPU.
    stop_after_stage: bool = True

    @field_validator("gguf", "mmproj", mode="after")
    @classmethod
    def _expand(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None


class AsrCfg(BaseModel):
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None  # None = auto-detect


class AudioCfg(BaseModel):
    sr: int = 22050
    hop: int = 512
    beats_per_bar: int = 4
    # Smallest structural section we are willing to report (bars).
    min_section_bars: int = 4
    asr: AsrCfg = Field(default_factory=AsrCfg)


class ImageCfg(BaseModel):
    """Keyframe / character / location generation (Qwen-Image-2.1 via ComfyUI)."""

    width: int = 1920
    height: int = 1088  # multiple of 32
    steps: int = 25
    unet: str = "qwen_image_2.1_int8_convrot.safetensors"
    clip: str = "qwen3vl_8b_int8_convrot.safetensors"
    vae: str = "qwen_image_2.1_vae_bf16.safetensors"
    max_refs: int = 4  # character/location reference images per shot


class VideoCfg(BaseModel):
    # Family used for ordinary (non-lipsync) shots. ltx-2.5 = best visual quality.
    default_family: Literal["ltx-2.5", "ltx-2.3"] = "ltx-2.5"
    # Family used for shots that must follow the vocals (image+audio -> video).
    # Only LTX-2.3 ships an official image+audio template at the time of writing.
    audio_family: Literal["ltx-2.3"] = "ltx-2.3"
    fps: int = 24
    width: int = 1280
    height: int = 704  # multiple of 32; 1280x720 is not (720 % 32 != 0)
    # Hard upper bound for one generated clip. LTX handles ~20 s, but VRAM/time grow fast.
    max_clip_s: float = 10.0
    min_shot_s: float = 1.5
    audio_conditioning: Literal["auto", "always", "never"] = "auto"
    seed: int = 1234


class PlanningCfg(BaseModel):
    cut_style: Literal["calm", "medium", "fast"] = "medium"
    # identical: reuse the clip of the first chorus when repeat is timing-identical
    # variants : same keyframes, new motion/seed per occurrence
    # escalate : like variants, but the planner is told to raise intensity each time
    refrain_mode: Literal["identical", "variants", "escalate"] = "escalate"
    max_revision_rounds: int = 3
    beat_tolerance_s: float = 0.12
    # Ratio of performance (singer on camera) shots in vocal sections, 0..1
    performance_ratio: float = 0.35


class Config(BaseModel):
    comfy: ComfyCfg = Field(default_factory=ComfyCfg)
    llm: LlmCfg = Field(default_factory=LlmCfg)
    audio: AudioCfg = Field(default_factory=AudioCfg)
    image: ImageCfg = Field(default_factory=ImageCfg)
    video: VideoCfg = Field(default_factory=VideoCfg)
    planning: PlanningCfg = Field(default_factory=PlanningCfg)


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def repo_root() -> Path:
    """Directory that contains config/ and workflows/ (the repo checkout)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "config" / "default.toml").is_file():
            return p
    return here.parents[2]


def load_config(project_dir: Path | None = None) -> Config:
    data: dict[str, Any] = {}
    data = _deep_merge(data, _read_toml(repo_root() / "config" / "default.toml"))
    data = _deep_merge(data, _read_toml(Path.home() / ".config" / "mvstudio" / "config.toml"))
    if project_dir is not None:
        data = _deep_merge(data, _read_toml(project_dir / "project.toml"))

    env = os.environ
    if env.get("MVS_COMFY_URL"):
        data = _deep_merge(data, {"comfy": {"url": env["MVS_COMFY_URL"]}})
    if env.get("MVS_COMFY_DIR"):
        data = _deep_merge(data, {"comfy": {"dir": env["MVS_COMFY_DIR"]}})
    if env.get("MVS_LLM_URL"):
        data = _deep_merge(data, {"llm": {"url": env["MVS_LLM_URL"]}})
    return Config.model_validate(data)
