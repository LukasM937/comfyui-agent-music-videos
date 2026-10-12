"""Model manifest: where each file lives, whether it is complete, and how to fetch it."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Config, repo_root

Log = Callable[[str], None]


@dataclass(frozen=True)
class Model:
    id: str
    group: str
    repo: str
    dest: str
    size: int
    file: str = ""
    kind: str = "file"
    gated: bool = False
    note: str = ""

    @property
    def gb(self) -> float:
        return self.size / 1e9


def load_manifest(path: Path | None = None) -> list[Model]:
    path = path or repo_root() / "config" / "models.toml"
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return [Model(**m) for m in data["model"]]


def llm_dir(cfg: Config) -> Path:
    return cfg.llm.gguf.parent


def dest_path(m: Model, cfg: Config) -> Path | None:
    scheme, _, rel = m.dest.partition(":")
    if scheme == "comfy":
        return cfg.comfy.dir / "models" / rel
    if scheme == "llm":
        return llm_dir(cfg) / rel
    return None  # hf-cache snapshots are located by huggingface_hub


def status(m: Model, cfg: Config) -> str:
    if m.kind == "snapshot":
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(m.repo, local_files_only=True)
            return "ok"
        except Exception:  # noqa: BLE001 - any failure means 'not cached'
            return "missing"
    p = dest_path(m, cfg)
    if p is None or not p.exists():
        return "missing"
    size = p.stat().st_size
    if size == m.size:
        return "ok"
    return "partial" if size < m.size else "size-mismatch"


def download(m: Model, cfg: Config, log: Log = print) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        if m.kind == "snapshot":
            log(f"  {m.id}: snapshot {m.repo}")
            snapshot_download(m.repo, token=token)
            return
        dst = dest_path(m, cfg)
        assert dst is not None
        dst.parent.mkdir(parents=True, exist_ok=True)
        stage = dst.parent / ".hf-staging"
        log(f"  {m.id}: {m.repo}/{m.file} ({m.gb:.1f} GB)")
        got = hf_hub_download(m.repo, m.file, token=token, local_dir=stage)
        os.replace(got, dst)
    except GatedRepoError as e:
        raise SystemExit(
            f"{m.repo} is gated. Open https://huggingface.co/{m.repo}, accept the licence, and "
            f"export HF_TOKEN=<token with read access>. ({e})") from e
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403):
            raise SystemExit(
                f"{m.repo}: access denied. Accept the licence at https://huggingface.co/{m.repo} "
                "and set HF_TOKEN.") from e
        raise


def select(manifest: list[Model], groups: list[str] | None) -> list[Model]:
    if not groups:
        return [m for m in manifest if m.group == "core"]
    if "all" in groups:
        return manifest
    return [m for m in manifest if m.group in groups]
