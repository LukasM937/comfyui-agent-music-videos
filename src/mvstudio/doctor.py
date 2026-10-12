"""Read-only environment checks: `mvs doctor`. Fast, side-effect free, says how to fix things."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

from . import modelstore
from .comfy.client import ComfyClient
from .comfy.validate import validate
from .comfy.workflows import ImageJob, VideoJob, build_image, build_video
from .config import Config

MIN_DRIVER = 570  # first branch with Blackwell (sm_120) support
MIN_COMFY = (0, 39, 0)  # versions below lack the LTX-2.5 / Qwen-Image-2.1 nodes we use


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | skip
    detail: str = ""
    fix: str = ""


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def check_os() -> list[Check]:
    osr = {}
    try:
        for line in open("/etc/os-release", encoding="utf-8"):  # noqa: SIM115
            k, _, v = line.strip().partition("=")
            osr[k] = v.strip('"')
    except OSError:
        pass
    pretty = osr.get("PRETTY_NAME", platform.platform())
    out = [Check("OS", "ok" if osr.get("ID") == "arch" else "warn",
                 f"{pretty}, kernel {platform.release()}",
                 "" if osr.get("ID") == "arch" else "scripts/setup.sh targets Arch; other distros "
                 "need the package names adapted")]
    v = sys.version_info
    out.append(Check("Python", "ok" if v >= (3, 12) else "fail", f"{v.major}.{v.minor}.{v.micro}",
                     "" if v >= (3, 12) else "need Python >= 3.12 (pacman -S python)"))
    return out


def check_gpu() -> list[Check]:
    q = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
              "--format=csv,noheader,nounits"])
    if not q:
        return [Check("GPU", "fail", "nvidia-smi not available or no GPU",
                      "install the NVIDIA driver: pacman -S nvidia-open-dkms nvidia-utils "
                      "(RTX 50xx needs the open kernel modules)")]
    out = []
    for i, line in enumerate(q.splitlines()):
        name, mem, drv, cc = [x.strip() for x in line.split(",")]
        mem_gb = float(mem) / 1024
        st = "ok" if mem_gb >= 30 else "warn" if mem_gb >= 22 else "fail"
        out.append(Check(f"GPU {i}", st, f"{name}, {mem_gb:.0f} GB, driver {drv}, sm_{cc}",
                         "" if st == "ok" else "pipeline is tuned for 32 GB; expect offloading"))
        major = int(drv.split(".")[0]) if drv.split(".")[0].isdigit() else 0
        if cc.startswith("12") and major < MIN_DRIVER:
            out.append(Check("Driver", "fail", f"{drv} is too old for Blackwell",
                             f"need >= {MIN_DRIVER}: pacman -Syu nvidia-open-dkms nvidia-utils"))
    return out


def check_resources(cfg: Config) -> list[Check]:
    out = []
    try:
        kb = int(next(ln for ln in open("/proc/meminfo", encoding="utf-8")  # noqa: SIM115
                      if ln.startswith("MemTotal")).split()[1])
        gb = kb / 1024 / 1024
        out.append(Check("RAM", "ok" if gb >= 96 else "warn" if gb >= 60 else "fail",
                         f"{gb:.0f} GB", "" if gb >= 96 else "weights are streamed from RAM; "
                         "64 GB works for one model at a time, 128 GB is comfortable"))
    except (OSError, StopIteration):
        out.append(Check("RAM", "skip", "cannot read /proc/meminfo"))
    target = cfg.comfy.dir if cfg.comfy.dir.exists() else cfg.comfy.dir.parent
    if target.exists():
        free = shutil.disk_usage(target).free / 1e9
        out.append(Check("Disk", "ok" if free >= 150 else "warn" if free >= 60 else "fail",
                         f"{free:.0f} GB free at {target}",
                         "" if free >= 150 else "core+lipsync models need ~110 GB, outputs more"))
    return out


def check_tools(cfg: Config) -> list[Check]:
    out = []
    for tool, fix in [("ffmpeg", "pacman -S ffmpeg"), ("ffprobe", "pacman -S ffmpeg"),
                      ("git", "pacman -S git"), ("uv", "pacman -S uv")]:
        p = shutil.which(tool)
        out.append(Check(tool, "ok" if p else "fail", p or "not found", "" if p else fix))
    p = shutil.which(cfg.llm.binary)
    out.append(Check("llama-server", "ok" if p else "fail", p or "not found",
                     "" if p else "scripts/setup.sh builds llama.cpp with CUDA (sm_120)"))
    return out


def check_comfy(cfg: Config) -> tuple[list[Check], dict | None]:  # type: ignore[type-arg]
    out = []
    d = cfg.comfy.dir
    out.append(Check("ComfyUI dir", "ok" if (d / "main.py").exists() else "fail", str(d),
                     "" if (d / "main.py").exists() else "scripts/setup.sh comfyui"))
    client = ComfyClient(cfg.comfy.url)
    if not client.alive():
        out.append(Check("ComfyUI server", "warn", f"not reachable at {cfg.comfy.url}",
                         "systemctl --user start comfyui  (or: cd ~/ComfyUI && .venv/bin/python "
                         "main.py)"))
        return out, None
    s = client.system_stats()
    ver = s["system"].get("comfyui_version", "?")
    try:
        tup = tuple(int(x) for x in ver.split(".")[:3])
    except ValueError:
        tup = (0, 0, 0)
    out.append(Check("ComfyUI server", "ok" if tup >= MIN_COMFY else "fail",
                     f"v{ver}, torch {s['system'].get('pytorch_version', '?')}",
                     "" if tup >= MIN_COMFY else "git pull in the ComfyUI dir; need >= 0.39"))
    for dev in s.get("devices", []):
        out.append(Check("ComfyUI device", "ok" if dev["type"] == "cuda" else "fail",
                         f"{dev['name']}, {dev['vram_total'] / 2**30:.0f} GiB",
                         "" if dev["type"] == "cuda" else "ComfyUI is running on CPU"))
    return out, client.object_info()


def check_workflows(cfg: Config, oi: dict) -> list[Check]:  # type: ignore[type-arg]
    jobs = {
        "ltx-2.5 i2v": build_video(VideoJob(image="a.png", prompt="x", width=1280, height=704,
                                            frames=97, fps=24, seed=1), cfg),
        "ltx-2.3 image+audio": build_video(VideoJob(image="a.png", audio="a.wav", prompt="x",
                                                    width=1280, height=704, frames=97, fps=24,
                                                    seed=1, family="ltx-2.3"), cfg),
        "qwen-image t2i": build_image(ImageJob(prompt="x", width=1024, height=1024, seed=1), cfg),
        "qwen-image refs": build_image(ImageJob(prompt="x", width=1024, height=1024, seed=1,
                                                refs=["a.png"]), cfg),
    }
    out = []
    for name, wf in jobs.items():
        problems = [p for p in validate(wf, oi)
                    if "model file missing" not in p]  # missing models are reported separately
        out.append(Check(f"workflow {name}", "ok" if not problems else "fail",
                         "valid against this ComfyUI" if not problems else "; ".join(problems[:3]),
                         "" if not problems else "scripts/sync_templates.py after updating ComfyUI"))
    return out


def check_models(cfg: Config, groups: list[str]) -> list[Check]:
    manifest = modelstore.select(modelstore.load_manifest(), groups)
    out, missing_gb = [], 0.0
    by_group: dict[str, list[modelstore.Model]] = {}
    for m in manifest:
        by_group.setdefault(m.group, []).append(m)
    for g, ms in by_group.items():
        bad = [m for m in ms if modelstore.status(m, cfg) != "ok"]
        missing_gb += sum(m.gb for m in bad)
        if not bad:
            out.append(Check(f"models: {g}", "ok", f"{len(ms)} files, {sum(m.gb for m in ms):.0f} GB"))
        else:
            out.append(Check(f"models: {g}", "warn",
                             f"{len(bad)}/{len(ms)} missing: {', '.join(m.id for m in bad[:4])}"
                             f"{'...' if len(bad) > 4 else ''} ({sum(m.gb for m in bad):.0f} GB)",
                             f"scripts/setup.sh models {g}"))
    if os.environ.get("HF_TOKEN") is None and missing_gb:
        out.append(Check("HF_TOKEN", "warn", "not set",
                         "LTX-2.x repos are gated: accept the licence on huggingface.co and "
                         "export HF_TOKEN"))
    return out


def run_all(cfg: Config, groups: list[str] | None = None, workflows: bool = True
            ) -> list[Check]:
    groups = groups or ["core", "lipsync", "asr"]
    checks = check_os() + check_gpu() + check_resources(cfg) + check_tools(cfg)
    comfy_checks, oi = check_comfy(cfg)
    checks += comfy_checks
    if workflows and oi is not None:
        checks += check_workflows(cfg, oi)
    elif workflows:
        checks.append(Check("workflows", "skip", "ComfyUI not running: cannot validate nodes"))
    checks += check_models(cfg, groups)
    return checks
