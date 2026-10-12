"""`mvs selftest`: real inference smoke tests on tiny settings. Proves that driver, ComfyUI,
models, node versions and the patched workflows work *together* on this machine."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .comfy.client import ComfyClient, ComfyError
from .comfy.workflows import ImageJob, VideoJob, build_image, build_video, valid_frames
from .config import Config
from .doctor import Check
from .llm.server import llm_session

Log = Callable[[str], None]
STEPS = ["llm", "image", "image-refs", "video-ltx25", "video-ltx23-audio"]


class Pong(BaseModel):
    answer: Literal["pong"]
    number: int


def _probe(path: Path) -> dict[str, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_read_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return {"w": s["width"], "h": s["height"], "frames": s["nb_read_frames"],
            "dur": j["format"]["duration"]}


def run(cfg: Config, only: list[str] | None = None, log: Log = print) -> list[Check]:
    steps = only or STEPS
    results: list[Check] = []
    tmp = Path(tempfile.mkdtemp(prefix="mvs-selftest-"))
    client = ComfyClient(cfg.comfy.url, cfg.comfy.poll_interval_s, cfg.comfy.job_timeout_s)

    def step(name: str, fn: Callable[[], str]) -> None:
        if name not in steps:
            return
        t0 = time.time()
        log(f"selftest: {name} ...")
        try:
            detail = fn()
            results.append(Check(name, "ok", f"{detail} ({time.time() - t0:.0f}s)"))
        except FileNotFoundError as e:
            results.append(Check(name, "skip", str(e)))
        except (ComfyError, RuntimeError, OSError, subprocess.CalledProcessError) as e:
            results.append(Check(name, "fail", f"{type(e).__name__}: {e}"[:400],
                                 "run `mvs doctor`; check the ComfyUI console for the traceback"))

    def t_llm() -> str:
        if not cfg.llm.gguf.exists():
            raise FileNotFoundError(f"{cfg.llm.gguf} not downloaded (scripts/setup.sh models)")
        with llm_session(cfg, log) as llm:
            r = llm.generate(Pong, "Reply with JSON only.",
                             'Return answer "pong" and the number 7.')
        return f"schema-constrained JSON ok ({r.model_dump_json()})"

    img = tmp / "apple.png"

    def need_comfy() -> None:
        if not client.alive():
            raise FileNotFoundError(f"ComfyUI not running at {cfg.comfy.url}")

    def t_image() -> str:
        need_comfy()
        job = ImageJob(prompt="A red apple on a wooden table, soft window light.", width=512,
                       height=512, seed=1, steps=8, prefix="mvs/selftest/apple")
        client.run(build_image(job, cfg), img)
        return f"{img.stat().st_size // 1024} KB png"

    def t_refs() -> str:
        need_comfy()
        if not img.exists():
            t_image()
        up = client.upload(img, "mvs_selftest_apple.png")
        job = ImageJob(prompt="The apple from <image1> on a marble kitchen counter.", width=512,
                       height=512, seed=2, steps=8, refs=[up], prefix="mvs/selftest/apple_ref")
        dst = tmp / "apple_ref.png"
        client.run(build_image(job, cfg), dst)
        return f"{dst.stat().st_size // 1024} KB png"

    def t_ltx25() -> str:
        need_comfy()
        if not img.exists():
            t_image()
        up = client.upload(img, "mvs_selftest_apple.png")
        frames = valid_frames(2.0, cfg.video.fps)
        job = VideoJob(image=up, prompt="The camera slowly pushes in on the apple. Soft light.",
                       width=512, height=288, frames=frames, fps=cfg.video.fps, seed=3,
                       prefix="mvs/selftest/ltx25")
        dst = tmp / "ltx25.mp4"
        client.run(build_video(job, cfg), dst)
        p = _probe(dst)
        if int(p["frames"]) < frames - 1:
            raise RuntimeError(f"expected {frames} frames, got {p['frames']}")
        return f"{p['w']}x{p['h']}, {p['frames']} frames"

    def t_ltx23() -> str:
        need_comfy()
        if not img.exists():
            t_image()
        wav = tmp / "tone.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=330:duration=3", "-ac", "2", "-ar", "44100", str(wav)],
                       check=True)
        up_i = client.upload(img, "mvs_selftest_apple.png")
        up_a = client.upload(wav, "mvs_selftest_tone.wav")
        frames = valid_frames(2.0, cfg.video.fps)
        job = VideoJob(image=up_i, audio=up_a, prompt="The apple pulses gently with the sound.",
                       width=512, height=288, frames=frames, fps=cfg.video.fps, seed=4,
                       family="ltx-2.3", prefix="mvs/selftest/ltx23a")
        dst = tmp / "ltx23a.mp4"
        client.run(build_video(job, cfg), dst)
        p = _probe(dst)
        return f"{p['w']}x{p['h']}, {p['frames']} frames"

    step("llm", t_llm)
    step("image", t_image)
    step("image-refs", t_refs)
    step("video-ltx25", t_ltx25)
    step("video-ltx23-audio", t_ltx23)
    log(f"selftest outputs kept in {tmp}")
    return results
