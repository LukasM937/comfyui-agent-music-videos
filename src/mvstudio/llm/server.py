"""Start/stop llama-server and hand the GPU back and forth with ComfyUI.

A 5090 has 32 GB. Qwen3.8-27B Q4 (~16.5 GB + KV cache + vision projector) and LTX/Qwen-Image
(~20 GB weights streamed from RAM, plus activations) do not coexist reliably, so the pipeline is
strictly staged: plan with the LLM, shut it down, then render."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse

import requests

from ..comfy.client import ComfyClient
from ..config import Config
from .client import LLM, LLMError


def server_command(cfg: Config) -> list[str]:
    u = urlparse(cfg.llm.url)
    cmd = [
        cfg.llm.binary, "-m", str(cfg.llm.gguf), "-c", str(cfg.llm.ctx),
        "-ngl", str(cfg.llm.n_gpu_layers), "--host", u.hostname or "127.0.0.1",
        "--port", str(u.port or 8080), "--alias", cfg.llm.model, "--jinja", "-np", "1",
    ]
    if cfg.llm.mmproj and cfg.llm.mmproj.is_file():
        cmd += ["--mmproj", str(cfg.llm.mmproj)]
    return cmd


class LlamaServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self, wait_s: float = 600.0) -> None:
        if shutil.which(self.cfg.llm.binary) is None:
            raise LLMError(f"'{self.cfg.llm.binary}' not found in PATH (run scripts/setup.sh)")
        if not self.cfg.llm.gguf.is_file():
            raise LLMError(f"model file missing: {self.cfg.llm.gguf} (run scripts/setup.sh models)")
        self.proc = subprocess.Popen(server_command(self.cfg), stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        llm = LLM(self.cfg.llm)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.proc.poll() is not None:
                raise LLMError(f"llama-server exited with code {self.proc.returncode}; run "
                               f"`{' '.join(server_command(self.cfg))}` by hand to see why")
            if llm.alive():
                return
            time.sleep(1.0)
        self.stop()
        raise LLMError("llama-server did not become healthy in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


@contextmanager
def llm_session(cfg: Config, log=print) -> Iterator[LLM]:  # noqa: ANN001
    """Yield a ready LLM. Starts llama-server (after freeing ComfyUI's VRAM) unless one is already
    running, and stops it afterwards if we started it and `stop_after_stage` is set."""
    llm = LLM(cfg.llm)
    server: LlamaServer | None = None
    if not llm.alive():
        comfy = ComfyClient(cfg.comfy.url)
        if comfy.alive():
            try:
                comfy.free()
            except requests.RequestException:
                pass
        log("llm: starting llama-server ...")
        server = LlamaServer(cfg)
        server.start()
        log("llm: ready")
    try:
        yield llm
    finally:
        if server is not None and cfg.llm.stop_after_stage:
            log("llm: stopping llama-server (GPU goes back to ComfyUI)")
            server.stop()
