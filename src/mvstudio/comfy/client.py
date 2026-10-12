"""Minimal ComfyUI HTTP client (queue, poll, download). No websocket dependency on purpose:
polling /history is enough for jobs that run minutes, and survives reconnects."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str, poll_interval_s: float = 2.0, timeout_s: float = 3600.0,
                 comfy_dir: Path | None = None):
        self.url = url.rstrip("/")
        self.poll = poll_interval_s
        self.timeout = timeout_s
        self.comfy_dir = comfy_dir
        self.client_id = uuid.uuid4().hex
        self.http = requests.Session()

    # ------------------------------------------------------------------ basics
    def alive(self) -> bool:
        try:
            return self.http.get(f"{self.url}/system_stats", timeout=5).ok
        except requests.RequestException:
            return False

    def system_stats(self) -> dict[str, Any]:
        r = self.http.get(f"{self.url}/system_stats", timeout=10)
        r.raise_for_status()
        return r.json()

    def object_info(self, node: str | None = None) -> dict[str, Any]:
        path = f"/object_info/{node}" if node else "/object_info"
        r = self.http.get(self.url + path, timeout=120)
        r.raise_for_status()
        return r.json()

    def free(self, unload_models: bool = True) -> None:
        """Drop models from VRAM, e.g. before the LLM server needs the GPU."""
        self.http.post(f"{self.url}/free",
                       json={"unload_models": unload_models, "free_memory": True}, timeout=30)

    # ------------------------------------------------------------------ files
    def upload(self, path: Path, name: str | None = None, overwrite: bool = True) -> str:
        """Upload an image or audio file to ComfyUI's input dir; returns the stored file name.

        `/upload/image` accepts any file type; LoadAudio reads from the same input folder."""
        name = name or path.name
        with path.open("rb") as fh:
            r = self.http.post(
                f"{self.url}/upload/image",
                files={"image": (name, fh)},
                data={"overwrite": "true" if overwrite else "false", "type": "input"},
                timeout=300,
            )
        if not r.ok:
            raise ComfyError(f"upload of {path.name} failed: {r.status_code} {r.text[:200]}")
        return r.json()["name"]

    def download(self, ref: dict[str, str], dst: Path) -> Path:
        r = self.http.get(f"{self.url}/view", params={
            "filename": ref["filename"], "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output")}, timeout=600, stream=True)
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        return dst

    # ------------------------------------------------------------------ jobs
    def queue(self, prompt: dict[str, Any]) -> str:
        r = self.http.post(f"{self.url}/prompt",
                           json={"prompt": prompt, "client_id": self.client_id}, timeout=60)
        if not r.ok:
            raise ComfyError(self._explain(r))
        return r.json()["prompt_id"]

    @staticmethod
    def _explain(r: requests.Response) -> str:
        try:
            j = r.json()
        except ValueError:
            return f"{r.status_code}: {r.text[:300]}"
        lines = [j.get("error", {}).get("message", str(j)[:200])]
        for nid, e in (j.get("node_errors") or {}).items():
            for er in e.get("errors", []):
                lines.append(f"  node {nid} ({e.get('class_type')}): {er.get('message')} "
                             f"{er.get('details', '')}"[:240])
        return "\n".join(lines)

    def wait(self, prompt_id: str, on_tick: Callable[[float], None] | None = None
             ) -> dict[str, Any]:
        t0 = time.time()
        while True:
            r = self.http.get(f"{self.url}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            h = r.json().get(prompt_id)
            if h:
                st = h.get("status", {})
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[-1][1] if msgs else {}
                    raise ComfyError(
                        f"execution failed in node {detail.get('node_id')} "
                        f"({detail.get('node_type')}): {detail.get('exception_message', '?')}")
                if st.get("completed") or h.get("outputs"):
                    return h
            if time.time() - t0 > self.timeout:
                raise ComfyError(f"timeout after {self.timeout:.0f}s waiting for {prompt_id}")
            if on_tick:
                on_tick(time.time() - t0)
            time.sleep(self.poll)

    def run(self, prompt: dict[str, Any], dst: Path, on_tick: Callable[[float], None] | None = None
            ) -> Path:
        """Queue, wait, and download the first file produced by a Save* node."""
        pid = self.queue(prompt)
        hist = self.wait(pid, on_tick)
        for out in hist.get("outputs", {}).values():
            for files in out.values():
                if isinstance(files, list):
                    for f in files:
                        if isinstance(f, dict) and "filename" in f and f.get("type") == "output":
                            return self.download(f, dst)
        raise ComfyError(f"job {pid} finished without a saved output")
