"""OpenAI-compatible chat client with schema-constrained JSON output (llama.cpp `llama-server`)."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from ..config import LlmCfg

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMLike(Protocol):
    def generate(self, schema: type[T], system: str, user: str, *,
                 images: list[Path] | None = None, thinking: bool | None = None,
                 temperature: float | None = None) -> T: ...


def _data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLM:
    def __init__(self, cfg: LlmCfg):
        self.cfg = cfg
        self.http = requests.Session()
        self.calls = 0

    def alive(self) -> bool:
        try:
            base = self.cfg.url.removesuffix("/v1")
            return self.http.get(base + "/health", timeout=3).ok
        except requests.RequestException:
            return False

    def _post(self, body: dict[str, Any]) -> str:
        r = self.http.post(f"{self.cfg.url}/chat/completions", json=body,
                           timeout=self.cfg.timeout_s)
        if not r.ok:
            raise LLMError(f"LLM server returned {r.status_code}: {r.text[:300]}")
        msg = r.json()["choices"][0]["message"]
        return msg.get("content") or ""

    def generate(self, schema: type[T], system: str, user: str, *,
                 images: list[Path] | None = None, thinking: bool | None = None,
                 temperature: float | None = None, retries: int = 2) -> T:
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": _data_url(p)}} for p in images]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        use_thinking = self.cfg.thinking if thinking is None else thinking
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "chat_template_kwargs": {"enable_thinking": use_thinking},
            "messages": messages,
        }
        if not use_thinking:
            # Grammar-constrained decoding guarantees parseable JSON. It conflicts with a free
            # thinking block, so thinking calls rely on validation + retry instead.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        else:
            messages[0]["content"] += (
                "\n\nAnswer with a single JSON object matching this JSON Schema, nothing else:\n"
                + json.dumps(schema.model_json_schema()))
        last_err = ""
        for _ in range(retries + 1):
            self.calls += 1
            raw = self._post(body)
            text = _FENCE.sub("", raw.strip())
            if "</think>" in text:  # servers that inline the reasoning
                text = text.split("</think>", 1)[1].strip()
            try:
                return schema.model_validate_json(text)
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:600]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                                 f"That did not validate: {last_err}\nReturn corrected JSON only."})
        raise LLMError(f"{schema.__name__}: no valid JSON after {retries + 1} attempts: {last_err}")
