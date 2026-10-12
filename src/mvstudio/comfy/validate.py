"""Offline validation of an API workflow against a running ComfyUI's /object_info.

`/prompt` would validate too, but it also *queues* the job. This check is side-effect free, so
`mvs doctor` can run it on every workflow after each ComfyUI update."""

from __future__ import annotations

from typing import Any

DYNAMIC = {"COMFY_DYNAMICCOMBO_V3", "COMFY_AUTOGROW_V3", "COMFY_MATCHTYPE_V3"}


def _options(typ: Any, opts: dict[str, Any]) -> list[Any] | None:
    if isinstance(typ, list):
        return typ
    if typ == "COMBO":
        return list(opts.get("options", []))
    return None


def validate(wf: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for key, node in wf.items():
        cls = node["class_type"]
        spec = object_info.get(cls)
        if spec is None:
            problems.append(f"{key}: node type {cls} is not installed in ComfyUI")
            continue
        req = spec["input"].get("required", {})
        opt = spec["input"].get("optional", {})
        allowed = {**req, **opt}
        inputs = node["inputs"]
        for name in req:
            typ = req[name][0]
            if name not in inputs and not (isinstance(typ, str) and typ in DYNAMIC):
                problems.append(f"{key} ({cls}): required input '{name}' is missing")
        for name, val in inputs.items():
            base = name.split(".")[0]
            if base not in allowed:
                problems.append(f"{key} ({cls}): unknown input '{name}'")
                continue
            if "." in name or (isinstance(allowed[base][0], str) and allowed[base][0] in DYNAMIC):
                continue
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] not in wf:
                    problems.append(f"{key}.{name}: link to missing node {val[0]}")
                continue
            typ = allowed[name][0]
            opts = allowed[name][1] if len(allowed[name]) > 1 else {}
            choices = _options(typ, opts)
            if choices is not None and val not in choices:
                problems.append(f"{key} ({cls}).{name}: '{val}' not in {len(choices)} available "
                                "choices (model file missing?)")
            elif typ in ("INT", "FLOAT") and isinstance(val, (int, float)):
                lo, hi = opts.get("min"), opts.get("max")
                if (lo is not None and val < lo) or (hi is not None and val > hi):
                    problems.append(f"{key} ({cls}).{name}: {val} outside [{lo}, {hi}]")
    return problems
