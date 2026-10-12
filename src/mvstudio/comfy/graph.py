"""Small helper around an API-format workflow dict: locate nodes by role, patch, prune."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..config import repo_root

DEFAULT_DIR = repo_root() / "workflows" / "api"


class GraphError(RuntimeError):
    pass


class ApiGraph:
    def __init__(self, nodes: dict[str, dict[str, Any]]):
        self.nodes = copy.deepcopy(nodes)

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> ApiGraph:
        path = (directory or DEFAULT_DIR) / f"{name}.json"
        if not path.is_file():
            raise GraphError(f"workflow template missing: {path} (run scripts/sync_templates.py)")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # ---------------------------------------------------------------- lookup
    def find(self, class_type: str | None = None, title: str | None = None) -> list[str]:
        out = []
        for k, v in self.nodes.items():
            if class_type and v["class_type"] != class_type:
                continue
            if title is not None and v.get("_meta", {}).get("title") != title:
                continue
            out.append(k)
        return sorted(out)

    def one(self, class_type: str, title: str | None = None) -> str:
        hits = self.find(class_type, title)
        if len(hits) != 1:
            raise GraphError(
                f"expected exactly one {class_type}{f' titled {title!r}' if title else ''} in "
                f"workflow, found {len(hits)}: the upstream template changed; adapt "
                f"mvstudio/comfy/workflows.py")
        return hits[0]

    # ---------------------------------------------------------------- patching
    def set(self, key: str, **inputs: Any) -> None:
        self.nodes[key]["inputs"].update(inputs)

    def drop_input(self, key: str, name: str) -> None:
        self.nodes[key]["inputs"].pop(name, None)

    def bypass_switch(self, key: str, take: str) -> None:
        """Replace a ComfySwitchNode by one of its branches (`on_true` / `on_false`)."""
        node = self.nodes[key]
        if node["class_type"] != "ComfySwitchNode":
            raise GraphError(f"{key} is {node['class_type']}, not a ComfySwitchNode")
        branch = node["inputs"][take]
        for n in self.nodes.values():
            for name, v in n["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and v[0] == key:
                    n["inputs"][name] = list(branch)
        del self.nodes[key]

    def prune(self) -> None:
        outs = [k for k, v in self.nodes.items() if v["class_type"].startswith("Save")]
        keep: set[str] = set()
        stack = list(outs)
        while stack:
            k = stack.pop()
            if k in keep or k not in self.nodes:
                continue
            keep.add(k)
            for v in self.nodes[k]["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    stack.append(v[0])
        self.nodes = {k: v for k, v in self.nodes.items() if k in keep}

    def finalize(self) -> dict[str, dict[str, Any]]:
        """Ready for POST /prompt: pruned, no `_meta`, no null-valued inputs."""
        self.prune()
        out: dict[str, dict[str, Any]] = {}
        for k, v in self.nodes.items():
            out[k] = {"class_type": v["class_type"],
                      "inputs": {a: b for a, b in v["inputs"].items() if b is not None}}
        return out

    def referenced_files(self) -> dict[str, list[str]]:
        """{input_name: [file names]} for model-like inputs, used by doctor/model checks."""
        keys = ("ckpt_name", "unet_name", "vae_name", "clip_name", "text_encoder", "lora_name",
                "model_name")
        out: dict[str, list[str]] = {}
        for v in self.nodes.values():
            for name in keys:
                val = v["inputs"].get(name)
                if isinstance(val, str):
                    out.setdefault(name, []).append(val)
        return out
