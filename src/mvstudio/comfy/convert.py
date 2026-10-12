"""Convert a ComfyUI *UI-format* workflow (with subgraphs) into the flat *API format*.

Why: the official templates (Comfy-Org/workflow_templates) are the best-maintained recipes for
new models, but they are saved for the editor: nodes sit in subgraphs, widget values are
positional lists, notes/previews are mixed in. `/prompt` wants a flat {id: {class_type, inputs}}.
This module flattens a template deterministically so `scripts/sync_templates.py` can re-generate
`workflows/api/*.json` whenever upstream changes, instead of us hand-copying node graphs.

Only what the templates actually use is implemented: subgraphs (nested), Reroute, bypass/mute,
`widgets_values_named`, positional fallback via `object_info`. Anything unexpected raises.
"""

from __future__ import annotations

from typing import Any

# Never part of an API prompt.
FRONTEND_ONLY = {"MarkdownNote", "Note", "PrimitiveNode", "Reroute", "ImageCompare",
                 "PreviewImage", "RecordAudio"}
# Widget keys that exist only in the editor.
FRONTEND_WIDGET_KEYS = {"control_after_generate", "upload"}
OUTPUT_PREFIXES = ("Save",)  # SaveVideo, SaveImage, SaveImageAdvanced, SaveAudio...

BOUNDARY_IN = -10
BOUNDARY_OUT = -20

Ref = tuple[str, str, int]  # ("link", node_key, slot)
Val = tuple[str, Any]  # ("value", python value)


class ConvertError(RuntimeError):
    pass


def _spec_items(input_spec: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    items = []
    for section in ("required", "optional"):
        for name, d in input_spec.get(section, {}).items():
            items.append((name, d[0], d[1] if len(d) > 1 else {}))
    return items


def _is_widget(typ: Any, opts: dict[str, Any]) -> bool:
    if opts.get("forceInput"):
        return False
    return (isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
            or bool(opts.get("widgetType")))


def _links_by_id(links: list[Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for ln in links:
        if isinstance(ln, list):  # [id, origin_id, origin_slot, target_id, target_slot, type]
            out[ln[0]] = {"id": ln[0], "origin_id": ln[1], "origin_slot": ln[2],
                          "target_id": ln[3], "target_slot": ln[4], "type": ln[5]}
        else:
            out[ln["id"]] = ln
    return out


class _Graph:
    """One graph level (top level or a subgraph definition)."""

    def __init__(self, nodes: list[dict[str, Any]], links: list[Any], inputs: list[dict[str, Any]],
                 prefix: str, parent_inputs: dict[str, Any] | None):
        self.nodes = {n["id"]: n for n in nodes}
        self.links = _links_by_id(links)
        self.inputs_def = inputs  # subgraph input slots (empty at top level)
        self.prefix = prefix
        # resolved value/ref for each subgraph input slot name, provided by the parent
        self.parent_inputs = parent_inputs or {}


class Converter:
    def __init__(self, ui: dict[str, Any], object_info: dict[str, Any] | None = None):
        self.ui = ui
        self.oi = object_info or {}
        self.subgraphs = {sg["id"]: sg for sg in ui.get("definitions", {}).get("subgraphs", [])}
        self.api: dict[str, dict[str, Any]] = {}
        # (node_key) -> {output_slot: ("link", key, slot)}  for pass-through nodes (Reroute/bypass)
        self._passthrough: dict[str, dict[int, Ref]] = {}
        # subgraph instance key -> output slot -> Ref (filled after expanding)
        self._sg_outputs: dict[str, dict[int, Ref]] = {}

    # ------------------------------------------------------------------ public
    def convert(self) -> dict[str, dict[str, Any]]:
        top = _Graph(self.ui["nodes"], self.ui["links"], [], "", None)
        self._expand(top)
        self._prune()
        return self.api

    # ------------------------------------------------------------------ expansion
    def _key(self, g: _Graph, nid: int) -> str:
        return f"{g.prefix}{nid}"

    def _expand(self, g: _Graph) -> None:
        # Pass 1: register pass-through / subgraph nodes so links can resolve through them.
        for n in g.nodes.values():
            if n["type"] in self.subgraphs and n.get("mode", 0) not in (2, 4):
                self._expand_instance(g, n)
        # Pass 2: real nodes
        for n in g.nodes.values():
            t = n["type"]
            mode = n.get("mode", 0)
            if t in FRONTEND_ONLY or t in self.subgraphs:
                continue
            if mode == 2:  # muted
                continue
            if mode == 4:  # bypassed -> pass first matching-type input through
                self._register_bypass(g, n)
                continue
            self._emit(g, n)

    def _expand_instance(self, g: _Graph, inst: dict[str, Any]) -> None:
        sg = self.subgraphs[inst["type"]]
        key = self._key(g, inst["id"])
        # Resolve every subgraph input: external link wins, otherwise the instance's widget value.
        widget_types = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
        named = inst.get("widgets_values_named")
        if isinstance(named, dict) and named:
            by_name = dict(named)
        else:  # positional: one value per widget-typed subgraph input, in definition order
            names = [i["name"] for i in sg.get("inputs", []) if i["type"] in widget_types]
            by_name = dict(zip(names, inst.get("widgets_values") or [], strict=False))
        slots = {s["name"]: s for s in inst.get("inputs", [])}
        provided: dict[str, Ref | Val] = {}
        for sub_in in sg.get("inputs", []):
            name = sub_in["name"]
            slot = slots.get(name)
            if slot is not None and slot.get("link") is not None:
                provided[name] = self._resolve_link(g, slot["link"])
            elif name in by_name:
                provided[name] = ("value", by_name[name])
        sub = _Graph(sg["nodes"], sg["links"], sg.get("inputs", []), prefix=f"{key}:",
                     parent_inputs=provided)
        self._expand(sub)
        # subgraph outputs -> resolve internal source of each output slot
        outs: dict[int, Ref] = {}
        for oi, o in enumerate(sg.get("outputs", [])):
            for lid in o.get("linkIds", []):
                ln = sub.links[lid]
                if ln["origin_id"] == BOUNDARY_IN:  # input wired straight to output
                    r = provided[sub.inputs_def[ln["origin_slot"]]["name"]]
                    if r[0] == "link":
                        outs[oi] = r  # type: ignore[assignment]
                else:
                    outs[oi] = self._origin_ref(sub, ln["origin_id"], ln["origin_slot"])
                break
        self._sg_outputs[key] = outs

    def _register_bypass(self, g: _Graph, n: dict[str, Any]) -> None:
        key = self._key(g, n["id"])
        outs = n.get("outputs", [])
        ins = n.get("inputs", [])
        mapping: dict[int, Ref] = {}
        for oi, o in enumerate(outs):
            for slot in ins:
                if slot.get("type") == o.get("type") and slot.get("link") is not None:
                    r = self._resolve_link(g, slot["link"])
                    if r[0] == "link":
                        mapping[oi] = r  # type: ignore[assignment]
                    break
        self._passthrough[key] = mapping

    # ------------------------------------------------------------------ link resolution
    def _origin_ref(self, g: _Graph, origin_id: int, origin_slot: int) -> Ref:
        if origin_id == BOUNDARY_IN:
            name = g.inputs_def[origin_slot]["name"]
            r = g.parent_inputs.get(name)
            if r is None:
                raise ConvertError(f"subgraph input '{name}' not provided")
            if r[0] != "link":
                raise ConvertError(f"subgraph input '{name}' is a widget value, not a link")
            return r  # type: ignore[return-value]
        origin = g.nodes[origin_id]
        key = self._key(g, origin_id)
        t = origin["type"]
        if t == "Reroute":
            slot = origin["inputs"][0]
            if slot.get("link") is None:
                raise ConvertError(f"dangling Reroute {key}")
            return self._resolve_link(g, slot["link"])  # type: ignore[return-value]
        if t in self.subgraphs:
            m = self._sg_outputs.get(key, {})
            if origin_slot not in m:
                raise ConvertError(f"subgraph {key} output {origin_slot} unresolved")
            return m[origin_slot]
        if key in self._passthrough:
            if origin_slot not in self._passthrough[key]:
                raise ConvertError(f"bypassed node {key} has no pass-through for slot {origin_slot}")
            return self._passthrough[key][origin_slot]
        return ("link", key, origin_slot)

    def _resolve_link(self, g: _Graph, link_id: int) -> Ref:
        ln = g.links[link_id]
        return self._origin_ref(g, ln["origin_id"], ln["origin_slot"])

    # ------------------------------------------------------------------ emit node
    def _widget_values(self, n: dict[str, Any]) -> dict[str, Any]:
        named = n.get("widgets_values_named")
        if isinstance(named, dict):
            return {k: v for k, v in named.items() if k not in FRONTEND_WIDGET_KEYS}
        vals = n.get("widgets_values") or []
        if isinstance(vals, dict):
            return {k: v for k, v in vals.items() if k not in FRONTEND_WIDGET_KEYS}
        # positional fallback: needs object_info for the widget order
        spec = self.oi.get(n["type"])
        if spec is None:
            if vals:
                raise ConvertError(f"{n['type']}: positional widgets but no object_info")
            return {}
        out: dict[str, Any] = {}
        it = iter(vals)
        self._consume(_spec_items(spec["input"]), it, "", out)
        return out  # leftovers are editor-only widgets (upload buttons etc.)

    def _consume(self, items: list[tuple[str, Any, dict[str, Any]]], it: Any, prefix: str,
                 out: dict[str, Any]) -> None:
        """Assign positional widget values to input names, expanding dynamic combos in place."""
        for name, typ, opts in items:
            if typ == "COMFY_DYNAMICCOMBO_V3":
                val = next(it, None)
                out[prefix + name] = val
                opt = next((o for o in opts.get("options", []) if o["key"] == val), None)
                if opt is not None:
                    self._consume(_spec_items(opt["inputs"]), it, f"{prefix}{name}.", out)
            elif _is_widget(typ, opts):
                try:
                    out[prefix + name] = next(it)
                except StopIteration:
                    return
                if opts.get("control_after_generate"):
                    next(it, None)  # editor-only 'fixed/randomize' companion value

    def _emit(self, g: _Graph, n: dict[str, Any]) -> None:
        key = self._key(g, n["id"])
        inputs: dict[str, Any] = {}
        driven: set[str] = set()  # inputs fed by a link or by a promoted subgraph widget
        for slot in n.get("inputs", []):
            lid = slot.get("link")
            if lid is None:
                continue
            name = slot["name"]
            ln = g.links[lid]
            if ln["origin_id"] == BOUNDARY_IN:
                src = g.inputs_def[ln["origin_slot"]]["name"]
                prov = g.parent_inputs.get(src)
                if prov is None:  # optional/autogrow slot left unconnected on the instance
                    continue
                if prov[0] == "value":
                    inputs[name] = prov[1]
                    driven.add(name)
                    continue
            ref = self._resolve_link(g, lid)
            inputs[name] = [ref[1], ref[2]]
            driven.add(name)
        for name, v in self._widget_values(n).items():
            if name not in driven:
                inputs[name] = v
        self.api[key] = {"class_type": n["type"], "inputs": inputs,
                         "_meta": {"title": n.get("title") or n["type"]}}

    # ------------------------------------------------------------------ prune
    def _prune(self) -> None:
        """Drop everything the Save* nodes do not depend on (previews, dead branches)."""
        outputs = [k for k, v in self.api.items() if v["class_type"].startswith(OUTPUT_PREFIXES)]
        if not outputs:
            raise ConvertError("template has no Save* output node")
        keep: set[str] = set()
        stack = list(outputs)
        while stack:
            k = stack.pop()
            if k in keep or k not in self.api:
                continue
            keep.add(k)
            for v in self.api[k]["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    stack.append(v[0])
        self.api = {k: v for k, v in self.api.items() if k in keep}


_CONTROL_VALUES = {"fixed", "randomize", "increment", "decrement"}


def sanitize(api: dict[str, dict[str, Any]], object_info: dict[str, Any] | None
             ) -> list[str]:
    """Drop editor-only leftovers (e.g. the 'fixed' seed-control value). Returns what was dropped."""
    dropped: list[str] = []
    for key, node in api.items():
        spec = (object_info or {}).get(node["class_type"])
        allowed: set[str] | None = None
        if spec:
            inp = spec.get("input", {})
            allowed = set(inp.get("required", {})) | set(inp.get("optional", {}))
        for name in list(node["inputs"]):
            val = node["inputs"][name]
            if allowed is not None:
                ok = name in allowed or name.split(".")[0] in allowed
            else:
                ok = not (isinstance(val, str) and val in _CONTROL_VALUES and name in _CONTROL_VALUES)
            if not ok:
                dropped.append(f"{key}:{node['class_type']}.{name}")
                del node["inputs"][name]
    return dropped


def convert_template(ui: dict[str, Any], object_info: dict[str, Any] | None = None
                     ) -> dict[str, dict[str, Any]]:
    api = Converter(ui, object_info).convert()
    sanitize(api, object_info)
    return api
