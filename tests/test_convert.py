import json

from mvstudio.comfy.convert import convert_template, sanitize

SG = "11111111-2222-3333-4444-555555555555"


def template():
    """LoadImage -> [subgraph: Reroute -> Resize(width promoted from the instance)] -> Save."""
    return {
        "nodes": [
            {"id": 1, "type": "LoadImage", "mode": 0, "inputs": [],
             "widgets_values_named": {"image": "a.png", "upload": "image"}},
            {"id": 2, "type": "MarkdownNote", "mode": 0, "inputs": [], "widgets_values": ["x"]},
            {"id": 3, "type": SG, "mode": 0,
             "inputs": [{"name": "img", "type": "IMAGE", "link": 10},
                        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None}],
             "widgets_values": [640]},
            {"id": 4, "type": "SaveImage", "mode": 0,
             "inputs": [{"name": "images", "type": "IMAGE", "link": 11}],
             "widgets_values_named": {"filename_prefix": "x"}},
            {"id": 5, "type": "PreviewImage", "mode": 0,
             "inputs": [{"name": "images", "type": "IMAGE", "link": 12}]},
        ],
        "links": [[10, 1, 0, 3, 0, "IMAGE"], [11, 3, 0, 4, 0, "IMAGE"], [12, 3, 0, 5, 0, "IMAGE"]],
        "definitions": {"subgraphs": [{
            "id": SG, "name": "sg",
            "inputs": [{"name": "img", "type": "IMAGE", "linkIds": [20]},
                       {"name": "width", "type": "INT", "linkIds": [21]}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "linkIds": [22]}],
            "nodes": [
                {"id": 7, "type": "Reroute", "mode": 0,
                 "inputs": [{"name": "", "type": "*", "link": 20}]},
                {"id": 8, "type": "ImageScale", "mode": 0,
                 "inputs": [{"name": "image", "type": "IMAGE", "link": 23},
                            {"name": "width", "type": "INT", "widget": {"name": "width"},
                             "link": 21}],
                 "widgets_values_named": {"upscale_method": "lanczos", "width": 1, "height": 360,
                                          "crop": "center"}},
                {"id": 9, "type": "PrimitiveInt", "mode": 0, "inputs": [],
                 "widgets_values_named": {"value": 5, "fixed": "fixed"}},
            ],
            "links": [
                {"id": 20, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0,
                 "type": "IMAGE"},
                {"id": 21, "origin_id": -10, "origin_slot": 1, "target_id": 8, "target_slot": 1,
                 "type": "INT"},
                {"id": 23, "origin_id": 7, "origin_slot": 0, "target_id": 8, "target_slot": 0,
                 "type": "IMAGE"},
                {"id": 22, "origin_id": 8, "origin_slot": 0, "target_id": -20, "target_slot": 0,
                 "type": "IMAGE"},
            ]}]},
    }


def test_subgraph_is_flattened_with_promoted_widgets_and_reroutes():
    api = convert_template(template())
    assert set(api) == {"1", "3:8", "4"}  # note, preview and the dangling primitive are gone
    assert api["3:8"]["inputs"]["image"] == ["1", 0]  # through Reroute and the subgraph border
    assert api["3:8"]["inputs"]["width"] == 640  # instance widget value, not the inner 1
    assert api["3:8"]["inputs"]["height"] == 360
    assert api["4"]["inputs"]["images"] == ["3:8", 0]
    assert "upload" not in api["1"]["inputs"]


def test_sanitize_drops_editor_only_control_values():
    api = {"1": {"class_type": "PrimitiveInt", "inputs": {"value": 5, "fixed": "fixed"}}}
    oi = {"PrimitiveInt": {"input": {"required": {"value": ["INT", {}]}}}}
    assert sanitize(api, oi) == ["1:PrimitiveInt.fixed"]
    assert api["1"]["inputs"] == {"value": 5}
    api = {"1": {"class_type": "X", "inputs": {"a": 1, "fixed": "fixed"}}}
    sanitize(api, None)
    assert api["1"]["inputs"] == {"a": 1}


def test_committed_workflows_are_flat_valid_json():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "workflows" / "api"
    files = sorted(root.glob("*.json"))
    assert len(files) >= 6
    for f in files:
        wf = json.loads(f.read_text())
        for k, v in wf.items():
            assert "class_type" in v and "inputs" in v
            for val in v["inputs"].values():
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                    assert val[0] in wf, f"{f.name}:{k} dangling link {val}"
