#!/usr/bin/env python3
"""Download official ComfyUI templates and flatten them into workflows/api/*.json.

    python scripts/sync_templates.py                 # all templates in TEMPLATES
    python scripts/sync_templates.py --object-info http://127.0.0.1:8188

Run it after updating ComfyUI to pick up upstream workflow improvements, then run
`mvs doctor --workflows` to validate the result against your ComfyUI's node definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvstudio.comfy.convert import convert_template  # noqa: E402

RAW = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{}.json"

# local name -> official template name
TEMPLATES = {
    "ltx25_i2v": "video_ltx2_5_i2v",
    "ltx25_flf2v": "video_ltx2_5_flf2v",
    "ltx23_ia2v": "video_ltx2_3_ia2v",
    "ltx23_i2v": "video_ltx2_3_i2v",
    "qwen21_t2i": "image_qwen_image_2_1_t2i",
    "qwen21_edit": "image_qwen_image_2_1_image_edit",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-info", help="ComfyUI base URL; enables positional-widget fallback")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "workflows" / "api"))
    ap.add_argument("--only", nargs="*", help="local names to sync")
    args = ap.parse_args()

    oi = None
    if args.object_info:
        oi = requests.get(args.object_info.rstrip("/") + "/object_info", timeout=60).json()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for local, remote in TEMPLATES.items():
        if args.only and local not in args.only:
            continue
        r = requests.get(RAW.format(remote), timeout=60)
        r.raise_for_status()
        api = convert_template(r.json(), oi)
        (out / f"{local}.json").write_text(json.dumps(api, indent=1, sort_keys=True) + "\n")
        print(f"{local:12s} <- {remote:36s} {len(api):3d} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
