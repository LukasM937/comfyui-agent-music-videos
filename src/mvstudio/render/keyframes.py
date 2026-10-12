"""First frames for every shot (Qwen-Image-2.1, conditioned on character and location refs)."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from ..comfy.client import ComfyClient
from ..comfy.workflows import ImageJob, build_image
from ..config import Config
from ..models import Brief, Scenebook, Shot
from ..project import Project
from . import prompting, refs
from .common import State, image_size, stable_seed

Log = Callable[[str], None]


def keyframe_path(p: Project, shot: Shot) -> Path:
    return p.keyframes_dir / f"{shot.id}.png"


def source_shot(sb: Scenebook, shot: Shot) -> Shot:
    """The shot whose keyframe file this shot uses."""
    if shot.same_keyframe and shot.variant_of:
        base = sb.shot(shot.variant_of)
        return source_shot(sb, base)
    return shot


def plan_refs(project: Project, brief: Brief, sb: Scenebook, shot: Shot, cfg: Config,
              base_keyframe: Path | None) -> tuple[list[Path], dict[str, int], int | None]:
    """Ordered reference files + {character id: image slot} + plate slot."""
    files: list[Path] = []
    slots: dict[str, int] = {}
    budget = cfg.image.max_refs
    plate_needed = bool(shot.location_id) and refs.plate_path(project, shot.location_id).exists()
    reserve = int(plate_needed) + int(base_keyframe is not None)
    for cid in shot.subject_ids:
        c = brief.character(cid)
        if c is None:
            continue
        cands = [refs.portrait_path(project, c)]
        body = refs.body_path(project, c, shot.wardrobe_id)
        close = shot.shot_type.lower().startswith(("close", "extreme"))
        if not close and body.exists():
            cands = [body, cands[0]]  # full-body first for wide/medium framing
        for f in cands:
            if f.exists() and len(files) < budget - reserve and f not in files:
                files.append(f)
                slots.setdefault(cid, len(files))
                break  # one reference per character keeps the budget for the others
    plate_slot = None
    if plate_needed and len(files) < budget:
        files.append(refs.plate_path(project, shot.location_id))
        plate_slot = len(files)
    if base_keyframe is not None and len(files) < budget:
        files.append(base_keyframe)
    return files, slots, plate_slot


def make_keyframes(client: ComfyClient, project: Project, brief: Brief, sb: Scenebook,
                   cfg: Config, only: set[str] | None = None, force: bool = False,
                   log: Log = print) -> None:
    st = State(project)
    iw, ih = image_size(brief.aspect, cfg)
    todo = [s for s in sb.shots if not (s.reuse_of or (s.same_keyframe and s.variant_of))]
    for i, shot in enumerate(todo, start=1):
        if only and shot.id not in only:
            continue
        dst = keyframe_path(project, shot)
        base_kf = None
        if shot.variant_of and not shot.same_keyframe:
            b = keyframe_path(project, source_shot(sb, sb.shot(shot.variant_of)))
            base_kf = b if b.exists() else None
        files, slot, plate_slot = plan_refs(project, brief, sb, shot, cfg, base_kf)
        prompt = prompting.keyframe_prompt(shot, brief, sb.treatment, slot, plate_slot)
        if base_kf is not None:
            prompt += (f" This is a return of an earlier image (<image{len(files)}>): keep its "
                       "composition idea and look, with the changes described.")
        key = {"p": prompt, "refs": [f.name + str(f.stat().st_mtime_ns) for f in files],
               "size": [iw, ih], "steps": cfg.image.steps, "unet": cfg.image.unet}
        if not force and st.fresh(dst, key):
            continue
        log(f"keyframe {shot.id} ({i}/{len(todo)}): {shot.shot_type}, {len(files)} ref(s)")
        up = [client.upload(f, f"{project.root.name}__{f.parent.name}__{f.name}") for f in files]
        job = ImageJob(prompt=prompt, width=iw, height=ih, refs=up, steps=cfg.image.steps,
                       seed=stable_seed(cfg.video.seed, shot.id),
                       prefix=f"mvs/{project.root.name}/kf_{shot.id}")
        client.run(build_image(job, cfg), dst)
        st.record(dst, key)

    for shot in sb.shots:  # shots that share a keyframe get a copy under their own id
        src = source_shot(sb, shot)
        if src is not shot:
            s, d = keyframe_path(project, src), keyframe_path(project, shot)
            if s.exists():
                shutil.copyfile(s, d)
