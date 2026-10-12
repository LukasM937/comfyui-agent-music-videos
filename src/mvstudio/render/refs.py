"""Reference images: one identity per character (+ wardrobe), one plate per location.

User-supplied photos are used as-is. Anything missing is generated once with Qwen-Image-2.1 and
cached, so every later keyframe is conditioned on the *same* faces and places."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..comfy.client import ComfyClient
from ..comfy.workflows import ImageJob, build_image
from ..config import Config
from ..models import Brief, Character, Treatment
from ..project import Project
from .common import State, image_size, stable_seed
from .prompting import wardrobe_text

Log = Callable[[str], None]


def char_dir(p: Project) -> Path:
    return p.refs_dir / "characters"


def portrait_path(p: Project, c: Character) -> Path:
    if c.images:
        return p.root / c.images[0]
    return char_dir(p) / f"{c.id}__portrait.png"


def body_path(p: Project, c: Character, wardrobe_id: str) -> Path:
    # a user image doubles as the full-body reference of the first wardrobe
    if len(c.images) > 1 and wardrobe_id == c.wardrobes[0].id:
        return p.root / c.images[1]
    return char_dir(p) / f"{c.id}__{wardrobe_id}.png"


def plate_path(p: Project, loc_id: str) -> Path:
    return p.refs_dir / "locations" / f"{loc_id}.png"


def _gen(client: ComfyClient, cfg: Config, job: ImageJob, dst: Path) -> Path:
    return client.run(build_image(job, cfg), dst)


def make_refs(client: ComfyClient, project: Project, brief: Brief, t: Treatment, cfg: Config,
              log: Log = print, force: bool = False) -> None:
    st = State(project)
    iw, ih = image_size(brief.aspect, cfg)
    style = t.visual_style

    for c in brief.characters:
        portrait = portrait_path(project, c)
        key = {"desc": c.description, "style": style}
        if not c.images and (force or not st.fresh(portrait, key)):
            log(f"refs: portrait of {c.name}")
            job = ImageJob(
                prompt=(f"Neutral reference portrait photograph, plain grey studio background, "
                        f"soft even light, sharp focus, three-quarter view. {c.description}. "
                        f"Wearing {wardrobe_text(c, c.wardrobes[0].id)}. No text."),
                width=1024, height=1024, seed=stable_seed(cfg.video.seed, c.id),
                steps=cfg.image.steps, prefix=f"mvs/{project.root.name}/ref_{c.id}")
            _gen(client, cfg, job, portrait)
            st.record(portrait, key)
        if not portrait.exists():
            raise FileNotFoundError(f"character image missing: {portrait}")
        pid = client.upload(portrait, f"{project.root.name}__{c.id}__portrait{portrait.suffix}")
        for w in c.wardrobes:
            body = body_path(project, c, w.id)
            key = {"desc": c.description, "wardrobe": w.description, "style": style}
            if body.parent == char_dir(project) and (force or not st.fresh(body, key)):
                log(f"refs: full-body of {c.name} / {w.id}")
                job = ImageJob(
                    prompt=(f"Full-body photograph of the same person as <image1>, standing, "
                            f"front view, plain grey studio background, even light. Wearing "
                            f"{w.description}. Keep face, hair and build identical to <image1>. "
                            "No text."),
                    width=1024, height=1536, seed=stable_seed(cfg.video.seed, f"{c.id}{w.id}"),
                    steps=cfg.image.steps, refs=[pid],
                    prefix=f"mvs/{project.root.name}/ref_{c.id}_{w.id}")
                _gen(client, cfg, job, body)
                st.record(body, key)

    for loc in t.locations:
        plate = plate_path(project, loc.id)
        key = {"loc": loc.description, "tod": loc.time_of_day, "style": style}
        if force or not st.fresh(plate, key):
            log(f"refs: plate for {loc.name}")
            job = ImageJob(
                prompt=(f"{style} Establishing wide shot, no people. {loc.name}. "
                        f"{loc.description} Time of day: {loc.time_of_day}. No text."),
                width=iw, height=ih, seed=stable_seed(cfg.video.seed, loc.id),
                steps=cfg.image.steps, prefix=f"mvs/{project.root.name}/plate_{loc.id}")
            _gen(client, cfg, job, plate)
            st.record(plate, key)
    st.record(project.refs_dir, {"done": True})
