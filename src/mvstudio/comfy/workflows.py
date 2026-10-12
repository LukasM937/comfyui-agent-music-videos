"""Parametrised workflows built from the official ComfyUI templates (see scripts/sync_templates.py).

Each builder loads a flattened template, patches only the inputs it owns, removes optional
prompt-enhancer branches (we write the prompts ourselves with Qwen3.8, and the enhancers would
force extra multi-GB model downloads), and returns a ready-to-POST prompt dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from .graph import ApiGraph, GraphError

DEFAULT_NEGATIVE = ("pc game, console game, video game, cartoon, childish, ugly, blurry, "
                    "watermark, subtitles, text overlay, deformed hands, extra fingers")


def valid_frames(duration_s: float, fps: int) -> int:
    """Smallest LTX-legal frame count (8n+1) covering `duration_s`."""
    n = max(1, round(duration_s * fps))
    k = -(-(n - 1) // 8)  # ceil((n-1)/8)
    return 8 * max(k, 1) + 1


# ====================================================================== images (Qwen-Image-2.1)


@dataclass
class ImageJob:
    prompt: str
    width: int
    height: int
    seed: int
    steps: int = 25
    refs: list[str] = field(default_factory=list)  # file names already uploaded to ComfyUI
    prefix: str = "mvs/image"
    # 1024 = official default: references are scaled to ~1 MP before encoding
    ref_resolution: int = 1024


def build_image(job: ImageJob, cfg: Config) -> dict[str, Any]:
    """Text-to-image if no refs, otherwise reference-conditioned generation/edit."""
    name = "qwen21_edit" if job.refs else "qwen21_t2i"
    g = ApiGraph.load(name)

    enc = g.one("TextEncodeQwenImage21")
    g.set(enc, prompt=job.prompt, negative_prompt="", resolution=job.ref_resolution)
    ks = g.one("KSampler")
    g.set(ks, seed=job.seed, steps=job.steps)

    # custom canvas instead of the size of reference image_1
    latent = g.one("EmptyLatentImage")
    g.set(latent, width=job.width, height=job.height, batch_size=1)
    if job.refs:
        sw = [k for k in g.find("ComfySwitchNode")
              if g.nodes[k]["inputs"].get("on_true") == [latent, 0]]
        if len(sw) != 1:
            raise GraphError("qwen21_edit: custom-size switch not found")
        g.bypass_switch(sw[0], "on_true")
        # replace the template's sample inputs by our reference images
        for k in list(g.nodes[enc]["inputs"]):
            if k.startswith("images."):
                del g.nodes[enc]["inputs"][k]
        for old in g.find("LoadImage"):
            del g.nodes[old]
        for i, ref in enumerate(job.refs[:10], start=1):
            nid = f"ref{i}"
            g.nodes[nid] = {"class_type": "LoadImage", "inputs": {"image": ref},
                            "_meta": {"title": f"ref{i}"}}
            g.nodes[enc]["inputs"][f"images.image_{i}"] = [nid, 0]

    # models from config
    g.set(g.one("UNETLoader"), unet_name=cfg.image.unet)
    for k in g.find("CLIPLoader"):  # the PE loader has no consumer after the prompt is literal
        g.nodes[k]["inputs"]["clip_name"] = cfg.image.clip
    g.set(g.one("VAELoader"), vae_name=cfg.image.vae)
    g.set(g.one("SaveImageAdvanced"), filename_prefix=job.prefix)
    return g.finalize()


# ====================================================================== video (LTX)


@dataclass
class VideoJob:
    image: str  # uploaded first-frame file name
    prompt: str
    width: int
    height: int
    frames: int  # 8n+1
    fps: int
    seed: int
    audio: str | None = None  # uploaded wav, exactly the span covered by `frames`
    family: str = "ltx-2.5"
    negative: str = DEFAULT_NEGATIVE
    prefix: str = "mvs/clip"

    @property
    def duration_s(self) -> float:
        return (self.frames - 1) / self.fps


def _common_video_patch(g: ApiGraph, job: VideoJob) -> None:
    g.set(g.one("PrimitiveStringMultiline", "Prompt"), value=job.prompt)
    g.set(g.one("PrimitiveInt", "Width"), value=job.width)
    g.set(g.one("PrimitiveInt", "Height"), value=job.height)
    g.set(g.one("PrimitiveInt", "Frame Rate"), value=job.fps)
    noise = g.find("RandomNoise")
    if len(noise) != 2:
        raise GraphError(f"expected 2 RandomNoise nodes, found {len(noise)}")
    for i, k in enumerate(noise):
        g.set(k, noise_seed=job.seed + i)
    # negative prompt = the CLIPTextEncode with a literal string
    for k in g.find("CLIPTextEncode"):
        if isinstance(g.nodes[k]["inputs"].get("text"), str):
            g.set(k, text=job.negative)
    g.set(g.one("LoadImage"), image=job.image)
    save = g.one("SaveVideo")
    g.set(save, filename_prefix=job.prefix)
    g.drop_input(save, "codec")


def _drop_prompt_enhancer(g: ApiGraph) -> None:
    sw = [k for k in g.find("ComfySwitchNode")
          if g.nodes[k]["inputs"].get("on_true", [None])[0] in g.find("TextGenerateLTX2Prompt")]
    if len(sw) != 1:
        raise GraphError("LTX prompt-enhancer switch not found")
    g.bypass_switch(sw[0], "on_false")


def build_video(job: VideoJob, cfg: Config) -> dict[str, Any]:
    if job.family == "ltx-2.5":
        if job.audio is not None:
            raise GraphError("LTX-2.5 has no official audio-conditioned template; use ltx-2.3")
        g = ApiGraph.load("ltx25_i2v")
        _drop_prompt_enhancer(g)
        _common_video_patch(g, job)
        # exact frame count instead of the template's integer-seconds control
        length = g.one("ComfyMathExpression", "Math Expression (length)")
        g.set(length, expression="a", **{"values.a": job.frames})
        g.drop_input(length, "values.b")
        return g.finalize()

    if job.family == "ltx-2.3":
        if job.audio is not None:
            g = ApiGraph.load("ltx23_ia2v")
            _drop_prompt_enhancer(g)
            _common_video_patch(g, job)
            g.set(g.one("LoadAudio"), audio=job.audio)
            g.set(g.one("TrimAudioDuration"), start_index=0.0)
            g.set(g.one("PrimitiveFloat", "Duration"), value=job.duration_s)
            frames = [k for k in g.find("ComfyMathExpression")
                      if g.nodes[k]["inputs"].get("expression") == "a * b + 1"]
            if len(frames) != 1:
                raise GraphError("ltx23_ia2v: frame-count expression not found")
            # round() guards against 151.99999 -> 151 from float multiplication
            g.set(frames[0], expression="round(a * b) + 1")
            return g.finalize()
        g = ApiGraph.load("ltx23_i2v")
        _drop_prompt_enhancer(g)
        _common_video_patch(g, job)
        length = [k for k in g.find("ComfyMathExpression")
                  if g.nodes[k]["inputs"].get("expression") == "a * b + 1"]
        if len(length) != 1:
            raise GraphError("ltx23_i2v: frame-count expression not found")
        g.set(length[0], expression="a", **{"values.a": job.frames})
        g.drop_input(length[0], "values.b")
        return g.finalize()

    raise GraphError(f"unknown video family {job.family!r}")
