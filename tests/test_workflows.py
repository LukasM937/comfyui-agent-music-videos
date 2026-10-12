import pytest

from mvstudio.comfy.graph import ApiGraph
from mvstudio.comfy.workflows import (
    ImageJob,
    VideoJob,
    build_image,
    build_video,
    valid_frames,
)
from mvstudio.config import Config

CFG = Config()


def classes(wf):
    return {v["class_type"] for v in wf.values()}


def refs_ok(wf):
    """every link points at an existing node and slot index is an int"""
    for k, v in wf.items():
        for name, val in v["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in wf, f"{k}.{name} -> missing node {val[0]}"
                assert isinstance(val[1], int)


@pytest.mark.parametrize("seconds,fps,expect", [(1.0, 24, 25), (2.0, 24, 49), (4.0, 24, 97),
                                                 (6.5, 24, 161), (10.0, 24, 241), (0.2, 24, 9)])
def test_valid_frames_are_8n_plus_1_and_cover_the_shot(seconds, fps, expect):
    f = valid_frames(seconds, fps)
    assert f == expect and (f - 1) % 8 == 0
    assert f >= round(seconds * fps)


def video_job(**kw):
    base = dict(image="kf.png", prompt="A woman walks.", width=1280, height=704,
                frames=valid_frames(6.5, 24), fps=24, seed=11)
    base.update(kw)
    return VideoJob(**base)


def test_ltx25_i2v_is_patched_and_enhancer_is_gone():
    wf = build_video(video_job(), CFG)
    refs_ok(wf)
    assert "TextGenerateLTX2Prompt" not in classes(wf)
    assert "ComfySwitchNode" not in classes(wf)
    prompts = [v["inputs"]["value"] for v in wf.values() if v["class_type"] == "PrimitiveStringMultiline"]
    assert prompts == ["A woman walks."]
    length = [v for v in wf.values() if v["class_type"] == "ComfyMathExpression"
              and v["inputs"].get("expression") == "a"]
    assert any(v["inputs"].get("values.a") == 161 for v in length)
    assert [v["inputs"]["noise_seed"] for v in wf.values() if v["class_type"] == "RandomNoise"] \
        == sorted(v["inputs"]["noise_seed"] for v in wf.values() if v["class_type"] == "RandomNoise")
    save = next(v for v in wf.values() if v["class_type"] == "SaveVideo")
    assert "codec" not in save["inputs"]
    # only the models we ship in the manifest
    files = {n for v in wf.values() for k, n in v["inputs"].items()
             if k in ("unet_name", "vae_name", "clip_name", "model_name") and isinstance(n, str)}
    assert files == {"ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                     "ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-audio-vae-bf16.safetensors",
                     "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                     "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"}


def test_ltx23_audio_workflow_takes_audio_and_exact_duration():
    wf = build_video(video_job(family="ltx-2.3", audio="slice.wav"), CFG)
    refs_ok(wf)
    assert "LTXVAudioVAEEncode" in classes(wf) and "TrimAudioDuration" in classes(wf)
    assert "TextGenerateLTX2Prompt" not in classes(wf)
    audio = next(v for v in wf.values() if v["class_type"] == "LoadAudio")
    assert audio["inputs"]["audio"] == "slice.wav"
    dur = next(v for v in wf.values() if v["class_type"] == "PrimitiveFloat")
    assert dur["inputs"]["value"] == pytest.approx(160 / 24)
    expr = [v["inputs"]["expression"] for v in wf.values() if v["class_type"] == "ComfyMathExpression"]
    assert "round(a * b) + 1" in expr


def test_ltx25_refuses_audio_instead_of_silently_ignoring_it():
    with pytest.raises(Exception, match="no official audio"):
        build_video(video_job(audio="slice.wav"), CFG)


def test_ltx23_without_audio_uses_plain_i2v():
    wf = build_video(video_job(family="ltx-2.3"), CFG)
    refs_ok(wf)
    assert "LTXVAudioVAEEncode" not in classes(wf)


def test_qwen_t2i_and_edit():
    t2i = build_image(ImageJob(prompt="a cat", width=1920, height=1088, seed=5), CFG)
    refs_ok(t2i)
    assert "LoadImage" not in classes(t2i) and "TextGenerate" not in classes(t2i)
    lat = next(v for v in t2i.values() if v["class_type"] == "EmptyLatentImage")
    assert (lat["inputs"]["width"], lat["inputs"]["height"]) == (1920, 1088)

    edit = build_image(ImageJob(prompt="<image1> in <image2>", width=1920, height=1088, seed=5,
                                refs=["a.png", "b.png", "c.png"]), CFG)
    refs_ok(edit)
    enc = next(v for v in edit.values() if v["class_type"] == "TextEncodeQwenImage21")
    assert sorted(k for k in enc["inputs"] if k.startswith("images.")) == [
        "images.image_1", "images.image_2", "images.image_3"]
    assert "TextGenerate" not in classes(edit)
    ks = next(v for v in edit.values() if v["class_type"] == "KSampler")
    latent_src = edit[ks["inputs"]["latent_image"][0]]
    assert latent_src["class_type"] == "EmptyLatentImage"  # custom canvas, not image_1's size
    loads = sorted(v["inputs"]["image"] for v in edit.values() if v["class_type"] == "LoadImage")
    assert loads == ["a.png", "b.png", "c.png"]


def test_templates_still_have_the_nodes_our_patches_rely_on():
    g = ApiGraph.load("ltx25_i2v")
    for title in ("Prompt", "Width", "Height", "Frame Rate"):
        assert g.find(title=title), title
    g = ApiGraph.load("ltx23_ia2v")
    assert g.one("PrimitiveFloat", "Duration") and g.one("TrimAudioDuration") and g.one("LoadAudio")


def test_every_model_file_the_workflows_need_is_in_the_manifest():
    from mvstudio.modelstore import load_manifest
    shipped = {m.dest.split("/", 1)[-1] for m in load_manifest() if m.dest.startswith("comfy:")}
    needed: set[str] = set()
    jobs = [video_job(), video_job(family="ltx-2.3"),
            video_job(family="ltx-2.3", audio="s.wav")]
    wfs = [build_video(j, CFG) for j in jobs]
    wfs += [build_image(ImageJob(prompt="x", width=512, height=512, seed=1), CFG),
            build_image(ImageJob(prompt="x", width=512, height=512, seed=1, refs=["a.png"]), CFG)]
    keys = ("unet_name", "vae_name", "clip_name", "text_encoder", "lora_name", "model_name",
            "ckpt_name")
    for wf in wfs:
        for v in wf.values():
            for k in keys:
                if isinstance(v["inputs"].get(k), str):
                    needed.add(v["inputs"][k])
    assert needed <= shipped, f"not in config/models.toml: {sorted(needed - shipped)}"
