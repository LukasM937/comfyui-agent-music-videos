import huggingface_hub
import pytest

from mvstudio import modelstore
from mvstudio.config import Config


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.comfy.dir = tmp_path / "ComfyUI"
    c.llm.gguf = tmp_path / "models" / "llm" / "m.gguf"
    return c


def model(**kw):
    base = dict(id="x", group="core", repo="a/b", file="sub/f.bin", dest="comfy:vae/f.bin", size=4)
    base.update(kw)
    return modelstore.Model(**base)


def test_manifest_loads_and_has_exact_sizes():
    ms = modelstore.load_manifest()
    assert {m.group for m in ms} == {"core", "lipsync", "asr", "extras"}
    assert all(m.size > 0 for m in ms)
    assert len({m.id for m in ms}) == len(ms)
    core = sum(m.gb for m in modelstore.select(ms, None))
    assert 60 < core < 90  # documented as ~75 GB


def test_status_distinguishes_missing_partial_ok_mismatch(cfg):
    m = model()
    p = modelstore.dest_path(m, cfg)
    assert modelstore.status(m, cfg) == "missing"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"12")
    assert modelstore.status(m, cfg) == "partial"
    p.write_bytes(b"1234")
    assert modelstore.status(m, cfg) == "ok"
    p.write_bytes(b"123456")
    assert modelstore.status(m, cfg) == "size-mismatch"


def test_download_stages_then_moves_into_place(cfg, monkeypatch):
    def fake(repo, filename, token=None, local_dir=None):
        out = local_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"1234")
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    m = model()
    modelstore.download(m, cfg, log=lambda *_: None)
    assert modelstore.status(m, cfg) == "ok"
    assert modelstore.dest_path(m, cfg).read_bytes() == b"1234"


def test_llm_models_go_next_to_the_configured_gguf(cfg):
    m = model(dest="llm:x.gguf")
    assert modelstore.dest_path(m, cfg) == cfg.llm.gguf.parent / "x.gguf"


def test_select_groups():
    ms = modelstore.load_manifest()
    assert {m.group for m in modelstore.select(ms, ["lipsync"])} == {"lipsync"}
    assert modelstore.select(ms, ["all"]) == ms
