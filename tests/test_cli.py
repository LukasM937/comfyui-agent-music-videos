from contextlib import contextmanager

import soundfile as sf
import synth
from test_planning import FakeLLM
from typer.testing import CliRunner

from mvstudio import cli

runner = CliRunner()


def test_init_analyze_plan_check_approve_end_to_end(tmp_path, monkeypatch):
    proj = tmp_path / "night-road"
    r = runner.invoke(cli.app, ["init", str(proj), "--title", "Night Road"])
    assert r.exit_code == 0, r.output
    assert (proj / "brief.yaml").exists()

    sf.write(proj / "song.wav", synth.make_song(), synth.SR)
    r = runner.invoke(cli.app, ["analyze", str(proj), "--no-lyrics"])
    assert r.exit_code == 0, r.output
    assert (proj / "work" / "analysis.json").exists()

    fake = FakeLLM()

    @contextmanager
    def fake_session(cfg, log=print):
        yield fake

    monkeypatch.setattr(cli, "llm_session", fake_session)
    r = runner.invoke(cli.app, ["plan", str(proj)])
    assert r.exit_code == 0, r.output
    assert (proj / "work" / "scenebook.json").exists()
    assert (proj / "work" / "scenebook.md").read_text().startswith("# Night Road")
    html = (proj / "work" / "board.html").read_text()
    assert "<table>" in html and "NOT approved" in html

    # rendering is gated on approval
    r = runner.invoke(cli.app, ["render", str(proj)])
    assert r.exit_code == 1 and "not approved" in r.output

    r = runner.invoke(cli.app, ["check", str(proj), "--no-llm"])
    assert r.exit_code == 0, r.output

    r = runner.invoke(cli.app, ["approve", str(proj)])
    assert r.exit_code == 0, r.output
    assert "approved scenebook" in r.output

    # editing the scenebook afterwards revokes the approval
    import json
    f = proj / "work" / "scenebook.json"
    data = json.loads(f.read_text())
    data["shots"][0]["action"] = "edited by hand"
    f.write_text(json.dumps(data))
    r = runner.invoke(cli.app, ["render", str(proj)])
    assert r.exit_code == 1 and "not approved" in r.output


def test_revise_applies_note_to_selected_section(tmp_path, monkeypatch):
    proj = tmp_path / "p"
    runner.invoke(cli.app, ["init", str(proj)])
    sf.write(proj / "song.wav", synth.make_song(), synth.SR)
    runner.invoke(cli.app, ["analyze", str(proj), "--no-lyrics"])
    fake = FakeLLM()

    @contextmanager
    def fake_session(cfg, log=print):
        yield fake

    monkeypatch.setattr(cli, "llm_session", fake_session)
    runner.invoke(cli.app, ["plan", str(proj)])
    r = runner.invoke(cli.app, ["revise", str(proj), "--note", "make it rain", "--section", "S02"])
    assert r.exit_code == 0, r.output
    assert any("make it rain" in s for s in fake.feedback_seen)


def test_models_status_lists_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("MVS_COMFY_DIR", str(tmp_path / "ComfyUI"))
    r = runner.invoke(cli.app, ["models", "status", "--groups", "core"])
    assert r.exit_code == 0
    assert "ltx25_unet_int8" in r.output and "missing" in r.output
