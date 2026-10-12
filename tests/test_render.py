import shutil
import subprocess

import pytest

from mvstudio.config import Config
from mvstudio.models import Brief, Location, Scenebook, Shot, Treatment
from mvstudio.project import Project
from mvstudio.render import assemble, clips, prompting
from mvstudio.render.keyframes import source_shot

T = Treatment(logline="l", theme_statement="t", visual_style="35mm warm", locations=[
    Location(id="road", name="Road", description="empty road at dusk", time_of_day="dusk")], arc=[])


def shot(i, a, b, **kw):
    base = dict(id=f"S01-{i:02d}", section_id="S01", start=a, end=b, location_id="road",
                subject_ids=["singer"], keyframe_prompt="[singer] stands", video_prompt="[singer] sings")
    base.update(kw)
    return Shot(**base)


def brief():
    from test_planning import BRIEF
    return BRIEF


def test_keyframe_prompt_expands_characters_with_reference_slots():
    s = shot(1, 0, 3, shot_type="medium")
    p = prompting.keyframe_prompt(s, brief(), T, {"singer": 1}, plate_slot=2)
    assert "Mara (the person from <image1>, wearing mustard cardigan)" in p
    assert "<image2>" in p and "35mm warm" in p and "empty road at dusk" in p
    p2 = prompting.keyframe_prompt(s, brief(), T, {}, None)
    assert "woman, 30s, dark curls" in p2 and "<image" not in p2


def test_video_prompt_adds_lipsync_instruction():
    s = shot(1, 0, 3, performance="lipsync", camera="slow push-in")
    p = prompting.video_prompt(s, brief(), T)
    assert "Mara is singing" in p and "slow push-in" in p and "first frame" in p


def test_audio_conditioning_policy():
    cfg = Config()
    lip, plain = shot(1, 0, 3, performance="lipsync"), shot(2, 3, 6)
    assert clips.family_for(lip, cfg) == "ltx-2.3" and clips.family_for(plain, cfg) == "ltx-2.5"
    cfg.video.audio_conditioning = "always"
    assert clips.family_for(plain, cfg) == "ltx-2.3"
    cfg.video.audio_conditioning = "never"
    assert clips.family_for(lip, cfg) == "ltx-2.3"  # lipsync cannot work without audio
    assert clips.family_for(plain, cfg) == "ltx-2.5"


def test_variant_keyframe_sharing_resolves_through_chains():
    a = shot(1, 0, 3)
    b = shot(2, 3, 6, variant_of=a.id, same_keyframe=True)
    c = shot(3, 6, 9, variant_of=b.id, same_keyframe=True)
    d = shot(4, 9, 12, variant_of=a.id, same_keyframe=False)
    sb = Scenebook(title="x", treatment=T, shots=[a, b, c, d])
    assert source_shot(sb, c).id == a.id
    assert source_shot(sb, d).id == d.id


def test_frame_spans_tile_the_song_exactly():
    sb = Scenebook(title="x", treatment=T, shots=[shot(1, 0, 2.51), shot(2, 2.51, 5.2),
                                                  shot(3, 5.2, 9.0)])
    spans = assemble.frame_spans(sb, 24)
    assert spans[0][1] == 0
    assert sum(n for *_, n in spans) == round(9.0 * 24)
    for (_, a, n), (_, b, _) in zip(spans, spans[1:], strict=False):
        assert a + n == b


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_assemble_is_frame_exact_and_muxes_the_original_audio(tmp_path):
    p = Project(tmp_path)
    p.clips_dir.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6", "-ac", "2", str(p.work / "audio.wav")],
                   check=True)
    shots = [shot(1, 0, 2.5), shot(2, 2.5, 4.0, reuse_of="S01-01"), shot(3, 4.0, 6.0)]
    sb = Scenebook(title="x", treatment=T, shots=shots)
    for i, dur in ((1, 3.0), (3, 2.0)):  # clips are longer/shorter than their slot
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc=duration={dur}:size=1280x704:rate=24", "-pix_fmt", "yuv420p",
                        str(p.clips_dir / f"S01-{i:02d}.mp4")], check=True)
    cfg = Config()
    out = assemble.assemble(p, Brief(title="x"), sb, cfg, log=lambda *_: None)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames,width,height", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    assert (int(probe[0]), int(probe[1]), int(probe[2])) == (1920, 1080, 144)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                              "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert "audio" in streams
