import numpy as np
import pytest
from helpers import analysis_from_truth

from mvstudio.config import Config
from mvstudio.planning.skeleton import make_skeleton


@pytest.fixture
def cfg():
    return Config()


def test_tiles_whole_song_without_gaps(cfg):
    shots = make_skeleton(analysis_from_truth(), cfg)
    assert shots[0].start == 0.0
    assert shots[-1].end == pytest.approx(112.0)
    for a, b in zip(shots, shots[1:], strict=False):
        assert b.start == pytest.approx(a.end)
        assert a.duration > 0


def test_respects_min_and_max_clip_length(cfg):
    for style in ("calm", "medium", "fast"):
        cfg.planning.cut_style = style
        for s in make_skeleton(analysis_from_truth(), cfg):
            assert cfg.video.min_shot_s - 1e-6 <= s.duration <= cfg.video.max_clip_s + 1e-6, (
                style, s.id, s.duration)


def test_cuts_land_on_beats_and_never_cross_sections(cfg):
    a = analysis_from_truth()
    shots = make_skeleton(a, cfg)
    beats = np.array(a.beats)
    for s in shots:
        sec = a.section(s.section_id)
        assert sec.start - 1e-6 <= s.start and s.end <= sec.end + 1e-6
        assert np.min(np.abs(beats - s.start)) < 1e-3 or s.start == 0.0
        assert np.min(np.abs(beats - s.end)) < 1e-3 or s.end == pytest.approx(112.0)


def test_faster_style_means_more_shots(cfg):
    counts = {}
    for style in ("calm", "medium", "fast"):
        cfg.planning.cut_style = style
        counts[style] = len(make_skeleton(analysis_from_truth(), cfg))
    assert counts["calm"] < counts["medium"] < counts["fast"]


def test_high_energy_sections_are_cut_faster(cfg):
    shots = make_skeleton(analysis_from_truth(), cfg)
    avg = lambda sid: np.mean([s.duration for s in shots if s.section_id == sid])  # noqa: E731
    assert avg("S03") < avg("S02")  # chorus (energy 1.0) vs verse (0.5)


def test_refrain_repeats_the_cut_pattern_and_is_tagged(cfg):
    shots = make_skeleton(analysis_from_truth(), cfg)
    pat = lambda sid: [round(s.duration, 2) for s in shots if s.section_id == sid]  # noqa: E731
    assert pat("S03") == pat("S05") == pat("S07")
    ch = [s for s in shots if s.refrain_group]
    assert {s.section_id for s in ch} == {"S03", "S05", "S07"}
    assert {s.occurrence for s in ch} == {1, 2, 3}
    assert all(s.refrain_slot is not None for s in ch)


def test_cuts_avoid_the_middle_of_sung_words(cfg):
    a = analysis_from_truth()
    shots = make_skeleton(a, cfg)
    spans = [(w.start, w.end) for ln in a.lyrics for w in ln.words]
    inside = [s for s in shots[1:] if any(ws + .08 < s.start < we - .08 for ws, we in spans)]
    assert len(inside) <= 1


def test_lyrics_are_attached_to_shots(cfg):
    shots = make_skeleton(analysis_from_truth(), cfg)
    assert any(s.lyric_text for s in shots)
    assert not [s for s in shots if s.section_id == "S01" and s.lyric_text]
