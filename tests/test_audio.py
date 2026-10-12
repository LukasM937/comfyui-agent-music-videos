import numpy as np
import pytest
import synth

from mvstudio.audio import features, lyrics, structure
from mvstudio.models import LyricLine, Word


@pytest.fixture(scope="module")
def song():
    y = synth.make_song()
    grid = features.tempo_grid(y, synth.SR, 512, 4)
    edges = structure.bar_edges(grid.downbeats, len(y) / synth.SR, grid.bpm, grid.beats_per_bar)
    bf = structure.bar_features(y, synth.SR, 512, edges, grid.onset_env)
    sections = structure.build_sections(bf, edges, min_bars=4)
    return y, grid, sections


def test_bar_grid_covers_whole_song(song):
    y, grid, _ = song
    edges = structure.bar_edges(grid.downbeats, len(y) / synth.SR, grid.bpm, grid.beats_per_bar)
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(len(y) / synth.SR)
    assert np.all(np.diff(edges) > 0)


def test_finds_the_eight_sections_within_a_bar_and_a_half(song):
    _, _, sections = song
    truth = synth.truth()
    assert len(sections) == len(truth)
    for sec, (_, start, _) in zip(sections, truth, strict=True):
        assert abs(sec.start - start) <= 3.0, (sec.id, sec.start, start)


def test_repeating_material_shares_a_cluster(song):
    _, _, s = song
    # verse, chorus, verse, chorus, bridge, chorus
    assert s[1].cluster == s[3].cluster
    assert s[2].cluster == s[4].cluster == s[6].cluster
    assert len({s[1].cluster, s[2].cluster, s[5].cluster}) == 3
    assert s[3].repeat_of == s[1].id
    assert s[4].repeat_of == s[2].id


def test_choruses_are_reported_as_exact_repeats(song):
    y, grid, s = song
    al = structure.repeat_alignment(grid.onset_env, grid.onset_times, s[2], s[4])
    assert al.similarity > 0.75
    assert abs(al.lag_s) < 0.2


def _fake_lyrics(sections):
    verse = "walking down the empty road tonight with nothing but my shadow"
    chorus = "take me home take me home where the light never fades"
    bridge = "and if the river runs dry we will find another way"
    texts = {1: verse, 2: chorus, 3: verse, 4: chorus, 5: bridge, 6: chorus}
    lines = []
    for i, txt in texts.items():
        sec = sections[i]
        lines.append(LyricLine(start=sec.start + 1, end=sec.end - 1, text=txt))
    return lines


def test_labels_from_vocals_and_repetition(song):
    _, _, sections = song
    secs = [s.model_copy() for s in sections]
    structure.attach_lyrics(secs, _fake_lyrics(secs))
    structure.merge_by_lyric_repeat(secs)
    structure.label_sections(secs, have_vocals=True)
    assert [s.label for s in secs] == [
        "intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]


def test_without_lyrics_we_still_label_something(song):
    _, _, sections = song
    secs = [s.model_copy() for s in sections]
    structure.label_sections(secs, have_vocals=False)
    assert all(s.label for s in secs)
    assert secs[0].label == "intro" and secs[-1].label == "outro"


def test_align_provided_lyrics_uses_asr_times_and_survives_misheard_words():
    asr_words = [("take", 10.0), ("me", 10.4), ("home", 10.8), ("take", 12.0), ("me", 12.4),
                 ("homes", 12.8), ("where", 13.6), ("the", 13.9), ("light", 14.2)]
    words = [Word(start=t, end=t + 0.3, text=w) for w, t in asr_words]
    asr = [LyricLine(start=10.0, end=14.5, text=" ".join(w for w, _ in asr_words), words=words)]
    provided = "Take me home\nTake me home\nWhere the light"
    out = lyrics.align_provided(provided, asr, duration=60)
    assert [ln.text for ln in out] == ["Take me home", "Take me home", "Where the light"]
    assert out[0].start == pytest.approx(10.0)
    assert out[1].start == pytest.approx(12.0)
    assert out[1].end > out[1].start  # the misheard 'homes' got an interpolated time
    assert out[2].start == pytest.approx(13.6)


def test_lrc_parsing():
    txt = "[00:12.50] first line\n[00:20.00]second line\n[bad] x\n"
    out = lyrics.parse_lrc(txt, duration=30)
    assert [ln.text for ln in out] == ["first line", "second line"]
    assert out[0].start == pytest.approx(12.5)
    assert out[0].end == pytest.approx(20.0)


def test_analysis_reports_alignment_for_chorus_pairs(tmp_path):
    import soundfile as sf

    from mvstudio.audio.analyze import analyze, describe
    from mvstudio.config import Config
    from mvstudio.models import Brief
    from mvstudio.project import Project

    p = Project(tmp_path)
    sf.write(tmp_path / "song.wav", synth.make_song(), synth.SR)
    a = analyze(p, Brief(title="x"), Config(), log=lambda *_: None, with_lyrics=False)
    ch = [s for s in a.sections if s.label == "chorus"]
    assert len(ch) == 3
    pairs = {(r.a, r.b): r for r in a.repeats}
    assert (ch[0].id, ch[1].id) in pairs and pairs[(ch[0].id, ch[1].id)].similarity > 0.7
    assert "timing similarity" in describe(a)
