"""Audio -> AudioAnalysis (tempo grid, sections, repeats, lyrics)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import librosa

from ..config import Config
from ..models import AudioAnalysis, Brief, LyricLine
from ..project import Project
from . import features, lyrics, structure

Log = Callable[[str], None]


def _lyrics(project: Project, brief: Brief, wav: Path, duration: float, cfg: Config,
            log: Log) -> tuple[list[LyricLine], str, str | None]:
    if project.lyrics_lrc.exists():
        log("lyrics: using provided .lrc timings")
        return lyrics.parse_lrc(project.lyrics_lrc.read_text(encoding="utf-8"), duration), "lrc", \
            brief.language
    try:
        log("lyrics: transcribing with faster-whisper (word timestamps)")
        asr = lyrics.transcribe(wav, cfg.audio.asr, brief.language)
    except Exception as e:  # noqa: BLE001 - missing extra, no model download, CUDA problems...
        log(f"lyrics: ASR unavailable ({type(e).__name__}: {str(e)[:160]}); "
            "continuing without timed lyrics (provide lyrics.lrc to get timings anyway)")
        asr = []
    if brief.lyrics and asr:
        log("lyrics: aligning provided lyrics to ASR words")
        return lyrics.align_provided(brief.lyrics, asr, duration), "provided+asr", brief.language
    if asr:
        return asr, "asr", brief.language
    return [], "none", brief.language


def analyze(project: Project, brief: Brief, cfg: Config, log: Log = print,
            with_lyrics: bool = True) -> AudioAnalysis:
    src = project.audio_src
    log(f"audio: decoding {src.name}")
    wav = features.to_wav(src, project.audio_wav)
    sr = cfg.audio.sr
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    duration = len(y) / sr
    log(f"audio: {duration:.1f}s")

    log("audio: tempo grid")
    grid = features.tempo_grid(y, sr, cfg.audio.hop, cfg.audio.beats_per_bar)
    edges = structure.bar_edges(grid.downbeats, duration, grid.bpm, grid.beats_per_bar)

    log(f"audio: structure ({len(edges) - 1} bars, {grid.bpm:.0f} bpm, {grid.beats_per_bar}/4)")
    bf = structure.bar_features(y, sr, cfg.audio.hop, edges, grid.onset_env)
    sections = structure.build_sections(bf, edges, cfg.audio.min_section_bars)

    lines: list[LyricLine] = []
    source = "none"
    language = brief.language
    if with_lyrics:
        lines, source, language = _lyrics(project, brief, wav, duration, cfg, log)
    structure.attach_lyrics(sections, lines)
    if lines:
        structure.merge_by_lyric_repeat(sections)
    structure.label_sections(sections, have_vocals=bool(lines))

    # Alignment of every pair of similarly long sections, not just acoustic repeats: the director
    # may later decide that two differently arranged choruses belong together.
    repeats = [structure.repeat_alignment(grid.onset_env, grid.onset_times, a, b)
               for i, a in enumerate(sections) for b in sections[i + 1:]
               if abs(a.duration - b.duration) <= 0.15 * max(a.duration, b.duration)]

    energy = features.energy_per_second(y, sr)
    return AudioAnalysis(
        audio_file=str(src.name), duration=duration, sr=sr, bpm=round(grid.bpm, 2),
        beats=[round(float(b), 3) for b in grid.beats],
        downbeats=[round(float(b), 3) for b in grid.downbeats],
        sections=sections, lyrics=lines, lyrics_source=source,  # type: ignore[arg-type]
        energy_curve=[round(float(e), 3) for e in energy], repeats=repeats, language=language,
    )


def describe(a: AudioAnalysis) -> str:
    """Compact human/LLM-readable summary used in prompts and in the CLI."""
    out = [f"duration {a.duration:.1f}s, {a.bpm:.0f} bpm, lyrics: {a.lyrics_source}"]
    for s in a.sections:
        rep = f" (repeat of {s.repeat_of})" if s.repeat_of else ""
        out.append(
            f"{s.id} {s.start:6.1f}-{s.end:6.1f}s {s.bars:4.0f} bars  cluster {s.cluster}{rep}  "
            f"energy {s.energy:.2f}  vocals {s.vocal_ratio:.2f}  label~{s.label}"
            + (f'\n      lyrics: "{s.lyric_text[:240]}"' if s.lyric_text else ""))
    close = [r for r in a.repeats if r.similarity >= 0.5]
    if close:
        out.append("timing similarity of section pairs: " + ", ".join(
            f"{r.a}~{r.b} lag {r.lag_s:+.2f}s sim {r.similarity:.2f}"
            f"{' IDENTICAL' if r.identical_ok else ''}" for r in close))
    return "\n".join(out)
