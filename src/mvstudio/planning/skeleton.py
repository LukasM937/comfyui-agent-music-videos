"""Deterministic shot grid: *when* to cut. The LLM later decides *what* is shown.

Timing is never left to the language model: cuts land on beats (preferably downbeats), never
straddle section boundaries, avoid slicing through sung words, respect the video model's clip
limit, and repeat the cut pattern for refrains so that the chorus feels familiar."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..models import AudioAnalysis, Section, Shot

BASE_BARS = {"calm": 4.0, "medium": 2.0, "fast": 1.0}
REFRAIN_LABELS = {"chorus", "refrain", "hook"}


@dataclass
class Slot:
    start: float
    end: float


def _bar_len(a: AudioAnalysis) -> float:
    if len(a.downbeats) >= 2:
        return float(np.median(np.diff(a.downbeats)))
    return 4 * 60.0 / max(a.bpm, 1.0)


def _beat_len(a: AudioAnalysis) -> float:
    if len(a.beats) >= 2:
        return float(np.median(np.diff(a.beats)))
    return 60.0 / max(a.bpm, 1.0)


def target_length(sec: Section, a: AudioAnalysis, cfg: Config) -> float:
    bars = BASE_BARS[cfg.planning.cut_style]
    if sec.energy >= 0.75:
        bars /= 2
    elif sec.energy <= 0.35:
        bars *= 2
    length = bars * _bar_len(a)
    return float(np.clip(length, max(cfg.video.min_shot_s, 2 * _beat_len(a)), cfg.video.max_clip_s))


def _word_spans(a: AudioAnalysis) -> list[tuple[float, float]]:
    spans = [(w.start, w.end) for ln in a.lyrics for w in ln.words]
    if spans:
        return spans
    return [(ln.start, ln.end) for ln in a.lyrics]


def _cuts_in_section(sec: Section, a: AudioAnalysis, cfg: Config) -> list[Slot]:
    t0, t1 = sec.start, sec.end
    min_s, max_s = cfg.video.min_shot_s, cfg.video.max_clip_s
    length = target_length(sec, a, cfg)
    beats = np.array([b for b in a.beats if t0 < b < t1])
    downs = set(round(float(d), 3) for d in a.downbeats)
    spans = _word_spans(a)
    beat_len, bar_len = _beat_len(a), _bar_len(a)
    word_weight = bar_len if a.lyrics_source != "none" else 0.0

    def penalty(c: float, target: float) -> float:
        p = abs(c - target)
        if round(float(c), 3) not in downs:
            p += 0.35 * beat_len
        for s, e in spans:
            if s + 0.08 < c < e - 0.08:
                p += word_weight
                break
        return p

    slots: list[Slot] = []
    cur = t0
    while (t1 - cur) > max_s or (t1 - cur) > 1.6 * length:
        target = cur + length
        cands = [float(b) for b in beats if cur + min_s <= b <= t1 - min_s and b - cur <= max_s]
        if not cands:
            break
        cut = min(cands, key=lambda c: penalty(c, target))
        slots.append(Slot(cur, cut))
        cur = cut
    rem = t1 - cur
    if rem > max_s:  # no usable beat: split evenly
        n = math.ceil(rem / max_s)
        step = rem / n
        for i in range(n):
            slots.append(Slot(cur + i * step, cur + (i + 1) * step))
    else:
        slots.append(Slot(cur, t1))
    return slots


def _snap(t: float, beats: list[float]) -> float:
    return min(beats, key=lambda b: abs(b - t)) if beats else t


def _text_between(a: AudioAnalysis, start: float, end: float) -> str:
    words = [w for ln in a.lyrics for w in ln.words if start <= (w.start + w.end) / 2 < end]
    if words:
        # re-join per original line to keep line breaks readable
        out = []
        for ln in a.lyrics:
            ws = [w.text for w in ln.words if start <= (w.start + w.end) / 2 < end]
            if ws:
                out.append(" ".join(ws))
        return " / ".join(out)
    parts = []
    for ln in a.lyrics:
        ov = min(ln.end, end) - max(ln.start, start)
        if ov > 0.5 * (ln.end - ln.start):
            parts.append(ln.text.strip())
    return " / ".join(parts)


def make_skeleton(a: AudioAnalysis, cfg: Config) -> list[Shot]:
    shots: list[Shot] = []
    by_section: dict[str, list[Slot]] = {}
    occurrence: dict[str, int] = {}

    for sec in a.sections:
        slots: list[Slot] | None = None
        is_refrain = sec.label in REFRAIN_LABELS
        group = None
        if is_refrain:
            root = sec.repeat_of or sec.id
            group = f"refrain:{root}"
            occurrence[root] = occurrence.get(root, 0) + 1
        if is_refrain and sec.repeat_of and sec.repeat_of in by_section:
            src = a.section(sec.repeat_of)
            if abs(src.duration - sec.duration) <= 0.75 * _beat_len(a) * 2:
                # mirror the first occurrence's cut pattern, snapped to this section's beats
                offs = [(s.start - src.start, s.end - src.start) for s in by_section[sec.repeat_of]]
                beats = [b for b in a.beats if sec.start <= b <= sec.end]
                cuts = [sec.start] + [_snap(sec.start + e, beats) for _, e in offs[:-1]] + [sec.end]
                if all(c2 - c1 >= cfg.video.min_shot_s * 0.99 for c1, c2 in zip(cuts, cuts[1:])):
                    slots = [Slot(c1, c2) for c1, c2 in zip(cuts, cuts[1:])]
        if slots is None:
            slots = _cuts_in_section(sec, a, cfg)
        by_section[sec.id] = slots

        for i, s in enumerate(slots, start=1):
            shots.append(Shot(
                id=f"{sec.id}-{i:02d}", section_id=sec.id, start=round(s.start, 3),
                end=round(s.end, 3), lyric_text=_text_between(a, s.start, s.end),
                refrain_group=group, refrain_slot=i - 1 if group else None,
                occurrence=occurrence.get(sec.repeat_of or sec.id, 1) if group else 1,
            ))
    # make the shots tile exactly (rounding): each starts where the previous ended
    for prev, nxt in zip(shots, shots[1:], strict=False):
        nxt.start = prev.end
    if shots:
        shots[0].start = 0.0
        shots[-1].end = round(a.duration, 3)
    return shots
