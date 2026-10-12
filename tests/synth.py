"""Synthetic song with known structure, for deterministic audio-analysis tests."""

from __future__ import annotations

import numpy as np

SR = 22050
BPM = 120.0
BAR = 4 * 60.0 / BPM  # 2.0 s

# (label, bars, chord progression (root semitones above A2), drums, level)
PLAN = [
    ("intro", 4, [0, 7], False, 0.35),
    ("verse", 8, [0, 5, 7, 5], True, 0.6),
    ("chorus", 8, [5, 7, 12, 10], True, 1.0),
    ("verse", 8, [0, 5, 7, 5], True, 0.6),
    ("chorus", 8, [5, 7, 12, 10], True, 1.0),
    ("bridge", 8, [3, 3, 8, 10], False, 0.5),
    ("chorus", 8, [5, 7, 12, 10], True, 1.0),
    ("outro", 4, [0, 7], False, 0.3),
]


def truth() -> list[tuple[str, float, float]]:
    out, t = [], 0.0
    for label, bars, *_ in PLAN:
        out.append((label, t, t + bars * BAR))
        t += bars * BAR
    return out


def _tone(freq: float, n: int, bright: float) -> np.ndarray:
    t = np.arange(n) / SR
    y = np.zeros(n)
    for h in range(1, 6):
        y += (bright ** (h - 1)) * np.sin(2 * np.pi * freq * h * t) / h
    return y


def make_song(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = int(sum(b for _, b, *_ in PLAN) * BAR * SR)
    y = np.zeros(total)
    pos = 0
    for label, bars, chords, drums, level in PLAN:
        bright = {"intro": 0.2, "verse": 0.5, "chorus": 0.8, "bridge": 0.3, "outro": 0.2}[label]
        for b in range(bars):
            root = 110.0 * 2 ** (chords[b % len(chords)] / 12)
            n = int(BAR * SR)
            seg = np.zeros(n)
            for k in (0, 4, 7):  # triad
                seg += _tone(root * 2 ** (k / 12), n, bright)
            seg *= level * 0.15
            seg *= np.minimum(1.0, np.arange(n) / (0.02 * SR))
            if drums:
                for beat in range(4):
                    i0 = int(beat * BAR / 4 * SR)
                    m = int(0.12 * SR)
                    env = np.exp(-np.arange(m) / (0.03 * SR))
                    kick = np.sin(2 * np.pi * 60 * np.arange(m) / SR) * env * (0.9 if beat % 2 == 0 else 0.5)
                    seg[i0:i0 + m] += kick * level * 0.5
                    if beat % 2 == 1:
                        hat = rng.normal(0, 0.3, m) * env
                        seg[i0:i0 + m] += hat * level * 0.3
            y[pos:pos + n] += seg
            pos += n
    return (y / (np.abs(y).max() + 1e-9) * 0.8).astype(np.float32)
