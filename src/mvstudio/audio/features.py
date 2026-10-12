"""Tempo grid and loudness features."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


def to_wav(src: Path, dst: Path, sr: int = 44100) -> Path:
    """Decode anything ffmpeg understands to a stereo PCM wav (also used for clip slicing/muxing)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "2",
           "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


@dataclass
class TempoGrid:
    bpm: float
    beats: np.ndarray  # seconds
    downbeats: np.ndarray  # seconds
    beats_per_bar: int
    onset_env: np.ndarray
    onset_times: np.ndarray


def _pick_meter(strength: np.ndarray, default: int) -> int:
    """Choose 3 or 4 beats per bar from how strongly one beat phase stands out."""
    if len(strength) < 24:
        return default
    best, best_score = default, -1.0
    for m in (3, 4):
        n = len(strength) // m * m
        phase_means = strength[:n].reshape(-1, m).mean(axis=0)
        score = float(phase_means.max() / (phase_means.mean() + 1e-9))
        if m != default:
            score /= 1.10  # bias towards the configured meter
        if score > best_score:
            best, best_score = m, score
    return best


def tempo_grid(y: np.ndarray, sr: int, hop: int, beats_per_bar: int = 4,
               auto_meter: bool = True) -> TempoGrid:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop,
                                                 tightness=120)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)

    if len(beats) == 0:  # no rhythm found: fall back to a metronome at 100 bpm
        dur = len(y) / sr
        bpm = 100.0
        beats = np.arange(0.0, dur, 60.0 / bpm)

    strength = np.interp(beats, onset_times, onset_env)
    m = _pick_meter(strength, beats_per_bar) if auto_meter else beats_per_bar
    # downbeat phase = the beat position (mod m) with the strongest accents
    n = len(strength) // m * m
    if n >= m:
        phase = int(np.argmax(strength[:n].reshape(-1, m).mean(axis=0)))
    else:
        phase = 0
    downbeats = beats[phase::m]
    return TempoGrid(bpm=bpm, beats=beats, downbeats=downbeats, beats_per_bar=m,
                     onset_env=onset_env, onset_times=onset_times)


def energy_per_second(y: np.ndarray, sr: int) -> np.ndarray:
    """RMS per second, scaled so the 95th percentile is 1.0."""
    n = int(np.ceil(len(y) / sr))
    out = np.zeros(n)
    for i in range(n):
        seg = y[i * sr:(i + 1) * sr]
        out[i] = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
    ref = np.percentile(out, 95) + 1e-9
    return np.clip(out / ref, 0.0, 1.0)
