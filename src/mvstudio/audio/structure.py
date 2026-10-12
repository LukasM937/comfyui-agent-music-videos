"""Song structure: bar-synchronous self-similarity -> novelty peaks -> sections -> clusters.

Deliberately classical (Foote novelty on a bar-level self-similarity matrix). It is deterministic,
CPU-only and testable; the LLM later *names* the sections with the lyrics as extra evidence, it does
not invent boundaries.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering

from ..models import LyricLine, RepeatAlignment, Section


def bar_edges(downbeats: np.ndarray, duration: float, bpm: float, beats_per_bar: int) -> np.ndarray:
    """Bar boundaries covering [0, duration]; extrapolates before the first / after the last
    downbeat with the average bar length so the grid always spans the whole song."""
    bar_len = 60.0 / max(bpm, 1.0) * beats_per_bar
    if len(downbeats) >= 2:
        bar_len = float(np.median(np.diff(downbeats)))
    edges = list(map(float, downbeats))
    if not edges:
        edges = [0.0]
    while edges[0] - bar_len > 0.25 * bar_len:
        edges.insert(0, edges[0] - bar_len)
    if edges[0] > 0.0:
        edges[0] = 0.0
    while edges[-1] + bar_len < duration - 0.25 * bar_len:
        edges.append(edges[-1] + bar_len)
    edges.append(duration)
    # strictly increasing
    out = [edges[0]]
    for e in edges[1:]:
        if e > out[-1] + 1e-3:
            out.append(e)
    return np.array(out)


@dataclass
class BarFeatures:
    feat: np.ndarray  # (n_bars, d) L2-normalised
    rms: np.ndarray  # (n_bars,)
    onset: np.ndarray  # (n_bars,)


def bar_features(y: np.ndarray, sr: int, hop: int, edges: np.ndarray,
                 onset_env: np.ndarray) -> BarFeatures:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)[1:]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    n_frames = min(chroma.shape[1], mfcc.shape[1], rms.shape[0], len(onset_env))
    chroma, mfcc, rms, onset_env = (chroma[:, :n_frames], mfcc[:, :n_frames], rms[:n_frames],
                                    onset_env[:n_frames])
    fr = librosa.time_to_frames(edges, sr=sr, hop_length=hop)
    fr = np.clip(fr, 0, n_frames)

    c_list, m_list, r_list, o_list = [], [], [], []
    for a, b in zip(fr[:-1], fr[1:], strict=True):
        b = max(b, a + 1)
        c_list.append(chroma[:, a:b].mean(axis=1))
        m_list.append(mfcc[:, a:b].mean(axis=1))
        r_list.append(float(rms[a:b].mean()))
        o_list.append(float(onset_env[a:b].mean()))
    c = np.array(c_list)
    m = np.array(m_list)
    # z-score timbre over the song, then weight the blocks
    m = (m - m.mean(axis=0)) / (m.std(axis=0) + 1e-6)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    # dynamics: loudness + onset density tell verse from chorus when harmony is shared
    dyn = np.stack([np.log(np.array(r_list) + 1e-6), np.array(o_list)], axis=1)
    dyn = (dyn - dyn.mean(axis=0)) / (dyn.std(axis=0) + 1e-6)
    dyn = dyn / (np.linalg.norm(dyn, axis=1, keepdims=True) + 1e-9)
    feat = np.hstack([c, m, 0.8 * dyn])
    feat = feat - feat.mean(axis=0)  # centre: cosine similarity then reflects *differences*
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    return BarFeatures(feat=feat, rms=np.array(r_list), onset=np.array(o_list))


def self_similarity(feat: np.ndarray) -> np.ndarray:
    return np.clip(feat @ feat.T, -1.0, 1.0)


def _checkerboard(half: int) -> np.ndarray:
    n = 2 * half
    x = np.arange(n) - (half - 0.5)
    g = np.exp(-0.5 * (x / (half * 0.8)) ** 2)
    gauss = np.outer(g, g)
    sign = np.ones((n, n))
    sign[:half, half:] = -1
    sign[half:, :half] = -1
    return gauss * sign


def novelty(ssm: np.ndarray, half: int) -> np.ndarray:
    n = ssm.shape[0]
    k = _checkerboard(half)
    pad = np.pad(ssm, half, mode="edge")
    out = np.zeros(n)
    for i in range(n):
        # window centred between bar i-1 and i  -> boundary *before* bar i
        win = pad[i:i + 2 * half, i:i + 2 * half]
        out[i] = float((win * k).sum())
    out[0] = 0.0
    return np.maximum(out, 0.0)


def segment_boundaries(feat: np.ndarray, min_bars: int) -> list[int]:
    """Indices of bars where a new section starts (always includes 0)."""
    n = len(feat)
    if n <= min_bars:
        return [0]
    ssm = self_similarity(feat)
    half = int(np.clip(min_bars // 2, 2, 8))
    nov = novelty(ssm, half)
    if nov.max() <= 0:
        return [0]
    thr = nov.mean() + 0.35 * nov.std()
    peaks, _ = find_peaks(nov, height=thr, distance=max(2, min_bars - 1), prominence=0.15 * nov.max())
    bounds = [0] + [int(p) for p in peaks if p >= min_bars // 2 and p <= n - min_bars // 2]
    bounds = refine_boundaries(feat, sorted(set(bounds)))
    # merge tiny leading/trailing fragments
    bounds = sorted(set(bounds))
    cleaned = [bounds[0]]
    for b in bounds[1:]:
        if b - cleaned[-1] >= max(2, min_bars // 2):
            cleaned.append(b)
    if len(cleaned) > 1 and n - cleaned[-1] < max(2, min_bars // 2):
        cleaned.pop()
    return cleaned


def refine_boundaries(feat: np.ndarray, bounds: list[int], reach: int = 1, win: int = 3) -> list[int]:
    """Nudge each boundary by up to `reach` bars to where the two sides are most homogeneous.

    Novelty peaks are blurred by the kernel and by bars that two sections share (same chord);
    the within-window variance criterion puts the cut where the material really switches."""
    n = len(feat)
    out = [bounds[0]]
    for b in bounds[1:]:
        best, best_cost = b, np.inf
        for c in range(b - reach, b + reach + 1):
            if c <= out[-1] + 1 or c >= n - 1:
                continue
            left, right = feat[max(0, c - win):c], feat[c:min(n, c + win)]
            cost = float(((left - left.mean(0)) ** 2).sum() + ((right - right.mean(0)) ** 2).sum())
            if cost < best_cost - 1e-9:
                best, best_cost = c, cost
        out.append(best)
    return out


def cluster_segments(seg_feat: np.ndarray, threshold: float = 0.18) -> list[int]:
    if len(seg_feat) == 1:
        return [0]
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                    distance_threshold=threshold)
    labels = model.fit_predict(seg_feat)
    # relabel by first appearance
    order: dict[int, int] = {}
    out = []
    for lab in labels:
        order.setdefault(int(lab), len(order))
        out.append(order[int(lab)])
    return out


def _letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def build_sections(bf: BarFeatures, edges: np.ndarray, min_bars: int,
                   cluster_threshold: float = 0.18) -> list[Section]:
    bounds = segment_boundaries(bf.feat, min_bars)
    bounds_end = bounds[1:] + [len(bf.feat)]
    seg_feat, seg_rms, seg_onset = [], [], []
    for a, b in zip(bounds, bounds_end, strict=True):
        v = bf.feat[a:b].mean(axis=0)
        seg_feat.append(v / (np.linalg.norm(v) + 1e-9))
        seg_rms.append(float(bf.rms[a:b].mean()))
        seg_onset.append(float(bf.onset[a:b].mean()))
    clusters = cluster_segments(np.array(seg_feat), cluster_threshold)
    rms = np.array(seg_rms) / (max(seg_rms) + 1e-9)
    onset = np.array(seg_onset) / (max(seg_onset) + 1e-9)
    energy = 0.7 * rms + 0.3 * onset
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-9) * 0.9 + 0.1

    sections: list[Section] = []
    first_of: dict[int, str] = {}
    for i, (a, b) in enumerate(zip(bounds, bounds_end, strict=True)):
        sid = f"S{i + 1:02d}"
        cl = clusters[i]
        sec = Section(id=sid, start=float(edges[a]), end=float(edges[b]), bars=float(b - a),
                      cluster=_letter(cl), energy=float(energy[i]))
        if cl in first_of:
            sec.repeat_of = first_of[cl]
        else:
            first_of[cl] = sid
        sections.append(sec)
    return sections


# ------------------------------------------------------------------ lyrics <-> sections

_WORD = re.compile(r"[^\w\s']", re.UNICODE)


def norm_tokens(text: str) -> list[str]:
    return _WORD.sub(" ", text.lower()).split()


def text_similarity(a: str, b: str) -> float:
    ta, tb = norm_tokens(a), norm_tokens(b)
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).ratio()


def attach_lyrics(sections: list[Section], lines: list[LyricLine]) -> None:
    """Fill lyric_text and vocal_ratio. A line belongs to the section holding its midpoint."""
    for sec in sections:
        inside = [ln for ln in lines if sec.start <= (ln.start + ln.end) / 2 < sec.end]
        sec.lyric_text = " / ".join(ln.text.strip() for ln in inside)
        covered = 0.0
        for ln in inside:
            covered += max(0.0, min(ln.end, sec.end) - max(ln.start, sec.start))
        sec.vocal_ratio = float(min(1.0, covered / max(sec.duration, 1e-6)))


def merge_by_lyric_repeat(sections: list[Section], threshold: float = 0.6) -> None:
    """Sections whose lyrics repeat belong together even if the arrangement differs (e.g. a
    stripped-down second chorus)."""
    for i, sec in enumerate(sections):
        if sec.repeat_of or not sec.lyric_text or len(norm_tokens(sec.lyric_text)) < 4:
            continue
        for prev in sections[:i]:
            if prev.repeat_of is None and prev.lyric_text and \
                    text_similarity(prev.lyric_text, sec.lyric_text) >= threshold:
                sec.repeat_of = prev.id
                sec.cluster = prev.cluster
                break


def label_sections(sections: list[Section], have_vocals: bool) -> None:
    """Heuristic first guess: intro/verse/chorus/bridge/outro/instrumental. The LLM refines it."""
    n = len(sections)
    if n == 0:
        return
    groups: dict[str, list[Section]] = {}
    for s in sections:
        groups.setdefault(s.cluster, []).append(s)

    def is_vocal(s: Section) -> bool:
        return s.vocal_ratio >= 0.25 if have_vocals else True

    chorus_cluster = None
    best = -1.0
    for cl, secs in groups.items():
        if len(secs) < 2 or not any(is_vocal(s) for s in secs):
            continue
        rep = np.mean([text_similarity(secs[0].lyric_text, s.lyric_text) for s in secs[1:]]) \
            if have_vocals and secs[0].lyric_text else 0.0
        score = float(np.mean([s.energy for s in secs])) + 0.5 * rep
        if score > best:
            best, chorus_cluster = score, cl

    for i, s in enumerate(sections):
        if i == 0 and have_vocals and s.vocal_ratio < 0.25 and n > 2 and s.cluster != chorus_cluster:
            s.label = "intro"
        elif i == n - 1 and have_vocals and s.vocal_ratio < 0.25 and n > 2:
            s.label = "outro"
        elif s.cluster == chorus_cluster:
            s.label = "chorus"
        elif have_vocals and s.vocal_ratio < 0.25:
            s.label = "instrumental"
        else:
            repeated = len(groups[s.cluster]) > 1
            late = i / max(n - 1, 1) > 0.55
            s.label = "verse" if repeated or not late else "bridge"
    if not have_vocals:
        # without vocal info, fall back to position: first = intro, last = outro
        if n > 2:
            sections[0].label = sections[0].label if sections[0].label == "chorus" else "intro"
            sections[-1].label = sections[-1].label if sections[-1].label == "chorus" else "outro"


# ------------------------------------------------------------------ repeat alignment


def repeat_alignment(onset_env: np.ndarray, onset_times: np.ndarray, a: Section, b: Section,
                     max_lag_s: float = 0.5) -> RepeatAlignment:
    """Cross-correlate the onset envelopes of two repeats to decide if a clip can be reused."""
    dt = float(np.median(np.diff(onset_times))) if len(onset_times) > 1 else 0.023

    def seg(s: Section) -> np.ndarray:
        i0, i1 = int(s.start / dt), int(s.end / dt)
        x = onset_env[i0:i1].astype(float)
        return (x - x.mean()) / (x.std() + 1e-9)

    xa, xb = seg(a), seg(b)
    n = min(len(xa), len(xb))
    if n < 8:
        return RepeatAlignment(a=a.id, b=b.id, lag_s=0.0, similarity=0.0, identical_ok=False)
    xa, xb = xa[:n], xb[:n]
    max_lag = int(max_lag_s / dt)
    best_lag, best = 0, -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            u, v = xa[: n - lag], xb[lag:]
        else:
            u, v = xa[-lag:], xb[: n + lag]
        if len(u) < n // 2:
            continue
        c = float(np.dot(u, v) / len(u))
        if c > best:
            best, best_lag = c, lag
    lag_s = best_lag * dt
    dur_ok = abs(a.duration - b.duration) < 0.15
    return RepeatAlignment(
        a=a.id, b=b.id, lag_s=round(lag_s, 3), similarity=round(max(best, 0.0), 3),
        identical_ok=bool(dur_ok and abs(lag_s) <= 0.06 and best >= 0.75),
    )
