"""Lyrics with timings: LRC parsing, ASR (faster-whisper), and alignment of user-provided text."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from ..config import AsrCfg
from ..models import LyricLine, Word
from .structure import norm_tokens

_LRC = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


def parse_lrc(text: str, duration: float) -> list[LyricLine]:
    """Standard line-level .lrc. Line end = next line start (capped at 8 s)."""
    stamped: list[tuple[float, str]] = []
    for raw in text.splitlines():
        m = _LRC.match(raw.strip())
        if m and m.group(3).strip():
            stamped.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    stamped.sort()
    lines = []
    for i, (t, txt) in enumerate(stamped):
        nxt = stamped[i + 1][0] if i + 1 < len(stamped) else duration
        end = min(nxt, t + 8.0, duration)
        lines.append(LyricLine(start=t, end=max(end, t + 0.5), text=txt))
    return lines


def _preload_cuda12_libs() -> None:
    """ctranslate2 wheels link against CUDA 12 (cublas/cudnn); a CUDA 13 system toolkit does not
    provide them. The `nvidia-*-cu12` pip wheels do, so load them before ctranslate2 does."""
    import ctypes
    import glob
    import importlib.util
    for pkg, pattern in (("nvidia.cublas", "libcublas.so.12*"), ("nvidia.cudnn", "libcudnn.so.9*")):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        for base in (spec.submodule_search_locations or []) if spec else []:
            for lib in glob.glob(f"{base}/lib/{pattern}"):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _load_whisper(cfg: AsrCfg):  # noqa: ANN202
    from faster_whisper import WhisperModel
    if cfg.device != "cpu":
        _preload_cuda12_libs()
        try:
            return WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        except (RuntimeError, ValueError) as e:  # missing CUDA libs / unsupported compute type
            print(f"lyrics: GPU ASR unavailable ({str(e)[:120]}); falling back to CPU int8")
    return WhisperModel(cfg.model, device="cpu", compute_type="int8")


def transcribe(wav: Path, cfg: AsrCfg, language: str | None) -> list[LyricLine]:
    """Word-timestamped transcription. Raises a clear error if faster-whisper is not installed."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError("faster-whisper missing: pip install 'mvstudio[asr]'") from e

    model = _load_whisper(cfg)
    segments, info = model.transcribe(
        str(wav),
        language=language or cfg.language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # avoids runaway repetition on sung text
        no_speech_threshold=0.5,
        temperature=0.0,
    )
    lines: list[LyricLine] = []
    for seg in segments:
        words = [Word(start=w.start, end=w.end, text=w.word.strip()) for w in (seg.words or [])
                 if w.word.strip()]
        text = seg.text.strip()
        if not text or not words:
            continue
        lines.append(LyricLine(start=words[0].start, end=words[-1].end, text=text, words=words))
    return lines


def align_provided(lyrics: str, asr_lines: list[LyricLine], duration: float) -> list[LyricLine]:
    """Give each line of the user's lyrics a time span by matching its words against the ASR words.

    Unmatched words get linearly interpolated times between their matched neighbours, so a
    mis-heard ASR word does not lose the line. Returns [] if no word of the lyrics could be
    matched at all (wrong song / wrong language): better no timings than invented ones.
    """
    asr_words = [w for ln in asr_lines for w in ln.words]
    asr_tok = [norm_tokens(w.text) for w in asr_words]
    a_flat = [t[0] if t else "" for t in asr_tok]

    user_lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
    user_lines = [ln for ln in user_lines if not re.fullmatch(r"\[.*\]|\(.*\)", ln)]  # [Chorus]
    flat: list[str] = []
    owner: list[int] = []
    for li, ln in enumerate(user_lines):
        for tok in norm_tokens(ln):
            flat.append(tok)
            owner.append(li)

    times: list[tuple[float, float] | None] = [None] * len(flat)
    sm = difflib.SequenceMatcher(a=flat, b=a_flat, autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            w = asr_words[blk.b + k]
            times[blk.a + k] = (w.start, w.end)

    # interpolate gaps
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        return []
    for i in range(len(times)):
        if times[i] is not None:
            continue
        left = max((k for k in known if k < i), default=None)
        right = min((k for k in known if k > i), default=None)
        if left is None:
            r = times[right]
            span = 0.35 * (right - i)
            times[i] = (max(0.0, r[0] - span), max(0.0, r[0] - span + 0.3))
        elif right is None:
            lft = times[left]
            off = 0.35 * (i - left)
            times[i] = (min(duration, lft[1] + off), min(duration, lft[1] + off + 0.3))
        else:
            l_, r_ = times[left], times[right]
            frac = (i - left) / (right - left)
            s = l_[1] + frac * (r_[0] - l_[1])
            times[i] = (s, s + 0.3)

    out: list[LyricLine] = []
    for li, text in enumerate(user_lines):
        idx = [i for i, o in enumerate(owner) if o == li]
        if not idx:
            continue
        words = [Word(start=times[i][0], end=times[i][1], text=flat[i]) for i in idx]
        out.append(LyricLine(start=words[0].start, end=words[-1].end, text=text, words=words))
    return out
