"""Stage: Treatment + audio analysis -> Scenebook (timing from code, content from the LLM)."""

from __future__ import annotations

from collections.abc import Callable

from ..config import Config
from ..llm.client import LLMLike
from ..models import AudioAnalysis, Brief, Scenebook, Section, Shot, Treatment
from . import prompts
from .drafts import SectionDraft, ShotContent
from .skeleton import REFRAIN_LABELS, make_skeleton

Log = Callable[[str], None]

CONTENT_FIELDS = ["shot_type", "camera", "subject_ids", "location_id", "wardrobe_id",
                  "time_of_day", "action", "mood", "intensity", "performance", "transition_in",
                  "location_change_reason", "continuity_notes", "keyframe_prompt", "video_prompt"]


def performance_slots(slots: list[Shot], has_performer: bool, ratio: float) -> set[int]:
    """Which slots of a sung section should show the singer (1-based). Always the first lyric
    slot; the rest spread evenly so performance and story shots alternate."""
    if not has_performer:
        return set()
    sung = [i for i, s in enumerate(slots, start=1) if s.lyric_text]
    if not sung:
        return set()
    want = max(1, round(len(sung) * ratio))
    step = len(sung) / want
    return {sung[min(int(k * step), len(sung) - 1)] for k in range(want)}


def _apply(shot: Shot, c: ShotContent) -> None:
    for f in CONTENT_FIELDS:
        setattr(shot, f, getattr(c, f))


def _copy_content(dst: Shot, src: Shot) -> None:
    for f in CONTENT_FIELDS:
        v = getattr(src, f)
        setattr(dst, f, list(v) if isinstance(v, list) else v)


def fill_section(llm: LLMLike, brief: Brief, t: Treatment, a: AudioAnalysis, cfg: Config,
                 sec: Section, slots: list[Shot], previous: Shot | None,
                 base: list[Shot] | None, feedback: list[str] | None = None) -> None:
    perf = performance_slots(slots, brief.performer is not None, cfg.planning.performance_ratio)
    user = prompts.section_user(brief, t, a, sec, slots, perf, previous, base,
                                cfg.planning.refrain_mode, feedback)
    problem = ""
    draft: SectionDraft | None = None
    for _ in range(3):
        draft = llm.generate(SectionDraft, prompts.DIRECTOR,
                             user + (f"\n\nFix: {problem}" if problem else ""))
        got = sorted(c.slot for c in draft.shots)
        if got == list(range(1, len(slots) + 1)):
            break
        problem = f"you returned slots {got}; return exactly slots 1..{len(slots)}"
    else:
        # best effort: positional mapping
        assert draft is not None
        for i, c in enumerate(draft.shots[: len(slots)], start=1):
            c.slot = i
    assert draft is not None
    by_slot = {c.slot: c for c in draft.shots}
    for i, shot in enumerate(slots, start=1):
        c = by_slot.get(i)
        if c is None:
            raise ValueError(f"{sec.id}: LLM produced no content for slot {i}")
        _apply(shot, c)


def build_scenebook(llm: LLMLike, brief: Brief, t: Treatment, a: AudioAnalysis, cfg: Config,
                    log: Log = print) -> Scenebook:
    shots = make_skeleton(a, cfg)
    by_section: dict[str, list[Shot]] = {}
    for s in shots:
        by_section.setdefault(s.section_id, []).append(s)
    align = {(r.a, r.b): r for r in a.repeats}

    previous: Shot | None = None
    done: dict[str, list[Shot]] = {}
    for sec in a.sections:
        slots = by_section[sec.id]
        base = done.get(sec.repeat_of) if (sec.repeat_of and sec.label in REFRAIN_LABELS) else None
        if base is not None and len(base) != len(slots):
            base = None  # cut patterns differ: treat as fresh material, still same refrain group
        mode = cfg.planning.refrain_mode
        al = align.get((sec.repeat_of, sec.id)) if sec.repeat_of else None

        if base and mode == "identical" and al and al.identical_ok and \
                all(abs(b.duration - s.duration) < 0.05 for b, s in zip(base, slots, strict=True)):
            log(f"  {sec.id}: exact repeat of {sec.repeat_of} -> reusing its clips")
            for b, s in zip(base, slots, strict=True):
                _copy_content(s, b)
                s.reuse_of = b.reuse_of or b.id
                s.variant_of = b.variant_of or b.id
                s.same_keyframe = True
        else:
            log(f"  {sec.id}: {len(slots)} shots ({sec.label or '?'}"
                f"{', refrain return' if base else ''})")
            fill_section(llm, brief, t, a, cfg, sec, slots, previous, base)
            if base:
                for b, s in zip(base, slots, strict=True):
                    s.variant_of = b.variant_of or b.id
                    if mode in ("variants", "identical"):  # identical fell through: not exact
                        s.same_keyframe = True
                        s.keyframe_prompt = b.keyframe_prompt
        done[sec.id] = slots
        previous = slots[-1]

    return Scenebook(title=brief.title, treatment=t, shots=shots)


def refill_sections(llm: LLMLike, brief: Brief, sb: Scenebook, a: AudioAnalysis, cfg: Config,
                    feedback: dict[str, list[str]], log: Log = print) -> None:
    """Re-generate whole sections with the critic's feedback (keeps timing untouched)."""
    for sec in a.sections:
        if sec.id not in feedback:
            continue
        slots = [s for s in sb.shots if s.section_id == sec.id]
        idx = sb.shots.index(slots[0])
        previous = sb.shots[idx - 1] if idx > 0 else None
        base = None
        if sec.repeat_of and sec.label in REFRAIN_LABELS:
            cand = [s for s in sb.shots if s.section_id == sec.repeat_of]
            base = cand if len(cand) == len(slots) else None
        log(f"  revising {sec.id} ({len(feedback[sec.id])} issue(s))")
        fill_section(llm, brief, sb.treatment, a, cfg, sec, slots, previous, base, feedback[sec.id])
        for s in slots:
            s.reuse_of = None
        if base:
            for b, s in zip(base, slots, strict=True):
                s.variant_of = b.variant_of or b.id
