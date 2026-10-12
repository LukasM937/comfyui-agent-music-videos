"""Scenebook review: deterministic rules first (cheap, exact), then an LLM script supervisor."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from ..config import Config
from ..llm.client import LLMLike
from ..models import AudioAnalysis, Brief, Finding, Report, Scenebook, Shot
from . import prompts
from .drafts import CriticDraft

_REF = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def _f(cat: str, sev: str, msg: str, shots: list[str] | None = None, sec: str | None = None,
       sug: str = "") -> Finding:
    return Finding(source="rule", category=cat, severity=sev,  # type: ignore[arg-type]
                   message=msg, shot_ids=shots or [], section_id=sec, suggestion=sug)


def rule_checks(sb: Scenebook, brief: Brief, a: AudioAnalysis, cfg: Config) -> list[Finding]:
    out: list[Finding] = []
    shots = sb.shots
    tol = cfg.planning.beat_tolerance_s
    chars = {c.id: c for c in brief.characters}
    locs = {x.id for x in sb.treatment.locations}
    beats = np.array(a.beats) if a.beats else np.array([0.0])

    # ---- coverage / timing
    if not shots:
        return [_f("coverage", "error", "scenebook has no shots")]
    if abs(shots[0].start) > 1e-3:
        out.append(_f("coverage", "error", f"first shot starts at {shots[0].start:.2f}s, not 0"))
    if abs(shots[-1].end - a.duration) > 0.05:
        out.append(_f("coverage", "error",
                      f"last shot ends at {shots[-1].end:.2f}s but the song is {a.duration:.2f}s"))
    for p, n in zip(shots, shots[1:], strict=False):
        if abs(n.start - p.end) > 1e-3:
            out.append(_f("coverage", "error", f"gap/overlap between {p.id} and {n.id}",
                          [p.id, n.id]))
    for s in shots:
        d = s.duration
        if d > cfg.video.max_clip_s + 1e-3:
            out.append(_f("timing", "error", f"{s.id} is {d:.1f}s, model limit is "
                          f"{cfg.video.max_clip_s:.0f}s", [s.id]))
        if d < cfg.video.min_shot_s - 0.05:
            out.append(_f("timing", "warning", f"{s.id} is only {d:.1f}s", [s.id]))
        sec = a.section(s.section_id)
        if s.start < sec.start - 0.05 or s.end > sec.end + 0.05:
            out.append(_f("timing", "error", f"{s.id} crosses the boundary of {sec.id}", [s.id]))
        if s.start > 0 and np.min(np.abs(beats - s.start)) > tol:
            out.append(_f("timing", "warning", f"{s.id} starts off the beat grid", [s.id]))

    # ---- references
    for s in shots:
        if s.location_id not in locs:
            out.append(_f("references", "error", f"{s.id}: unknown location '{s.location_id}'",
                          [s.id]))
        for cid in s.subject_ids:
            c = chars.get(cid)
            if c is None:
                out.append(_f("references", "error", f"{s.id}: unknown character '{cid}'", [s.id]))
            elif s.wardrobe_id not in {w.id for w in c.wardrobes}:
                out.append(_f("references", "error",
                              f"{s.id}: {cid} has no wardrobe '{s.wardrobe_id}'", [s.id]))
        for field in ("keyframe_prompt", "video_prompt"):
            text = getattr(s, field)
            if not text.strip():
                out.append(_f("prompt", "error", f"{s.id}: empty {field}", [s.id]))
            for tok in _REF.findall(text):
                if tok not in chars:
                    out.append(_f("prompt", "warning",
                                  f"{s.id}: {field} mentions [{tok}] which is not in the cast",
                                  [s.id]))
        for cid in s.subject_ids:
            if f"[{cid}]" not in s.keyframe_prompt:
                out.append(_f("prompt", "warning",
                              f"{s.id}: {cid} is a subject but [{cid}] is not in the keyframe "
                              "prompt (character refs would be lost)", [s.id]))
    used = {c for s in shots for c in s.subject_ids}
    for cid, c in chars.items():
        if cid not in used and c.role != "extra":
            out.append(_f("references", "warning", f"cast member '{cid}' never appears"))

    # ---- performance
    perf = brief.performer
    for s in shots:
        if s.performance == "lipsync":
            if perf is None or perf.id not in s.subject_ids:
                out.append(_f("logic", "error", f"{s.id}: lipsync shot without the performer",
                              [s.id]))
            if not s.lyric_text:
                out.append(_f("logic", "warning", f"{s.id}: lipsync shot but nobody sings here",
                              [s.id]))
            if s.shot_type.lower().startswith(("wide", "aerial", "extreme wide")):
                out.append(_f("logic", "warning", f"{s.id}: lipsync in a wide shot will not read",
                              [s.id]))

    # ---- continuity
    for p, n in zip(shots, shots[1:], strict=False):
        if p.location_id != n.location_id and not n.location_change_reason.strip() \
                and n.transition_in == "cut" and p.section_id == n.section_id:
            out.append(_f("continuity", "warning",
                          f"{p.id}->{n.id}: location changes ({p.location_id} -> {n.location_id}) "
                          "within a section with no reason or transition", [p.id, n.id],
                          sug="add location_change_reason or keep the location"))
        shared = set(p.subject_ids) & set(n.subject_ids)
        for cid in shared:
            if p.wardrobe_id != n.wardrobe_id and not n.continuity_notes.strip():
                out.append(_f("continuity", "warning",
                              f"{p.id}->{n.id}: {cid} changes wardrobe ({p.wardrobe_id} -> "
                              f"{n.wardrobe_id}) without a note", [p.id, n.id]))
    # a character cannot be in two places inside the same second
    for p, n in zip(shots, shots[1:], strict=False):
        if p.location_id != n.location_id and set(p.subject_ids) & set(n.subject_ids) \
                and n.transition_in == "match_cut":
            out.append(_f("continuity", "info", f"{p.id}->{n.id}: match cut between locations "
                          "with the same character; make sure the eyeline/pose matches",
                          [p.id, n.id]))

    # ---- refrain
    groups: dict[str, list[Shot]] = defaultdict(list)
    for s in shots:
        if s.refrain_group:
            groups[s.refrain_group].append(s)
    for members in groups.values():
        occ = sorted({m.occurrence for m in members})
        if len(occ) < 2:
            continue
        first = [m for m in members if m.occurrence == occ[0]]
        for o in occ[1:]:
            later = [m for m in members if m.occurrence == o]
            if len(later) != len(first):
                out.append(_f("refrain", "warning",
                              f"refrain occurrence {o} has {len(later)} shots, the first has "
                              f"{len(first)}: the returns should rhyme visually",
                              [m.id for m in later]))
                continue
            for b, m in zip(first, later, strict=True):
                if b.location_id != m.location_id:
                    out.append(_f("refrain", "warning",
                                  f"{m.id} moves the refrain image to another location than {b.id}",
                                  [b.id, m.id]))
            if cfg.planning.refrain_mode == "escalate":
                i0 = float(np.mean([b.intensity for b in first]))
                i1 = float(np.mean([m.intensity for m in later]))
                if i1 + 0.05 < i0:
                    out.append(_f("refrain", "warning",
                                  f"refrain occurrence {o} is calmer ({i1:.2f}) than the first "
                                  f"({i0:.2f}) although mode is 'escalate'",
                                  [m.id for m in later]))
    for s in shots:
        if s.reuse_of:
            src = next((x for x in shots if x.id == s.reuse_of), None)
            if src is None or src.reuse_of:
                out.append(_f("refrain", "error", f"{s.id}: reuse_of must point at a rendered "
                              f"shot, got {s.reuse_of}", [s.id]))
            elif abs(src.duration - s.duration) > 0.1:
                out.append(_f("refrain", "error", f"{s.id}: reuse_of {src.id} has a different "
                              "duration", [s.id, src.id]))

    # ---- pacing against the music
    by_sec: dict[str, list[Shot]] = defaultdict(list)
    for s in shots:
        by_sec[s.section_id].append(s)
    secs = a.sections
    if len(secs) >= 3:
        order = sorted(secs, key=lambda x: x.energy)
        calm, loud = order[0], order[-1]
        d_calm = np.mean([s.duration for s in by_sec[calm.id]])
        d_loud = np.mean([s.duration for s in by_sec[loud.id]])
        if d_loud > d_calm * 1.15:
            out.append(_f("pacing", "warning",
                          f"the loudest section {loud.id} is cut slower ({d_loud:.1f}s/shot) than "
                          f"the calmest {calm.id} ({d_calm:.1f}s/shot)", sec=loud.id))
    types = [s.shot_type.lower() for s in shots]
    for i in range(len(types) - 3):
        if len(set(types[i:i + 4])) == 1:
            out.append(_f("pacing", "info", f"four consecutive '{types[i]}' shots from "
                          f"{shots[i].id}: vary the shot sizes", [s.id for s in shots[i:i + 4]]))
            break

    # ---- hard no-go list
    for term in (t.lower() for t in brief.avoid):
        for s in shots:
            blob = " ".join([s.action, s.keyframe_prompt, s.video_prompt]).lower()
            if term and term in blob and not re.search(rf"(no|without|avoid)\s+{re.escape(term)}",
                                                       blob):
                out.append(_f("safety", "error", f"{s.id} mentions '{term}' (brief says avoid)",
                              [s.id]))
    return out


def critic_pass(llm: LLMLike, brief: Brief, sb: Scenebook, a: AudioAnalysis,
                rule_findings: list[Finding]) -> tuple[list[Finding], dict[str, float]]:
    known = [f"{f.severity}: {f.message}" for f in rule_findings]
    d = llm.generate(CriticDraft, prompts.CRITIC, prompts.critic_user(brief, sb, a, known),
                     thinking=False, temperature=0.3)
    valid_shots = {s.id for s in sb.shots}
    findings = []
    for x in d.findings:
        sev = x.severity if x.severity in ("error", "warning", "info") else "warning"
        findings.append(Finding(
            source="critic", category=x.category, severity=sev,  # type: ignore[arg-type]
            message=x.message, shot_ids=[i for i in x.shot_ids if i in valid_shots],
            section_id=x.section_id, suggestion=x.suggestion))
    return findings, d.scores


def review(llm: LLMLike | None, brief: Brief, sb: Scenebook, a: AudioAnalysis,
           cfg: Config) -> Report:
    rules = rule_checks(sb, brief, a, cfg)
    findings = list(rules)
    scores: dict[str, float] = {}
    if llm is not None:
        crit, scores = critic_pass(llm, brief, sb, a, rules)
        findings += crit
    rep = Report(findings=findings, scores=scores)
    rep.verdict = "pass" if not rep.errors else "revise"
    return rep


def feedback_by_section(rep: Report, sb: Scenebook) -> dict[str, list[str]]:
    """Group errors (and warnings the critic raised) by the section that has to be redone."""
    sec_of = {s.id: s.section_id for s in sb.shots}
    out: dict[str, list[str]] = defaultdict(list)
    for f in rep.findings:
        if f.severity == "info":
            continue
        if f.severity == "warning" and f.source == "rule" and f.category in ("pacing", "timing"):
            continue  # timing is code-owned; the LLM cannot fix it by rewriting content
        secs = {sec_of[i] for i in f.shot_ids if i in sec_of}
        if f.section_id:
            secs.add(f.section_id)
        text = f"{', '.join(f.shot_ids) or f.section_id or 'general'}: {f.message}" + \
               (f" -> {f.suggestion}" if f.suggestion else "")
        for sid in secs:
            out[sid].append(text)
    return dict(out)

