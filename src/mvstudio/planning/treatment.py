"""Stage: brief + audio analysis -> Treatment."""

from __future__ import annotations

from ..config import Config
from ..llm.client import LLMLike
from ..models import AudioAnalysis, Brief, Location, Treatment
from . import prompts
from .drafts import TreatmentDraft
from .skeleton import REFRAIN_LABELS


def make_treatment(llm: LLMLike, brief: Brief, a: AudioAnalysis, cfg: Config,
                   feedback: list[str] | None = None) -> Treatment:
    user = prompts.treatment_user(brief, a, cfg.planning.cut_style)
    if feedback:
        user += "\n\n## Problems found in your previous treatment\n- " + "\n- ".join(feedback)
    last_problems: list[str] = []
    for _ in range(3):
        d = llm.generate(TreatmentDraft, prompts.DIRECTOR,
                         user + ("\n\nFix: " + "; ".join(last_problems) if last_problems else ""))
        last_problems = check_draft(d, a)
        if not last_problems:
            break
    t = Treatment(
        logline=d.logline, theme_statement=d.theme_statement, visual_style=d.visual_style,
        palette=d.palette, motifs=d.motifs,
        locations=[Location(**x.model_dump()) for x in d.locations],
        arc=_complete(d.arc, a, "arc"), section_roles=_complete(d.section_roles, a, "roles"),
        refrain_concept=d.refrain_concept, continuity_rules=d.continuity_rules)
    return t


def check_draft(d: TreatmentDraft, a: AudioAnalysis) -> list[str]:
    ids = [s.id for s in a.sections]
    problems = []
    for name, items in (("arc", d.arc), ("section_roles", d.section_roles)):
        got = [x.section_id for x in items]
        if sorted(got) != sorted(ids):
            problems.append(f"{name} must contain exactly one entry per section id {ids}, got {got}")
    seen = set()
    for loc in d.locations:
        if loc.id in seen:
            problems.append(f"duplicate location id {loc.id}")
        seen.add(loc.id)
    if not d.locations:
        problems.append("at least one location is required")
    return problems


def _complete(items, a: AudioAnalysis, what: str):  # noqa: ANN001, ANN202
    """Order by section; drop unknown ids (the LLM got them wrong twice already)."""
    order = {s.id: i for i, s in enumerate(a.sections)}
    kept = [x for x in items if x.section_id in order]
    return sorted(kept, key=lambda x: order[x.section_id])


def apply_roles(a: AudioAnalysis, t: Treatment) -> None:
    """The director's labels beat our heuristics (it sees lyrics and story)."""
    for r in t.section_roles:
        a.section(r.section_id).label = r.label.strip().lower()
    # every later chorus is a return of the first one, even if the arrangement differs
    first_chorus: str | None = None
    for s in a.sections:
        if s.label not in REFRAIN_LABELS:
            continue
        if first_chorus is None:
            first_chorus = s.id
        elif s.repeat_of is None:
            s.repeat_of = first_chorus
