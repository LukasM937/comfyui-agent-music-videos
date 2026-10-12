"""Plan -> check -> revise loop."""

from __future__ import annotations

from collections.abc import Callable

from ..config import Config
from ..llm.client import LLMLike
from ..models import AudioAnalysis, Brief, Report, Scenebook, Treatment, save_json
from ..project import Project
from . import check, scenebook, treatment

Log = Callable[[str], None]


def make_plan(llm: LLMLike, project: Project, brief: Brief, a: AudioAnalysis, cfg: Config,
              log: Log = print) -> tuple[Scenebook, Report]:
    log("plan: treatment")
    t: Treatment = treatment.make_treatment(llm, brief, a, cfg)
    treatment.apply_roles(a, t)
    save_json(t, project.treatment_file)
    save_json(a, project.analysis_file)  # labels / repeat links refined by the director

    log("plan: scenebook (timing from the beat grid, content from the director)")
    sb = scenebook.build_scenebook(llm, brief, t, a, cfg, log)
    return revise_loop(llm, project, brief, sb, a, cfg, log)


def revise_loop(llm: LLMLike | None, project: Project, brief: Brief, sb: Scenebook,
                a: AudioAnalysis, cfg: Config, log: Log = print) -> tuple[Scenebook, Report]:
    rep = Report()
    for rnd in range(cfg.planning.max_revision_rounds + 1):
        log(f"check: round {rnd + 1}")
        rep = check.review(llm, brief, sb, a, cfg)
        log(f"  {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)"
            + (f", scores {rep.scores}" if rep.scores else ""))
        sb.revision = rnd
        save_json(sb, project.scenebook_file)
        save_json(rep, project.report_file)
        if rep.verdict == "pass" or llm is None or rnd == cfg.planning.max_revision_rounds:
            break
        fb = check.feedback_by_section(rep, sb)
        if not fb:
            break
        scenebook.refill_sections(llm, brief, sb, a, cfg, fb, log)
    return sb, rep
