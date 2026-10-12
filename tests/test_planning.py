import json
import re

import pytest
from helpers import analysis_from_truth

from mvstudio.config import Config
from mvstudio.models import AudioAnalysis, Brief, Character, RepeatAlignment, Wardrobe, load_model
from mvstudio.planning import check, treatment
from mvstudio.planning.drafts import (
    CriticDraft,
    LocationDraft,
    SectionDraft,
    ShotContent,
    TreatmentDraft,
)
from mvstudio.planning.pipeline import make_plan
from mvstudio.project import Project

BRIEF = Brief(
    title="Night Road", theme="leaving home", style="35mm, warm tungsten", avoid=["gore"],
    characters=[
        Character(id="singer", name="Mara", role="performer", is_performer=True,
                  description="woman, 30s, dark curls",
                  wardrobes=[Wardrobe(id="default", description="mustard cardigan")]),
        Character(id="friend", name="Joe", description="tall man, beard",
                  wardrobes=[Wardrobe(id="default", description="denim jacket")]),
    ])


class FakeLLM:
    """Deterministic stand-in for the LLM. Reads the fixed slots out of the prompt."""

    def __init__(self, bad_location_once: bool = False, critic_error_once: bool = False):
        self.calls: list[str] = []
        self.bad_location_once = bad_location_once
        self.critic_error_once = critic_error_once
        self.feedback_seen: list[str] = []

    def generate(self, schema, system, user, **kw):
        self.calls.append(schema.__name__)
        if schema is TreatmentDraft:
            ids = re.findall(r"^(S\d\d) ", user, re.M)
            from mvstudio.models import Beat, SectionRole
            labels = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
            return TreatmentDraft(
                logline="A woman leaves.", theme_statement="Leaving is a beginning.",
                visual_style="35mm warm", palette=["amber"], motifs=["road"],
                locations=[LocationDraft(id="road", name="Road", description="empty road"),
                           LocationDraft(id="room", name="Room", description="small room")],
                arc=[Beat(section_id=i, story="walks") for i in ids],
                section_roles=[SectionRole(section_id=i, label=labels[k])
                               for k, i in enumerate(ids)],
                refrain_concept="the road at dusk, wider each time",
                continuity_rules=["cardigan stays mustard"])
        if schema is SectionDraft:
            if "Fix these problems" in user:
                self.feedback_seen.append(user.split("Fix these problems", 1)[1][:200])
            rows = json.loads(user.split("## Time slots", 1)[1].split("\n", 1)[1]
                              .split("\n\n## ")[0].split("\n\n")[0])
            sec = re.search(r"## This section: (S\d\d)", user).group(1)
            out = []
            for r in rows:
                loc = "road"
                if self.bad_location_once and sec == "S02" and r["slot"] == 1 \
                        and not self.feedback_seen:
                    loc = "moon"
                perf = "lipsync" if r["singer_on_camera_suggested"] else "none"
                out.append(ShotContent(
                    slot=r["slot"], shot_type="close-up" if perf != "none" else "wide",
                    camera="static", subject_ids=["singer"], location_id=loc,
                    action="she walks", intensity=0.9 if "RETURN" in user else 0.5,
                    performance=perf,
                    keyframe_prompt="[singer] on the road", video_prompt="[singer] walks"))
            return SectionDraft(shots=out)
        if schema is CriticDraft:
            if self.critic_error_once:
                self.critic_error_once = False
                from mvstudio.planning.drafts import CriticFinding
                return CriticDraft(scores={"theme_fit": 4}, verdict="revise", findings=[
                    CriticFinding(category="theme", severity="error", shot_ids=["S04-01"],
                                  message="verse 2 ignores the leaving theme")])
            return CriticDraft(scores={"theme_fit": 8}, findings=[], verdict="pass")
        raise AssertionError(schema)


@pytest.fixture
def setup(tmp_path):
    p = Project(tmp_path)
    p.work.mkdir()
    return p, analysis_from_truth(), Config()


def test_full_plan_passes_all_rules(setup):
    p, a, cfg = setup
    sb, rep = make_plan(FakeLLM(), p, BRIEF, a, cfg, log=lambda *_: None)
    assert rep.verdict == "pass", [f.message for f in rep.errors]
    assert p.scenebook_file.exists() and p.report_file.exists() and p.treatment_file.exists()
    assert sb.shots[0].start == 0 and sb.shots[-1].end == pytest.approx(a.duration)
    assert all(s.keyframe_prompt and s.video_prompt for s in sb.shots)
    reloaded = load_model(type(sb), p.scenebook_file)
    assert reloaded.content_hash() == sb.content_hash()


def test_refrain_returns_are_linked_to_first_occurrence_and_escalate(setup):
    p, a, cfg = setup
    cfg.planning.refrain_mode = "escalate"
    sb, _ = make_plan(FakeLLM(), p, BRIEF, a, cfg, log=lambda *_: None)
    first = [s for s in sb.shots if s.section_id == "S03"]
    second = [s for s in sb.shots if s.section_id == "S05"]
    assert len(first) == len(second) > 1
    assert all(s.variant_of is None for s in first)
    assert [s.variant_of for s in second] == [s.id for s in first]
    assert not any(s.same_keyframe for s in second)
    assert min(s.intensity for s in second) > max(s.intensity for s in first)


def test_variants_mode_shares_keyframes(setup):
    p, a, cfg = setup
    cfg.planning.refrain_mode = "variants"
    sb, _ = make_plan(FakeLLM(), p, BRIEF, a, cfg, log=lambda *_: None)
    second = [s for s in sb.shots if s.section_id == "S05"]
    assert all(s.same_keyframe for s in second)


def test_identical_mode_reuses_clips_without_asking_the_llm(setup):
    p, a, cfg = setup
    cfg.planning.refrain_mode = "identical"
    a.repeats = [RepeatAlignment(a="S03", b="S05", lag_s=0.0, similarity=0.95, identical_ok=True),
                 RepeatAlignment(a="S03", b="S07", lag_s=0.0, similarity=0.95, identical_ok=True)]
    llm = FakeLLM()
    sb, rep = make_plan(llm, p, BRIEF, a, cfg, log=lambda *_: None)
    s5 = [s for s in sb.shots if s.section_id == "S05"]
    first = [s for s in sb.shots if s.section_id == "S03"]
    assert [s.reuse_of for s in s5] == [s.id for s in first]
    assert [s.action for s in s5] == [s.action for s in first]
    # intro, verse, chorus, verse, bridge, outro -> 6 section calls; S05/S07 reused
    assert llm.calls.count("SectionDraft") == 6
    assert rep.verdict == "pass"


def test_revision_loop_fixes_unknown_location_using_feedback(setup):
    p, a, cfg = setup
    llm = FakeLLM(bad_location_once=True)
    sb, rep = make_plan(llm, p, BRIEF, a, cfg, log=lambda *_: None)
    assert rep.verdict == "pass"
    assert llm.feedback_seen and "moon" in llm.feedback_seen[0]
    assert sb.shot("S02-01").location_id == "road"


def test_critic_errors_trigger_revision_of_the_named_section(setup):
    p, a, cfg = setup
    llm = FakeLLM(critic_error_once=True)
    _, rep = make_plan(llm, p, BRIEF, a, cfg, log=lambda *_: None)
    assert rep.verdict == "pass"
    assert any("leaving theme" in s for s in llm.feedback_seen)


def test_rule_checks_catch_structural_problems(setup):
    p, a, cfg = setup
    sb, _ = make_plan(FakeLLM(), p, BRIEF, a, cfg, log=lambda *_: None)
    bad = sb.model_copy(deep=True)
    bad.shots[3].end += 0.7  # gap/overlap with next
    bad.shots[5].location_id = "nowhere"
    bad.shots[6].subject_ids = ["ghost"]
    bad.shots[7].performance = "lipsync"
    bad.shots[7].subject_ids = ["friend"]
    bad.shots[8].keyframe_prompt = ""
    msgs = " | ".join(f.message for f in check.rule_checks(bad, BRIEF, a, cfg))
    assert "gap/overlap" in msgs
    assert "unknown location 'nowhere'" in msgs
    assert "unknown character 'ghost'" in msgs
    assert "lipsync shot without the performer" in msgs
    assert "empty keyframe_prompt" in msgs


def test_avoid_list_is_enforced(setup):
    p, a, cfg = setup
    sb, _ = make_plan(FakeLLM(), p, BRIEF, a, cfg, log=lambda *_: None)
    sb.shots[2].action = "a gore-filled scene"
    errs = [f for f in check.rule_checks(sb, BRIEF, a, cfg) if f.category == "safety"]
    assert errs and errs[0].shot_ids == [sb.shots[2].id]


def test_approval_hash_invalidates_on_edit(setup):
    p, a, cfg = setup
    sb, _ = make_plan(FakeLLM(), p, BRIEF, a, cfg, log=lambda *_: None)
    assert not sb.approved
    sb.approved_hash = sb.content_hash()
    assert sb.approved
    sb.shots[0].action = "changed"
    assert not sb.approved


def test_treatment_check_demands_every_section():
    a: AudioAnalysis = analysis_from_truth()
    d = FakeLLM().generate(TreatmentDraft, "", "\n".join(f"{s.id} x" for s in a.sections))
    assert treatment.check_draft(d, a) == []
    d.arc = d.arc[:-1]
    assert "arc must contain exactly one entry" in treatment.check_draft(d, a)[0]


def test_demo_brief_and_brief_template_are_valid():
    import pathlib

    import yaml

    from mvstudio.project import BRIEF_TEMPLATE
    root = pathlib.Path(__file__).resolve().parents[1]
    demo = Brief.model_validate(yaml.safe_load((root / "examples/demo-project/brief.yaml").read_text()))
    assert demo.performer and demo.performer.id == "singer" and len(demo.characters) == 2
    tpl = Brief.model_validate(yaml.safe_load(BRIEF_TEMPLATE.format(title="x")))
    assert tpl.title == "x" and tpl.performer is not None
