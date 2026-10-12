"""LLM-facing output schemas (kept separate from the stored models so the LLM is never asked for
fields it cannot know, like rendered file paths)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Beat, Performance, SectionRole, Transition


class LocationDraft(BaseModel):
    id: str = Field(description="snake_case id, e.g. rooftop_dusk")
    name: str
    description: str = Field(description="concrete, filmable description of the place (<= 50 words)")
    time_of_day: str = "unspecified"


class TreatmentDraft(BaseModel):
    logline: str
    theme_statement: str = Field(description="what the video says about the song's theme")
    visual_style: str = Field(description="one paragraph (<= 80 words): lens, light, grade, texture")
    palette: list[str] = Field(description="3-6 colour words")
    motifs: list[str] = Field(description="3-6 recurring visual motifs")
    locations: list[LocationDraft]
    arc: list[Beat] = Field(description="exactly one beat per section id, in order")
    section_roles: list[SectionRole] = Field(description="one per section id, in order")
    refrain_concept: str = Field(
        description="what the recurring refrain imagery is and how it evolves with each return")
    continuity_rules: list[str] = Field(description="3-6 hard rules that must hold in every shot")


class ShotContent(BaseModel):
    slot: int = Field(description="1-based index of the time slot this content is for")
    shot_type: str = Field(description="wide | medium | close-up | extreme close-up | insert | "
                                       "aerial | over-the-shoulder | low angle | ...")
    camera: str = Field(description="camera move, e.g. static, slow push-in, handheld, orbit")
    subject_ids: list[str] = Field(description="character ids visible (may be empty)")
    location_id: str
    wardrobe_id: str = "default"
    time_of_day: str = ""
    action: str = Field(description="what happens, concrete and filmable, 1-2 sentences")
    mood: str = ""
    intensity: float = Field(ge=0.0, le=1.0, description="visual energy 0..1")
    performance: Performance = "none"
    transition_in: Transition = "cut"
    location_change_reason: str = Field(
        default="", description="REQUIRED when location differs from the previous shot")
    continuity_notes: str = ""
    keyframe_prompt: str = Field(
        description="still image of the first frame: subject, setting, composition, light. "
                    "Refer to characters as [id], never describe their faces/clothes")
    video_prompt: str = Field(
        description="motion over the clip: what moves, camera move, pacing. "
                    "Refer to characters as [id]")


class SectionDraft(BaseModel):
    shots: list[ShotContent]


class CriticFinding(BaseModel):
    category: str = Field(description="theme | logic | continuity | pacing | refrain | cliche | "
                                      "safety | prompt | other")
    severity: str = Field(description="error | warning | info")
    shot_ids: list[str] = Field(default_factory=list)
    section_id: str | None = None
    message: str
    suggestion: str = ""


class CriticDraft(BaseModel):
    scores: dict[str, float] = Field(
        description="0-10 for: theme_fit, narrative_logic, continuity, pacing, refrain, originality")
    findings: list[CriticFinding]
    verdict: str = Field(description="pass | revise")
