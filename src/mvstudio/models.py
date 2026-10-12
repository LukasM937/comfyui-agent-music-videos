"""Data model shared by all stages. Everything on disk is JSON/YAML validated by these classes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- user input


class Wardrobe(BaseModel):
    id: str
    description: str


class Character(BaseModel):
    """A recurring person/creature. Appearance text is *canonical*: it is injected verbatim into
    every keyframe prompt, so keep it concrete (age, build, hair, clothes, signature props)."""

    id: str
    name: str
    description: str
    role: str = "supporting"  # performer | protagonist | supporting | extra
    is_performer: bool = False  # the singer / band member who is shown singing
    wardrobes: list[Wardrobe] = Field(default_factory=list)
    # Images supplied by the user, relative to the project dir.
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_wardrobe(self) -> Character:
        if not self.wardrobes:
            self.wardrobes = [Wardrobe(id="default", description="as described")]
        return self


class Brief(BaseModel):
    title: str
    theme: str = ""  # one-sentence theme / what the song is about
    style: str = ""  # visual style, e.g. "16mm film, grainy, warm tungsten"
    story_notes: str = ""  # free text: what the user roughly wants to see
    mood: list[str] = Field(default_factory=list)
    language: str | None = None  # lyrics language hint
    lyrics: str | None = None  # optional full lyrics (strongly improves alignment)
    avoid: list[str] = Field(default_factory=list)  # hard no-go list
    aspect: Literal["16:9", "9:16", "1:1", "21:9"] = "16:9"
    characters: list[Character] = Field(default_factory=list)

    def character(self, cid: str) -> Character | None:
        return next((c for c in self.characters if c.id == cid), None)

    @property
    def performer(self) -> Character | None:
        return next((c for c in self.characters if c.is_performer), None)


# --------------------------------------------------------------------------- audio analysis


class Word(BaseModel):
    start: float
    end: float
    text: str


class LyricLine(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Section(BaseModel):
    id: str  # S01, S02, ...
    start: float
    end: float
    bars: float
    cluster: str  # acoustic similarity cluster: A, B, C... (same letter = repeats)
    energy: float  # 0..1 relative loudness/onset density within this song
    vocal_ratio: float = 0.0  # fraction of the section where vocals are present
    label: str = ""  # intro/verse/chorus/... (set from heuristics, refined by LLM)
    repeat_of: str | None = None  # id of the first section this one repeats
    lyric_text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class RepeatAlignment(BaseModel):
    """How exactly two similarly long sections line up in time. Decides whether a clip can be
    reused verbatim for the later one."""

    a: str
    b: str
    lag_s: float  # shift of b relative to a (positive = b is later) after aligning
    similarity: float  # 0..1 correlation of the onset envelopes
    identical_ok: bool


class AudioAnalysis(BaseModel):
    audio_file: str
    duration: float
    sr: int
    bpm: float
    beats: list[float]
    downbeats: list[float]
    sections: list[Section]
    lyrics: list[LyricLine] = Field(default_factory=list)
    lyrics_source: Literal["none", "asr", "provided+asr", "lrc"] = "none"
    energy_curve: list[float] = Field(default_factory=list)  # 1 value per second, 0..1
    repeats: list[RepeatAlignment] = Field(default_factory=list)
    language: str | None = None

    def section(self, sid: str) -> Section:
        return next(s for s in self.sections if s.id == sid)

    def nearest_beat(self, t: float, grid: str = "beat") -> float:
        pts = self.downbeats if grid == "downbeat" else self.beats
        if not pts:
            return t
        return min(pts, key=lambda b: abs(b - t))


# --------------------------------------------------------------------------- creative plan


class Location(BaseModel):
    id: str
    name: str
    description: str
    time_of_day: str = "unspecified"
    # filled by `mvs locations`: plate image used as reference for all shots at this location
    plate: str | None = None


class Beat(BaseModel):
    """One step of the narrative arc, tied to a section."""

    section_id: str
    story: str  # what happens
    emotion: str = ""


class SectionRole(BaseModel):
    section_id: str
    label: str  # intro | verse | pre-chorus | chorus | bridge | instrumental | outro
    purpose: str = ""  # dramatic function of this section in the story


class Treatment(BaseModel):
    logline: str
    theme_statement: str
    visual_style: str  # one paragraph, prepended to every image/video prompt
    palette: list[str] = Field(default_factory=list)
    motifs: list[str] = Field(default_factory=list)
    locations: list[Location]
    arc: list[Beat]
    section_roles: list[SectionRole] = Field(default_factory=list)
    refrain_concept: str = ""  # what the recurring chorus imagery is and how it evolves
    continuity_rules: list[str] = Field(default_factory=list)

    def location(self, lid: str) -> Location | None:
        return next((x for x in self.locations if x.id == lid), None)


Performance = Literal["none", "lipsync", "dance", "instrument", "gesture"]
Transition = Literal["cut", "dissolve", "match_cut", "whip", "fade"]


class Shot(BaseModel):
    id: str
    section_id: str
    start: float
    end: float
    lyric_text: str = ""
    # --- creative content (LLM) ---
    shot_type: str = "medium"  # wide/medium/close-up/insert/aerial/...
    camera: str = "static"
    subject_ids: list[str] = Field(default_factory=list)
    location_id: str = ""
    wardrobe_id: str = "default"
    time_of_day: str = ""
    action: str = ""
    mood: str = ""
    intensity: float = 0.5
    performance: Performance = "none"
    transition_in: Transition = "cut"
    location_change_reason: str = ""  # required if location differs from previous shot
    continuity_notes: str = ""
    keyframe_prompt: str = ""
    video_prompt: str = ""
    # --- refrain handling ---
    refrain_group: str | None = None  # e.g. "chorus"
    refrain_slot: int | None = None  # index of the shot inside the refrain pattern
    occurrence: int = 1  # 1st, 2nd, ... time this refrain is shown
    reuse_of: str | None = None  # render once, reuse the clip of this shot id
    variant_of: str | None = None  # same refrain slot in the first occurrence
    same_keyframe: bool = False  # variant shares the keyframe of `variant_of`
    # --- bookkeeping ---
    notes: list[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Scenebook(BaseModel):
    title: str
    treatment: Treatment
    shots: list[Shot]
    approved_hash: str | None = None
    revision: int = 0

    def shot(self, sid: str) -> Shot:
        return next(s for s in self.shots if s.id == sid)

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"approved_hash", "revision"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def approved(self) -> bool:
        return self.approved_hash is not None and self.approved_hash == self.content_hash()


# --------------------------------------------------------------------------- checks


class Finding(BaseModel):
    source: Literal["rule", "critic"]
    category: str  # coverage|timing|references|continuity|refrain|pacing|theme|logic|prompt|...
    severity: Literal["error", "warning", "info"]
    message: str
    shot_ids: list[str] = Field(default_factory=list)
    section_id: str | None = None
    suggestion: str = ""


class Report(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)  # critic scores 0..10
    verdict: Literal["pass", "revise"] = "revise"

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]


# --------------------------------------------------------------------------- helpers


def save_json(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_model[T: BaseModel](cls: type[T], path: Path) -> T:
    return cls.model_validate_json(path.read_text(encoding="utf-8"))
