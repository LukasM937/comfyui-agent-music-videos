"""Turn scenebook entries into image/video prompts. Characters are written as [id] by the
director and expanded here, so appearance text lives in exactly one place (the brief)."""

from __future__ import annotations

import re

from ..models import Brief, Character, Shot, Treatment

_REF = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def wardrobe_text(c: Character, wid: str) -> str:
    w = next((x for x in c.wardrobes if x.id == wid), c.wardrobes[0])
    return w.description


def short_descriptor(c: Character) -> str:
    words = c.description.replace(";", ",").split(",")[0].split()
    return " ".join(words[:7])


def expand_keyframe(text: str, shot: Shot, brief: Brief, ref_slot: dict[str, int]) -> str:
    """[id] -> 'Name (the person from <imageK>, wearing ...)' when a reference image exists,
    otherwise the full canonical description."""
    def sub(m: re.Match[str]) -> str:
        c = brief.character(m.group(1))
        if c is None:
            return m.group(1)
        wear = wardrobe_text(c, shot.wardrobe_id)
        if c.id in ref_slot:
            return f"{c.name} (the person from <image{ref_slot[c.id]}>, wearing {wear})"
        return f"{c.name} ({c.description}, wearing {wear})"
    return _REF.sub(sub, text)


def keyframe_prompt(shot: Shot, brief: Brief, t: Treatment, ref_slot: dict[str, int],
                    plate_slot: int | None) -> str:
    loc = t.location(shot.location_id)
    parts = [t.visual_style.strip(), f"{shot.shot_type} shot.",
             expand_keyframe(shot.keyframe_prompt, shot, brief, ref_slot)]
    if loc:
        parts.append(f"Setting: {loc.name}. {loc.description}")
    if shot.time_of_day or (loc and loc.time_of_day != "unspecified"):
        parts.append(f"Time of day: {shot.time_of_day or loc.time_of_day}.")  # type: ignore[union-attr]
    if shot.mood:
        parts.append(f"Mood: {shot.mood}.")
    if plate_slot:
        parts.append(f"The environment follows <image{plate_slot}>, but use a new camera position "
                     "that fits the shot size; do not copy the framing of the references.")
    if ref_slot:
        parts.append("Keep each person's face, hair, build and clothing consistent with their "
                     "reference image.")
    parts.append("No text, no subtitles, no watermark.")
    return " ".join(p for p in parts if p)


def video_prompt(shot: Shot, brief: Brief, t: Treatment) -> str:
    def sub(m: re.Match[str]) -> str:
        c = brief.character(m.group(1))
        return f"{c.name}, the {short_descriptor(c)}" if c else m.group(1)
    body = _REF.sub(sub, shot.video_prompt)
    parts = [f"Use the provided start image as the first frame. {body}",
             f"Camera: {shot.camera}."]
    if shot.performance == "lipsync":
        singer = brief.performer
        who = singer.name if singer else "the singer"
        parts.append(f"{who} is singing: the mouth moves in sync with the vocal track, natural "
                     "head and shoulder movement, emotional expression.")
    elif shot.performance == "dance":
        parts.append("The body movement follows the rhythm of the music.")
    elif shot.performance == "instrument":
        parts.append("The hands play the instrument in time with the music.")
    parts.append("Cinematic, coherent motion, stable identity, no text.")
    return " ".join(parts)
