"""Prompt texts and context formatting for the planning LLM calls."""

from __future__ import annotations

import json

from ..audio.analyze import describe
from ..models import AudioAnalysis, Brief, Scenebook, Section, Shot, Treatment

DIRECTOR = """\
You are a music-video director and script supervisor working with a very capable but literal
video model. You turn a song's audio analysis, lyrics and a rough brief into a coherent, filmable
visual plan.

Principles
- Ground every choice in the song: lyrics, mood, tempo, energy. Do not ignore the brief's theme.
- Be concrete and filmable: nouns, light, motion. No abstract adjectives without an image.
- Story logic matters: people and objects cannot teleport; if the location changes, say why
  (journey, memory, cut-away) and use a fitting transition.
- Use ONLY the cast and locations you are given. Refer to people as [character_id].
- Respect the brief's "avoid" list absolutely. No on-screen text, subtitles or logos.
- Refrains return: same core imagery each time, evolving, never random.
- Write in English regardless of the lyrics' language.
- Answer with JSON matching the provided schema, nothing else."""

CRITIC = """\
You are a strict script supervisor reviewing a music-video scenebook before an expensive render.
Find real problems; do not praise. Check:
1. theme_fit: does the visual story serve the stated theme, style and lyrics? Are lyrics
   illustrated sensibly (literal where it helps, metaphorical where it works)?
2. narrative_logic: causality, geography, time of day, props, who is where; unexplained jumps.
3. continuity: wardrobe, location and time-of-day changes between consecutive shots.
4. pacing: do shot lengths and intensity follow the song's energy? Variety of shot sizes?
5. refrain: do refrains return with recognisable imagery, and do they evolve sensibly?
6. originality: clichés (generic neon city, random rain) without purpose.
7. safety: anything in the brief's avoid list; anything that would be impossible to render
   (readable text, exact choreography, crowds of named people, hands doing fine manipulation).
Severity: error = must fix before rendering; warning = should fix; info = optional.
Reference shot ids precisely. Score 0-10 per category. verdict = pass only if no errors."""


def cast_block(brief: Brief) -> str:
    out = []
    for c in brief.characters:
        w = "; ".join(f"{x.id} = {x.description}" for x in c.wardrobes)
        flag = " (the singer, may be shown singing)" if c.is_performer else ""
        out.append(f"- {c.id}: {c.name}, {c.role}{flag}. {c.description}. Wardrobes: {w}")
    return "\n".join(out) or "(no named characters; use anonymous figures sparingly)"


def brief_block(brief: Brief) -> str:
    parts = [f"Title: {brief.title}"]
    if brief.theme:
        parts.append(f"Theme: {brief.theme}")
    if brief.style:
        parts.append(f"Visual style: {brief.style}")
    if brief.mood:
        parts.append(f"Mood: {', '.join(brief.mood)}")
    if brief.story_notes.strip():
        parts.append(f"Story notes from the artist:\n{brief.story_notes.strip()}")
    if brief.avoid:
        parts.append("AVOID (hard): " + "; ".join(brief.avoid))
    parts.append(f"Aspect ratio: {brief.aspect}")
    return "\n".join(parts)


def treatment_user(brief: Brief, a: AudioAnalysis, cut_style: str) -> str:
    return (
        f"## Brief\n{brief_block(brief)}\n\n## Cast\n{cast_block(brief)}\n\n"
        f"## Audio analysis (sections are fixed; labels are heuristic guesses you may correct)\n"
        f"{describe(a)}\n\n## Task\nCut style: {cut_style}.\n"
        "Write the treatment: logline, theme statement, one visual-style paragraph, palette, "
        "motifs, 3-7 locations, ONE arc beat and ONE section role per section id (all of them, in "
        "order), a refrain concept (how the chorus imagery recurs and evolves), and 3-6 continuity "
        "rules. Use section roles' labels from: intro, verse, pre-chorus, chorus, bridge, "
        "instrumental, outro. Sections that repeat acoustically or lyrically (see 'repeat of') "
        "usually share a role."
    )


def treatment_block(t: Treatment) -> str:
    locs = "\n".join(f"- {x.id}: {x.name} ({x.time_of_day}). {x.description}" for x in t.locations)
    return (f"Logline: {t.logline}\nTheme: {t.theme_statement}\nStyle: {t.visual_style}\n"
            f"Motifs: {', '.join(t.motifs)}\nRefrain concept: {t.refrain_concept}\n"
            f"Rules: {'; '.join(t.continuity_rules)}\nLocations:\n{locs}")


def shot_line(s: Shot) -> str:
    return (f"{s.id} [{s.start:.1f}-{s.end:.1f}s] {s.shot_type}, {s.camera} | loc={s.location_id} "
            f"ward={s.wardrobe_id} tod={s.time_of_day} subj={','.join(s.subject_ids) or '-'} "
            f"perf={s.performance} int={s.intensity:.1f} | {s.action}")


def section_user(brief: Brief, t: Treatment, a: AudioAnalysis, sec: Section, slots: list[Shot],
                 perf_slots: set[int], previous: Shot | None, base: list[Shot] | None,
                 refrain_mode: str, feedback: list[str] | None) -> str:
    beat = next((b for b in t.arc if b.section_id == sec.id), None)
    role = next((r for r in t.section_roles if r.section_id == sec.id), None)
    slot_rows = []
    for i, s in enumerate(slots, start=1):
        slot_rows.append({
            "slot": i, "start": round(s.start, 2), "end": round(s.end, 2),
            "duration": round(s.duration, 2), "lyrics": s.lyric_text,
            "singer_on_camera_suggested": i in perf_slots})
    txt = [
        f"## Treatment\n{treatment_block(t)}",
        f"## Cast\n{cast_block(brief)}",
        f"## This section: {sec.id} ({role.label if role else sec.label}) "
        f"{sec.start:.1f}-{sec.end:.1f}s, energy {sec.energy:.2f}\n"
        f"Purpose: {role.purpose if role else ''}\nArc beat: {beat.story if beat else ''} "
        f"({beat.emotion if beat else ''})\nLyrics: {sec.lyric_text or '(instrumental)'}",
        "## Time slots (fixed; produce exactly one ShotContent per slot, same order)\n"
        + json.dumps(slot_rows, ensure_ascii=False, indent=1),
    ]
    if previous:
        txt.append("## Previous shot (continue from it)\n" + shot_line(previous))
    if base:
        verb = {"variants": "Make an alternate take of EACH shot: same location, subjects, "
                            "blocking and shot sizes; vary lens/angle/camera move and small "
                            "details. Keep the keyframe_prompt the same.",
                "escalate": "Each time a refrain returns it must feel bigger or more resolved: "
                            "keep the same location and the same core image per slot, but raise "
                            "the intensity and change something visible (light, weather, crowd, "
                            "distance, colour). Same subjects and wardrobe unless the story "
                            "justifies a change."}.get(refrain_mode, "")
        txt.append("## This section is a RETURN of an earlier refrain. First occurrence:\n"
                   + "\n".join(shot_line(s) for s in base) + f"\n{verb}")
    if feedback:
        txt.append("## Fix these problems found in your previous attempt\n- "
                   + "\n- ".join(feedback))
    txt.append("Singer-on-camera suggestions are hints for performance shots (lip-sync). Use "
               "performance=lipsync only when the singer is clearly visible and singing. "
               "Output {\"shots\": [...]} with slot numbers 1..N.")
    return "\n\n".join(txt)


def critic_user(brief: Brief, sb: Scenebook, a: AudioAnalysis, rule_findings: list[str]) -> str:
    rows = "\n".join(
        f"{s.id} {s.section_id} {s.start:.1f}-{s.end:.1f}s occ{s.occurrence}"
        f"{' REFRAIN' if s.refrain_group else ''} | lyrics: {s.lyric_text[:90]} | "
        + shot_line(s).split("|", 1)[1] for s in sb.shots)
    return (f"## Brief\n{brief_block(brief)}\n\n## Cast\n{cast_block(brief)}\n\n"
            f"## Treatment\n{treatment_block(sb.treatment)}\n\n## Audio\n{describe(a)}\n\n"
            f"## Scenebook ({len(sb.shots)} shots)\n{rows}\n\n"
            "## Findings from automatic checks (already known; do not repeat)\n"
            + ("\n".join(rule_findings) or "none"))
