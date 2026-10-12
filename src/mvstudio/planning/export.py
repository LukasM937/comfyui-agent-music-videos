"""Human-readable views of the scenebook: Markdown (review/edit) and a static HTML board."""

from __future__ import annotations

import html

from ..models import AudioAnalysis, Report, Scenebook

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2", "#edc948",
           "#9c755f", "#bab0ac"]


def _t(x: float) -> str:
    m, s = divmod(x, 60)
    return f"{int(m)}:{s:04.1f}"


def to_markdown(sb: Scenebook, a: AudioAnalysis, rep: Report | None) -> str:
    t = sb.treatment
    out = [f"# {sb.title} — scenebook (rev {sb.revision})", "",
           f"**Logline:** {t.logline}", "", f"**Theme:** {t.theme_statement}", "",
           f"**Style:** {t.visual_style}", "",
           f"**Refrain:** {t.refrain_concept}", "",
           "**Locations:** " + "; ".join(f"`{x.id}` {x.name}" for x in t.locations), ""]
    if rep:
        out += [f"## Review — verdict: **{rep.verdict}**", ""]
        if rep.scores:
            out.append("Scores: " + ", ".join(f"{k} {v:g}" for k, v in rep.scores.items()))
            out.append("")
        for f in rep.findings:
            if f.severity == "info":
                continue
            where = f" ({', '.join(f.shot_ids)})" if f.shot_ids else ""
            out.append(f"- **{f.severity}** [{f.category}/{f.source}]{where} {f.message}"
                       + (f" → _{f.suggestion}_" if f.suggestion else ""))
        out.append("")
    cur = None
    for s in sb.shots:
        if s.section_id != cur:
            cur = s.section_id
            sec = a.section(cur)
            role = next((r for r in t.section_roles if r.section_id == cur), None)
            beat = next((b for b in t.arc if b.section_id == cur), None)
            out += ["", f"## {cur} · {sec.label or '?'} · {_t(sec.start)}–{_t(sec.end)} "
                    f"· energy {sec.energy:.2f}", ""]
            if beat:
                out.append(f"_{beat.story}_ ({beat.emotion})  ")
            if role and role.purpose:
                out.append(f"Purpose: {role.purpose}")
            out.append("")
        flags = []
        if s.refrain_group:
            flags.append(f"refrain #{s.occurrence}")
        if s.reuse_of:
            flags.append(f"reuses {s.reuse_of}")
        elif s.variant_of:
            flags.append(f"variant of {s.variant_of}" + (" (same keyframe)" if s.same_keyframe else ""))
        if s.performance != "none":
            flags.append(s.performance)
        out += [f"### {s.id} · {_t(s.start)}–{_t(s.end)} ({s.duration:.1f}s)"
                + (f" · {' · '.join(flags)}" if flags else ""),
                f"- **{s.shot_type}**, {s.camera} · {s.location_id} · "
                f"{', '.join(s.subject_ids) or 'no characters'} ({s.wardrobe_id}) · "
                f"intensity {s.intensity:.1f} · {s.transition_in}",
                f"- action: {s.action}"]
        if s.lyric_text:
            out.append(f"- lyrics: _{s.lyric_text}_")
        out += [f"- keyframe: {s.keyframe_prompt}", f"- video: {s.video_prompt}"]
        if s.location_change_reason or s.continuity_notes:
            out.append(f"- continuity: {s.location_change_reason} {s.continuity_notes}".rstrip())
        out.append("")
    return "\n".join(out)


def to_html(sb: Scenebook, a: AudioAnalysis, rep: Report | None) -> str:
    e = html.escape
    total = a.duration
    labels = sorted({s.cluster for s in a.sections})
    color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(labels)}
    sec_bar = "".join(
        f'<div class="sec" style="width:{s.duration / total * 100:.3f}%;background:{color[s.cluster]}"'
        f' title="{e(s.id)} {e(s.label)} {_t(s.start)}-{_t(s.end)}">{e(s.label or s.id)}</div>'
        for s in a.sections)
    energy = a.energy_curve
    pts = " ".join(f"{i / max(len(energy) - 1, 1) * 1000:.1f},{40 - v * 38:.1f}"
                   for i, v in enumerate(energy))
    shot_bar = "".join(
        f'<div class="shot{" refrain" if s.refrain_group else ""}{" reuse" if s.reuse_of else ""}"'
        f' style="width:{s.duration / total * 100:.3f}%" title="{e(s.id)} {_t(s.start)} '
        f'{e(s.shot_type)}: {e(s.action)}">{e(s.id.split("-")[1])}</div>' for s in sb.shots)
    rows = "".join(
        f"<tr><td>{e(s.id)}</td><td>{_t(s.start)}</td><td>{s.duration:.1f}</td>"
        f"<td>{e(s.shot_type)}<br><small>{e(s.camera)}</small></td>"
        f"<td>{e(s.location_id)}<br><small>{e(', '.join(s.subject_ids))}</small></td>"
        f"<td>{e(s.action)}<br><small><i>{e(s.lyric_text)}</i></small></td>"
        f"<td>{e(s.performance)}{' ♻' + e(s.reuse_of) if s.reuse_of else ''}"
        f"{' ↺' if s.refrain_group else ''}</td></tr>" for s in sb.shots)
    finds = ""
    if rep:
        finds = "<h2>Review: " + e(rep.verdict) + "</h2><ul>" + "".join(
            f'<li class="{e(f.severity)}"><b>{e(f.severity)}</b> [{e(f.category)}] '
            f'{e(", ".join(f.shot_ids))} {e(f.message)}</li>'
            for f in rep.findings if f.severity != "info") + "</ul>"
    return f"""<!doctype html><meta charset="utf-8"><title>{e(sb.title)} board</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#111;color:#ddd}}
.row{{display:flex;height:34px;margin:4px 0;border-radius:4px;overflow:hidden}}
.sec{{color:#fff;font-size:12px;padding:8px 4px;overflow:hidden;white-space:nowrap;
border-right:1px solid #111}}
.shot{{background:#2a2f3a;border-right:1px solid #111;font-size:10px;padding:10px 2px;
overflow:hidden;text-align:center}}
.shot.refrain{{background:#3b5b3f}} .shot.reuse{{background:#6b5b2a}}
table{{border-collapse:collapse;width:100%;margin-top:18px}}
td,th{{border-bottom:1px solid #333;padding:6px 8px;text-align:left;vertical-align:top}}
small{{color:#999}} li.error b{{color:#ff6b6b}} li.warning b{{color:#f2c744}}
svg{{width:100%;height:40px;background:#1a1a1a;border-radius:4px}}
</style>
<h1>{e(sb.title)}</h1>
<p>{e(sb.treatment.logline)}<br><small>{a.duration:.0f}s · {a.bpm:.0f} bpm · {len(sb.shots)} shots ·
{"approved" if sb.approved else "NOT approved"}</small></p>
<div class="row">{sec_bar}</div>
<svg viewBox="0 0 1000 40" preserveAspectRatio="none"><polyline fill="none" stroke="#7fd"
points="{pts}"/></svg>
<div class="row">{shot_bar}</div>
<small>green = refrain returns, amber = reused clips</small>
{finds}
<table><tr><th>shot</th><th>at</th><th>s</th><th>framing</th><th>where/who</th><th>action / lyrics</th>
<th>mode</th></tr>{rows}</table>"""
