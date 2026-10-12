"""`mvs` command line: one command per pipeline stage, plus doctor/selftest/models."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import doctor as doctor_mod
from . import modelstore, selftest
from .audio import analyze as audio_analyze
from .comfy.client import ComfyClient
from .config import Config, load_config
from .llm.server import llm_session
from .models import Report, Scenebook, save_json
from .planning import check as plan_check
from .planning import export, pipeline, scenebook
from .project import Project
from .render import assemble as asm
from .render import clips, keyframes, refs

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Audio-driven AI music videos on ComfyUI.")
llm_app = typer.Typer(no_args_is_help=True, help="Planner LLM (llama.cpp) helpers.")
models_app = typer.Typer(no_args_is_help=True, help="Model manifest.")
app.add_typer(llm_app, name="llm")
app.add_typer(models_app, name="models")
console = Console()

PathArg = Annotated[Path, typer.Argument(help="Project folder")]


def log(msg: str) -> None:
    console.print(msg)


def _ctx(path: Path) -> tuple[Project, Config]:
    p = Project(path)
    return p, load_config(p.root)


def _need(path: Path, what: str, hint: str) -> None:
    if not path.exists():
        console.print(f"[red]{what} missing:[/red] {path}\n  -> {hint}")
        raise typer.Exit(1)


def _comfy(cfg: Config) -> ComfyClient:
    c = ComfyClient(cfg.comfy.url, cfg.comfy.poll_interval_s, cfg.comfy.job_timeout_s,
                    cfg.comfy.dir)
    if not c.alive():
        console.print(f"[red]ComfyUI is not reachable at {cfg.comfy.url}[/red]\n"
                      "  -> systemctl --user start comfyui   (see scripts/setup.sh services)")
        raise typer.Exit(1)
    return c


def _show_checks(checks: list[doctor_mod.Check]) -> int:
    t = Table(show_header=True, header_style="bold")
    t.add_column("check")
    t.add_column("")
    t.add_column("detail", overflow="fold")
    icon = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]",
            "fail": "[red]FAIL[/red]", "skip": "[dim]skip[/dim]"}
    for c in checks:
        t.add_row(c.name, icon[c.status], c.detail + (f"\n[dim]fix: {c.fix}[/dim]" if c.fix else ""))
    console.print(t)
    return sum(c.status == "fail" for c in checks)


# ------------------------------------------------------------------ project lifecycle


@app.command()
def init(path: PathArg, title: Annotated[str, typer.Option(help="Video title")] = "") -> None:
    """Create a project folder with a commented brief.yaml."""
    p = Project(path)
    p.init(title or p.root.name)
    console.print(f"Created {p.root}\n  1. copy your song into it (mp3/wav/flac/...)\n"
                  f"  2. edit {p.brief_file.name} (theme, style, characters)\n"
                  "  3. optional: lyrics.txt or lyrics.lrc, reference images in characters/\n"
                  f"  4. mvs run {p.root}")


@app.command()
def analyze(path: PathArg, lyrics: Annotated[bool, typer.Option(help="Transcribe lyrics")] = True
            ) -> None:
    """Understand the audio: tempo grid, sections, repeats, lyrics with timings."""
    p, cfg = _ctx(path)
    _need(p.brief_file, "brief.yaml", f"mvs init {path}")
    brief = p.load_brief()
    a = audio_analyze.analyze(p, brief, cfg, log, with_lyrics=lyrics)
    save_json(a, p.analysis_file)
    console.print(audio_analyze.describe(a))
    console.print(f"\nsaved {p.analysis_file}")


@app.command()
def plan(path: PathArg) -> None:
    """Treatment + scenebook, checked and revised by the supervisor loop."""
    p, cfg = _ctx(path)
    _need(p.analysis_file, "analysis", f"mvs analyze {path}")
    brief, a = p.load_brief(), p.load_analysis()
    with llm_session(cfg, log) as llm:
        sb, rep = pipeline.make_plan(llm, p, brief, a, cfg, log)
    _publish(p, sb, rep)


def _publish(p: Project, sb: Scenebook, rep: Report | None) -> None:
    a = p.load_analysis()
    p.scenebook_md.write_text(export.to_markdown(sb, a, rep), encoding="utf-8")
    p.board_html.write_text(export.to_html(sb, a, rep), encoding="utf-8")
    verdict = rep.verdict if rep else "unchecked"
    col = "green" if verdict == "pass" else "yellow"
    console.print(f"\nscenebook: {len(sb.shots)} shots, verdict [{col}]{verdict}[/{col}]"
                  f"  ({len(rep.errors) if rep else 0} errors, "
                  f"{len(rep.warnings) if rep else 0} warnings)")
    console.print(f"  read   : {p.scenebook_md}\n  board  : {p.board_html}\n"
                  f"  edit   : {p.scenebook_file}  (then `mvs check`)\n"
                  f"  accept : mvs approve {p.root}")


@app.command()
def check(path: PathArg, llm: Annotated[bool, typer.Option("--llm/--no-llm")] = True) -> None:
    """Re-run the supervisor on the (hand-edited) scenebook without regenerating anything."""
    p, cfg = _ctx(path)
    _need(p.scenebook_file, "scenebook", f"mvs plan {path}")
    brief, a, sb = p.load_brief(), p.load_analysis(), p.load_scenebook()
    if llm:
        with llm_session(cfg, log) as client:
            rep = plan_check.review(client, brief, sb, a, cfg)
    else:
        rep = plan_check.review(None, brief, sb, a, cfg)
    save_json(rep, p.report_file)
    _publish(p, sb, rep)
    for f in rep.findings:
        if f.severity != "info":
            console.print(f"  [{'red' if f.severity == 'error' else 'yellow'}]{f.severity}[/] "
                          f"{','.join(f.shot_ids) or f.section_id or ''} {f.message}")


@app.command()
def revise(path: PathArg,
           note: Annotated[str, typer.Option(help="What to change, in plain words")],
           section: Annotated[list[str] | None, typer.Option(help="Section id(s), e.g. S03")] = None
           ) -> None:
    """Redo sections following your instruction, then re-check. Timing stays untouched."""
    p, cfg = _ctx(path)
    _need(p.scenebook_file, "scenebook", f"mvs plan {path}")
    brief, a, sb = p.load_brief(), p.load_analysis(), p.load_scenebook()
    targets = section or [s.id for s in a.sections]
    with llm_session(cfg, log) as llm:
        scenebook.refill_sections(llm, brief, sb, a, cfg,
                                  {sid: [f"Artist instruction: {note}"] for sid in targets}, log)
        sb.approved_hash = None
        sb, rep = pipeline.revise_loop(llm, p, brief, sb, a, cfg, log)
    _publish(p, sb, rep)


@app.command()
def approve(path: PathArg, force: Annotated[bool, typer.Option(help="Ignore errors")] = False
            ) -> None:
    """Lock the scenebook. Rendering refuses a scenebook that changed after approval."""
    p, cfg = _ctx(path)
    _need(p.scenebook_file, "scenebook", f"mvs plan {path}")
    sb = p.load_scenebook()
    rep = plan_check.review(None, p.load_brief(), sb, p.load_analysis(), cfg)
    if rep.errors and not force:
        for f in rep.errors:
            console.print(f"[red]error[/red] {','.join(f.shot_ids)} {f.message}")
        console.print("Fix these (or --force).")
        raise typer.Exit(1)
    sb.approved_hash = sb.content_hash()
    save_json(sb, p.scenebook_file)
    console.print(f"approved scenebook {sb.approved_hash}")


# ------------------------------------------------------------------ render stages


def _approved(p: Project, force: bool) -> Scenebook:
    _need(p.scenebook_file, "scenebook", f"mvs plan {p.root}")
    sb = p.load_scenebook()
    if not sb.approved and not force:
        console.print("[red]scenebook is not approved[/red] (or changed since approval)\n"
                      f"  -> review {p.scenebook_md}, then `mvs approve {p.root}` (or --force)")
        raise typer.Exit(1)
    return sb


@app.command(name="refs")
def refs_cmd(path: PathArg, force: bool = False, unapproved: bool = False) -> None:
    """Character and location reference images (user photos are used as they are)."""
    p, cfg = _ctx(path)
    sb = _approved(p, unapproved)
    refs.make_refs(_comfy(cfg), p, p.load_brief(), sb.treatment, cfg, log, force)


@app.command(name="keyframes")
def keyframes_cmd(path: PathArg, shots: Annotated[list[str] | None, typer.Option()] = None,
                  force: bool = False, unapproved: bool = False) -> None:
    """First frame of every shot."""
    p, cfg = _ctx(path)
    sb = _approved(p, unapproved)
    client = _comfy(cfg)
    brief = p.load_brief()
    refs.make_refs(client, p, brief, sb.treatment, cfg, log)
    keyframes.make_keyframes(client, p, brief, sb, cfg, set(shots) if shots else None, force, log)
    console.print(f"keyframes in {p.keyframes_dir}  (tweak a prompt, then --shots <id> --force)")


@app.command(name="render")
def render_cmd(path: PathArg, shots: Annotated[list[str] | None, typer.Option()] = None,
               force: bool = False, unapproved: bool = False) -> None:
    """Video clips for every shot."""
    p, cfg = _ctx(path)
    sb = _approved(p, unapproved)
    clips.render_clips(_comfy(cfg), p, p.load_brief(), sb, cfg, set(shots) if shots else None,
                       force, log)


@app.command(name="assemble")
def assemble_cmd(path: PathArg, unapproved: bool = False) -> None:
    """Cut to the music and mux the original audio."""
    p, cfg = _ctx(path)
    sb = _approved(p, unapproved)
    out = asm.assemble(p, p.load_brief(), sb, cfg, log)
    console.print(f"[green]done:[/green] {out}")


@app.command()
def run(path: PathArg,
        auto_approve: Annotated[bool, typer.Option(help="Skip the human review gate")] = False,
        lyrics: bool = True) -> None:
    """Everything: analyze -> plan -> (you review) -> refs -> keyframes -> clips -> assemble."""
    p, cfg = _ctx(path)
    if not p.analysis_file.exists():
        analyze(path, lyrics)
    if not p.scenebook_file.exists():
        plan(path)
    sb = p.load_scenebook()
    if not sb.approved:
        if not auto_approve:
            console.print("\n[yellow]Review gate:[/yellow] read the scenebook, adjust, then run "
                          f"`mvs approve {path}` and `mvs run {path}` again.")
            raise typer.Exit(0)
        approve(path, force=False)
    refs_cmd(path)
    keyframes_cmd(path)
    render_cmd(path)
    assemble_cmd(path)


# ------------------------------------------------------------------ environment


@app.command()
def doctor(groups: Annotated[list[str] | None, typer.Option(help="model groups")] = None,
           workflows: bool = True) -> None:
    """Check GPU, tools, ComfyUI, models and workflow validity (read-only)."""
    cfg = load_config()
    n = _show_checks(doctor_mod.run_all(cfg, groups, workflows))
    raise typer.Exit(1 if n else 0)


@app.command(name="selftest")
def selftest_cmd(only: Annotated[list[str] | None, typer.Option(
    help=f"subset of: {', '.join(selftest.STEPS)}")] = None) -> None:
    """Tiny real renders (LLM JSON, image, image+refs, LTX-2.5 clip, LTX-2.3 audio clip)."""
    cfg = load_config()
    n = _show_checks(selftest.run(cfg, only, log))
    raise typer.Exit(1 if n else 0)


@llm_app.command("serve")
def llm_serve() -> None:
    """Run llama-server in the foreground (Ctrl-C to stop)."""
    from .llm.server import server_command
    cfg = load_config()
    cmd = server_command(cfg)
    exe = shutil.which(cmd[0])
    if not exe:
        console.print(f"[red]{cmd[0]} not found[/red]")
        raise typer.Exit(1)
    console.print(" ".join(cmd))
    os.execv(exe, cmd)


@models_app.command("status")
def models_status(groups: Annotated[list[str] | None, typer.Option()] = None) -> None:
    cfg = load_config()
    t = Table("id", "group", "GB", "status", "repo")
    for m in modelstore.select(modelstore.load_manifest(), groups or ["all"]):
        s = modelstore.status(m, cfg)
        t.add_row(m.id, m.group, f"{m.gb:.1f}", "[green]ok[/green]" if s == "ok" else f"[red]{s}[/red]",
                  m.repo)
    console.print(t)


@models_app.command("download")
def models_download(groups: Annotated[list[str] | None, typer.Option()] = None) -> None:
    """Download missing files (default group: core). Needs HF_TOKEN for gated repos."""
    cfg = load_config()
    todo = [m for m in modelstore.select(modelstore.load_manifest(), groups)
            if modelstore.status(m, cfg) != "ok"]
    console.print(f"{len(todo)} file(s), {sum(m.gb for m in todo):.1f} GB")
    for m in todo:
        modelstore.download(m, cfg, log)


if __name__ == "__main__":
    app()
