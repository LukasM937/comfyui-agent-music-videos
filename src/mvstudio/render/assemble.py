"""Frame-exact assembly: cut every clip to its slot, conform to one size, mux the original song.

All cut points are computed in *frames* from the shot times (round(t*fps)), so the sum of
segment lengths equals the song length and picture never drifts against the music."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from ..config import Config
from ..models import Brief, Scenebook
from ..project import Project
from .clips import clip_path
from .common import OUTPUT

Log = Callable[[str], None]


def frame_spans(sb: Scenebook, fps: int) -> list[tuple[str, int, int]]:
    """[(shot id, first frame, frame count)] tiling [0, total)."""
    out = []
    for s in sb.shots:
        a, b = round(s.start * fps), round(s.end * fps)
        out.append((s.id, a, max(b - a, 1)))
    return out


def source_clip(sb: Scenebook, shot_id: str) -> str:
    s = sb.shot(shot_id)
    return s.reuse_of or s.id


def assemble(project: Project, brief: Brief, sb: Scenebook, cfg: Config, log: Log = print,
             out_name: str | None = None) -> Path:
    fps = cfg.video.fps
    ow, oh = OUTPUT[brief.aspect]
    seg_dir = project.work / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    spans = frame_spans(sb, fps)
    segs: list[Path] = []
    for sid, _, n in spans:
        src = clip_path(project, sb.shot(source_clip(sb, sid)))
        if not src.exists():
            raise FileNotFoundError(f"clip missing for {sid}: {src}")
        dst = seg_dir / f"{sid}.mp4"
        vf = (f"fps={fps},scale={ow}:{oh}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={ow}:{oh},setsar=1,format=yuv420p")
        # tpad repeats the last frame if a clip came out shorter than its slot
        vf += f",tpad=stop_mode=clone:stop_duration=1,trim=end_frame={n},setpts=PTS-STARTPTS"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", vf,
                        "-frames:v", str(n), "-c:v", "libx264", "-crf", "14", "-preset", "slow",
                        str(dst)], check=True)
        segs.append(dst)

    listing = seg_dir / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    total_frames = sum(n for _, _, n in spans)
    project.out.mkdir(parents=True, exist_ok=True)
    out = project.out / (out_name or f"{project.root.name}.mp4")
    log(f"assemble: {len(segs)} shots, {total_frames} frames ({total_frames / fps:.1f}s) -> {out}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-i", str(project.audio_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "320k", "-t", f"{total_frames / fps:.3f}", "-movflags",
         "+faststart", str(out)], check=True)
    return out
