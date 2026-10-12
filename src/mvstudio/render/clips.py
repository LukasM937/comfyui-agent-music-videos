"""Video clips: keyframe (+ the song's own audio slice) -> LTX, one clip per shot."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ..comfy.client import ComfyClient
from ..comfy.workflows import VideoJob, build_video, valid_frames
from ..config import Config
from ..models import Brief, Scenebook, Shot
from ..project import Project
from . import prompting
from .common import State, stable_seed, video_size
from .keyframes import keyframe_path

Log = Callable[[str], None]


def clip_path(p: Project, shot: Shot) -> Path:
    return p.clips_dir / f"{shot.id}.mp4"


def uses_audio(shot: Shot, cfg: Config) -> bool:
    """Audio-conditioned (image+audio -> video) shots follow the vocals / rhythm."""
    mode = cfg.video.audio_conditioning
    if mode == "never":
        return shot.performance == "lipsync"  # lip-sync is impossible without the audio
    if mode == "always":
        return True
    return shot.performance in ("lipsync", "dance", "instrument")


def family_for(shot: Shot, cfg: Config) -> str:
    return cfg.video.audio_family if uses_audio(shot, cfg) else cfg.video.default_family


def slice_audio(wav: Path, start: float, duration: float, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(wav), "-af", f"apad=whole_dur={duration:.3f}", "-t", f"{duration:.3f}",
         "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


def render_clips(client: ComfyClient, project: Project, brief: Brief, sb: Scenebook, cfg: Config,
                 only: set[str] | None = None, force: bool = False, log: Log = print) -> None:
    st = State(project)
    vw, vh = video_size(brief.aspect, cfg)
    todo = [s for s in sb.shots if not s.reuse_of and (not only or s.id in only)]
    # one model family at a time: swapping 20+ GB of weights between shots wastes minutes
    todo.sort(key=lambda s: (family_for(s, cfg), s.start))
    t_all = time.time()
    for i, shot in enumerate(todo, start=1):
        kf = keyframe_path(project, shot)
        if not kf.exists():
            raise FileNotFoundError(f"keyframe missing for {shot.id}: run `mvs keyframes` first")
        fam = family_for(shot, cfg)
        frames = valid_frames(shot.duration, cfg.video.fps)
        prompt = prompting.video_prompt(shot, brief, sb.treatment)
        audio = uses_audio(shot, cfg)
        dst = clip_path(project, shot)
        key = {"p": prompt, "kf": kf.stat().st_mtime_ns, "fam": fam, "frames": frames,
               "size": [vw, vh], "audio": audio, "start": shot.start,
               "seed": cfg.video.seed}
        if not force and st.fresh(dst, key):
            continue
        t0 = time.time()
        log(f"clip {shot.id} ({i}/{len(todo)}) {fam}{' +audio' if audio else ''} "
            f"{shot.duration:.1f}s ({frames} frames)")
        img = client.upload(kf, f"{project.root.name}__kf_{shot.id}.png")
        aud = None
        if audio:
            sl = slice_audio(project.audio_wav, shot.start, (frames - 1) / cfg.video.fps,
                             project.slices_dir / f"{shot.id}.wav")
            aud = client.upload(sl, f"{project.root.name}__sl_{shot.id}.wav")
        job = VideoJob(image=img, audio=aud, prompt=prompt, width=vw, height=vh, frames=frames,
                       fps=cfg.video.fps, seed=stable_seed(cfg.video.seed, shot.id), family=fam,
                       prefix=f"mvs/{project.root.name}/clip_{shot.id}")
        client.run(build_video(job, cfg), dst,
                   on_tick=lambda s: None if int(s) % 60 else log(f"   ... {int(s)}s"))
        st.record(dst, key)
        log(f"   done in {time.time() - t0:.0f}s")
    log(f"clips: finished in {(time.time() - t_all) / 60:.1f} min")
