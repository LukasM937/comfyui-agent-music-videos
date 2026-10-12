# mvstudio — audio-driven AI music videos on ComfyUI

You provide **a song, a rough brief, and your characters** (text and/or photos). The pipeline
**listens to the song**, writes a **scenebook** (a shot list with exact beat-aligned timing), has a
**script supervisor check** it against your theme and for story logic, lets **you approve** it, and
only then renders keyframes and clips with **ComfyUI** and cuts them to the music.

Built for one box: **Arch Linux · 1× RTX 5090 (32 GB) · 128 GB RAM**.

```
song.mp3 ─┐                       ┌─────────────── human gate ───────────────┐
brief.yaml├─► analyze ─► plan ──► check/revise ─► you read board.html ─► approve
chars/*.jpg┘   (beats,    (director   (rules +        edit scenebook.json       │
                sections,   LLM)        critic LLM)    or `mvs revise --note`    ▼
                repeats,                                       refs ─► keyframes ─► clips ─► assemble ─► out/video.mp4
                lyrics)                                      Qwen-Image-2.1     LTX-2.5 / LTX-2.3     ffmpeg, frame-exact
```

## Why this design

| Problem in typical "AI music video" scripts | What this does instead |
|---|---|
| The LLM invents timings; cuts land randomly | **Timing is code.** Cuts snap to the beat grid, never cross section borders, avoid slicing sung words, respect the video model's clip limit. The LLM only fills in *what* is seen. |
| Chorus is just "another scene" | Refrains are detected (audio self-similarity **and** lyric repetition), keep the same cut pattern, and return as **identical** reuse, **variants** (same keyframe, new take) or **escalations** (same image, bigger each time). Whether a clip can be reused verbatim is decided by measuring how exactly the repeats line up. |
| Story drifts, characters change clothes, people teleport | A **scenebook supervisor**: ~20 deterministic rules (coverage, timing, references, continuity, refrain, pacing, avoid-list) + an LLM critic for theme fit and narrative logic, in a revise loop. |
| 6 hours of rendering before you see a bad idea | Nothing renders until you **approve** a content hash of the scenebook. Edit anything afterwards and the approval is void. |
| Character looks change every shot | Per-character **reference images** (yours, or generated once) feed every keyframe; wardrobes are explicit and checked. |
| Audio sync drifts | Assembly is **frame-exact** (`round(t·fps)` per shot) and muxes your original audio. |

## Models (details and sources: [docs/model-selection.md](docs/model-selection.md))

| Role | Model | Why |
|---|---|---|
| Video | **LTX-2.5** 22B distilled, int8 (default shots) | Best open quality that fits 32 GB; native 24 fps, up to 10 s per clip here |
| Video, singer on camera / audio-driven | **LTX-2.3** image+audio→video | The only LTX release with an official audio-conditioned workflow (lip-sync) |
| Keyframes, characters, locations | **Qwen-Image-2.1** (7B DiT, ≤10 reference images, native 2K) | Reference-consistent edits; small enough to run next to everything else |
| Director / critic | **Qwen3.8-27B** Q4 via llama.cpp (vision-capable) | Strong planner that fits beside nothing else — hence staged GPU use |
| Lyrics | **faster-whisper large-v3-turbo** | Word timestamps; your own lyrics are aligned to them |

**MiniMax-H3** was evaluated and is *not* the default: 33B dense (~66 GB in BF16), reference
deployment uses 4 GPUs, its context-IR module is API-only, and the open-weight licence requires an
application for EU users. Its hard 15 s limit would also force cutting the song — which this
pipeline does anyway (per shot, on musical boundaries), so a backend can be added later.

## Quick start

```bash
git clone <this repo> && cd comfyui-agent-music-videos
scripts/setup.sh check                  # read-only: what is there, what is missing
export HF_TOKEN=hf_...                  # LTX repos are gated: accept the licence on huggingface.co first
scripts/setup.sh                        # installs missing packages, ComfyUI, llama.cpp, models (~75 GB), tests
scripts/setup.sh models lipsync         # +42 GB: LTX-2.3 for singer / audio-driven shots
systemctl --user enable --now comfyui

mvs init ~/videos/night-road            # creates brief.yaml
cp song.mp3 ~/videos/night-road/        # + optional lyrics.txt / lyrics.lrc, photos in characters/
$EDITOR ~/videos/night-road/brief.yaml  # theme, style, story notes, characters
mvs run ~/videos/night-road             # analyze -> plan -> stops at the review gate
xdg-open ~/videos/night-road/work/board.html
mvs revise ~/videos/night-road --section S03 --note "make the chorus happen on the road at sunset"
mvs approve ~/videos/night-road
mvs run ~/videos/night-road             # refs -> keyframes -> clips -> out/night-road.mp4
```

`examples/demo-project/brief.yaml` is a complete brief to start from.

### What the setup script does ([scripts/setup.sh](scripts/setup.sh))

`check` · `system` (pacman, only what is missing) · `python` · `comfyui` (pinned, validated commit; PyTorch
wheels chosen by driver version) · `llama` (llama.cpp built for `sm_120`) · `models [group]` · `services`
(systemd user unit) · `test` (starts ComfyUI if needed, runs `mvs doctor` and `mvs selftest`).
`--dry-run` prints every command; `--install-driver` is required before it touches the NVIDIA driver.

### Health checks

* `mvs doctor` — GPU/driver (Blackwell ≥ 570), RAM, disk, tools, ComfyUI version and device,
  **every bundled workflow validated against your ComfyUI's node definitions**, model files by exact byte size.
* `mvs selftest` — tiny real runs: schema-constrained LLM JSON, Qwen-Image text→image and with references,
  an LTX-2.5 clip, an LTX-2.3 audio-conditioned clip. Steps with missing models are skipped, not failed.

## Commands

| Command | Does |
|---|---|
| `mvs init DIR` | project skeleton + commented `brief.yaml` |
| `mvs analyze DIR` | tempo grid, sections, repeats, lyrics → `work/analysis.json` |
| `mvs plan DIR` | treatment → scenebook → check/revise loop → `scenebook.json/.md`, `board.html`, `report.json` |
| `mvs check DIR [--no-llm]` | re-review after you edited `scenebook.json` |
| `mvs revise DIR --note "…" [--section S03]` | LLM redoes sections following your words; timing untouched |
| `mvs approve DIR` | lock the scenebook (refuses on rule errors) |
| `mvs refs / keyframes / render / assemble DIR` | render stages; `--shots S03-02 --force` to redo single shots |
| `mvs run DIR [--auto-approve]` | everything, stopping at the review gate |
| `mvs doctor`, `mvs selftest`, `mvs models status|download`, `mvs llm serve` | environment |

Every stage resumes: artifacts are cached by a hash of their inputs (`work/state.json`).

## Configuration

Defaults in [`config/default.toml`](config/default.toml); override in `~/.config/mvstudio/config.toml` or
`<project>/project.toml`. Useful knobs: `planning.cut_style` (calm/medium/fast), `planning.refrain_mode`
(identical/variants/escalate), `planning.performance_ratio`, `video.max_clip_s`, `video.audio_conditioning`
(auto/always/never), `video.default_family`.

## Honest status

* **Verified here (no GPU in the dev sandbox):** all 57 unit/integration tests; audio analysis on a
  synthetic song with known structure; planning with a scripted stand-in LLM; the template converter;
  **all patched workflows validated against a real ComfyUI 0.39 `/object_info` and accepted by `/prompt`**;
  frame-exact assembly with ffmpeg; the setup script (`shellcheck`-clean, dry-run).
* **Not verified — needs your first run:** actual generation quality and VRAM/time on the 5090,
  llama.cpp's support of Qwen3.8 + the JSON-schema grammar on your build, model downloads (the sandbox
  blocks huggingface.co), ASR on real vocals, structure detection on real, dense mixes.
  `mvs selftest` is the first thing to run; see [docs/architecture.md](docs/architecture.md#known-limits).
* Licences: Qwen-Image-2.1 is under the **Qwen Research License** (non-commercial); LTX-2.x under the
  LTX community licence; check both before commercial use.

The original research notes (2× RTX 3090 era) are in [`docs/legacy/`](docs/legacy/).
