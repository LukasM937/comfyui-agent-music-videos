# Model selection (checked 2026-10-12 on the Hugging Face Hub)

Target box: 1× RTX 5090 (32 GB VRAM, Blackwell `sm_120`), 128 GB RAM, Arch Linux.
Constraint that shapes everything: **one GPU, so heavy models run one after another**, with weights
streamed from the 128 GB of RAM. ComfyUI offloads automatically; the planner LLM is stopped before
rendering (`llm.stop_after_stage`).

## Video

| Candidate | Size / precision | Fits 32 GB? | Audio input | Verdict |
|---|---|---|---|---|
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (2026-07) | 33B dense transformer (+Qwen3-VL-32B text encoder), BF16 | **No.** ~66 GB transformer weights; reference serving is `--num-gpus 4`. FP8/GGUF community builds would be needed and are untested | Reference audio ≤ 15 s total (Ref2VA) | Not default. 15 s hard limit; *H3-Context-IR* (the prompt-understanding stage the authors "strongly recommend") and *Regenerate-2K* are **API-only**; open weights are licensed for **EU/UK/US/KR only after an application** ([licence Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md)) — relevant for an EU user |
| [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (2026-07) | 22B distilled; **int8 21.5 GB**, NVFP4 18.7 GB; Gemma-4-12B text encoder int8 15.4 GB | **Yes** with RAM offload | Joint audio-video model; official ComfyUI templates: T2V, I2V, first/last frame | **Default** for ordinary shots |
| [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (2026-03) | 22B; fp8 checkpoint 29 GB | Yes (tight, offload) | **Official image+audio→video template with lip-sync** (`video_ltx2_3_ia2v`) | Used for singer / audio-driven shots |

Resulting rule: *ordinary shots* → LTX-2.5 I2V from a keyframe; *shots that must follow the vocals*
(`performance: lipsync | dance | instrument`) → LTX-2.3 image+audio→video, fed with the exact audio
slice of that shot. LTX-2.5 has no official audio-conditioned workflow yet; when one appears,
`video.audio_family` can be switched. Shots are cut on musical boundaries and capped at
`video.max_clip_s` (10 s), so the "cut the song" problem is solved for any clip-length-limited model.

The NVFP4 build (`extras` group) is Blackwell-native and smaller; it is in the manifest but **not the
default** because its quality relative to int8 has not been compared here. `mvs selftest` renders the
same tiny clip with whichever file `video.*` points to.

## Images (keyframes, character sheets, location plates)

[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) — 7B DiT, text-to-image **and**
editing in one checkpoint, up to 10 reference images, native 2K, identity preservation; ComfyUI int8 build
is 7.3 GB (+9.4 GB Qwen3-VL-8B text encoder). Licence: **Qwen Research License (non-commercial)**.
Alternative if that matters: FLUX.2-klein-4B ([Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B))
— would need a new workflow; not wired up.

## Planner LLM (director + supervisor)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0, dense, native image/video
understanding, 262k context) as [unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
`UD-Q4_K_M` (16.5 GB) + vision projector, served by **llama.cpp** (`llama-server`): grammar-constrained JSON
from a Pydantic schema, no Python-ecosystem GPU conflicts (vLLM would pin its own torch). 27B dense at Q4
leaves room for a 32k context, but not for ComfyUI at the same time — hence staging.
The vision projector is used for keyframe QA (planned, see limits).

## Audio understanding

There is no single model that "understands music"; the pipeline combines:

* **DSP** (librosa): beat/downbeat grid, bar-synchronous chroma + timbre + dynamics → self-similarity →
  Foote novelty → sections and acoustic repeat clusters; onset-envelope cross-correlation between repeats
  (decides if a refrain clip can be reused verbatim).
* **ASR** ([faster-whisper large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)):
  word timestamps. Your own lyrics (`lyrics.txt`) are aligned to the ASR words; an `.lrc` file bypasses ASR.
* **LLM**: names the sections (intro/verse/chorus/…) from structure + lyrics and links later choruses to the first.

## Disk budget

core ≈ 75 GB (LTX-2.5 + Qwen-Image + planner LLM), lipsync +42 GB (LTX-2.3), asr +1.6 GB,
extras +18.7 GB. Recommended free space: 150 GB.
