# Architecture

## Stages and artifacts

```
project/
  brief.yaml  song.*  lyrics.txt|lrc  characters/*        (you)
  work/
    audio.wav                       decoded master, also used for slicing and the final mux
    analysis.json                   tempo grid, sections, repeats, lyrics        <- mvs analyze
    treatment.json                  logline, style, locations, arc, refrain idea  <- mvs plan
    scenebook.json / .md            the shot list (source of truth, hand-editable)
    report.json / board.html        supervisor findings, timeline view
    refs/ keyframes/ slices/ clips/ segments/    render caches
    state.json                      artifact -> hash of inputs (resume)
  out/<project>.mp4
```

All models are Pydantic classes in [`models.py`](../src/mvstudio/models.py); every file on disk is
validated on load.

## Planning

1. **Skeleton** ([`planning/skeleton.py`](../src/mvstudio/planning/skeleton.py)) — deterministic.
   Target shot length from `cut_style` (calm/medium/fast ≙ 4/2/1 bars), halved for high-energy and doubled
   for low-energy sections. Cut candidates are beats; score = distance to target + off-downbeat penalty +
   a bar-length penalty for cutting through a sung word. Last shot of a section ends exactly on the
   section end. Returning refrains copy the first occurrence's cut pattern (snapped to their own beats).
2. **Treatment** (LLM, schema-constrained) — logline, style paragraph, palette, motifs, locations, one arc
   beat and one role per section, refrain concept, continuity rules. Incomplete section coverage is
   rejected and retried.
3. **Scenebook** (LLM per section) — receives the fixed time slots (with lyrics and a suggestion which
   slots should show the singer), the previous shot for continuity, and for refrain returns the first
   occurrence's shots plus a mode-specific instruction:
   * `identical` — if measured alignment says the repeat is sample-tight (lag ≤ 60 ms, onset correlation ≥ 0.75,
     equal length) the clips are **rendered once and reused**; otherwise it falls back to `variants`.
   * `variants` — same location/blocking/keyframe, new lens/camera/motion (new seed).
   * `escalate` — same core image per slot, intensity rises; the first occurrence's keyframe is passed as a
     reference image so the look stays recognisable.
4. **Supervisor** ([`planning/check.py`](../src/mvstudio/planning/check.py))
   * *rules*: tiling of the whole song, clip-length limits, section borders, beat alignment, unknown
     location/character/wardrobe ids, `[id]` tokens in prompts, empty prompts, cast members never used,
     lip-sync without the performer / without sung text / in wide shots, location and wardrobe jumps without
     a reason, refrain rhyme (shot counts, locations, escalating intensity), valid clip reuse, loudest section
     cut slower than the calmest, four identical shot sizes in a row, avoid-list violations.
   * *critic LLM*: theme fit, narrative logic, continuity, pacing, refrain, originality, render-ability; scores 0–10.
   * *loop*: errors (and critic warnings) are grouped per section and fed back into a regeneration of only
     those sections (`planning.max_revision_rounds`). Timing is never altered by the LLM.
5. **Approval** — `approved_hash` = hash of the scenebook content. Any edit invalidates it.

## Rendering

* **refs** — user images are used as given; missing portraits/full-body/wardrobes and one plate per
  location are generated once (Qwen-Image-2.1; full-body shots use the portrait as identity reference).
* **keyframes** — prompt = visual-style paragraph + shot type + director's prompt with `[id]` expanded to
  *"Mara (the person from <image1>, wearing …)"* + setting + time of day. Up to 4 reference images per shot
  (character, plate, return-of-refrain base).
* **clips** — shots sorted by model family (no weight swapping every shot); frames = smallest 8n+1 ≥ shot
  length; for audio-conditioned shots the audio slice of exactly that span is uploaded; seed derived from the shot id.
* **assemble** — each clip is conformed to the output size, cut to `round(end·fps) − round(start·fps)`
  frames (the last frame is held if a clip is short), concatenated, and the original audio is muxed.
  Cuts only: dissolves would break frame-exact sync.

## GPU time-slicing on one card

`mvs plan/check/revise` → `llm_session` frees ComfyUI (`POST /free`), starts `llama-server`, and stops it
afterwards. Render stages require ComfyUI and never run while the LLM is up. Peak numbers on a 32 GB card are
**estimates** until measured with `mvs selftest` (watch `nvidia-smi`).

## Workflows from the official templates

[`scripts/sync_templates.py`](../scripts/sync_templates.py) downloads templates from
`Comfy-Org/workflow_templates`, flattens them (subgraphs, reroutes, promoted widgets, dynamic combos —
[`comfy/convert.py`](../src/mvstudio/comfy/convert.py)) into `workflows/api/*.json`, and
[`comfy/workflows.py`](../src/mvstudio/comfy/workflows.py) patches only what it owns (prompt, size, frame count,
seed, input files) and cuts the optional prompt-enhancer branches (they would pull ~10 GB of unused models).
After updating ComfyUI: `scripts/sync_templates.py --object-info http://127.0.0.1:8188`, then `mvs doctor`.

## Known limits

* Not run on a GPU by the author: generation quality, VRAM peaks and timings are unmeasured.
* Structure detection is classical DSP; ±1 bar errors on boundaries are normal (section borders are advisory,
  cuts are snapped to beats anyway). Downbeat phase can be off by a beat in drum-light intros.
* LTX-2.3 and LTX-2.5 differ in look; a mix inside one video relies on the shared keyframes to hold the style.
* Keyframe QA with the vision LLM (does the frame match the character sheet?) is designed (LLM already loads the
  projector) but **not implemented**; check `work/keyframes/` before `render`.
* No dissolves/wipes, no upscaler stage (output is a lanczos upscale of 1280×704 clips).
* No vocal separation before ASR (demucs): dense mixes may transcribe worse. Provide `lyrics.txt`/`lyrics.lrc` for such songs.
