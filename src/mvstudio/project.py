"""On-disk project layout.

    my-song/
      brief.yaml            <- you write this (theme, style, characters, ...)
      song.mp3              <- you drop the audio here (any ffmpeg-readable format)
      lyrics.txt | .lrc     <- optional
      characters/           <- optional reference photos/drawings, referenced from brief.yaml
      project.toml          <- optional per-project config overrides
      work/                 <- everything generated (safe to delete; stages resume from it)
      out/                  <- final video
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import AudioAnalysis, Brief, Scenebook, Treatment, load_model

AUDIO_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac")

BRIEF_TEMPLATE = """\
# Brief for your music video. Everything except `title` is optional, but the more
# concrete you are, the better the scenebook.
title: "{title}"
theme: "What is the song about, in one sentence?"
style: "Visual style, e.g. 16mm film, grainy, warm tungsten light, shallow depth of field"
story_notes: |
  Roughly what do you want to see? Locations, story, recurring images, things you love.
mood: [melancholic, hopeful]
language: null          # lyrics language hint, e.g. en / de. null = auto-detect
avoid:                  # hard no-go list
  - text overlays
  - gore
aspect: "16:9"          # 16:9 | 9:16 | 1:1 | 21:9
# lyrics: |             # paste lyrics here or put them into lyrics.txt (or lyrics.lrc with times)
characters:
  - id: singer
    name: "Mara"
    role: performer
    is_performer: true
    description: "Woman, early 30s, shoulder-length dark curly hair, olive skin, small scar over left eyebrow"
    wardrobes:
      - {{id: default, description: "oversized mustard cardigan over white t-shirt, blue jeans"}}
      - {{id: night, description: "black leather jacket, silver hoop earrings"}}
    images: []          # e.g. ["characters/mara_01.jpg"]  (relative to this folder)
"""


class Project:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    # ----- user inputs
    @property
    def brief_file(self) -> Path:
        return self.root / "brief.yaml"

    @property
    def audio_src(self) -> Path:
        cands = sorted(p for p in self.root.iterdir() if p.suffix.lower() in AUDIO_EXT) \
            if self.root.is_dir() else []
        if not cands:
            raise FileNotFoundError(f"no audio file ({', '.join(AUDIO_EXT)}) in {self.root}")
        return cands[0]

    @property
    def lyrics_txt(self) -> Path:
        return self.root / "lyrics.txt"

    @property
    def lyrics_lrc(self) -> Path:
        return self.root / "lyrics.lrc"

    # ----- generated
    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def audio_wav(self) -> Path:
        return self.work / "audio.wav"

    @property
    def analysis_file(self) -> Path:
        return self.work / "analysis.json"

    @property
    def treatment_file(self) -> Path:
        return self.work / "treatment.json"

    @property
    def scenebook_file(self) -> Path:
        return self.work / "scenebook.json"

    @property
    def report_file(self) -> Path:
        return self.work / "report.json"

    @property
    def scenebook_md(self) -> Path:
        return self.work / "scenebook.md"

    @property
    def board_html(self) -> Path:
        return self.work / "board.html"

    @property
    def refs_dir(self) -> Path:
        return self.work / "refs"

    @property
    def keyframes_dir(self) -> Path:
        return self.work / "keyframes"

    @property
    def clips_dir(self) -> Path:
        return self.work / "clips"

    @property
    def slices_dir(self) -> Path:
        return self.work / "slices"

    @property
    def state_file(self) -> Path:
        return self.work / "state.json"

    # ----- io
    def init(self, title: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "characters").mkdir(exist_ok=True)
        if not self.brief_file.exists():
            self.brief_file.write_text(BRIEF_TEMPLATE.format(title=title), encoding="utf-8")

    def load_brief(self) -> Brief:
        data = yaml.safe_load(self.brief_file.read_text(encoding="utf-8")) or {}
        brief = Brief.model_validate(data)
        if brief.lyrics is None:
            if self.lyrics_txt.exists():
                brief.lyrics = self.lyrics_txt.read_text(encoding="utf-8")
        return brief

    def load_analysis(self) -> AudioAnalysis:
        return load_model(AudioAnalysis, self.analysis_file)

    def load_treatment(self) -> Treatment:
        return load_model(Treatment, self.treatment_file)

    def load_scenebook(self) -> Scenebook:
        return load_model(Scenebook, self.scenebook_file)
