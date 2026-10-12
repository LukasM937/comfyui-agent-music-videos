import synth

from mvstudio.models import AudioAnalysis, LyricLine, Section, Word

BEAT = 0.5
LABELS = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]


def analysis_from_truth(with_lyrics=True) -> AudioAnalysis:
    dur = 112.0
    beats = [round(i * BEAT, 3) for i in range(int(dur / BEAT))]
    downs = beats[::4]
    sections, cl = [], {"intro": "A", "verse": "B", "chorus": "C", "bridge": "D", "outro": "A"}
    first: dict[str, str] = {}
    for i, (lab, s, e) in enumerate(synth.truth(), start=1):
        sid = f"S{i:02d}"
        sec = Section(id=sid, start=s, end=e, bars=(e - s) / 2, cluster=cl[lab],
                      energy={"intro": .2, "verse": .5, "chorus": 1., "bridge": .4, "outro": .2}[lab],
                      label=lab, vocal_ratio=.8 if lab in ("verse", "chorus", "bridge") else 0.)
        if cl[lab] in first:
            sec.repeat_of = first[cl[lab]]
        else:
            first[cl[lab]] = sid
        sections.append(sec)
    lines = []
    if with_lyrics:
        for sec in sections:
            if sec.vocal_ratio > 0:
                t = sec.start + 0.5
                while t + 2.0 < sec.end:
                    words = [Word(start=t + k * 0.4, end=t + k * 0.4 + 0.3, text=f"w{k}")
                             for k in range(5)]
                    lines.append(LyricLine(start=t, end=t + 2.0, text="la la la la la", words=words))
                    t += 2.5
    return AudioAnalysis(audio_file="x.wav", duration=dur, sr=22050, bpm=120, beats=beats,
                         downbeats=downs, sections=sections, lyrics=lines,
                         lyrics_source="asr" if with_lyrics else "none")
