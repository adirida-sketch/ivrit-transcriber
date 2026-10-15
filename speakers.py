"""Speaker separation (diarization) - runs locally with sherpa-onnx.

pyannote segmentation-3.0 finds who-speaks-when, wespeaker (ResNet34) tells the
voices apart. The two ONNX models (~33 MB) are downloaded once into models/ and
verified by SHA-256; after that everything works offline.

Clustering: sherpa-onnx with a deliberately low threshold (it over-splits), then
our own merge step by voice similarity: tiny clusters (a cough, noise) go to the
most similar voice, and with a speaker count the smallest clusters are merged into
the most similar voice until that count is left. Knowing the count matters: two
male voices on a phone line can be as similar as two parts of one speaker, so the
automatic mode may split one person in two. Merging by voice
beat merging by time (the neighbour in time is usually the *other* side of a call),
and sherpa-onnx's own fixed-count mode was unreliable on phone calls.
"""
import hashlib
import io
import os
import tarfile
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = {
    "file": MODELS_DIR / "pyannote-segmentation-3.0.onnx",
    "url": f"{_REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "url_sha256": "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    "member": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    "sha256": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
}
EMBEDDING = {
    "file": MODELS_DIR / "wespeaker-voxceleb-resnet34-LM.onnx",
    "url": f"{_REL}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
    "sha256": "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
}
THRESHOLD = 0.8          # 0.9 also worked on test calls; 1.0 merged two real speakers
TINY_SHARE = 0.03        # a cluster with < 3% of the speech ...
TINY_MAX_SEC = 20.0      # ... and < 20 s in total is folded into the most similar voice
SAMPLE_RATE = 16000
LABEL = "דובר"   # "[00:01:12] דובר 1: ..."
# Confidence check: a line counts as "clearly this speaker" when its voice is closer to
# that speaker than to any other by this cosine margin. On a hand-labelled call, 87 of
# 88 lines above the margin were right, against 11 of 18 below it.
CONFIDENT_MARGIN = 0.03
MIN_CHECK_SEC = 0.8      # shorter lines ("כן", "אהה") carry too little voice to check


class SpeakersError(Exception):
    """A failure with a message that can be shown to the user as-is (Hebrew)."""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def models_ready():
    return all(m["file"].exists() for m in (SEGMENTATION, EMBEDDING))


def _download(url, expected_sha):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except OSError as e:
        raise SpeakersError(
            "לא הצלחתי להוריד את מודל הפרדת הדוברים (כ-33MB, פעם אחת בלבד). "
            "בדקו את החיבור לאינטרנט ונסו שוב."
        ) from e
    if _sha256(data) != expected_sha:
        raise SpeakersError("מודל הפרדת הדוברים שהורד פגום. נסו שוב.")
    return data


def ensure_models(on_message=None):
    """Download the models once (verified by SHA-256)."""
    MODELS_DIR.mkdir(exist_ok=True)
    if not SEGMENTATION["file"].exists():
        if on_message:
            on_message("Downloading the speaker segmentation model (7 MB, once)...")
        archive = _download(SEGMENTATION["url"], SEGMENTATION["url_sha256"])
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:bz2") as tar:
            data = tar.extractfile(SEGMENTATION["member"]).read()   # one known member only
        if _sha256(data) != SEGMENTATION["sha256"]:
            raise SpeakersError("מודל הפרדת הדוברים שהורד פגום. נסו שוב.")
        _write(SEGMENTATION["file"], data)
    if not EMBEDDING["file"].exists():
        if on_message:
            on_message("Downloading the speaker voice model (26 MB, once)...")
        _write(EMBEDDING["file"], _download(EMBEDDING["url"], EMBEDDING["sha256"]))


def _write(path, data):
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)


def _threads():
    return max(2, min(8, os.cpu_count() or 4))


def _embedder(threads):
    import sherpa_onnx
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING["file"]), num_threads=threads))


def _voices(samples, segs, ext, per_cluster=20, min_len=0.5):
    """Length-weighted sum of voice embeddings per cluster (from its longest segments)."""
    import numpy as np
    sums = {}
    used = {}
    for s, e, k in sorted(segs, key=lambda t: t[0] - t[1]):        # longest first
        if used.get(k, 0) >= per_cluster or (e - s < min_len and k in sums):
            continue
        st = ext.create_stream()
        st.accept_waveform(SAMPLE_RATE, samples[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)])
        st.input_finished()
        v = np.array(ext.compute(st))
        v = v / (np.linalg.norm(v) + 1e-9) * (e - s)
        sums[k] = sums.get(k, 0) + v
        used[k] = used.get(k, 0) + 1
    return sums


def _cos(a, b):
    import numpy as np
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def diarize(samples, num_speakers=None, on_progress=None):
    """Return [(start, end, speaker)] with speakers numbered 1, 2, ... by first appearance.

    samples: 16 kHz mono float32. num_speakers: None (automatic) or the expected count.
    """
    try:
        import sherpa_onnx
    except ImportError as e:
        raise SpeakersError("חסרה ספריית sherpa-onnx. הריצו שוב את קובץ ההתקנה כדי לעדכן את המתמלל.") from e
    ensure_models()
    threads = _threads()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEGMENTATION["file"])),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING["file"]), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=THRESHOLD),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise SpeakersError("הגדרות הפרדת הדוברים לא תקינות.")

    def callback(done, total):
        if on_progress and total:
            on_progress(done / total)
        return 0

    result = sherpa_onnx.OfflineSpeakerDiarization(config).process(samples, callback=callback)
    segs = [(r.start, r.end, r.speaker) for r in result.sort_by_start_time()]
    if len({k for *_, k in segs}) > 1:
        segs = _merge(segs, _voices(samples, segs, _embedder(threads)), num_speakers)
    return _renumber(segs)


def confidence(samples, lines, diar):
    """Share of the checkable lines (>= MIN_CHECK_SEC) whose voice is clearly closest to their own speaker.

    Also marks every line: line["sure"] is True only for those lines (Claude may re-decide the others).
    None when there is nothing to compare (one speaker, or no long lines).
    """
    import numpy as np
    ext = _embedder(_threads())
    centre = {k: v / np.linalg.norm(v) for k, v in _voices(samples, diar, ext).items()}
    for ln in lines:
        ln["sure"] = False
    if len(centre) < 2:
        return None
    sure = checked = 0
    for ln in lines:
        k = ln.get("speaker")
        if k not in centre or ln["end"] - ln["start"] < MIN_CHECK_SEC:
            continue
        st = ext.create_stream()
        st.accept_waveform(SAMPLE_RATE, samples[int(ln["start"] * SAMPLE_RATE):int(ln["end"] * SAMPLE_RATE)])
        st.input_finished()
        v = np.array(ext.compute(st))
        v = v / (np.linalg.norm(v) + 1e-9)
        own = float(centre[k] @ v)
        other = max(float(centre[o] @ v) for o in centre if o != k)
        checked += 1
        ln["sure"] = own - other > CONFIDENT_MARGIN
        sure += ln["sure"]
    return round(sure / checked, 3) if checked else None


def _ts(sec, srt=False):
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), int(sec % 60)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{int((sec - int(sec)) * 1000):03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_labeled(txt_path, srt_path, lines, names=None):
    """Write .txt / .srt with "[HH:MM:SS] <speaker>: text" lines.

    names: {"1": "דנה", ...}; a speaker without a name is "דובר N"; a line without a speaker has no label.
    """
    names = names or {}
    with open(txt_path, "w", encoding="utf-8") as txt, open(srt_path, "w", encoding="utf-8") as srt:
        for i, ln in enumerate(lines, start=1):
            text = ln["text"]
            k = ln.get("speaker")
            if k is not None:
                text = f"{names.get(str(k)) or f'{LABEL} {k}'}: {text}"
            txt.write(f"[{_ts(ln['start'])}] {text}\n")
            srt.write(f"{i}\n{_ts(ln['start'], True)} --> {_ts(ln['end'], True)}\n{text}\n\n")


def _talk(segs):
    talk = {}
    for s, e, k in segs:
        talk[k] = talk.get(k, 0.0) + (e - s)
    return talk


def _merge(segs, voices, num_speakers):
    """Merge clusters by voice similarity (see the module docstring for the rules)."""
    while True:
        talk = _talk(segs)
        if len(talk) <= 1:
            return segs
        total = sum(talk.values())
        gone = min(talk, key=talk.get)
        too_many = num_speakers is not None and len(talk) > num_speakers
        tiny = talk[gone] < min(TINY_SHARE * total, TINY_MAX_SEC)
        if not (too_many or tiny):
            return segs
        others = [k for k in talk if k != gone]
        keep = max(others, key=lambda k: _cos(voices[k], voices[gone]) if k in voices and gone in voices else -1)
        segs = [(s, e, keep if k == gone else k) for s, e, k in segs]
        if gone in voices:
            voices[keep] = voices.get(keep, 0) + voices.pop(gone)


def _renumber(segs):
    order = {}
    for _s, _e, k in segs:
        order.setdefault(k, len(order) + 1)
    return [(s, e, order[k]) for s, e, k in segs]


def speaker_at(segs, start, end):
    """The speaker who talks most during [start, end]; the nearest one if nobody overlaps."""
    overlap = {}
    for s, e, k in segs:
        o = min(e, end) - max(s, start)
        if o > 0:
            overlap[k] = overlap.get(k, 0.0) + o
    if overlap:
        return max(overlap, key=overlap.get)
    if not segs:
        return None
    mid = (start + end) / 2
    return min(segs, key=lambda x: min(abs(x[0] - mid), abs(x[1] - mid)))[2]


def split_by_speaker(segments, diar):
    """Split transcript segments where the speaker changes, using word timestamps.

    segments: [{"start", "end", "text", "words": [(start, end, word), ...]}]
    Returns [{"start", "end", "text", "speaker"}] - consecutive pieces keep their order.
    """
    out = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"],
                        "speaker": speaker_at(diar, seg["start"], seg["end"])})
            continue
        runs = []   # [speaker, [words]]
        for w in words:
            k = speaker_at(diar, w[0], w[1])
            if runs and runs[-1][0] == k:
                runs[-1][1].append(w)
            else:
                runs.append([k, [w]])
        # a lone very short word between two runs of the same speaker is usually a timing blip
        i = 1
        while i < len(runs) - 1:
            k, ws = runs[i]
            if len(ws) == 1 and ws[0][1] - ws[0][0] < 0.3 and runs[i - 1][0] == runs[i + 1][0]:
                runs[i - 1][1].extend(ws + runs[i + 1][1])
                del runs[i:i + 2]
            else:
                i += 1
        for k, ws in runs:
            if "".join(w[2] for w in ws).strip():
                out.append({"speaker": k, "words": ws})
    return [_line(p) if "words" in p else p for p in _snap_turns(out)]


SENTENCE_END = (".", "?", "!", "…")
SNAP_MAX_WORDS = 4      # move at most this many words across a speaker change
SNAP_MAX_GAP = 1.0      # only between pieces this close in time (seconds)


def _ends_sentence(w):
    return w[2].strip().endswith(SENTENCE_END)


def _mid_word(w):
    """No punctuation at all after this word - a comma is also a natural place for a turn to end."""
    return w[2].strip()[-1:].isalnum()


def _snap_turns(pieces):
    """Move a speaker change that falls mid-sentence to the nearest sentence end.

    The voice tends to switch a word or a few late or early ("...אני אשלח לך את" | "הקובץ, טוב"),
    so at each change that is not at a sentence end: the words after the last sentence end of the
    piece before ("tail") can go to the next speaker, or the words up to the first sentence end of
    the piece after ("head") can go back. The shorter move wins (tail on a tie: the voice is more often
    late), up to SNAP_MAX_WORDS. A change right after a comma is left alone.
    """
    def mid_sentence(a, b):
        wa, wb = a.get("words"), b.get("words")
        return bool(wa and wb) and not _ends_sentence(wa[-1]) and wb[0][0] - wa[-1][1] <= SNAP_MAX_GAP

    i = 0
    for _ in range(10 * len(pieces) + 10):   # each move settles a boundary; the cap is only a safeguard
        if i >= len(pieces) - 1:
            break
        a, b = pieces[i], pieces[i + 1]
        if a["speaker"] == b["speaker"] or not mid_sentence(a, b) or not _mid_word(a["words"][-1]):
            i += 1
            continue
        wa, wb = a["words"], b["words"]
        ends = [j for j, w in enumerate(wa) if _ends_sentence(w)]
        tail = len(wa) - (ends[-1] + 1 if ends else 0)
        head = next((j + 1 for j, w in enumerate(wb) if _ends_sentence(w)), None)
        tail = tail if tail <= SNAP_MAX_WORDS else None
        head = head if head is not None and head <= SNAP_MAX_WORDS else None
        if head is not None and (tail is None or head < tail):
            a["words"], b["words"] = wa + wb[:head], wb[head:]
        elif tail is not None:
            a["words"], b["words"] = wa[:len(wa) - tail], wa[len(wa) - tail:] + wb
        else:
            i += 1
            continue
        # drop an emptied piece, then join neighbours that now have the same speaker mid-sentence
        pieces[i:i + 2] = [p for p in (a, b) if p["words"]]
        j = max(i - 1, 0)
        while j < min(i + 2, len(pieces) - 1):
            p, q = pieces[j], pieces[j + 1]
            short = min(len(p["words"]), len(q["words"])) <= SNAP_MAX_WORDS   # keep long lines as they were
            if p["speaker"] == q["speaker"] and mid_sentence(p, q) and short:
                p["words"] = p["words"] + pieces.pop(j + 1)["words"]
            else:
                j += 1
        i = max(i - 1, 0)
    return pieces


def _line(piece):
    ws = piece["words"]
    return {"start": ws[0][0], "end": ws[-1][1], "text": "".join(w[2] for w in ws).strip(),
            "speaker": piece["speaker"]}
