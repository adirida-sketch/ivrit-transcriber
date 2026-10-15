import argparse
import math
import sys
import time
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel

MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
try:
    VERSION = (Path(__file__).resolve().parent / "VERSION").read_text(encoding="utf-8").strip()
except OSError:
    VERSION = "dev"
SAMPLE_RATE = 16000
SPEAKER_LABEL = "דובר"   # "[00:01:12] דובר 1: ..." when speakers are separated

class Cancelled(Exception):
    """Raised by the caller's checkpoint() to stop a transcription."""


# Keeps the loaded model in memory so the GUI can transcribe several files
# without reloading it every time.
_model_cache = {}


def fmt_ts(seconds, srt=False):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_audio(path, sr=SAMPLE_RATE, on_progress=None):
    """Decode any audio file to 16 kHz mono float32.
    Done here instead of inside faster-whisper, to avoid the
    'metadata_errors' incompatibility with newer PyAV versions."""
    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        total = container.duration / av.time_base if container.duration else None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        last_report = 0.0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
            if on_progress and total and frame.time is not None:
                now = time.monotonic()
                if now - last_report > 0.2:
                    on_progress(min(frame.time / total, 1.0))
                    last_report = now
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError("No audio decoded from file")
    if on_progress:
        on_progress(1.0)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def pick_device(requested):
    if requested != "auto":
        return requested
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_model(device):
    compute_type = "float16" if device == "cuda" else "int8"
    key = (device, compute_type)
    if key not in _model_cache:
        _model_cache.clear()
        try:
            # Use the local copy if it was downloaded before - works offline
            model = WhisperModel(MODEL_ID, device=device, compute_type=compute_type, local_files_only=True)
        except Exception:
            model = WhisperModel(MODEL_ID, device=device, compute_type=compute_type)
        _model_cache[key] = model
    return _model_cache[key]


def _plain(lines):
    """Transcript lines for the caller: no word timings, and no speaker unless the voices were told apart."""
    return [{"start": ln["start"], "end": ln["end"], "text": ln["text"],
             "speaker": ln.get("speaker"), "sure": bool(ln.get("sure"))} for ln in lines]


def _separate_speakers(audio, segments, num_speakers, emit):
    """Label transcript lines by speaker. Returns (lines, count, error message or None, confidence).

    lines always come back (without speakers when the voices could not be told apart), so the
    caller can still let Claude attribute them by content.
    """
    import speakers
    emit("stage", stage="speakers", message="Separating speakers...")
    try:
        if not speakers.models_ready():
            speakers.ensure_models(on_message=lambda m: emit("info", message=m))
        diar = speakers.diarize(audio, num_speakers or None, on_progress=lambda p: emit("progress", progress=p))
    except Exception as e:   # never lose the transcript because of this step
        msg = str(e) if isinstance(e, speakers.SpeakersError) else f"הפרדת הדוברים נכשלה: {e or e.__class__.__name__}"
        emit("info", message=f"Speaker separation failed: {e}")
        return _plain([{**s, "speaker": None} for s in segments]), 0, msg, None
    lines = speakers.split_by_speaker(segments, diar)
    if len({ln.get("speaker") for ln in lines} - {None}) < 2:
        # one voice found: labelling every line "speaker 1" would be misleading (and Claude would then
        # name that one speaker), so keep the transcript without labels and say why
        emit("info", message="Speaker separation found a single voice - no labels")
        if num_speakers:
            msg = (f"ביקשת {num_speakers} דוברים, אבל לפי הקול זוהה רק דובר אחד. זה קורה בשיחות קצרות "
                   "או כשהקולות דומים מאוד, ולכן התמלול נשאר בלי סימוני דוברים.")
        else:
            msg = "זוהה דובר אחד בלבד, ולכן התמלול נשאר בלי סימוני דוברים."
        return _plain([{**s, "speaker": None} for s in segments]), 0, msg, None
    try:
        conf = speakers.confidence(audio, lines, diar)
    except Exception as e:   # only a display figure - never fail the job for it
        emit("info", message=f"Speaker confidence not computed: {e}")
        conf = None
    return _plain(lines), len({ln["speaker"] for ln in lines if ln.get("speaker") is not None}), None, conf


def _run(audio, audio_path, device, beam, emit, num_speakers=None, check=lambda: None):
    compute_type = "float16" if device == "cuda" else "int8"
    emit("stage", stage="model", device=device,
         message=f"Loading model {MODEL_ID} on {device} ({compute_type})...\n"
                 "First run downloads about 1.6 GB, later runs use the local cache.")

    start = time.time()
    model = load_model(device)
    check()

    emit("stage", stage="transcribe", device=device, message="Transcribing...")
    with_speakers = num_speakers is not None
    segments, info = model.transcribe(
        audio,
        language="he",
        beam_size=beam,
        vad_filter=True,
        word_timestamps=with_speakers,   # lets a speaker change split a sentence
    )

    duration = len(audio) / SAMPLE_RATE
    txt_path = audio_path.with_suffix(".txt")
    srt_path = audio_path.with_suffix(".srt")

    collected = []
    conf_sum = conf_len = 0.0   # model confidence: mean token probability, weighted by duration
    with open(txt_path, "w", encoding="utf-8") as txt, open(srt_path, "w", encoding="utf-8") as srt:
        for i, seg in enumerate(segments, start=1):
            check()   # pause / cancel take effect between sentences
            text = seg.text.strip()
            line = f"[{fmt_ts(seg.start)}] {text}"
            txt.write(line + "\n")
            txt.flush()
            srt.write(f"{i}\n{fmt_ts(seg.start, True)} --> {fmt_ts(seg.end, True)}\n{text}\n\n")
            conf_sum += math.exp(seg.avg_logprob) * (seg.end - seg.start)
            conf_len += seg.end - seg.start
            if with_speakers:
                collected.append({"start": seg.start, "end": seg.end, "text": text,
                                  "words": [(w.start, w.end, w.word) for w in (seg.words or [])]})
            emit("segment", index=i, start=seg.start, end=seg.end, text=text, line=line,
                 progress=min(seg.end / duration, 1.0) if duration else None)

    found, speakers_error, speakers_conf, lines = None, None, None, None
    if with_speakers:
        import speakers
        lines, found, speakers_error, speakers_conf = _separate_speakers(audio, collected, num_speakers, emit)
        if not speakers_error:
            speakers.write_labeled(txt_path, srt_path, lines)

    elapsed = time.time() - start
    emit("done", txt=str(txt_path), srt=str(srt_path), elapsed=elapsed,
         speakers=found, speakers_error=speakers_error, speakers_confidence=speakers_conf, lines=lines,
         confidence=round(conf_sum / conf_len, 3) if conf_len else None)
    return txt_path, srt_path


def transcribe_file(audio_path, device="auto", beam=5, on_event=None, speakers=None, checkpoint=None):
    """Transcribe one file and write <name>.txt and <name>.srt next to it.

    speakers: None = no speaker separation, 0 = automatic count, N = N speakers.
    checkpoint(): called often while decoding and transcribing; it may block (pause)
    or raise Cancelled (cancel).
    on_event(kind, **data) is called with kind in:
    stage, decode, audio, segment, restart, progress, info, done.
    """
    emit = on_event or (lambda kind, **data: None)
    check = checkpoint or (lambda: None)
    audio_path = Path(audio_path)

    def decode_progress(p):
        check()
        emit("decode", progress=p)

    emit("stage", stage="decode", message="Decoding audio...")
    audio = load_audio(audio_path, on_progress=decode_progress)
    duration = len(audio) / SAMPLE_RATE
    emit("audio", duration=duration, message=f"Audio length: {duration / 60:.1f} min")

    device = pick_device(device)
    try:
        return _run(audio, audio_path, device, beam, emit, speakers, check)
    except Cancelled:
        raise
    except Exception as e:
        if device != "cuda":
            raise
        # GPU detected but CUDA libraries are missing/broken: fall back to CPU
        emit("restart", message=f"GPU failed ({e}), retrying on CPU...")
        return _run(audio, audio_path, "cpu", beam, emit, speakers, check)


def main():
    parser = argparse.ArgumentParser(description="Hebrew transcription with ivrit.ai")
    parser.add_argument("audio", help="Path to the audio file (m4a, mp3, wav...)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--version", action="version", version=f"Ivrit Transcriber {VERSION}")
    parser.add_argument("--beam", type=int, default=5, help="Beam size (1 = faster, 5 = more accurate)")
    parser.add_argument("--speakers", default=None, choices=["auto", "2", "3", "4", "5"],
                        help="Separate speakers: 'auto' or the number of speakers (more accurate)")
    args = parser.parse_args()

    # Make Hebrew print correctly in the Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    def on_event(kind, **data):
        if kind == "segment":
            print(data["line"], flush=True)
        elif kind == "done":
            if data.get("speakers_error"):
                print(f"\nSpeaker separation failed - the transcript has no speaker labels: {data['speakers_error']}")
            elif data.get("speakers"):
                print(f"\nSpeakers found: {data['speakers']}")
            print(f"\nDone in {data['elapsed'] / 60:.1f} min")
            print(f"Text: {data['txt']}")
            print(f"Subtitles: {data['srt']}")
        elif "message" in data:
            print(data["message"], flush=True)

    num_speakers = None if args.speakers is None else (0 if args.speakers == "auto" else int(args.speakers))
    transcribe_file(audio_path, device=args.device, beam=args.beam, on_event=on_event, speakers=num_speakers)


if __name__ == "__main__":
    main()
