"""Ivrit transcriber - local GUI.

Runs a small web server on 127.0.0.1 and opens the UI (ui/index.html) in an
Edge/Chrome app window. Each transcription gets its own folder,
Transcripts/<date_time>/, holding the audio file and the .txt / .srt results
(plus a summary file when the user asked for a title / summary / tasks).
"""
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import insights
import speakers
import winfront

APP_DIR = Path(__file__).resolve().parent
UI_FILE = APP_DIR / "ui" / "index.html"
LOG_FILE = APP_DIR / "transcriber.log"
TRANSCRIPTS_DIR = APP_DIR / "Transcripts"
INCOMING_DIR = APP_DIR / ".incoming"   # uploaded file waits here until "transcribe"
DEFAULT_PORT = 8765
APP_TAG = "ivrit-transcriber"
AUTHOR = "Adir Avraham"
REPO_URL = "https://github.com/adirida-sketch/ivrit-transcriber"
try:
    VERSION = (APP_DIR / "VERSION").read_text(encoding="utf-8").strip()
except OSError:
    VERSION = "dev"
MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".ogg", ".oga", ".opus", ".flac", ".aac",
              ".wma", ".amr", ".3gp", ".mp4", ".m4v", ".mov", ".webm", ".mkv"}
IDLE_TIMEOUT = 180  # seconds without any request from the window -> exit
BYE_GRACE = 5       # seconds after the window closed (allows page reloads)
PAUSED_ABANDON = 600  # a paused job with no window for this long is cancelled, so the server can exit
NO_NET_MSG = ("אין חיבור לאינטרנט.\n"
              "בתמלול הראשון צריך להוריד את מודל התמלול (כ-1.6GB), פעם אחת בלבד.\n"
              "התחברו לאינטרנט ונסו שוב. אחרי ההורדה המתמלל יעבוד גם בלי אינטרנט.")
NET_LOST_MSG = ("החיבור לאינטרנט נותק בזמן הורדת המודל.\n"
                "התחברו שוב ולחצו 'תמלל', וההורדה תמשיך מהנקודה שבה נעצרה.")

# pythonw.exe has no console: send prints/tracebacks to a log file instead
if sys.stdout is None or sys.stderr is None:
    _log = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _log

LOCK = threading.Lock()
STATE = {}
LAST_SEEN = time.monotonic()
BYE = False
INSIGHTS_LOCK = threading.Lock()   # one Claude request at a time
# transcript lines of the current job with voice labels + certainty, for Claude's attribution by content
# (kept out of STATE so /status stays small)
LINES = {"job": None, "lines": None, "count": None}


class JobControl:
    """Pause / cancel for the running job: the engine calls it between sentences."""

    def __init__(self):
        self.go = threading.Event()
        self.go.set()
        self.cancelled = False

    def __call__(self):
        while not self.go.wait(0.3):
            if self.cancelled:
                break
        if self.cancelled:
            import transcribe_ivrit
            raise transcribe_ivrit.Cancelled()


CONTROL = JobControl()


def new_control():
    global CONTROL
    CONTROL = JobControl()


def reset_state(**extra):
    STATE.clear()
    STATE.update(
        job=0, running=False, stage="idle", progress=0.0,
        message="", lines=[], duration=None, download_mb=None, device=None,
        file=None, folder=None, started=None, error=None, result=None, extras=[], speakers=None,
        paused=False, paused_at=None, paused_total=0.0, cancelled=False,
    )
    STATE.update(extra)


reset_state(job=0)


def touch(bye=False):
    global LAST_SEEN, BYE
    LAST_SEEN = time.monotonic()
    BYE = bye


# ---------------------------------------------------------------- transcription

def model_cache_dir():
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = Path(HF_HUB_CACHE)
    except Exception:
        base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return base / ("models--" + MODEL_ID.replace("/", "--"))


def model_is_cached():
    snapshots = model_cache_dir() / "snapshots"
    return snapshots.is_dir() and any((d / "model.bin").exists() for d in snapshots.iterdir())


def has_internet(timeout=4):
    """True if huggingface.co answers at all (any HTTP status counts)."""
    try:
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        urllib.request.urlopen(req, timeout=timeout).close()
        return True
    except urllib.error.HTTPError:
        return True
    except urllib.error.URLError as e:
        return isinstance(e.reason, ssl.SSLError)   # reached a server, TLS issue only
    except Exception:
        return False


def dir_size_mb(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            try:
                if not os.path.islink(p):
                    total += os.path.getsize(p)
            except OSError:
                pass
    return total / (1024 * 1024)


def watch_download(job, done):
    """While the model is loading, report how much of it is in the local cache."""
    cache = model_cache_dir()
    while not done.wait(1.0):
        with LOCK:
            if STATE["job"] != job:
                return
            if STATE["stage"] != "model":
                continue
        mb = dir_size_mb(cache) if cache.exists() else 0.0
        with LOCK:
            if STATE["job"] == job:
                STATE["download_mb"] = round(mb)


def elapsed():
    """Seconds the current job has been working (time spent paused does not count). Call under LOCK."""
    if not STATE["started"]:
        return 0
    paused = STATE["paused_total"] + (time.time() - STATE["paused_at"] if STATE["paused_at"] else 0)
    return time.time() - STATE["started"] - paused


def set_paused(paused):
    with LOCK:
        if not STATE["running"] or STATE["paused"] == paused:
            return
        STATE["paused"] = paused
        if paused:
            STATE["paused_at"] = time.time()
        else:
            STATE["paused_total"] += time.time() - (STATE["paused_at"] or time.time())
            STATE["paused_at"] = None
    if paused:
        CONTROL.go.clear()
    else:
        CONTROL.go.set()


def cancel_job():
    CONTROL.cancelled = True
    CONTROL.go.set()   # wake a paused job so it can stop


def run_job(job, audio_path, beam, extras, num_speakers):
    def on_event(kind, **data):
        with LOCK:
            if kind == "stage":
                STATE["stage"] = data["stage"]
                STATE["progress"] = 0.0
                STATE["device"] = data.get("device", STATE["device"])
            elif kind in ("decode", "progress"):
                STATE["progress"] = data["progress"]
            elif kind == "audio":
                STATE["duration"] = data["duration"]
            elif kind == "segment":
                STATE["lines"].append({"start": data["start"], "text": data["text"]})
                if data["progress"] is not None:
                    STATE["progress"] = data["progress"]
            elif kind == "restart":
                STATE["lines"] = []
            elif kind == "done":
                STATE["progress"] = 1.0
                if data.get("lines"):
                    LINES.update(job=job, lines=data["lines"], count=num_speakers or None)
                STATE["result"] = {"txt": data["txt"], "srt": data["srt"], "elapsed": data["elapsed"],
                                   "speaker_names": speaker_names_from(data["txt"]) if data.get("speakers") else None,
                                   "speakers": data.get("speakers"), "speakers_error": data.get("speakers_error"),
                                   "speakers_confidence": data.get("speakers_confidence"),
                                   "confidence": data.get("confidence")}
            if "message" in data:
                STATE["message"] = data["message"]
        if "message" in data:
            print(data["message"], flush=True)

    done = threading.Event()
    threading.Thread(target=watch_download, args=(job, done), daemon=True).start()
    try:
        try:
            import transcribe_ivrit
        except ImportError as e:
            raise RuntimeError(
                f"חסרות ספריות פייתון ({e.name}). הריצו שוב את install.cmd מתיקיית ההתקנה."
            ) from e
        if not model_is_cached() and not has_internet():
            raise RuntimeError(NO_NET_MSG)
        transcribe_ivrit.transcribe_file(audio_path, beam=beam, on_event=on_event, speakers=num_speakers,
                                         checkpoint=CONTROL)
    except Exception as e:
        cancelled = e.__class__.__name__ == "Cancelled"
        if cancelled:
            print("Transcription cancelled by the user", flush=True)
            with LOCK:
                STATE["cancelled"] = True
                STATE["result"] = None
        else:
            traceback.print_exc()
            with LOCK:
                STATE["error"] = str(e) or e.__class__.__name__
                in_model_stage = STATE["stage"] == "model"
            if in_model_stage and not model_is_cached() and not has_internet():
                with LOCK:
                    STATE["error"] = NET_LOST_MSG
        # the audio goes back to .incoming (so "transcribe" can run again) and the job folder is removed
        try:
            INCOMING_DIR.mkdir(exist_ok=True)
            shutil.move(str(audio_path), str(INCOMING_DIR / audio_path.name))
            shutil.rmtree(audio_path.parent)
        except OSError:
            traceback.print_exc()
    else:
        if extras:
            with LOCK:
                STATE["stage"] = "insights"
                STATE["progress"] = 0.0
            with INSIGHTS_LOCK:
                run_insights(extras)
    finally:
        done.set()
        with LOCK:
            STATE["running"] = False


def run_insights(kinds):
    """Ask Claude for a title / summary / tasks of the current result.

    Writes <name>_סיכום.txt next to the transcript and, with a title, renames the
    job folder to "<date_time> - <title>". A failure never touches the transcript:
    it is stored as result["insights_error"] so the user can retry.
    """
    with LOCK:
        result = STATE["result"]
    if not result:
        return
    with LOCK:
        job = STATE["job"]
    lines = LINES["lines"] if LINES["job"] == job else None
    try:
        text = Path(result["txt"]).read_text(encoding="utf-8")
        data = insights.analyze(text, kinds, insights.load_key(), lines=lines, count=LINES["count"],
                                names=result.get("speaker_names"))
    except Exception as e:
        if not isinstance(e, insights.InsightsError):
            traceback.print_exc()
        print(f"insights failed: {e}", flush=True)
        with LOCK:
            STATE["result"] = {**result, "insights_error": str(e) or e.__class__.__name__,
                               "insights_kinds": list(kinds)}
        return

    names = data.pop("speaker_names", None)
    by_line = data.pop("line_speakers", None)
    merged = {**(result.get("insights") or {}), **data}
    txt = Path(result["txt"])
    summary = txt.with_name(txt.stem + "_סיכום.txt")
    summary.write_text(insights.to_text(merged), encoding="utf-8")
    new = {**result, "insights": merged, "summary": str(summary)}
    new.pop("insights_error", None)
    new.pop("insights_kinds", None)
    if by_line and lines:
        new = attribute_by_content(new, lines, by_line, job)
    if names and new.get("speaker_names"):
        wanted = dict(new["speaker_names"])
        for num, name in names.items():
            if num in wanted and wanted[num] == default_speaker(num) and name not in wanted.values():
                wanted[num] = name   # only fill in speakers the user has not named
        new = apply_speaker_names(new, wanted)
    if data.get("title"):
        new = rename_job_dir(new, data["title"])
    with LOCK:
        STATE["result"] = new
        STATE["folder"] = str(Path(new["txt"]).parent)


# ---------------------------------------------------------------- helpers

def safe_name(name):
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "audio.m4a"


def new_job_dir():
    """Transcripts/2026-10-13_22-45-10 (with _2, _3... if that second is taken)."""
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path, n = TRANSCRIPTS_DIR / stamp, 2
    while path.exists():
        path, n = TRANSCRIPTS_DIR / f"{stamp}_{n}", n + 1
    path.mkdir()
    return path


def attribute_by_content(result, lines, by_line, job):
    """Combine the voice labels with Claude's attribution by content and rewrite the .txt / .srt.

    A line the voice was sure about keeps its voice label (those were right 99% of the time in tests);
    unsure lines, and every line when the voices could not be told apart, take Claude's speaker.
    """
    final, changed = [], 0
    for ln, k in zip(lines, by_line):
        if k is not None and (ln.get("speaker") is None or not ln.get("sure")):
            changed += k != ln.get("speaker")
            ln = {**ln, "speaker": k}
        final.append(ln)
    found = sorted({ln["speaker"] for ln in final})
    if len(found) < 2:
        return result
    old = result.get("speaker_names") or {}
    names = {str(k): old.get(str(k)) or default_speaker(k) for k in found}
    speakers.write_labeled(result["txt"], result["srt"], final, names)
    LINES.update(job=job, lines=final)
    return {**result, "speaker_names": names, "speakers": len(found), "speakers_error": None,
            "speakers_by_content": (result.get("speakers_by_content") or 0) + changed}


def default_speaker(num):
    return f"{transcribe_label()} {num}"


def transcribe_label():
    return "דובר"   # same as transcribe_ivrit.SPEAKER_LABEL (that module needs faster-whisper to import)


def speaker_names_from(txt_path):
    """{"1": "דובר 1", ...} for the speaker labels found in a fresh transcript."""
    found = {}
    for m in re.finditer(r"^\[\d{2}:\d{2}:\d{2}\] דובר (\d+): ", Path(txt_path).read_text(encoding="utf-8"), re.M):
        found.setdefault(m.group(1), default_speaker(m.group(1)))
    return dict(sorted(found.items(), key=lambda kv: int(kv[0])))


def apply_speaker_names(result, names):
    """Rename speakers in the .txt / .srt (line prefixes only) and in the summary; returns the new result."""
    old = result.get("speaker_names") or {}
    names = {k: (insights.clean_name(names.get(k)) or default_speaker(k)) for k in old}
    if len(set(names.values())) != len(names):
        raise ValueError("לשני דוברים אי אפשר לתת אותו שם")
    swap = {old[k] + ": ": names[k] + ": " for k in old if old[k] != names[k]}
    if not swap:
        return {**result, "speaker_names": names}
    prefixes = sorted(swap, key=len, reverse=True)

    def relabel(line, ts_len):
        head, rest = line[:ts_len], line[ts_len:]
        for p in prefixes:
            if rest.startswith(p):
                return head + swap[p] + rest[len(p):]
        return line

    txt = Path(result["txt"])
    lines = txt.read_text(encoding="utf-8").split("\n")
    txt.write_text("\n".join(relabel(ln, 11) if re.match(r"\[\d{2}:\d{2}:\d{2}\] ", ln) else ln
                             for ln in lines), encoding="utf-8")
    if result.get("srt"):   # a saved transcript may have lost its .srt
        srt = Path(result["srt"])
        srt.write_text("\n".join(relabel(ln, 0) for ln in srt.read_text(encoding="utf-8").split("\n")),
                       encoding="utf-8")
    new = {**result, "speaker_names": names}
    if new.get("insights"):
        # all names at once (a swap must not chain); a name is a whole word, optionally after
        # Hebrew prefix letters ("ודנה", "ליוסי"), which are kept
        changed = {old[k]: names[k] for k in old if old[k] != names[k]}
        alts = "|".join(re.escape(o) for o in sorted(changed, key=len, reverse=True))
        pattern = re.compile(r"(?<!\w)([ובכלמהש]{0,3})(" + alts + r")(?!\w)")

        def sub(v):
            if isinstance(v, str):
                return pattern.sub(lambda m: m.group(1) + changed[m.group(2)], v)
            if isinstance(v, list):
                return [sub(x) for x in v]
            if isinstance(v, dict):
                return {a: sub(b) for a, b in v.items()}
            return v
        new["insights"] = sub(new["insights"])
        if new.get("summary"):
            Path(new["summary"]).write_text(insights.to_text(new["insights"]), encoding="utf-8")
    return new


def speaker_samples(result):
    """A first, reasonably long line of each speaker - helps the user tell who is who."""
    names = result.get("speaker_names") or {}
    out = {}
    for ln in Path(result["txt"]).read_text(encoding="utf-8").split("\n"):
        for k, name in names.items():
            p = f"] {name}: "
            if k not in out and p in ln[:11 + len(p)]:
                text = ln.split(p, 1)[1].strip()
                if len(text.split()) >= 4:
                    out[k] = text[:90]
    return out


def rename_job_dir(result, title):
    """Transcripts/<date_time> -> Transcripts/<date_time> - <title>; the paths in result follow."""
    old = Path(result["txt"]).parent
    stamp = old.name.split(" - ")[0]
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    clean = re.sub(r"\s+", " ", clean).strip(" .")[:60].strip(" .")
    new = old.with_name(f"{stamp} - {clean}")
    if not clean or new == old or new.exists():
        return result
    try:
        old.rename(new)
    except OSError:
        traceback.print_exc()   # e.g. a file in it is open - keep the plain folder name
        return result
    moved = dict(result)
    for k in ("txt", "srt", "summary"):
        if moved.get(k):
            moved[k] = str(new / Path(moved[k]).name)
    return moved


def main_txt(folder):
    """The transcript .txt in a job folder (not the summary), or None."""
    found = sorted(p for p in folder.glob("*.txt") if not p.stem.endswith("_סיכום"))
    return found[0] if found else None


def saved_transcripts():
    """The transcripts in Transcripts/, newest first, for the "open a saved transcript" list."""
    out = []
    if TRANSCRIPTS_DIR.is_dir():
        for d in TRANSCRIPTS_DIR.iterdir():
            txt = main_txt(d) if d.is_dir() else None
            if txt:
                stamp, _, title = d.name.partition(" - ")
                out.append({"id": d.name, "date": stamp, "title": title, "file": txt.stem,
                            "summary": txt.with_name(txt.stem + "_סיכום.txt").exists()})
    return sorted(out, key=lambda x: x["id"], reverse=True)


def load_saved(name):
    """A result for a saved transcript (as if it had just finished), so the done view and /insights work on it."""
    folder = TRANSCRIPTS_DIR / (name or "")
    if not name or not folder.is_dir() or folder.resolve().parent != TRANSCRIPTS_DIR.resolve():
        raise ValueError("התמלול לא נמצא")
    txt = main_txt(folder)
    if not txt:
        raise ValueError("בתיקייה הזו אין קובץ תמלול")
    srt, summary = txt.with_suffix(".srt"), txt.with_name(txt.stem + "_סיכום.txt")
    names = speaker_names_from(txt)
    audio = next((p.name for p in sorted(folder.iterdir()) if p.suffix.lower() in AUDIO_EXTS), txt.name)
    result = {"txt": str(txt), "srt": str(srt) if srt.exists() else None, "elapsed": None, "saved": True,
              "speaker_names": names or None, "speakers": len(names) or None,
              "insights": None, "summary": None}
    if summary.exists():
        result["insights"] = insights.from_text(summary.read_text(encoding="utf-8"))
        result["summary"] = str(summary)
    return result, audio


def clear_incoming():
    INCOMING_DIR.mkdir(exist_ok=True)
    for f in INCOMING_DIR.iterdir():
        try:
            f.unlink()
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        touch()
        url = urlparse(self.path)
        qs = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            body = UI_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        elif url.path == "/ping":
            self.send_json({"app": APP_TAG, "version": VERSION, "author": AUTHOR, "repo_url": REPO_URL, "folder": str(APP_DIR), "transcripts": str(TRANSCRIPTS_DIR)})

        elif url.path == "/status":
            since = int((qs.get("since") or ["0"])[0])
            with LOCK:
                s = dict(STATE)
                s["lines_total"] = len(STATE["lines"])
                s["lines"] = STATE["lines"][since:] if since <= len(STATE["lines"]) else STATE["lines"][:]
                s["elapsed"] = elapsed()
            self.send_json(s)

        elif url.path == "/netcheck":
            cached = model_is_cached()
            self.send_json({"model_cached": cached, "online": True if cached else has_internet()})

        elif url.path == "/result":
            with LOCK:
                result = STATE["result"]
            if not result:
                return self.send_json({"error": "no result"}, 404)
            extra = {"speaker_samples": speaker_samples(result)} if result.get("speaker_names") else {}
            self.send_json({"text": Path(result["txt"]).read_text(encoding="utf-8"), **result, **extra})

        elif url.path == "/saved":
            self.send_json({"items": saved_transcripts()})

        elif url.path == "/settings":
            key = insights.load_key()
            self.send_json({"has_key": bool(key), "key_hint": insights.key_hint(key)})

        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        touch()
        url = urlparse(self.path)
        qs = parse_qs(url.query)

        if url.path == "/upload":
            name = safe_name((qs.get("name") or [""])[0])
            if Path(name).suffix.lower() not in AUDIO_EXTS:
                return self.send_json({"error": "סוג קובץ לא נתמך"}, 400)
            size = int(self.headers.get("Content-Length") or 0)
            clear_incoming()
            path = INCOMING_DIR / name
            part = path.with_name(path.name + ".part")
            remaining = size
            with open(part, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            if remaining:
                part.unlink(missing_ok=True)
                return self.send_json({"error": "ההעלאה נקטעה"}, 400)
            os.replace(part, path)
            self.send_json({"name": path.name, "size": size, "path": str(path)})

        elif url.path == "/transcribe":
            data = self.read_json()
            src = INCOMING_DIR / safe_name(data.get("name", ""))
            beam = 1 if data.get("beam") == 1 else 5
            extras = [k for k in insights.KINDS if k in (data.get("extras") or [])]
            # speaker separation: None = off, 0 = automatic count, 2-5 = that many speakers
            num_speakers = data.get("speakers")
            if num_speakers not in (None, 0, 2, 3, 4, 5):
                return self.send_json({"error": "מספר דוברים לא תקין"}, 400)
            if extras and not insights.load_key():
                return self.send_json({"error": "לסיכום, לכותרת ולמשימות צריך מפתח API של Claude. הוסיפו אותו ב'אפשרויות נוספות'"}, 400)
            if not src.exists():
                return self.send_json({"error": "הקובץ לא נמצא. טענו אותו שוב"}, 404)
            with LOCK:
                if STATE["running"]:
                    return self.send_json({"error": "תמלול אחר כבר רץ"}, 409)
                job_dir = new_job_dir()
                path = Path(shutil.move(str(src), str(job_dir / src.name)))
                job = STATE["job"] + 1
                reset_state(job=job, running=True, stage="decode", file=path.name,
                            folder=str(job_dir), started=time.time(), extras=extras, speakers=num_speakers)
                new_control()
            threading.Thread(target=run_job, args=(job, path, beam, extras, num_speakers), daemon=True).start()
            self.send_json({"job": job})

        elif url.path == "/load":
            try:
                result, audio = load_saved(self.read_json().get("id"))
            except ValueError as e:
                return self.send_json({"error": str(e)}, 404)
            with LOCK:
                if STATE["running"] or INSIGHTS_LOCK.locked():
                    return self.send_json({"error": "המתמלל עסוק כרגע. נסו שוב בעוד רגע"}, 409)
                job = STATE["job"] + 1
                reset_state(job=job, file=audio, folder=str(Path(result["txt"]).parent), result=result)
            self.send_json({"job": job, "file": audio})

        elif url.path == "/pause":
            set_paused(bool(self.read_json().get("paused")))
            with LOCK:
                self.send_json({"paused": STATE["paused"]})

        elif url.path == "/cancel":
            with LOCK:
                running, stage = STATE["running"], STATE["stage"]
            if not running or stage not in ("decode", "model", "transcribe"):
                return self.send_json({"error": "אי אפשר לבטל בשלב הזה"}, 409)
            cancel_job()
            self.send_json({"ok": True})

        elif url.path == "/insights":
            wanted = self.read_json().get("kinds") or []
            kinds = [k for k in insights.KINDS if k in wanted]
            with LOCK:
                busy, result = STATE["running"], STATE["result"]
            if busy or not result:
                return self.send_json({"error": "אין תמלול מוכן לעיבוד"}, 409)
            if not kinds:
                return self.send_json({"error": "לא נבחרה אף אפשרות"}, 400)
            if not insights.load_key():
                return self.send_json({"error": "לא הוגדר מפתח API של Claude"}, 400)
            if not INSIGHTS_LOCK.acquire(blocking=False):
                return self.send_json({"error": "Claude כבר מעבד את התמלול"}, 409)
            try:
                run_insights(kinds)
            finally:
                INSIGHTS_LOCK.release()
            with LOCK:
                result = STATE["result"]
            if result.get("insights_error"):
                return self.send_json({"error": result["insights_error"]}, 502)
            self.send_json({"ok": True})

        elif url.path == "/speakers":
            names = self.read_json().get("names") or {}
            with LOCK:
                busy, result = STATE["running"], STATE["result"]
            if busy or not result or not result.get("speaker_names"):
                return self.send_json({"error": "אין תמלול עם דוברים"}, 409)
            try:
                new = apply_speaker_names(result, {str(k): str(v) for k, v in names.items()})
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            with LOCK:
                STATE["result"] = new
            self.send_json({"ok": True, "speaker_names": new["speaker_names"]})

        elif url.path == "/settings":
            key = (self.read_json().get("api_key") or "").strip()
            if key:
                if not key.startswith("sk-ant-"):
                    return self.send_json({"error": "זה לא נראה כמו מפתח API של Claude (הוא מתחיל ב-sk-ant-)"}, 400)
                try:
                    insights.check_key(key)
                except insights.InsightsError as e:
                    return self.send_json({"error": str(e)}, 400)
            insights.save_key(key)
            self.send_json({"has_key": bool(key), "key_hint": insights.key_hint(key)})

        elif url.path == "/open":
            data = self.read_json()
            with LOCK:
                result = STATE["result"]
            # folders and files open in front of the app window (see winfront.py)
            if data.get("what") == "transcripts" or (data.get("what") == "folder" and not result):
                TRANSCRIPTS_DIR.mkdir(exist_ok=True)
                winfront.open_in_front(lambda: os.startfile(TRANSCRIPTS_DIR), TRANSCRIPTS_DIR.name)
            elif data.get("what") == "repo":
                webbrowser.open(REPO_URL)   # fixed URLs only - never one sent by the page
            elif data.get("what") == "keys":
                webbrowser.open(insights.KEYS_URL)
            elif data.get("what") == "folder":
                txt = result["txt"]
                winfront.open_in_front(lambda: subprocess.Popen(["explorer", "/select,", txt]), Path(txt).parent.name)
            elif data.get("what") == "summary":
                if result and result.get("summary"):
                    winfront.open_in_front(lambda: os.startfile(result["summary"]), Path(result["summary"]).name)
            elif result:
                winfront.open_in_front(lambda: os.startfile(result["txt"]), Path(result["txt"]).name)
            self.send_json({"ok": True})

        elif url.path == "/bye":
            touch(bye=True)
            self.send_json({"ok": True})

        else:
            self.send_json({"error": "not found"}, 404)


# ---------------------------------------------------------------- startup

def find_app_browser():
    roots = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
    rels = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
    for rel in rels:
        for root in roots:
            if root and (Path(root) / rel).exists():
                return str(Path(root) / rel)
    return None


def open_window(url):
    browser = find_app_browser()
    if browser:
        subprocess.Popen([browser, f"--app={url}", "--window-size=980,860"])
    else:
        webbrowser.open(url)


def watchdog(server):
    while True:
        time.sleep(1)
        with LOCK:
            running, paused, stage = STATE["running"], STATE["paused"], STATE["stage"]
        idle = time.monotonic() - LAST_SEEN
        # paused and the window is gone: cancel (the audio goes back to .incoming), then exit as usual
        if (running and paused and idle > PAUSED_ABANDON and not CONTROL.cancelled
                and stage in ("decode", "model", "transcribe")):
            print("Paused job abandoned (no window), cancelling", flush=True)
            cancel_job()
        if not running and (idle > IDLE_TIMEOUT or (BYE and idle > BYE_GRACE)):
            server.shutdown()
            return


def already_running(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=1) as r:
            return json.load(r).get("app") == APP_TAG
    except Exception:
        return False


def main():
    os.chdir(APP_DIR)
    port = DEFAULT_PORT
    if already_running(port):
        open_window(f"http://127.0.0.1:{port}/")
        return
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"Ivrit transcriber {VERSION} running at {url} (folder: {APP_DIR})", flush=True)
    threading.Thread(target=watchdog, args=(server,), daemon=True).start()
    open_window(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
