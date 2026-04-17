import argparse
import json
import os
import sys
import threading
import uuid

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from parselogNew import LogParser, RVFILine, TimestampedLine

app = Flask(__name__, static_folder="static")

# Resolved at startup via --log-root CLI arg or LOG_ROOT env var
LOG_ROOT: str = ""

# Background parse-folder jobs: job_id -> {status, results, total}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def get_json_cache_path(logfile: str) -> str:
    return logfile + ".webapp_cache.json"


def try_load_json_cache(logfile: str) -> dict | None:
    cache_path = get_json_cache_path(logfile)
    if not os.path.exists(cache_path):
        return None
    try:
        log_mtime = os.path.getmtime(logfile)
        with open(cache_path) as f:
            data = json.load(f)
        if data.get("mtime") != log_mtime:
            return None
        return {"totals": data["totals"], "finalCycle": data.get("finalCycle")}
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def save_json_cache(logfile: str, totals: dict, finalCycle: int | None) -> None:
    cache_path = get_json_cache_path(logfile)
    log_mtime = os.path.getmtime(logfile)
    with open(cache_path, "w") as f:
        json.dump({"mtime": log_mtime, "totals": totals, "finalCycle": finalCycle}, f)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


_LOG_CACHE_SUFFIXES = (".webapp_cache.json", ".totals_cache")


def _scan_dir(path: str, depth: int = 0, max_depth: int = 16) -> dict:
    """Recursively build a directory tree with file cache status."""
    result: dict = {"name": os.path.basename(path), "path": path, "files": [], "dirs": []}
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name)
    except PermissionError:
        return result
    for entry in entries:
        if entry.is_dir():
            if depth < max_depth:
                result["dirs"].append(_scan_dir(entry.path, depth + 1, max_depth))
        elif entry.is_file() and not any(entry.name.endswith(s) for s in _LOG_CACHE_SUFFIXES):
            cached = os.path.exists(get_json_cache_path(entry.path))
            result["files"].append({"name": entry.name, "path": entry.path, "cached": cached})
    return result


@app.route("/api/browse")
def browse():
    if not LOG_ROOT:
        return jsonify({"error": "LOG_ROOT not configured. Start the server with --log-root or set the LOG_ROOT env var."}), 400
    if not os.path.isdir(LOG_ROOT):
        return jsonify({"error": f"LOG_ROOT directory not found: {LOG_ROOT}"}), 404

    try:
        entries = sorted(os.scandir(LOG_ROOT), key=lambda e: e.name)
    except PermissionError as e:
        return jsonify({"error": str(e)}), 500

    dirs = [_scan_dir(e.path) for e in entries if e.is_dir()]
    return jsonify({"root": LOG_ROOT, "dirs": dirs})


def _run_parse_folder(job_id: str, folder: str) -> None:
    """Runs in a background thread. Parses all uncached files in folder."""
    def update(result: dict) -> None:
        with _jobs_lock:
            _jobs[job_id]["results"].append(result)

    try:
        entries = []
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for fname in sorted(filenames):
                if not any(fname.endswith(s) for s in _LOG_CACHE_SUFFIXES):
                    entries.append(os.path.join(dirpath, fname))
    except PermissionError as e:
        with _jobs_lock:
            _jobs[job_id].update({"status": "error", "error": str(e)})
        return

    with _jobs_lock:
        _jobs[job_id]["total"] = len(entries)

    for fpath in entries:
        fname = os.path.basename(fpath)
        with _jobs_lock:
            _jobs[job_id]["current"] = fname

        cached = try_load_json_cache(fpath)
        if cached is not None:
            update({"file": fname, "path": fpath, "status": "cached"})
            continue
        try:
            lp = LogParser(
                log=fpath,
                lineTypesToPrune=[None],
                lineTypesToError=[TimestampedLine],
                RootLogLine=TimestampedLine,
                startWhen=(lambda ll: isinstance(ll, RVFILine) and ll.rvfi >= 10000),
                silent=True,
            )
            totals = {lt.__name__: stats for lt, stats in lp.totals.items()}
            save_json_cache(fpath, totals, lp.finalCycle)
            update({"file": fname, "path": fpath, "status": "ok"})
        except Exception as e:
            update({"file": fname, "path": fpath, "status": "error", "error": str(e)})

    with _jobs_lock:
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["current"] = None


@app.route("/api/parse-folder", methods=["POST"])
def parse_folder():
    data = request.get_json()
    folder = (data or {}).get("dir", "").strip()

    if not folder:
        return jsonify({"error": "No dir provided"}), 400
    if not os.path.isdir(folder):
        return jsonify({"error": f"Directory not found: {folder}"}), 404
    if LOG_ROOT and not os.path.abspath(folder).startswith(LOG_ROOT):
        return jsonify({"error": "Directory is outside LOG_ROOT"}), 403

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "results": [], "total": None, "current": None}

    thread = threading.Thread(target=_run_parse_folder, args=(job_id, folder), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/api/parse-folder/<job_id>", methods=["GET"])
def parse_folder_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/process", methods=["POST"])
def process_log():
    data = request.get_json()
    logfile = (data or {}).get("path", "").strip()

    if not logfile:
        return jsonify({"error": "No path provided"}), 400
    if not os.path.exists(logfile):
        return jsonify({"error": f"File not found: {logfile}"}), 404

    cached = try_load_json_cache(logfile)
    if cached is not None:
        return jsonify({"totals": cached["totals"], "finalCycle": cached["finalCycle"], "cached": True})

    try:
        lp = LogParser(
            log=logfile,
            lineTypesToPrune=[None],
            lineTypesToError=[TimestampedLine],
            RootLogLine=TimestampedLine,
            startWhen=(lambda ll: isinstance(ll, RVFILine) and ll.rvfi >= 10000),
            silent=True,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    totals = {lt.__name__: stats for lt, stats in lp.totals.items()}
    save_json_cache(logfile, totals, lp.finalCycle)
    return jsonify({"totals": totals, "finalCycle": lp.finalCycle, "cached": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Parser Web App")
    parser.add_argument("--log-root", default=os.environ.get("LOG_ROOT", ""),
                        help="Root directory containing CPU config subdirectories with log files")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    LOG_ROOT = os.path.abspath(args.log_root) if args.log_root else ""
    if LOG_ROOT:
        print(f"Log root: {LOG_ROOT}")
    else:
        print("No log root set. Use --log-root or LOG_ROOT env var to enable the file browser.")

    app.run(debug=True, port=args.port, host=args.host)
