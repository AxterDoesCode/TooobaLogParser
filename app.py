import argparse
import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from parselogNew import LogParser, RVFILine, TimestampedLine

app = Flask(__name__, static_folder="static")

# Resolved at startup via --log-root CLI arg or LOG_ROOT env var
LOG_ROOT: str = ""


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
        return data["totals"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def save_json_cache(logfile: str, totals: dict) -> None:
    cache_path = get_json_cache_path(logfile)
    log_mtime = os.path.getmtime(logfile)
    with open(cache_path, "w") as f:
        json.dump({"mtime": log_mtime, "totals": totals}, f)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/browse")
def browse():
    if not LOG_ROOT:
        return jsonify({"error": "LOG_ROOT not configured. Start the server with --log-root or set the LOG_ROOT env var."}), 400
    if not os.path.isdir(LOG_ROOT):
        return jsonify({"error": f"LOG_ROOT directory not found: {LOG_ROOT}"}), 404

    dirs = []
    try:
        entries = sorted(os.scandir(LOG_ROOT), key=lambda e: e.name)
    except PermissionError as e:
        return jsonify({"error": str(e)}), 500

    for entry in entries:
        if not entry.is_dir():
            continue
        files = []
        try:
            for f in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if f.is_file() and not f.name.endswith(".webapp_cache.json") and not f.name.endswith(".totals_cache"):
                    cached = os.path.exists(get_json_cache_path(f.path))
                    files.append({"name": f.name, "path": f.path, "cached": cached})
        except PermissionError:
            pass
        dirs.append({"name": entry.name, "path": entry.path, "files": files})

    return jsonify({"root": LOG_ROOT, "dirs": dirs})


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
        return jsonify({"totals": cached, "cached": True})

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
    save_json_cache(logfile, totals)
    return jsonify({"totals": totals, "cached": False})


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
