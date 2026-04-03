import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from parselogNew import LogParser, RVFILine, TimestampedLine

app = Flask(__name__, static_folder="static")


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
    app.run(debug=True, port=5000)
