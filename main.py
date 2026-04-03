import argparse
import io
import os
from parselogNew import *

def get_cache_path(logfile: str) -> str:
    return logfile + ".totals_cache"

def try_load_cache(logfile: str) -> str | None:
    cache_path = get_cache_path(logfile)
    if not os.path.exists(cache_path):
        return None
    try:
        log_mtime = str(os.path.getmtime(logfile))
        with open(cache_path, "r") as f:
            cached_mtime = f.readline().rstrip("\n")
            if cached_mtime != log_mtime:
                return None
            return f.read()
    except OSError:
        return None

def save_cache(logfile: str, output: str) -> None:
    cache_path = get_cache_path(logfile)
    log_mtime = str(os.path.getmtime(logfile))
    with open(cache_path, "w") as f:
        f.write(log_mtime + "\n")
        f.write(output)

def main():
    parser = argparse.ArgumentParser(description="Parse a log file and print totals.")
    parser.add_argument("logfile", help="Path to the log file (can be .gz)")
    args = parser.parse_args()

    cached = try_load_cache(args.logfile)
    if cached is not None:
        print(cached, end="")
        return

    lp = LogParser(
        log=args.logfile,
        # lineTypesToPrune=[None, NonRVFILine],
        lineTypesToPrune=[None],
        # If a timestamped line matches, then either RVFILine or NonRVFILine will match
        lineTypesToError=[TimestampedLine],
        RootLogLine=TimestampedLine,
        startWhen=(lambda ll: isinstance(ll, RVFILine) and ll.rvfi >= 10000),
    )

    buf = io.StringIO()
    import sys
    orig_stdout = sys.stdout
    sys.stdout = buf
    lp.printTotals()
    sys.stdout = orig_stdout

    output = buf.getvalue()
    save_cache(args.logfile, output)
    print(output, end="")

if __name__ == "__main__":
    main()
