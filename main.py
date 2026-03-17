import argparse
from parselogNew import *

def main():
    parser = argparse.ArgumentParser(description="Parse a log file and print totals.")
    parser.add_argument("logfile", help="Path to the log file (can be .gz)")
    args = parser.parse_args()

    lp = LogParser(
        log=args.logfile,
        lineTypesToPrune=[None, NonRVFILine],
        # If a timestamped line matches, then either RVFILine or NonRVFILine will match
        lineTypesToError=[TimestampedLine],
        RootLogLine=TimestampedLine,
        startWhen=(lambda ll: isinstance(ll, RVFILine) and ll.rvfi >= 10000),
    )

    lp.printTotals()

if __name__ == "__main__":
    main()
