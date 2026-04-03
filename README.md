# TooobaLogParser

A log parser and web-based comparison tool for RISC-V core simulation logs.

## CLI Usage

Parse a log file and print statistics:

```bash
python main.py /path/to/sim.log
python main.py /path/to/sim.log.gz   # gzip supported
```

Output is cached alongside the log file (`<logfile>.totals_cache`) and reused on subsequent runs unless the log file has changed.

## Web App

A browser-based UI for loading and comparing multiple logs side by side.

### Setup

```bash
pip install -r requirements.txt
```

### Running

```bash
# With a log root directory (enables the file browser sidebar)
python app.py --log-root /path/to/logs

# Or via environment variable
LOG_ROOT=/path/to/logs python app.py

# Options
python app.py --help
  --log-root DIR   Root directory containing CPU config subdirectories
  --host HOST      Bind host (default: 127.0.0.1)
  --port PORT      Bind port (default: 5000)
```

Then open `http://localhost:5000` in your browser.

### Log Directory Structure

The file browser expects the root directory to contain one subdirectory per CPU configuration:

```
/path/to/logs/
├── config_a/
│   ├── run1.log.gz
│   └── run2.log.gz
├── config_b/
│   └── run1.log.gz
└── ...
```

Each subdirectory appears as a collapsible group in the sidebar. Click any log file to load it into the comparison table.

### Features

- **Side-by-side comparison** — stats are grouped by line type, one column per log
- **Diff highlighting** — values that differ across logs are highlighted
- **Caching** — processed results are cached as `<logfile>.webapp_cache.json`; cached files are marked with a green dot in the sidebar and load instantly
- **Manual path input** — type an absolute path directly if not using a log root
