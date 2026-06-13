#!/bin/bash
# Launch the intraday monitor as a long-running daemon (15-min cron during US market hours).
cd "$(dirname "$0")"
exec .venv/bin/python -m src.notifier.intraday_monitor --daemon "$@"
