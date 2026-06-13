#!/bin/bash
# Launch the Telegram scheduler (blocking).
cd "$(dirname "$0")"
exec .venv/bin/python -m src.notifier.scheduler "$@"
