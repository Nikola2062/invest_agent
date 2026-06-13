#!/bin/bash
# Launch the Streamlit dashboard using the shared repo-root .venv.
cd "$(dirname "$0")"
exec ../.venv/bin/streamlit run src/dashboard/app.py "$@"
