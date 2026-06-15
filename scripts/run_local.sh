#!/usr/bin/env bash
# Launch the interactive TradingAgents CLI with all output written to the
# project-local .tradingagents/ directory (logs, cache, decision memory)
# instead of ~/.tradingagents. Usage: ./scripts/run_local.sh [cli args...]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRADINGAGENTS_RESULTS_DIR="$PROJECT_DIR/.tradingagents/logs"
export TRADINGAGENTS_CACHE_DIR="$PROJECT_DIR/.tradingagents/cache"
export TRADINGAGENTS_MEMORY_LOG_PATH="$PROJECT_DIR/.tradingagents/memory/trading_memory.md"

exec "$PROJECT_DIR/.venv/bin/tradingagents" "$@"
