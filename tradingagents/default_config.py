import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_DEBATE_REPORT_MODE":   "debate_report_mode",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Point-in-time data cache: when True, vendor calls with a past as_of
    # date are cached to disk and never re-fetched. Speeds up backtests and
    # guarantees a given (vendor, method, ticker, as_of) call returns the
    # same result on every replay. Off by default since live runs should
    # always hit the vendor.
    "pit_cache_enabled": False,
    # Phase 4: calibrated reflection. When True, every resolved trade also
    # writes a multi-horizon outcome row (5d/21d/63d alpha) to a CSV store.
    # The PM prompt then receives aggregated, significance-gated lessons in
    # place of (or alongside) the noisy per-trade prose reflections. Off
    # by default until the store has enough trades to be statistically
    # meaningful (~20 minimum per bucket).
    "calibrated_reflection_enabled": False,
    "outcomes_store_path": None,             # defaults to data_cache_dir/outcomes.csv
    "reflection_horizons": (5, 21, 63),      # trading days stored per trade
    "reflection_horizon_days": 21,           # horizon used for prompt-injected lessons
    "reflection_min_trades": 20,             # bucket must have ≥ this many trades
    "reflection_t_threshold": 1.5,           # |t-stat| gate for emitting a lesson
    # SEC EDGAR User-Agent. SEC's data API requires an identifying header
    # ("Name email@example.com"); requests without one get a 403. Set via
    # this config key or the EDGAR_USER_AGENT env var.
    "edgar_user_agent": os.getenv("EDGAR_USER_AGENT", ""),
    # Polygon.io API key. Required when data_vendors.core_stock_apis or
    # news_data is set to "polygon". Free tier is 5 req/min — paid tier
    # starts at $30/mo. Sign up at https://polygon.io.
    "polygon_api_key": os.getenv("POLYGON_API_KEY", ""),
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    # Debate context size. The five debate voices (bull, bear, aggressive,
    # neutral, conservative) re-ingest all four analyst reports on every turn,
    # which is the single largest repeated input-token cost per run.
    #   "compact" (default): each report is condensed to debate_report_max_chars
    #     before being placed in debate prompts — large token saving, minimal
    #     analytical loss (opening thesis + closing summary/recommendation kept).
    #   "full": legacy behavior — entire reports in every debate prompt.
    # Either way the full reports are preserved in state and the HTML report.
    "debate_report_mode": "compact",      # "compact" | "full"
    "debate_report_max_chars": 1800,      # per-report budget when mode == "compact"
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 10,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 5,       # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",    # NSE India (Nifty 50)
        ".BO":  "^BSESN",   # BSE India (Sensex)
        ".T":   "^N225",    # Tokyo (Nikkei 225)
        ".HK":  "^HSI",     # Hong Kong (Hang Seng)
        ".L":   "^FTSE",    # London (FTSE 100)
        ".TO":  "^GSPTSE",  # Toronto (TSX Composite)
        ".AX":  "^AXJO",    # Australia (ASX 200)
        "":     "SPY",      # default for US-listed tickers (no suffix)
    },
})
