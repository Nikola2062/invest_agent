# TradingAgents — Active Implementation Plan

**Principles (apply to every step):**
- Numeric work stays in **deterministic Python, not the LLM** — better correctness and
  zero token cost. The LLM supplies the *directional view*; quant code turns it into a
  *risk-measured structure*.
- Overlays are **post-hoc and pure** — they never mutate an agent's verdict
  (model on `tradingagents/portfolio/position_overlay.py`, pure + unit-tested).
- Every behavior-changing step is **backtest-gated** via `tradingagents/backtest/`.
- Every data step fails to `N/A`, never to a wrong number.

---

## ✅ Completed (history — do not re-action)

- **investor_agent → TradingAgents migration, Phases 0–6.** Macro snapshot hardening;
  position-aware overlay; regime gate (2.1); cross-sectional RS ranking; reflection loop;
  unified `main.py` launcher (`--schedule`/digest/Telegram bot/i18n); `investor_agent/`
  deleted, repo green without it.
- **Report mobile fix** (`scripts/report_html.py`). Responsive tables (`.table-wrap` +
  `@media`); switching tickers no longer scroll-jumps to page top (`scrollIntoView`).
- **LLM cost reduction.** Debate-report compaction — the 5 debate voices now receive a
  length-bounded digest of each analyst report instead of the full text
  (`agent_utils.get_debate_reports` + 5 debater nodes). Measured ~36.5k input tokens
  saved/ticker on real logs. Toggle `debate_report_mode` (`compact` default | `full`) +
  `debate_report_max_chars`; news payload trimmed (`news_article_limit` 20→10,
  `global_news_article_limit` 10→5); knobs wired through `main._graph_config` +
  `measure_config.json`. 410 tests pass.

---

- **On-demand overview (scheduled-run cost cut).** The `--schedule` fire no longer
  runs the full pipeline on all 6 tickers. New `overview.mode` (`on_demand` default |
  `full` legacy): `on_demand` pushes a cheap **LLM-free** overview (macro + RS rank +
  per-holding price action) and auto-deep-dives only HELD names that trip a
  deterministic alarm (`positions.triage_alarms`, pure + tested), capped at
  `max_auto_deep` (default 2); all other names are pulled on demand via the existing
  bot (`/us`, `/hk`). New: `positions.TriageConfig`/`triage_alarms`,
  `digest.build_overview_digest`, `main.build_overview` + rewired `on_fire`,
  `overview` config block. Expected ~75–90% scheduled-cost cut (≈$0 on calm days).
  The drawdown alarm fires on **drawdown-from-recent-high** (`drawdown_from_peak` +
  `recent_drawdowns`, 3mo default), a fresh-move signal — NOT P&L vs cost basis,
  so chronically underwater holdings don't trip every fire. **Per-name dedupe**
  (`select_for_deepdive` + `load/save_triage_state`, state in
  `.tradingagents/triage_state.json`): an alarming name auto-runs again only on a
  new alarm category, a drawdown worsening >= `rerun_worsen_pct`, or after
  `rerun_after_days` — so a steady-state alarm goes quiet; digest marks ▶ running /
  ⏸ covered. Tests: `test_triage.py` (17), `test_dedupe.py` (10),
  `test_overview_digest.py` (10). 308 unit tests pass. Live-smoked twice: fire 1
  deep-dives FIG+0700.HK, fire 2 dedupes both to zero auto-runs, $0 LLM cost.

## 🎯 Active — Quant volatility / options overlay (HK: 02800.HK / HSI)

The repo already has a quant layer (`relative_strength.py`, `sizing.py`, `backtest/`).
The missing dimension is **volatility / derivatives**. Build it as a deterministic
overlay that reuses existing primitives, surfaces as a report card + digest line, and
sits behind a config flag (`enable_options_overlay`, default off).

### Q0 — Data probe (GATE — decides everything below)  ⬜
- One-off OpenD script: pull a couple of `02800.HK` option snapshots, confirm **IV +
  Greeks come back populated** (not empty / no-permission).
  - **Path A** (populated) → real chain-level engine (specific strikes, payoffs).
  - **Path B** (empty) → VHSI-proxy engine (implied-vs-realized vol → qualitative lean,
    no strikes).
- Entitlement gotcha (app data ≠ OpenAPI data) documented in `[[futu-options-data-access]]`.

### Q1 — Volatility primitives  ⬜  *(pure; no probe dependency — can build now)*
- New `tradingagents/portfolio/volatility.py`: `realized_vol`, `iv_rank`,
  `expected_move`, `vol_risk_premium`, `term_structure_slope`.
- Lift annualized realized vol from `sizing._realised_vol` — don't duplicate.
- **Gate:** unit tests (mirrors `tests/test_sizing.py`).

### Q2 — Options strategy engine  ⬜  *(pure)*
- New `tradingagents/portfolio/options_overlay.py`: decision matrix
  (pipeline rating × IV regime × held?) → strategy family → `select_strikes` (delta/DTE
  targeted) → `payoff` (max profit / max loss / breakevens). Defined-risk only.
- Mirrors the pure/tested shape of `position_overlay.py`.
- **Gate:** unit tests over the full matrix + payoff math.

### Q3 — Data adapter  ⬜  *(gated by Q0)*
- New `tradingagents/dataflows/futu_options.py`: OpenD chain + snapshot (Path A) with
  VHSI fallback (Path B). Register in `dataflows/interface.py` vendor map. Config:
  `enable_options_overlay` (default false), OpenD host/port.

### Q4 — Surface + backtest gate  ⬜
- Report card in `scripts/report_html.py` (after macro, before per-ticker tabs) + digest
  line in `tradingagents/runtime/digest.py`.
- New `backtest/` Strategy (overlay on/off) through `backtest/runner.walk_forward`; ship
  **only** with a tearsheet showing improved risk-adjusted return.

**Open design decisions (confirm before Q2/Q3):** (1) executable strikes vs vol-regime
signal only; (2) hedging-overlay focus on the held `02800.HK` vs standalone directional
ideas; (3) templated output (recommended, cheapest) vs thin LLM narrative.

---

## ⏸ Deferred / NOT doing (unchanged)

- **2.2 — macro into risk debators.** Inject macro snapshot only into
  `conservative_debator.py` as context; needs a backtest tearsheet first. Flagged off.
- **HK macro gate.** Current macro snapshot is US-only plumbing (S&P/FRED/Treasury);
  revisit with `capital_rotation`'s HK southbound + RS when HK context is wanted.
- **Adding more LLM agents.** Resist (false precision + token cost); prune redundant
  debators instead, only if a backtest justifies it.
