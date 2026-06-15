# investor_agent → TradingAgents — Sequenced Implementation Plan

**Goal:** keep TradingAgents as the per-name decision engine and layer onto it the
three capabilities it structurally lacks — top-down macro context, a real holder's
position logic, and cross-sectional allocation — drawing the *logic* (not the code)
from the `investor_agent/` sub-projects as reference.

**Principles (apply to every phase):**
- Every behavior-changing step is gated by the existing `tradingagents/backtest/` harness.
- Every data step fails to `N/A`, never to a wrong number (wrong-but-authoritative is worse).
- Overlays are **post-hoc and pure** — they never mutate an agent's own verdict
  (model on `investor_agent/orchestrator/rules.py`, which is pure + unit-tested).
- Work lands in **TradingAgents** (git repo). Branch off `main` before Phase 0.

**Home & end-state (confirmed):**
- **TradingAgents is the permanent home.** `investor_agent/` is a **temporary reference
  only** — its logic is ported into TradingAgents natively, then the whole folder is
  **deleted** (Phase 6). Nothing may import from it at the end.
- `TradingAgents/main.py` is the **single launcher**. End goal: it runs the user's
  command directly —
  `python main.py --schedule --telegram-token <T> --telegram-chat-id <C> --deepseek-key <D> --fred-key <F>`.
  Today `main.py` is a one-shot batch runner (`--llm-key`, no `--schedule`/`--deepseek-key`/
  digest/bot); Phase 5 brings those capabilities in natively (the orchestrator can't be
  copied — it only exists to drive the three sub-projects that get deleted).

Status note: the read-only **Market Risk Context** section is already ported into the
report (`scripts/macro_snapshot.py`, `scripts/report_html.py`, `main.py`). Phase 0
hardens it; nothing else is actioned yet.

---

## Phase 0 — Harden the macro snapshot already in the report  ✅ DONE
*Reliability only; no behavior change. Do first — everything downstream trusts these numbers.*
*Shipped: `_bounded` guards in every fetcher + the two FRED call sites; Horizon column +
"strategic backdrop" framing; grouped rows + overlap footnote; `macro` config block +
dated manual stamp. Verified by `tests/test_macro_snapshot.py` (20 tests) + a render smoke.*

| Step | File | Change | Gate |
|---|---|---|---|
| 0.1 **Sanity bounds** | `scripts/macro_snapshot.py` | Clamp each fetcher to plausibility ranges (CAPE 5–70, Buffett 50–400%, VIX 5–150, PMI 30–70, T10Y2Y −3…+5). Out-of-range → return `None` + log a warning. Stops a regex grabbing garbage from a changed page silently poisoning the tally. | unit test per fetcher: mocked bad page → asserts `None` |
| 0.2 **Horizon labels** | `scripts/report_html.py` | Tag each indicator **Tactical (days–wks)** / **Strategic (yrs)** / **Macro (monthly)**; one-line card framing "strategic backdrop, not a trading trigger". | render `--no-send` HTML |
| 0.3 **Overlap grouping** | `scripts/report_html.py` | Group rows (Valuation / Vol-Sentiment / Credit-Rates / Growth) + footnote that the tally is indicative, not a weighted model (CAPE≈Buffett, VIX≈F&G double-count). | same render |
| 0.4 **Manual overrides** | `scripts/measure_config.json` + `macro_snapshot.py` | Optional `macro.manual_overrides {ism_pmi, ism_pmi_below_50_months, margin_debt_yoy_pct}` + `fred_api_key`, each date-stamped so a stale hand value is visible. | run with an override set; confirm it appears + dated |

**Exit:** snapshot renders identically when healthy, degrades to `N/A` (never wrong)
when sources break, no longer reads "3 reds" as 3 independent confirmations. No agent
behavior changed.

---

## Phase 1 — Position-aware overlay  ✅ DONE
*Biggest behavioral win; still no agent change. Pure + unit-testable.*

TA outputs a generic rating; it doesn't know what you hold or at what cost basis. Port
`stock_analysier`'s holder logic as a post-hoc overlay (model on `orchestrator/rules.py`).

- **1.1** New `tradingagents/portfolio/position_overlay.py` — pure
  `overlay(rating, position, macro_level, regime) -> action` mapping TA's 5-tier rating
  onto a holder's tactical ladder (`HOLD → YELLOW_WATCH → TRIM → DEFENSIVE → EXIT`) using
  cost basis, unrealized P&L, holding period, concentration. Port the *logic* from
  `investor_agent/stock_analysier/src/agents/{cost_basis,tactical_exit,portfolio_fit}.py`.
- **1.2** Positions source: a `portfolio.yaml` (holdings: symbol/shares/cost_basis/
  purchase_date; cash) — reuse `stock_analysier/config/portfolio.yaml`'s schema so it's
  one mental model.
- **1.3** Wire into `scripts/report_html.py` / `main.py`: held names show the **final
  holder action** beside TA's raw rating; non-held show TA's rating as-is.

**Gate:** 1.1 is pure → unit tests like `orchestrator/tests` do for `rules.py`. No LLM, no network.

**Exit:** a held name at +120% in a calm tape reads "TRIM / take partial", not a naive
"SELL"; the agent verdict itself is untouched and auditable.

---

## Phase 2 — Regime gate (two tiers)  ✅ 2.1 DONE · ⏸ 2.2 INTENTIONALLY DEFERRED (needs backtest)
*Behavior change → backtest-gated. The disciplined version of "feed macro into agents".*

- **2.1 Overlay gate (low risk, first).** Extend Phase 1's overlay with the D1 danger
  rule from `rules.py`: danger = macro level ∈ {HIGH, CRITICAL} **or** regime ∈ risk-off
  set → held positions escalate one ladder rung, new entries downgrade to `HOLD_OFF`.
  Pure, unit-testable, reversible. **Symmetry check:** must also relax on calm — verify
  it is not a one-way permabear ratchet.
- **2.2 Macro into the risk debators (gated; only if 2.1 + backtest support it).** Inject
  the macro snapshot **only** into `tradingagents/agents/risk_mgmt/conservative_debator.py`
  (optionally neutral) as *context, not directive* — append a "Market Risk Context" block
  to the existing prompt. Do **not** touch analysts or the trader (avoids LLM
  over-anchoring on vivid macro facts and washing out per-name edge).
  - **Gate (hard):** behind a config flag `macro_into_risk_debate: false` by default. A/B
    via `tradingagents/backtest/runner.py` — `AgentStrategy` flag-on vs flag-off over a
    fixed window, against the `RandomStrategy` baseline. Keep **only if** it improves
    risk-adjusted return without systematically biasing bearish in the up-window (the
    CAPE-red-for-a-decade failure mode is what the backtest exists to catch).

**Exit:** 2.1 shipped; 2.2 shipped only with a tearsheet showing net benefit, else left
flagged-off with the negative result recorded.

---

## Phase 3 — Cross-sectional / opportunity-cost ranking  ✅ DONE
*Structural gap: TA ranks nothing against anything. Backtest-gated.*

- **3.1** New `tradingagents/portfolio/relative_strength.py` — rank the candidate set
  (held + watchlist) by RS vs benchmark + regime fit, reusing
  `investor_agent/capital_rotation_analysier/src/rotation/{signals,regime,metrics}.py`.
- **3.2** Surface a ranked "best use of capital" table; feed the rank into
  `portfolio_manager.py` as context for sizing (it already has `portfolio/sizing.py`).

**Gate:** backtest rank-weighted allocation vs equal-weight vs TA-as-is.

---

## Phase 4 — Close the loop  ✅ DONE
*Cheap, high-discipline.*

Wire TA's existing `portfolio/reflection_v2.py` + `outcomes_store.py` to score each
overlay action against realized outcome (mirrors `stock_analysier`'s `audit.sqlite`).
A process you don't grade is noise. Low effort, no new infra.

---

## Phase 5 — Unified launcher in `main.py`  ✅ DONE (startup verified live)
*Reimplement the orchestrator's launch surface against TradingAgents' own pipeline +
the new overlays — NOT by copying `investor_agent/orchestrator` (it subprocesses the
three sub-projects, which are deleted in Phase 6).*

- **5.1** Extend `parse_args` in `main.py`: add `--schedule`, `--deepseek-key`
  (translation/narrative), keep `--telegram-token/--telegram-chat-id/--fred-key`. Map
  `--deepseek-key`/`--llm-key` so the given command authenticates the provider.
- **5.2 Scheduler** (`--schedule`): holiday-aware, timezone-aware firing 1h before each
  open (XHKG / XNYS), port the loop from `orchestrator/scheduler.py` (pandas-market-
  calendars). Without `--schedule`, `main.py` keeps its current one-shot behavior.
- **5.3 Digest**: compact bilingual (EN + 繁體中文) push message = macro gate (Phase 0) +
  per-name holder action (Phase 1/2) + cross-sectional rank (Phase 3) + action summary.
  Port the compose/i18n approach from `orchestrator/{digest,i18n}.py`; the per-name verdicts
  now come from TradingAgents' own pipeline, not `stock_analysier`.
- **5.4 Telegram two-way bot**: on-demand `/us <ticker>` / `/hk <ticker>` runs the
  TradingAgents pipeline for that name and replies. Port the listener from
  `orchestrator/telegram_bot.py` (allowlist + cooldown + daily cap).
- **5.5 Translation**: `--deepseek-key` powers 繁體中文; absent → English only (port
  `orchestrator/translate.py`, graceful fallback).

**Gate:** parity smoke test — `python main.py --schedule … --run-now` fires one digest;
`--no-send` builds without Telegram; `/us NVDA` replies. Verified against the user's creds.

## Phase 6 — Remove `investor_agent/`  ✅ DONE (deleted; repo imports + 60 tests pass without it)
*Only after Phases 0–5 land and parity is verified.*

- **6.1** `grep -r investor_agent` across TradingAgents → must be **zero** imports/paths.
- **6.2** Delete `investor_agent/` (it was never git-tracked). Confirm
  `python main.py --schedule …` still runs end-to-end with the folder gone.

---

## Explicitly deferred / NOT doing
- **HK macro gate** — the snapshot is US plumbing (S&P/FRED/Treasury); barely applies to
  `0700.HK`. Use `capital_rotation`'s HK southbound + RS when HK context is wanted. Defer.
- **Adding more agents** — TA + `stock_analysier`'s 23 already risk false precision and
  token cost. Resist until a backtest justifies the agents in place; consider *pruning*
  redundant debators instead.

---

## Dependency / ordering

```
Phase 0 (reliability) ──> Phase 1 (position overlay) ──> Phase 2.1 (gate overlay)
                                                              └─> Phase 2.2 (debator inject, backtest-gated)
Phase 1 ──> Phase 3 (cross-sectional rank, backtest-gated) ──> Phase 4 (reflection loop)
Phases 0–4 (native capabilities) ──> Phase 5 (unified launcher) ──> Phase 6 (delete investor_agent)
```

Phases 0 and 1 are pure/reliability — ship immediately. Everything that changes a
decision (2.2, 3) is flagged-off-by-default until the backtest tearsheet justifies it.
Phase 5 makes the user's `--schedule` command run on `main.py`; Phase 6 removes the
reference folder once parity holds.

---

## Appendix — rationale (original macro-soundness review)

Why each macro item makes investment sense, kept for reference:

- **Read-only macro context — KEEP, reframe.** A per-stock BUY/SELL in a vacuum ignores
  the regime; the same rating means different things at VIX 16 / CAPE 41 than at a panic
  low. Caveats: horizon mismatch (CAPE/Buffett/margin-debt are 5–10yr signals with ~no
  days-to-weeks power — "red" for a decade while the market tripled); naive tally
  (equal-weights correlated indicators: CAPE≈Buffett, VIX≈F&G); mixed frequency (daily vs
  lagged monthly blended into one snapshot). → Phase 0.2/0.3 + Phase 2 discipline.
- **Feed macro into agents — only via the risk layer, backtest first.** LLMs over-anchor
  on injected vivid facts regardless of the name; horizon mismatch can bias bearish in a
  bull market. It changes decisions → must be validated, not assumed. → Phase 2.2, gated.
- **Manual overrides (PMI / margin debt).** Operational robustness; these are the most
  fragile scrapes and monthly (a hand value is valid for weeks). → Phase 0.4.
- **Sanity bounds (highest value).** Wrong-but-authoritative data is worse than N/A; a
  layout change could make a regex grab garbage. → Phase 0.1.
- **Horizon labels.** The single biggest misuse risk cured cheaply by telling the reader
  which gauges are timely. → Phase 0.2.
- **Indicator-overlap note.** Stops the tally reading as N independent confirmations when
  several measure the same thing. → Phase 0.3.
- **HK macro context.** The whole gate is US; only loosely relevant to HK tickers.
  `capital_rotation` already produces HK southbound + RS. → Deferred.
</content>
</invoke>
