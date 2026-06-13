"""Pydantic schemas for config files and agent I/O."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


# ----- Config: portfolio -----

class Holding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    market: Literal["US", "HK"]
    shares: float
    cost_basis_per_share: float
    currency: str
    purchase_date: Optional[date] = None
    notes: Optional[str] = None

    @property
    def cost_basis_total(self) -> float:
        return self.shares * self.cost_basis_per_share


class CashPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currency: str
    amount: float


class Portfolio(BaseModel):
    model_config = ConfigDict(extra="forbid")
    holdings: list[Holding] = Field(default_factory=list)
    cash: list[CashPosition] = Field(default_factory=list)

    def find(self, symbol: str) -> Optional[Holding]:
        for h in self.holdings:
            if h.symbol == symbol:
                return h
        return None


# ----- Config: universe -----

class WatchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    notes: Optional[str] = None


class Universe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    watchlist: dict[str, list[WatchEntry]] = Field(default_factory=dict)
    active_markets: list[str] = Field(default_factory=list)
    preferred_hedge_instruments: list[str] = Field(default_factory=list)


# ----- Config: risk policy -----

class TriggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drawdown_magnitude_pct: float
    probability_min: float
    persistence_days: int


class ActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: int
    label: str
    trigger: TriggerConfig
    action: str
    trim_pct_of_position: Optional[float] = None
    rebuy_at_drawdown_pct: Optional[list[float]] = None
    rebuy_pct_of_trimmed: Optional[float] = None
    hedge_remainder: Optional[bool] = None
    description: str


class TaxAwareness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_long_term_holding_check: bool
    long_term_holding_days_us: int
    wash_sale_avoidance_days: int
    prefer_trim_over_full_exit_when_close_to_long_term: bool
    long_term_proximity_window_days: int


class VolatilityAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool
    realized_vol_window_days: int
    high_vol_threshold_annualized_pct: float
    low_vol_threshold_annualized_pct: float
    high_vol_factor: float
    low_vol_factor: float


class HedgingMinimums(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_position_value_usd: float
    prefer_etf_short_when_below_usd: float


class IntradayThresholds(BaseModel):
    """Intraday Monitor alert thresholds — previously hardcoded in intraday_monitor.py."""
    model_config = ConfigDict(extra="forbid")
    drop_pct: float = -5.0      # alert when current price is this far below prior close
    spike_pct: float = 8.0      # alert when current price is this far above prior close


class RiskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_horizon_days: int
    signal_persistence_days_min: int
    cooldown_days_between_actions: int
    actions: list[ActionConfig]
    tax_awareness: TaxAwareness
    volatility_adjustment: VolatilityAdjustment
    hedging_minimums: HedgingMinimums
    cold_start_min_history_runs: int = 3       # audit runs needed before high-impact actions are unblocked
    cold_start_max_action_level: int = 2       # during cold-start, cap firing at this action level (2 = ORANGE_TRIM)
    rebuy_staleness_days: int = 30             # if rebuy band never reached after N days → Monitor emits rebuy_stale alert
    intraday_thresholds: IntradayThresholds = Field(default_factory=IntradayThresholds)


# ----- Config: schedule -----

class PushConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    cron: str
    market: str
    purpose: str
    detail_level: str
    # When true, the scheduler suppresses the digest send if no position has
    # a tactical action and no watchlist entry is actionable and no alerts
    # are pending. Out-of-band alerts (thesis-breaks, intraday) are still pushed.
    send_only_if_action: bool = False


class Schedule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host_timezone: str
    pushes: list[PushConfig]
    dashboard_refresh_seconds: int


# ----- Config: competence circle -----

class CompetenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    in_circle_sectors: list[str] = Field(default_factory=list)
    in_circle_keywords: list[str] = Field(default_factory=list)
    out_of_circle_sectors: list[str] = Field(default_factory=list)
    out_of_circle_keywords: list[str] = Field(default_factory=list)
    always_in_circle: list[str] = Field(default_factory=list)
    always_out_of_circle: list[str] = Field(default_factory=list)
    on_out_of_circle: dict[str, str] = Field(default_factory=lambda: {"policy": "analyze_but_flag"})
    notes: Optional[str] = None


# ----- Config: valuation (DCF defaults + entry thresholds) -----

class DCFMarketOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    discount_rate: Optional[float] = None
    terminal_growth: Optional[float] = None
    growth_cap_y1_5: Optional[float] = None
    growth_floor_y1_5: Optional[float] = None
    fallback_growth_y1_5: Optional[float] = None


class DCFDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    terminal_growth: float = 0.025
    discount_rate: float = 0.10
    growth_cap_y1_5: float = 0.25
    growth_floor_y1_5: float = -0.05
    fallback_growth_y1_5: float = 0.05


class DCFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default: DCFDefaults = Field(default_factory=DCFDefaults)
    per_market: dict[str, DCFMarketOverride] = Field(default_factory=dict)

    def resolved(self, market: Optional[str]) -> DCFDefaults:
        """Merge per-market overrides on top of `default` and return effective values."""
        base = self.default.model_dump()
        override = self.per_market.get(market or "", None)
        if override is not None:
            for k, v in override.model_dump().items():
                if v is not None:
                    base[k] = v
        return DCFDefaults.model_validate(base)


class EntryDecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    margin_of_safety_required_pct: float = 20.0
    wait_band_discount_pct: float = 10.0
    default_position_size_pct: float = 5.0


class ValuationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dcf: DCFConfig = Field(default_factory=DCFConfig)
    entry_decision: EntryDecisionConfig = Field(default_factory=EntryDecisionConfig)


# ----- Config: secrets -----

class DeepSeekSecrets(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model_default: str = "deepseek-chat"      # general-purpose agents
    model_reasoner: str = "deepseek-reasoner" # Synthesizer / Devil's Advocate


class TelegramSecrets(BaseModel):
    bot_token: str = ""
    chat_id: str = ""


class FinnhubSecrets(BaseModel):
    api_key: str = ""


class OptionalKey(BaseModel):
    api_key: str = ""


class Secrets(BaseModel):
    finnhub: FinnhubSecrets = Field(default_factory=FinnhubSecrets)
    deepseek: DeepSeekSecrets = Field(default_factory=DeepSeekSecrets)
    telegram: TelegramSecrets = Field(default_factory=TelegramSecrets)
    polygon: Optional[OptionalKey] = None
    newsapi: Optional[OptionalKey] = None
    fred: Optional[OptionalKey] = None


# ----- Agent I/O: Fundamental Analyst -----

class FundamentalAssessment(BaseModel):
    quality_score: float = Field(ge=0, le=10)
    moat_assessment: str
    moat_strength: Literal["wide", "narrow", "none"]
    balance_sheet_health: Literal["strong", "adequate", "weak"]
    growth_outlook: str
    capital_allocation: str
    red_flags: list[str] = Field(default_factory=list)
    thesis_one_liner: str
    # Optional numerics from deterministic computation
    roic_pct: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None


# ----- Agent I/O: Valuation -----

class ValuationResult(BaseModel):
    current_price: float
    currency: str
    intrinsic_low: float
    intrinsic_base: float
    intrinsic_high: float
    margin_of_safety_pct: float  # vs intrinsic_base; positive = undervalued
    methodology_notes: str
    confidence: Literal["high", "medium", "low"]
    # Deterministic DCF / multiples breakdown
    dcf_value: Optional[float] = None
    multiples_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None


# ----- Agent I/O: Forward Scenarios (price paths) -----

class PriceScenario(BaseModel):
    name: str                       # e.g. "base", "bull", "bear", "black_swan"
    probability: float = Field(ge=0, le=1)
    target_price_low: float          # lower bound of expected price range at horizon
    target_price_base: float         # central estimate
    target_price_high: float
    return_pct_base: float           # % from current price to target_price_base
    drawdown_pct_estimated: float    # max expected drawdown during the path (negative number)
    key_drivers: list[str] = Field(default_factory=list)
    rationale: str


class ForwardScenarios(BaseModel):
    symbol: str
    current_price: float
    currency: str
    horizon_days: int
    scenarios: list[PriceScenario] = Field(default_factory=list)
    probability_weighted_target: float = 0.0  # = sum(prob * target_base)
    expected_return_pct: float = 0.0          # = (weighted_target / current_price - 1) * 100
    summary: str = ""


# ----- Agent I/O: Information Retrieval / Forward Catalysts -----

class CatalystImpact(BaseModel):
    event: str
    expected_date: Optional[date] = None
    direction: Literal["positive", "negative", "uncertain"]
    expected_magnitude_pct: Optional[float] = None  # rough % move (+/-) on resolution
    confidence: Literal["high", "medium", "low"]
    rationale: str


class ForwardCatalysts(BaseModel):
    symbol: str
    horizon_days: int
    key_catalysts: list[CatalystImpact] = Field(default_factory=list)
    macro_overlay: list[str] = Field(default_factory=list)
    sentiment_summary: str = ""
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)  # -1..+1


# ----- Agent I/O: Portfolio Fit -----

class CorrelationCluster(BaseModel):
    symbols: list[str]
    avg_pairwise_correlation: float
    common_risk_source: str
    concentration_pct_of_book: float  # 0-100
    severity: Literal["low", "medium", "high"]


class PortfolioFitReport(BaseModel):
    total_positions: int
    total_book_value_usd: float
    clusters: list[CorrelationCluster] = Field(default_factory=list)
    concentration_warnings: list[str] = Field(default_factory=list)
    diversification_recommendations: list[str] = Field(default_factory=list)
    diversification_score: float = Field(ge=0, le=10)
    summary: str


# ----- Agent I/O: Competence Gate -----

class CompetenceVerdict(BaseModel):
    symbol: str
    verdict: Literal["in_circle", "borderline", "out_of_circle"]
    reasoning: str
    matched_categories: list[str] = Field(default_factory=list)


# ----- Agent I/O: Sentiment / Contrarian -----

class ContrarianAssessment(BaseModel):
    crowd_position: Literal["euphoric", "bullish", "neutral", "bearish", "despondent"]
    contrarian_signal: Literal["strong_buy", "buy", "neutral", "pass", "strong_pass"]
    reasoning: str
    data_quality: Literal["high", "medium", "low"]
    key_observations: list[str] = Field(default_factory=list)


# ----- Agent I/O: Devil's Advocate -----

class DevilFinding(BaseModel):
    category: Literal[
        "confirmation_bias",
        "anchoring",
        "overconfidence",
        "narrative_fallacy",
        "circle_of_competence",
        "moat_erosion",
        "macro_blind_spot",
        "behavioral_fomo",
        "data_quality",
        "valuation_optimism",
        "risk_underestimation",
        "technical_fundamental_contradiction",   # Phase 6
        "other",
    ]
    severity: Literal["info", "concern", "veto"]
    finding: str
    evidence: str
    recommendation: str


class DevilAdvocateReview(BaseModel):
    overall_verdict: Literal["pass", "pass_with_concerns", "veto"]
    summary: str
    findings: list[DevilFinding] = Field(default_factory=list)
    counter_thesis: str
    # When verdict=veto, what specifically caused it (free text)
    veto_reason: Optional[str] = None


# ----- Agent I/O: Hedging -----

class HedgeCandidate(BaseModel):
    instrument: str           # e.g. "NQ=F", "HSI=F", "IGV" (ETF short)
    instrument_kind: Literal["future", "etf_short", "option", "index"]
    correlation_90d: Optional[float] = None
    rationale: str
    suggested_notional_usd: Optional[float] = None
    contract_specs: Optional[str] = None  # human-readable: contract size, margin, expiry


class HedgePlan(BaseModel):
    symbol_being_hedged: str
    position_value_usd: float
    candidates: list[HedgeCandidate] = Field(default_factory=list)
    recommended_index: int = 0  # which candidate is the recommended one
    rationale: str
    notes: list[str] = Field(default_factory=list)


# ----- Agent I/O: Risk Analyzer -----

class Scenario(BaseModel):
    name: str
    probability: float = Field(ge=0, le=1)
    expected_return_pct: float
    expected_drawdown_pct: float
    rationale: str


class RiskAssessment(BaseModel):
    scenarios: list[Scenario]
    drawdown_probabilities: dict[str, float]  # "10","15","20","25" -> probability (post-clamp)
    realized_vol_annualized_pct: float
    key_macro_signals: list[str]
    horizon_days: int
    persistence_days_observed: int = 0
    # Non-LLM historical-bootstrap sanity baseline. When present, the LLM's
    # drawdown_probabilities have been clamped toward this prior if they diverged
    # by more than BOOTSTRAP_CLAMP_MAX_DIVERGENCE. Surfaced in the dashboard.
    bootstrap_drawdown_probabilities: Optional[dict[str, float]] = None
    bootstrap_clamp_notes: list[str] = Field(default_factory=list)


# ----- Agent I/O: Tactical Exit / Order Generator -----

class TacticalAction(BaseModel):
    level: Optional[int] = None
    label: Optional[str] = None
    action: Literal[
        "no_action",
        "monitor_only",
        "trim",
        "defensive_reduction",
        "full_exit",
    ]
    trim_pct_of_position: Optional[float] = None
    rebuy_band_low: Optional[float] = None
    rebuy_band_high: Optional[float] = None
    hedge_recommended: bool = False
    rationale: str
    tax_notes: list[str] = Field(default_factory=list)


class OrderSpec(BaseModel):
    side: Literal["BUY", "SELL"]
    quantity: float
    symbol: str
    order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Literal["DAY", "GTC"] = "GTC"
    rationale: str
    conditional: bool = False  # True for rebuy orders waiting for a price level


class HeldDecision(BaseModel):
    tactical: TacticalAction
    immediate_orders: list[OrderSpec] = Field(default_factory=list)
    rebuy_orders: list[OrderSpec] = Field(default_factory=list)


class NotHeldDecision(BaseModel):
    recommendation: Literal["BUY_NOW", "WAIT_FOR_PRICE", "PASS"]
    entry_orders: list[OrderSpec] = Field(default_factory=list)
    rationale: str


# ----- Phase 6: Technical Division -----

class StructurePivot(BaseModel):
    date: str                                # ISO date
    price: float
    kind: Literal["HH", "HL", "LH", "LL"]


class StructureAssessment(BaseModel):
    trend: Literal["strong_uptrend", "uptrend", "range", "downtrend", "strong_downtrend"]
    stage: Literal["early", "middle", "late", "exhausted"]
    confidence: float = Field(ge=0, le=1)
    last_swing_high: float
    last_swing_low: float
    pivots: list[StructurePivot] = Field(default_factory=list)
    structure_summary: str


class VolumeAssessment(BaseModel):
    institutional_flow: Literal["accumulation", "distribution", "neutral"]
    obv_trend: Literal["rising", "flat", "falling"]
    volume_expansion_pct: float                          # 20d avg vs 50d avg (%)
    up_down_volume_ratio: float                          # >1.0 = bullish bias
    last_earnings_volume_spike_x: Optional[float] = None # multiple of normal vol on last earnings day
    confidence: float = Field(ge=0, le=1)
    signals: list[str] = Field(default_factory=list)


class CostBasisLevel(BaseModel):
    price_low: float
    price_high: float
    volume_pct_of_window: float                          # 0-100
    position_vs_current: Literal["above", "below", "at"]
    role: Literal["support", "resistance", "neutral"]


class CostBasisMap(BaseModel):
    lookback_days: int
    hvn_levels: list[CostBasisLevel] = Field(default_factory=list)
    trapped_supply_pct: float                            # % of vol sitting above current at >+10%
    accumulation_pct: float                              # % of vol sitting below current within -10%
    summary: str


class RelativeStrengthAssessment(BaseModel):
    vs_sector_etf_90d: Optional[float] = None            # ratio; >1.0 = outperforming sector
    vs_sector_etf_365d: Optional[float] = None
    vs_index_90d: Optional[float] = None
    vs_index_365d: Optional[float] = None
    rs_rank_in_universe: Optional[int] = None            # 1-100 within active watchlist
    signal: Literal["strong_leader", "leader", "neutral", "laggard", "weak_laggard"]
    benchmark_sector_etf: Optional[str] = None
    benchmark_index: Optional[str] = None


class PriceMapZone(BaseModel):
    price_low: float
    price_high: float
    label: Literal[
        "aggressive_accumulation",
        "accumulation",
        "watch",
        "hold",
        "trim",
        "distribution",
        "high_risk",
    ]
    rationale: str


class PriceMap(BaseModel):
    zones: list[PriceMapZone] = Field(default_factory=list)
    current_zone_index: int = 0
    key_support: Optional[float] = None
    key_resistance: Optional[float] = None
    summary: str


class TechnicalAssessment(BaseModel):
    structure: StructureAssessment
    volume: VolumeAssessment
    cost_basis: CostBasisMap
    relative_strength: RelativeStrengthAssessment
    price_map: PriceMap
    composite_signal: Literal[
        "strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"
    ]
    composite_rationale: str


# ----- Phase 6: Technical Config -----

class StructureCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pivot_window_bars: int = 5
    pivots_to_evaluate: int = 8


class VolumeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    obv_lookback_days: int = 60
    expansion_window_short: int = 20
    expansion_window_long: int = 50


class CostBasisCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookback_days: int = 365
    hvn_min_volume_pct: float = 5.0
    bucket_pct_width: float = 1.0


class RelativeStrengthCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    windows_days: list[int] = Field(default_factory=lambda: [90, 365])
    benchmarks: dict = Field(default_factory=dict)       # {market: {market_index, sector_etfs}}


class PriceMapCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = True


class IntegrationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tactical_persistence_boost: bool = True
    order_anchor_to_price_map: bool = True
    max_band_anchor_distance_pct: float = 30.0           # support must be within this % below current
    min_band_anchor_distance_pct: float = 5.0            # ...and at least this far below


class TechnicalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structure: StructureCfg = Field(default_factory=StructureCfg)
    volume: VolumeCfg = Field(default_factory=VolumeCfg)
    cost_basis: CostBasisCfg = Field(default_factory=CostBasisCfg)
    relative_strength: RelativeStrengthCfg = Field(default_factory=RelativeStrengthCfg)
    price_map: PriceMapCfg = Field(default_factory=PriceMapCfg)
    integration: IntegrationCfg = Field(default_factory=IntegrationCfg)


# ----- Financial Report (last-N-periods comparison + LLM deep resolution) -----

class FinancialLine(BaseModel):
    label: str                            # human-readable, e.g. "Revenue"
    values: list[Optional[float]]         # most-recent period first; same length as periods


class FinancialPeriodSet(BaseModel):
    cadence: Literal["annual", "quarterly"]
    periods: list[str] = Field(default_factory=list)        # ISO date strings, most-recent first
    income: list[FinancialLine] = Field(default_factory=list)
    balance: list[FinancialLine] = Field(default_factory=list)
    cashflow: list[FinancialLine] = Field(default_factory=list)


class FinancialDeepResolution(BaseModel):
    revenue_trend: str
    margin_trend: str
    balance_sheet_trend: str
    cash_flow_quality: str
    capital_allocation_observed: str
    key_positives: list[str] = Field(default_factory=list)
    key_red_flags: list[str] = Field(default_factory=list)
    investment_implication: str           # the "huge inference on the investment"
    summary: str


class FinancialReport(BaseModel):
    currency: str
    annual: Optional[FinancialPeriodSet] = None
    quarterly: Optional[FinancialPeriodSet] = None
    deep_resolution: Optional[FinancialDeepResolution] = None
    fetch_notes: list[str] = Field(default_factory=list)    # diagnostics (e.g. "annual: only 1 period available")


class AnalysisResult(BaseModel):
    symbol: str
    market: Literal["US", "HK"]
    timestamp_utc: datetime
    current_price: float
    currency: str
    position: Optional[Holding] = None
    fundamental: FundamentalAssessment
    valuation: ValuationResult
    risk: RiskAssessment
    if_held: HeldDecision
    if_not_held: NotHeldDecision
    forward_catalysts: Optional[ForwardCatalysts] = None
    forward_scenarios: Optional[ForwardScenarios] = None
    hedge_plan: Optional[HedgePlan] = None
    devil_advocate: Optional[DevilAdvocateReview] = None
    competence: Optional[CompetenceVerdict] = None
    contrarian: Optional[ContrarianAssessment] = None
    technical: Optional[TechnicalAssessment] = None
    financial_report: Optional[FinancialReport] = None
    errors: list[str] = Field(default_factory=list)
