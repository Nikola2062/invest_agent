"""D1: FRED panel wiring into the inflation/liquidity/recession signals.

These tests synthesize the close + FRED panels, then confirm:
  - inflation_score picks up T10YIE when present
  - liquidity_score computes net_fed_liquidity from WALCL/WTREGEN/RRP and
    half-weights the standalone RRP contributor to avoid double-counting
  - recession_concern prefers FRED T10Y2Y / HY OAS over the ETF proxies
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rotation import signals as S


def _make_close(n=300, syms=("UUP", "SPY", "BTC-USD", "^MOVE",
                            "USO", "GLD", "SLV", "TLT", "IEF",
                            "HYG", "LQD", "XLY", "XLP", "IYT")):
    rng = pd.date_range("2025-01-01", periods=n, freq="B")
    rs = np.random.RandomState(42)
    data = {s: 100.0 + np.cumsum(rs.normal(0, 1.0, n)) for s in syms}
    return pd.DataFrame(data, index=rng)


def _make_fred(close: pd.DataFrame, with_net_liq=True, with_t10y2y=True,
               with_hy_oas=True, with_t10yie=True):
    rng = close.index
    rs = np.random.RandomState(7)
    cols = {}
    if with_t10yie:
        cols["T10YIE"] = 2.0 + 0.001 * np.cumsum(rs.normal(0, 1, len(rng)))
    if with_net_liq:
        cols["WALCL"]   = 6_700_000.0 + 1000.0 * np.cumsum(rs.normal(0, 1, len(rng)))
        cols["WTREGEN"] =   800_000.0 + 1000.0 * np.cumsum(rs.normal(0, 1, len(rng)))
    cols["RRPONTSYD"]   =       100.0 +    1.0 * np.cumsum(rs.normal(0, 1, len(rng)))
    if with_t10y2y:
        cols["T10Y2Y"]  = 0.4 + 0.01 * np.cumsum(rs.normal(0, 1, len(rng)))
    if with_hy_oas:
        cols["BAMLH0A0HYM2"] = 3.0 + 0.02 * np.cumsum(rs.normal(0, 1, len(rng)))
    return pd.DataFrame(cols, index=rng)


def test_inflation_uses_t10yie_when_fred_present():
    close = _make_close()
    fred = _make_fred(close)
    asof = close.index[-1]

    no_fred = S.inflation_score(close, asof, fred=None)
    with_fred = S.inflation_score(close, asof, fred=fred)

    assert "T10YIE_chg21" not in no_fred["components"]
    assert "T10YIE_chg21" in with_fred["components"]


def test_inflation_works_without_fred():
    close = _make_close()
    asof = close.index[-1]
    out = S.inflation_score(close, asof, fred=None)
    assert out["score"] is not None  # commodity basket alone is enough


def test_liquidity_adds_net_liq_and_half_weights_rrp():
    close = _make_close()
    asof = close.index[-1]

    # Same panel, then strip WALCL/WTREGEN to make the "no net-liq" variant —
    # keeping the RRPONTSYD column identical across both runs so we can test the
    # half-weighting precisely.
    fred_full = _make_fred(close, with_net_liq=True)
    fred_no_net = fred_full.drop(columns=["WALCL", "WTREGEN"])

    out_no = S.liquidity_score(close, asof, fred=fred_no_net)
    out_yes = S.liquidity_score(close, asof, fred=fred_full)

    # net liquidity present in the full panel
    assert "net_fed_liquidity_chg21" in out_yes["components"]
    assert "net_fed_liquidity_chg21" not in out_no["components"]

    # RRP contrib should be half magnitude when net-liq is also present
    rrp_no = out_no["components"]["rrp_inv"]
    rrp_yes = out_yes["components"]["rrp_inv"]
    assert rrp_yes == rrp_no * 0.5


def test_recession_prefers_fred_curve_over_etf_proxy():
    close = _make_close()
    asof = close.index[-1]

    fred_no_curve = _make_fred(close, with_t10y2y=False)
    fred_with = _make_fred(close, with_t10y2y=True)

    out_no = S.recession_concern(close, asof, fred=fred_no_curve)
    out_yes = S.recession_concern(close, asof, fred=fred_with)

    # FRED present: curve_t10y2y_inv used, ETF proxy NOT used
    assert "curve_t10y2y_inv" in out_yes["components"]
    assert "curve_tlt_minus_ief" not in out_yes["components"]

    # FRED missing: fall back to ETF proxy
    assert "curve_t10y2y_inv" not in out_no["components"]
    assert "curve_tlt_minus_ief" in out_no["components"]


def test_recession_prefers_fred_credit_over_etf_proxy():
    close = _make_close()
    asof = close.index[-1]

    fred_no_oas = _make_fred(close, with_hy_oas=False)
    fred_with = _make_fred(close, with_hy_oas=True)

    out_no = S.recession_concern(close, asof, fred=fred_no_oas)
    out_yes = S.recession_concern(close, asof, fred=fred_with)

    assert "credit_hy_oas" in out_yes["components"]
    assert "credit_hyg_lqd" not in out_yes["components"]

    assert "credit_hy_oas" not in out_no["components"]
    assert "credit_hyg_lqd" in out_no["components"]


def test_compute_all_routes_fred_to_all_consumers():
    close = _make_close()
    volume = close * 0.0 + 1_000_000.0
    fred = _make_fred(close)
    asof = close.index[-1]

    out = S.compute_all(close, volume, asof, fred=fred)

    # All three FRED consumers should have engaged the FRED panel
    assert "T10YIE_chg21" in out["inflation"]["components"]
    assert "net_fed_liquidity_chg21" in out["liquidity"]["components"]
    assert "curve_t10y2y_inv" in out["recession"]["components"]
