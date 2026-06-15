"""Unit tests for the macro snapshot's sanity bounds (Phase 0.1).

The point of the bounds is that a *wrong-but-authoritative* number (e.g. a source
layout change makes a regex grab a garbage value) is worse than N/A. Each fragile
fetcher is exercised with a mocked page that yields an out-of-range value and must
degrade to None; an in-range page must pass through unchanged.

No network: ``requests.get`` and ``yfinance.Ticker`` are monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import macro_snapshot as ms  # noqa: E402

pytestmark = pytest.mark.unit


class FakeResp:
    def __init__(self, text="", json_data=None):
        self.text = text
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _patch_get(monkeypatch, resp):
    monkeypatch.setattr(ms.requests, "get", lambda *a, **k: resp)


# ------------------------------- _bounded ------------------------------------

def test_bounded_passes_in_range():
    assert ms._bounded(38.5, 5, 70, "x") == 38.5


def test_bounded_drops_out_of_range():
    assert ms._bounded(999.0, 5, 70, "x") is None
    assert ms._bounded(-5.0, 0, 100, "x") is None


def test_bounded_passes_none_through():
    assert ms._bounded(None, 5, 70, "x") is None


def test_bounded_inclusive_edges():
    assert ms._bounded(5, 5, 70, "x") == 5
    assert ms._bounded(70, 5, 70, "x") == 70


# ------------------------------- Fetchers ------------------------------------

def test_cape_garbage_is_na(monkeypatch):
    _patch_get(monkeypatch, FakeResp(text='<div id="current">999</div>'))
    assert ms.fetch_shiller_cape() == (None, None)


def test_cape_valid_passes(monkeypatch):
    _patch_get(monkeypatch, FakeResp(text='<div id="current">38.5</div>'))
    assert ms.fetch_shiller_cape() == (38.5, None)


def test_buffett_garbage_is_na(monkeypatch):
    _patch_get(monkeypatch, FakeResp(text="we calculate the Buffett Indicator as 999%"))
    assert ms.fetch_buffett_indicator() == (None, None)


def test_buffett_valid_passes(monkeypatch):
    _patch_get(monkeypatch, FakeResp(text="we calculate the Buffett Indicator as 219%"))
    assert ms.fetch_buffett_indicator() == (219.0, None)


def test_t10y2y_garbage_is_na(monkeypatch):
    csv = "Date,2 Yr,10 Yr\n06/13/2026,0.00,99.00\n"
    _patch_get(monkeypatch, FakeResp(text=csv))
    val, _ = ms.fetch_treasury_t10y2y()
    assert val is None


def test_t10y2y_valid_passes(monkeypatch):
    csv = "Date,2 Yr,10 Yr\n06/13/2026,4.00,4.50\n"
    _patch_get(monkeypatch, FakeResp(text=csv))
    val, date = ms.fetch_treasury_t10y2y()
    assert round(val, 2) == 0.50
    assert date == "06/13/2026"


def test_fear_greed_garbage_is_na(monkeypatch):
    _patch_get(monkeypatch, FakeResp(json_data={"fear_and_greed": {"score": 999, "rating": "x"}}))
    assert ms.fetch_cnn_fear_greed() == (None, "x")


def test_fear_greed_valid_passes(monkeypatch):
    _patch_get(monkeypatch, FakeResp(json_data={"fear_and_greed": {"score": 50, "rating": "Neutral"}}))
    assert ms.fetch_cnn_fear_greed() == (50.0, "Neutral")


def test_pmi_garbage_is_na(monkeypatch):
    _patch_get(monkeypatch, FakeResp(text="ISM PMI is 99.9 this month"))
    assert ms.fetch_ism_pmi() == (None, 0)


def test_pmi_valid_passes(monkeypatch):
    _patch_get(monkeypatch, FakeResp(text="ISM PMI is 48.5 this month"))
    assert ms.fetch_ism_pmi() == (48.5, 0)


def test_margin_debt_garbage_is_na(monkeypatch):
    html = (
        "<table>"
        "<tr><th>Month</th><th>Debt</th></tr>"
        "<tr><td>Apr-26</td><td>1,000,000</td></tr>"
        "<tr><td>Apr-25</td><td>1</td></tr>"
        "</table>"
    )
    _patch_get(monkeypatch, FakeResp(text=html))
    assert ms.fetch_margin_debt_yoy() is None


def test_margin_debt_valid_passes(monkeypatch):
    html = (
        "<table>"
        "<tr><th>Month</th><th>Debt</th></tr>"
        "<tr><td>Apr-26</td><td>110</td></tr>"
        "<tr><td>Apr-25</td><td>100</td></tr>"
        "</table>"
    )
    _patch_get(monkeypatch, FakeResp(text=html))
    assert round(ms.fetch_margin_debt_yoy(), 2) == 10.0


class _FakeTicker:
    def __init__(self, close):
        self._close = close

    def history(self, *a, **k):
        idx = pd.to_datetime(["2026-06-12", "2026-06-13"])
        return pd.DataFrame({"Close": [self._close - 1.0, self._close]}, index=idx)


def test_vix_garbage_is_na(monkeypatch):
    monkeypatch.setattr(ms.yf, "Ticker", lambda sym: _FakeTicker(999.0))
    val, _ = ms.fetch_vix()
    assert val is None


def test_vix_valid_passes(monkeypatch):
    monkeypatch.setattr(ms.yf, "Ticker", lambda sym: _FakeTicker(18.0))
    val, date = ms.fetch_vix()
    assert val == 18.0
    assert date == "2026-06-13"


# --------------------------- manual override stamp ---------------------------

def _stub_network_fetchers(monkeypatch):
    """Force all network fetchers to N/A so build_snapshot runs offline."""
    monkeypatch.setattr(ms, "fetch_shiller_cape", lambda: (None, None))
    monkeypatch.setattr(ms, "fetch_buffett_indicator", lambda: (None, None))
    monkeypatch.setattr(ms, "fetch_treasury_t10y2y", lambda: (None, None))
    monkeypatch.setattr(ms, "fetch_cnn_fear_greed", lambda: (None, None))
    monkeypatch.setattr(ms, "fetch_vix", lambda: (None, None))
    monkeypatch.setattr(ms, "fetch_fred_api", lambda series, key: (None, None))
    monkeypatch.setattr(ms, "fetch_sp500_drawdown", lambda: (None, None, None))


def test_manual_overrides_are_dated(monkeypatch):
    _stub_network_fetchers(monkeypatch)
    snap = ms.build_snapshot(manual_overrides={
        "ism_pmi": 48.0,
        "margin_debt_yoy_pct": 12.0,
        "as_of": "2026-06-01",
    })
    by_name = {ind["name"]: ind for ind in snap["indicators"]}
    assert "manual (as of 2026-06-01)" in by_name["Margin Debt YoY"]["display"]
    assert "manual (as of 2026-06-01)" in by_name["PMI + Sahm Rule"]["display"]


def test_no_manual_no_stamp(monkeypatch):
    _stub_network_fetchers(monkeypatch)
    snap = ms.build_snapshot()
    for ind in snap["indicators"]:
        assert "manual" not in ind["display"]
