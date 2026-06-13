"""Tests for the glossary scanner — only tickers/metrics that actually appear
in the body should make it into Section 14."""
from __future__ import annotations

from rotation.glossary import (
    METRIC_GLOSSARY, TICKER_GLOSSARY, find_metrics_in, find_tickers_in,
    render_glossary,
)


def test_find_tickers_picks_up_real_symbols():
    body = "Top weakening: SPY -2%, QQQ -3%, BTC-USD -4%."
    out = find_tickers_in(body)
    assert "SPY" in out
    assert "QQQ" in out
    assert "BTC-USD" in out


def test_find_tickers_ignores_blacklist_words():
    body = "VS the prior day, the OK status shows AS expected."
    out = find_tickers_in(body)
    # VS, OK, AS are all-caps words but blacklisted
    assert out == set()


def test_find_tickers_ignores_unknown_caps():
    body = "Something something XYZ went up."  # XYZ not in TICKER_GLOSSARY
    out = find_tickers_in(body)
    assert "XYZ" not in out


def test_find_tickers_caret_move_index():
    body = "^MOVE z-score is elevated."
    out = find_tickers_in(body)
    assert "^MOVE" in out


def test_find_metrics_basic():
    body = "The r_d was -2%, r_w -3%, vol_30 spiked."
    out = find_metrics_in(body)
    assert "r_d" in out
    assert "r_w" in out
    assert "vol_30" in out


def test_find_metrics_does_not_match_inside_other_tokens():
    # `r_d` appears only as a substring of another identifier; the word-boundary
    # regex must NOT match it.
    body = "The variable foo_r_d_bar contains it as substring only."
    out = find_metrics_in(body)
    assert "r_d" not in out


def test_render_glossary_only_contains_present_tickers():
    body = "SPY -2%, TLT +1%, BTC-USD -5%."
    rendered = render_glossary(body, section_number=14)
    assert "## Section 14 — Glossary" in rendered
    assert "SPY" in rendered
    assert "TLT" in rendered
    assert "BTC-USD" in rendered
    # ETFs not in the body should NOT appear
    assert "FXF" not in rendered
    assert "CPER" not in rendered


def test_render_glossary_empty_body_produces_minimal_output():
    body = ""
    rendered = render_glossary(body, section_number=14)
    assert "## Section 14 — Glossary" in rendered
    # No tickers / metrics tables when nothing matches
    assert "Tickers referenced" not in rendered


def test_metric_glossary_keys_have_definitions():
    """No empty values."""
    for k, v in METRIC_GLOSSARY.items():
        assert v and len(v) > 5, f"metric {k} lacks definition"


def test_ticker_glossary_covers_full_universe():
    """Every ticker in the standard universe should have a glossary entry so the
    glossary scanner never silently omits a ticker."""
    must_have = {
        "SPY","QQQ","IWM","EZU","EEM","EWJ","GLD","SLV","TLT","UUP","FXF",
        "CPER","USO","SMH","XLI","IYT","BTC-USD","ETH-USD","IEF","HYG","LQD",
        "XLK","XLF","XLE","XLV","XLY","XLP","XLU","XLB","XLRE","XLC","^MOVE",
    }
    missing = must_have - set(TICKER_GLOSSARY.keys())
    assert not missing, f"glossary missing: {missing}"
