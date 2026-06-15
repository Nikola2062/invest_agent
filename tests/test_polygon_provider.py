"""Polygon.io provider tests — all HTTP mocked.

The two contracts that matter:

  1. **CSV output shape matches yfinance** so agents (whose prompts
     reference the yfinance CSV header lines) don't break when the
     vendor is swapped.
  2. **News filter is server-side by ``published_utc``** so PIT
     correctness doesn't depend on client-side filtering of full history.

Plus mechanical safety: missing API key → clear error, 429 → retry.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows import polygon

pytestmark = pytest.mark.unit


# --- auth -----------------------------------------------------------------


def test_missing_api_key_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(polygon, "get_config", lambda: {"polygon_api_key": ""})
    with pytest.raises(RuntimeError, match="POLYGON_API_KEY"):
        polygon._api_key()


def test_env_key_overrides_config(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "from-env")
    monkeypatch.setattr(polygon, "get_config", lambda: {"polygon_api_key": "from-config"})
    assert polygon._api_key() == "from-env"


def test_config_key_used_when_env_unset(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(polygon, "get_config", lambda: {"polygon_api_key": "configured"})
    assert polygon._api_key() == "configured"


# --- _get_json + 429 backoff ---------------------------------------------


def _mock_resp(status_code: int, json_payload=None):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_payload or {}
    m.raise_for_status = MagicMock()
    if status_code >= 400 and status_code != 429:
        m.raise_for_status.side_effect = RuntimeError(f"HTTP {status_code}")
    return m


def test_get_json_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {})

    responses = [_mock_resp(429), _mock_resp(429), _mock_resp(200, {"results": [1]})]

    # Skip the actual sleep so the test is fast.
    monkeypatch.setattr(polygon.time, "sleep", lambda _: None)

    with patch.object(polygon.requests, "get", side_effect=responses) as get_mock:
        out = polygon._get_json("/x")
    assert out == {"results": [1]}
    assert get_mock.call_count == 3


def test_get_json_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {})
    monkeypatch.setattr(polygon.time, "sleep", lambda _: None)

    # Four 429s → exceeds _MAX_429_RETRIES=3.
    responses = [_mock_resp(429)] * (polygon._MAX_429_RETRIES + 2)
    with patch.object(polygon.requests, "get", side_effect=responses):
        with pytest.raises(RuntimeError, match="429 after"):
            polygon._get_json("/x")


def test_get_json_appends_api_key(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "secret-key")
    monkeypatch.setattr(polygon, "get_config", lambda: {})
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _mock_resp(200, {"results": []})

    with patch.object(polygon.requests, "get", side_effect=fake_get):
        polygon._get_json("/x", params={"foo": "bar"})

    assert captured["params"]["apiKey"] == "secret-key"
    assert captured["params"]["foo"] == "bar"


# --- get_stock_data ------------------------------------------------------


def _bars(*rows):
    """Build a Polygon aggregates payload from (date_str, o, h, l, c, v) tuples."""
    import pandas as pd
    return {"results": [
        {
            "t": int(pd.Timestamp(d).timestamp() * 1000),
            "o": o, "h": h, "l": l, "c": c, "v": v, "vw": (h + l) / 2,
        }
        for d, o, h, l, c, v in rows
    ]}


def test_stock_data_csv_shape_matches_yfinance(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {})
    payload = _bars(
        ("2024-01-02", 100.0, 102.0, 99.0, 101.0, 5_000_000),
        ("2024-01-03", 101.0, 103.0, 100.0, 102.5, 4_800_000),
    )
    with patch.object(polygon, "_get_json", return_value=payload):
        out = polygon.get_stock_data("AAPL", "2024-01-01", "2024-01-05")

    # Match yfinance shape: comment header, columns include OHLCV.
    assert out.startswith("# Stock data for AAPL")
    assert "Source: Polygon.io" in out
    assert "Date,Open,High,Low,Close,Volume" in out
    assert "2024-01-02," in out
    assert "101.0" in out  # close on first row


def test_stock_data_empty_returns_clean_message(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {})
    with patch.object(polygon, "_get_json", return_value={"results": []}):
        out = polygon.get_stock_data("BAD", "2024-01-01", "2024-01-05")
    assert "No data found" in out


def test_stock_data_validates_date_format(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {})
    with pytest.raises(ValueError):
        polygon.get_stock_data("AAPL", "01/01/2024", "2024-01-05")


def test_stock_data_passes_adjusted_param(monkeypatch):
    """Splits/dividends adjustment must always be on — otherwise the equity
    curve has fake gaps every dividend day."""
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {})
    captured = {}

    def fake_get_json(path, params=None):
        captured["params"] = params
        return _bars(("2024-01-02", 100, 100, 100, 100, 1))

    with patch.object(polygon, "_get_json", side_effect=fake_get_json):
        polygon.get_stock_data("AAPL", "2024-01-01", "2024-01-05")

    assert captured["params"]["adjusted"] == "true"


# --- get_news ------------------------------------------------------------


def _news_payload(*titles_dates):
    return {"results": [
        {
            "title": t, "published_utc": d,
            "publisher": {"name": "TestSource"},
            "description": f"Body of {t}",
            "article_url": f"https://example.com/{t.replace(' ', '_')}",
        }
        for t, d in titles_dates
    ]}


def test_get_news_filters_via_server_side_published_utc(monkeypatch):
    """PIT correctness: the published_utc filter must be sent to the API,
    not applied client-side after pulling everything."""
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {"news_article_limit": 20})
    captured = {}

    def fake_get_json(path, params=None):
        captured["params"] = params
        return _news_payload(("Article A", "2024-01-03T12:00:00Z"))

    with patch.object(polygon, "_get_json", side_effect=fake_get_json):
        polygon.get_news("AAPL", "2024-01-01", "2024-01-05")

    assert captured["params"]["published_utc.gte"] == "2024-01-01"
    assert captured["params"]["published_utc.lte"] == "2024-01-05"
    assert captured["params"]["ticker"] == "AAPL"


def test_get_news_renders_markdown_for_each_article(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {"news_article_limit": 20})

    payload = _news_payload(
        ("First headline", "2024-01-03T12:00:00Z"),
        ("Second headline", "2024-01-04T08:00:00Z"),
    )
    with patch.object(polygon, "_get_json", return_value=payload):
        out = polygon.get_news("AAPL", "2024-01-01", "2024-01-05")

    assert "AAPL News, from 2024-01-01 to 2024-01-05" in out
    assert "### First headline (source: TestSource)" in out
    assert "### Second headline (source: TestSource)" in out
    assert "Body of First headline" in out
    assert "Link: https://example.com/First_headline" in out


def test_get_news_empty_payload_returns_clean_message(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {"news_article_limit": 20})
    with patch.object(polygon, "_get_json", return_value={"results": []}):
        out = polygon.get_news("AAPL", "2024-01-01", "2024-01-05")
    assert "No news found for AAPL" in out


def test_get_news_paginates_via_next_url(monkeypatch):
    """When the first page has next_url, the client must follow it."""
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {"news_article_limit": 4})

    page1 = {
        "results": [
            {"title": f"P1_{i}", "publisher": {"name": "X"}, "description": "",
             "article_url": "", "published_utc": "2024-01-01"} for i in range(2)
        ],
        "next_url": "https://api.polygon.io/v2/reference/news?cursor=abc",
    }
    page2 = {
        "results": [
            {"title": f"P2_{i}", "publisher": {"name": "X"}, "description": "",
             "article_url": "", "published_utc": "2024-01-01"} for i in range(2)
        ],
    }

    call_count = {"n": 0}

    def fake_initial(path, params=None):
        call_count["n"] += 1
        return page1

    def fake_followup(url, params=None, timeout=None):
        call_count["n"] += 1
        return _mock_resp(200, page2)

    with patch.object(polygon, "_get_json", side_effect=fake_initial), \
         patch.object(polygon.requests, "get", side_effect=fake_followup):
        out = polygon.get_news("AAPL", "2024-01-01", "2024-01-05")

    assert "P1_0" in out and "P2_0" in out
    assert call_count["n"] == 2


# --- get_global_news -----------------------------------------------------


def test_global_news_uses_lookback_window(monkeypatch):
    """curr_date - look_back_days defines the server-side published_utc.gte."""
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {
        "global_news_lookback_days": 7,
        "global_news_article_limit": 10,
    })
    captured = {}

    def fake_get_json(path, params=None):
        captured["params"] = params
        return _news_payload(("Macro headline", "2024-03-10T12:00:00Z"))

    with patch.object(polygon, "_get_json", side_effect=fake_get_json):
        polygon.get_global_news("2024-03-15")

    assert captured["params"]["published_utc.lte"] == "2024-03-15"
    assert captured["params"]["published_utc.gte"] == "2024-03-08"


def test_global_news_inherits_config_defaults(monkeypatch):
    """Calling without explicit limit/lookback uses the config defaults."""
    monkeypatch.setenv("POLYGON_API_KEY", "k")
    monkeypatch.setattr(polygon, "get_config", lambda: {
        "global_news_lookback_days": 14,
        "global_news_article_limit": 5,
    })
    captured = {}

    def fake_get_json(path, params=None):
        captured["params"] = params
        return _news_payload(("h", "2024-03-10"))

    with patch.object(polygon, "_get_json", side_effect=fake_get_json):
        polygon.get_global_news("2024-03-15")
    # 14-day lookback → gte = 2024-03-01.
    assert captured["params"]["published_utc.gte"] == "2024-03-01"


# --- provider registry --------------------------------------------------


def test_polygon_registered_as_pit_safe():
    from tradingagents.dataflows.providers import pit_safe_methods
    safe = pit_safe_methods("polygon")
    assert "get_stock_data" in safe
    assert "get_news" in safe
    assert "get_global_news" in safe


def test_polygon_in_vendor_methods():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "polygon" in VENDOR_METHODS["get_stock_data"]
    assert "polygon" in VENDOR_METHODS["get_news"]
    assert "polygon" in VENDOR_METHODS["get_global_news"]
