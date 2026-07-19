"""Pull financial data from yfinance and store in PostgreSQL."""

from __future__ import annotations

import logging
import time
import pandas as pd
from datetime import datetime, date
from typing import Any

import yfinance as yf
from sqlalchemy import text, create_engine

from src.config import ALL_TICKERS, YFINANCE_SLEEP_SECS, HISTORY_PERIOD, FINANCIAL_QUARTERS, get_db_url

log = logging.getLogger(__name__)


def _engine():
    return create_engine(get_db_url())


def _get_ticker_id(engine, ticker: str) -> int | None:
    with engine.begin() as conn:
        result = conn.execute(text("SELECT id FROM tickers WHERE symbol = :ticker"), {"ticker": ticker})
        row = result.fetchone()
        return row[0] if row else None


def _fmt(val: float | None, is_dollar: bool = False) -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1_000_000_000_000:
        return f"${val/1_000_000_000_000:.2f}T"
    if abs(val) >= 1_000_000_000:
        return f"${val/1_000_000_000:.2f}B"
    return f"${val:,.0f}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}%"


def _upsert_financial_doc(conn, ticker_id: int, ticker_symbol: str,
                           ticker_name: str, period: date, row: dict) -> None:
    """Write one quarter's metrics as a financial document (searchable in ChromaDB)."""
    revenue = row.get("revenue")
    net_income = row.get("net_income")
    fcf = row.get("fcf")
    debt = row.get("debt")
    equity = row.get("equity")
    mc = row.get("market_cap")

    pe = row.get("pe_ratio")
    div_yield = row.get("dividend_yield")
    rev_growth = row.get("revenue_growth")

    # Derived
    debt_equity = f"{(debt/equity*100):.1f}%" if debt and equity else "N/A"
    fcf_margin = f"{fcf/revenue*100:.1f}%" if fcf and revenue else "N/A"
    net_margin = f"{net_income/revenue*100:.1f}%" if net_income and revenue else "N/A"

    content = f"""Quarterly Financial Report — {ticker_symbol} ({ticker_name})
Period: {period.strftime('%B %d, %Y')}

KEY METRICS
Revenue: {_fmt(revenue)}
Net Income: {_fmt(net_income)}
Free Cash Flow: {_fmt(fcf)}
Net Profit Margin: {net_margin}
Free Cash Flow Margin: {fcf_margin}

VALUATION & RETURNS
Market Capitalization: {_fmt(mc) if mc else 'N/A'}
Price-to-Earnings Ratio (P/E): {pe if pe is not None else 'N/A'}
Dividend Yield: {_fmt_pct(div_yield)}
Revenue Growth YoY: {_fmt_pct(rev_growth)}

BALANCE SHEET
Total Debt: {_fmt(debt)}
Total Equity: {_fmt(equity)}
Debt-to-Equity Ratio: {debt_equity}
"""
    title = f"{ticker_symbol} ({ticker_name}) — Q{period.month//3} {period.year} Financial Results"
    conn.execute(
        text("""
            INSERT INTO documents (ticker_id, doc_type, title, content, doc_date)
            VALUES (:ticker_id, 'financial', :title, :content, :doc_date)
            ON CONFLICT DO NOTHING
        """),
        {
            "ticker_id": ticker_id,
            "title": title,
            "content": content,
            "doc_date": period,
        },
    )


def _upsert_price(conn, ticker_id: int, dt, row: pd.Series) -> None:
    conn.execute(
        text("""
            INSERT INTO prices (ticker_id, date, open_price, high_price, low_price, close_price, volume)
            VALUES (:ticker_id, :date, :open_price, :high_price, :low_price, :close_price, :volume)
            ON CONFLICT (ticker_id, date) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume
        """),
        {
            "ticker_id": ticker_id,
            "date": dt.date() if hasattr(dt, "date") else dt,
            "open_price": _float(row.get("Open")),
            "high_price": _float(row.get("High")),
            "low_price": _float(row.get("Low")),
            "close_price": _float(row.get("Close")),
            "volume": _int(row.get("Volume")),
        },
    )


def _float(val) -> float | None:
    if val is None or pd.isna(val):
        return None
    return float(val)


def _int(val) -> int | None:
    if val is None or pd.isna(val):
        return None
    return int(val)


def _df_val(df: pd.DataFrame, label: str, col, default=None) -> float | None:
    """Safe extract from yfinance DataFrame."""
    if df.empty or label not in df.index:
        return default
    try:
        val = df.loc[label, col]
        return float(val) if not pd.isna(val) else default
    except Exception:
        return default


def _parse_date(col) -> date:
    if hasattr(col, "to_pydatetime"):
        return col.to_pydatetime().date()
    if hasattr(col, "date"):
        return col.date()
    if isinstance(col, str):
        try:
            return pd.to_datetime(col).date()
        except Exception:
            pass
    return date.today()


def ingest_ticker(engine, ticker: str) -> dict[str, int]:
    """Ingest all data for a single ticker."""
    counts = {"prices": 0, "financials": 0, "news": 0}
    log.info("Ingesting %s", ticker)

    try:
        yt = yf.Ticker(ticker)
    except Exception as exc:
        log.warning("yfinance failed for %s: %s", ticker, exc)
        return counts

    ticker_id = _get_ticker_id(engine, ticker)
    if not ticker_id:
        log.warning("Ticker %s not found in DB", ticker)
        return counts

    # ── Prices ────────────────────────────────────────────────────────────────
    try:
        hist = yt.history(period=HISTORY_PERIOD)
        if not hist.empty:
            with engine.begin() as conn:
                for dt, row in hist.iterrows():
                    _upsert_price(conn, ticker_id, dt, row)
                    counts["prices"] += 1
    except Exception as exc:
        log.warning("Prices failed for %s: %s", ticker, exc)

    # ── Financials ─────────────────────────────────────────────────────────────
    try:
        income = yt.quarterly_income_stmt
        cashflow = yt.quarterly_cashflow
        balance = yt.quarterly_balance_sheet
        periods = list(income.columns[:FINANCIAL_QUARTERS]) if not income.empty else []

        if periods:
            with engine.begin() as conn:
                for col in periods:
                    period = _parse_date(col)
                    revenue = _df_val(income, "Total Revenue", col)
                    net_income = _df_val(income, "Net Income", col)
                    fcf = _df_val(cashflow, "Free Cash Flow", col)
                    debt = _df_val(balance, "Total Debt", col)
                    equity = _df_val(balance, "Stockholders Equity", col)

                    row_data = {
                        "revenue": revenue,
                        "net_income": net_income,
                        "fcf": fcf,
                        "debt": debt,
                        "equity": equity,
                    }
                    if revenue:
                        conn.execute(
                            text("""
                                INSERT INTO financial_metrics
                                    (ticker_id, period, revenue, net_income, free_cash_flow,
                                     total_debt, total_equity)
                                VALUES (:ticker_id, :period, :revenue, :net_income,
                                        :fcf, :debt, :equity)
                                ON CONFLICT (ticker_id, period) DO UPDATE SET
                                    revenue = EXCLUDED.revenue,
                                    net_income = EXCLUDED.net_income,
                                    free_cash_flow = EXCLUDED.free_cash_flow,
                                    total_debt = EXCLUDED.total_debt,
                                    total_equity = EXCLUDED.total_equity
                            """),
                            {
                                "ticker_id": ticker_id,
                                "period": period,
                                "revenue": int(revenue) if revenue else None,
                                "net_income": int(net_income) if net_income else None,
                                "fcf": int(fcf) if fcf else None,
                                "debt": int(debt) if debt else None,
                                "equity": int(equity) if equity else None,
                            },
                        )
                        counts["financials"] += 1

                    # Also store as searchable document for ChromaDB
                    ticker_name_row = conn.execute(
                        text("SELECT name FROM tickers WHERE id = :id"),
                        {"id": ticker_id},
                    ).fetchone()
                    ticker_name = ticker_name_row[0] if ticker_name_row else ticker
                    _upsert_financial_doc(conn, ticker_id, ticker, ticker_name, period, row_data)
    except Exception as exc:
        log.warning("Financials failed for %s: %s", ticker, exc)

    # ── Market cap from fast_info ────────────────────────────────────────────────
    try:
        fi = yt.fast_info
        mc = _int(fi.market_cap) if fi.market_cap is not None else None
        if mc:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE financial_metrics
                        SET market_cap = :mc
                        WHERE ticker_id = :ticker_id
                          AND market_cap IS NULL
                    """),
                    {"ticker_id": ticker_id, "mc": mc},
                )
    except Exception as exc:
        log.warning("fast_info failed for %s: %s", ticker, exc)

    # ── News → Documents ───────────────────────────────────────────────────────
    try:
        raw_news = yt.news or []
        with engine.begin() as conn:
            for item in raw_news[:20]:
                # yfinance 1.5+ nests fields inside item["content"]
                content_dict = item.get("content") or {}
                title = (content_dict.get("title") or item.get("title") or "")[:500]
                summary = (content_dict.get("summary") or "")[:10000]
                content = f"{title}\n\n{summary}".strip()

                pub_date = None
                pub = content_dict.get("pubDate") or item.get("pubDate")
                if pub:
                    try:
                        if isinstance(pub, str):
                            pub_date = pd.to_datetime(pub).date()
                        elif isinstance(pub, (int, float)):
                            pub_date = date.fromtimestamp(pub / 1000)
                    except Exception:
                        pub_date = None

                if len(content) > 50:
                    conn.execute(
                        text("""
                            INSERT INTO documents (ticker_id, doc_type, title, content, doc_date)
                            VALUES (:ticker_id, 'news', :title, :content, :doc_date)
                        """),
                        {"ticker_id": ticker_id, "title": title, "content": content, "doc_date": pub_date},
                    )
                    counts["news"] += 1
    except Exception as exc:
        log.warning("News failed for %s: %s", ticker, exc)

    return counts


def backfill_financial_docs() -> int:
    """Replace all financial docs with properly-formatted versions + earnings schedule."""
    engine = _engine()

    # Clear stale financial docs
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM documents WHERE doc_type = 'financial'"))

    # Backfill from financial_metrics
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT fm.ticker_id, t.symbol, t.name, fm.period,
                       fm.revenue, fm.net_income, fm.free_cash_flow,
                       fm.total_debt, fm.total_equity, fm.market_cap,
                       fm.pe_ratio, fm.dividend_yield, fm.revenue_growth
                FROM financial_metrics fm
                JOIN tickers t ON t.id = fm.ticker_id
                ORDER BY fm.ticker_id, fm.period DESC
            """)
        ).fetchall()

    count = 0
    for row in rows:
        with engine.begin() as conn:
            _upsert_financial_doc(
                conn,
                ticker_id=row[0], ticker_symbol=row[1], ticker_name=row[2],
                period=row[3],
                row={
                    "revenue": float(row[4]) if row[4] else None,
                    "net_income": float(row[5]) if row[5] else None,
                    "fcf": float(row[6]) if row[6] else None,
                    "debt": float(row[7]) if row[7] else None,
                    "equity": float(row[8]) if row[8] else None,
                    "market_cap": float(row[9]) if row[9] else None,
                    "pe_ratio": row[10], "dividend_yield": row[11], "revenue_growth": row[12],
                },
            )
            count += 1
    log.info("Backfilled %d financial docs", count)
    return count


def backfill_earnings_docs() -> int:
    """Create earnings date documents for all tickers from yfinance calendar."""
    engine = _engine()
    count = 0
    for ticker in ALL_TICKERS:
        try:
            yt = yf.Ticker(ticker)
            cal = getattr(yt, "calendar", None) or {}

            with engine.begin() as conn:
                row = conn.execute(
                    text("SELECT id, name FROM tickers WHERE symbol = :s"),
                    {"s": ticker},
                ).fetchone()
                if not row:
                    continue
                ticker_id, ticker_name = row

            earnings_dates = cal.get("earnings_dates", []) if cal else []
            date_strs = []
            for edate in (earnings_dates[:6] or []):
                s = None
                if hasattr(edate, "strftime"):
                    s = edate.strftime("%Y-%m-%d")
                elif isinstance(edate, str) and len(edate) == 10:
                    s = edate
                if s:
                    date_strs.append(s)

            content = (
                f"{ticker} ({ticker_name}) earnings report schedule. "
                f"Upcoming earnings dates: {', '.join(date_strs) or 'none listed by yfinance'}. "
                f"{ticker_name} fiscal calendar typically: Q1 results reported April, "
                f"Q2 in July, Q3 in October, Q4 in January or February. "
                f"Use yfinance earnings calendar for confirmed dates."
            )
            title = f"{ticker} ({ticker_name}) — Earnings Report Schedule"

            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO documents (ticker_id, doc_type, title, content, doc_date)
                        VALUES (:ticker_id, 'financial', :title, :content, CURRENT_DATE)
                    """),
                    {"ticker_id": ticker_id, "title": title, "content": content},
                )
            count += 1
        except Exception as exc:
            log.warning("Earnings backfill failed for %s: %s", ticker, exc)
    log.info("Backfilled %d earnings docs", count)
    return count


def run_full_ingestion() -> dict[str, dict[str, int]]:
    """Ingest data for all configured tickers."""
    log.info("Starting full ingestion for %d tickers", len(ALL_TICKERS))
    engine = _engine()
    totals: dict[str, dict[str, int]] = {}

    for ticker in ALL_TICKERS:
        counts = ingest_ticker(engine, ticker)
        totals[ticker] = counts
        log.info("  %s → prices=%d financials=%d news=%d", ticker, *counts.values())
        time.sleep(YFINANCE_SLEEP_SECS)

    log.info("Ingestion complete for %d tickers.", len(ALL_TICKERS))
    return totals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_full_ingestion()
    for t, c in result.items():
        print(f"{t}: {c}")