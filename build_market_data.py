#!/usr/bin/env python3
"""Build market-data.json for MapleFolio from Yahoo Finance.

Reads tickers.txt (one MapleFolio ticker per line) and writes market-data.json
next to this script. Run it by hand (pip install yfinance pandas) or let the
GitHub Action in .github/workflows/update-data.yml run it every weekday.
"""
import json
import math
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAX_YEARS = 20       # longest history kept
MIN_MONTHS = 24      # skip tickers with less history than this
BATCH = 40
# Young funds borrow an older, similar asset's history before they existed.
PROXY = {"IBIT": "BTC-USD", "SGOV": "BIL", "BUG": "CIBR"}


def yahoo_symbol(t):
    """MapleFolio ticker -> Yahoo symbol (BRK.B -> BRK-B, BEP.UN.TO -> BEP-UN.TO)."""
    if t.endswith(".TO"):
        return t[:-3].replace(".", "-") + ".TO"
    return t.replace(".", "-")


def last_full_month(today):
    return pd.Period(today, "M") - 1


def monthly_returns(adj, last_full):
    m = adj.dropna()
    m = m[~m.index.duplicated()].resample("ME").last()
    r = m.pct_change().dropna()
    r.index = r.index.to_period("M")
    return r[r.index <= last_full]


def fetch(tickers, last_full):
    """Return {ticker: {"ret": monthly return Series, "yield": float}}."""
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        syms = [yahoo_symbol(t) for t in batch]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(syms, period="max", interval="1d", auto_adjust=False,
                                  actions=True, group_by="ticker", threads=True,
                                  progress=False)
                break
            except Exception as e:  # network hiccup: wait and retry
                print(f"batch {i // BATCH + 1} attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for t, s in zip(batch, syms):
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(subset=["Adj Close"])
                if df.empty:
                    continue
                r = monthly_returns(df["Adj Close"], last_full)
                yr_ago = df.index[-1] - pd.Timedelta(days=365)
                div = df.loc[df.index > yr_ago, "Dividends"].fillna(0).sum()
                last = float(df["Close"].iloc[-1])
                out[t] = {"ret": r, "yield": (float(div) / last * 100) if last > 0 else 0.0}
            except Exception as e:
                print(f"{t}: skipped ({e})", file=sys.stderr)
        print(f"fetched {min(i + BATCH, len(tickers))}/{len(tickers)}", file=sys.stderr)
    return out


def build(frames, tickers, today):
    last_full = last_full_month(today)
    start = max(last_full - (MAX_YEARS * 12 - 1),
                min(f["ret"].index[0] for f in frames.values() if len(f["ret"])))
    axis = pd.period_range(start, last_full, freq="M")
    months = [str(p) for p in axis]
    pos = {p: i for i, p in enumerate(axis)}
    data = {}
    for t in tickers:
        f = frames.get(t)
        if f is None or len(f["ret"]) == 0:
            continue
        r = f["ret"][f["ret"].index >= axis[0]]
        p, x = 0, None
        px = frames.get(PROXY.get(t, ""))
        if px is not None and len(px["ret"]) and len(r) and px["ret"].index[0] < r.index[0]:
            pre = px["ret"][(px["ret"].index >= axis[0]) & (px["ret"].index < r.index[0])]
            if len(pre):
                r = pd.concat([pre, r])
                p, x = len(pre), PROXY[t]
        if len(r) < MIN_MONTHS or r.index[-1] < last_full - 2:
            continue  # too short, or stopped trading
        r = r.reindex(pd.period_range(r.index[0], r.index[-1], freq="M")).fillna(0.0)
        vals = r.astype(float).values
        n = len(vals)
        growth = float(pd.Series(1 + vals).prod())
        ret = (growth ** (12 / n) - 1) * 100 if growth > 0 else float("nan")
        vol = float(pd.Series(vals).std(ddof=1)) * math.sqrt(12) * 100
        if not (math.isfinite(ret) and math.isfinite(vol)):
            continue
        d = {"ret": round(ret, 2), "vol": round(vol, 2), "y": round(f["yield"], 2),
             "s": pos[r.index[0]], "r": [round(float(v), 4) for v in vals]}
        if p:
            d["p"], d["x"] = p, x
        data[t] = d
    return {"asOf": today.isoformat(), "years": round(len(months) / 12, 1),
            "months": months, "data": data}


def main():
    tickers = [l.strip() for l in (HERE / "tickers.txt").read_text().splitlines() if l.strip()]
    today = dt.date.today()
    last_full = last_full_month(today)
    need = list(dict.fromkeys(tickers + list(PROXY.values())))
    frames = fetch(need, last_full)
    result = build(frames, tickers, today)
    got = len(result["data"])
    print(f"{got} of {len(tickers)} tickers built", file=sys.stderr)
    if got < 0.9 * len(tickers):  # something is wrong upstream: keep the old file
        print("Too few tickers came back; leaving market-data.json unchanged.", file=sys.stderr)
        sys.exit(1)
    (HERE / "market-data.json").write_text(json.dumps(result, separators=(",", ":")))
    print("wrote market-data.json", file=sys.stderr)


if __name__ == "__main__":
    main()
