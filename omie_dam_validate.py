#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "session_date",
    "period",
    "price_pt_eur_mwh",
    "price_es_eur_mwh",
    "source_filename",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate OMIE DAM consolidated output")
    p.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    p.add_argument(
        "--start",
        default=None,
        help="Optional expected start date YYYY-MM-DD (for coverage check)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Optional expected end date YYYY-MM-DD (for coverage check)",
    )
    return p.parse_args()


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, Path]:
    parquet_path = data_dir / "omie_dam_prices.parquet"
    csv_path = data_dir / "omie_dam_prices.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path), parquet_path
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path
    raise FileNotFoundError(f"No dataset found at {parquet_path} or {csv_path}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    df, source_path = load_dataset(data_dir)

    print(f"Loaded: {source_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return 1

    df = df.copy()
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["price_pt_eur_mwh"] = pd.to_numeric(df["price_pt_eur_mwh"], errors="coerce")
    df["price_es_eur_mwh"] = pd.to_numeric(df["price_es_eur_mwh"], errors="coerce")

    print(f"Date range: {df['session_date'].min()} -> {df['session_date'].max()}")
    print(f"Unique dates: {df['session_date'].nunique():,}")
    print(f"Unique (session_date, period): {df[['session_date','period']].drop_duplicates().shape[0]:,}")

    dupes = df.duplicated(subset=["session_date", "period"]).sum()
    print(f"Duplicate (session_date, period): {int(dupes):,}")

    null_pt = int(df["price_pt_eur_mwh"].isna().sum())
    null_es = int(df["price_es_eur_mwh"].isna().sum())
    print(f"Null prices -> PT: {null_pt:,}, ES: {null_es:,}")

    sentinel_like = int(
        ((df["price_pt_eur_mwh"].fillna(0) <= -9999) | (df["price_es_eur_mwh"].fillna(0) <= -9999)).sum()
    )
    print(f"Remaining sentinel-like values (<= -9999): {sentinel_like:,}")

    if args.start or args.end:
        observed_dates = set(df["session_date"].dropna())
        expected_start = pd.to_datetime(args.start).date() if args.start else min(observed_dates)
        expected_end = pd.to_datetime(args.end).date() if args.end else max(observed_dates)
        expected = set(pd.date_range(expected_start, expected_end, freq="D").date)
        missing_dates = sorted(expected - observed_dates)
        print(f"Expected coverage: {expected_start} -> {expected_end}")
        print(f"Missing session dates in expected range: {len(missing_dates):,}")
        if missing_dates:
            preview = ", ".join(str(d) for d in missing_dates[:20])
            suffix = " ..." if len(missing_dates) > 20 else ""
            print(f"Missing dates preview: {preview}{suffix}")

    by_date_counts = df.groupby("session_date", dropna=True)["period"].nunique()
    if not by_date_counts.empty:
        print(
            "Periods/day (unique): "
            f"min={int(by_date_counts.min())}, median={float(by_date_counts.median()):.0f}, max={int(by_date_counts.max())}"
        )

    print("Validation summary complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
