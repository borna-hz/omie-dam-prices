#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.omie.es/en/file-download"
HEADER_PREFIX = "MARGINALPDBC"
VERSION_CANDIDATES = range(9, 0, -1)
SENTINEL_THRESHOLD = -9999.0
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 3

PARENTS_CONFIG = (
    {"parents": "marginalpdbc", "prefix": "marginalpdbc_"},
    {"parents": "marginalpdbcpt", "prefix": "marginalpdbcpt_"},
)


@dataclass(frozen=True)
class RawFileRef:
    session_date: date
    parents: str
    filename: str
    version: int
    path: Path


def parse_args() -> argparse.Namespace:
    today_str = date.today().isoformat()
    parser = argparse.ArgumentParser(
        description="Download and consolidate OMIE DAM marginal prices."
    )
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=today_str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload raw files even if they exist locally",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"HTTP retries (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_session(retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "omie-dam-pipeline/1.0"})
    return session


def build_filename(prefix: str, d: date, version: int) -> str:
    return f"{prefix}{d.strftime('%Y%m%d')}.{version}"


def download_candidate(
    session: requests.Session,
    parents: str,
    filename: str,
    destination: Path,
    timeout: int,
    refresh: bool,
) -> bool:
    if destination.exists() and not refresh:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {"filename": filename, "parents": parents}
    logging.debug("Fetching %s (%s)", filename, parents)

    try:
        response = session.get(BASE_URL, params=params, timeout=timeout)
    except requests.RequestException as exc:
        logging.debug("Request failed for %s: %s", filename, exc)
        return False

    if response.status_code != 200:
        logging.debug("Non-200 for %s: %s", filename, response.status_code)
        return False

    text = response.text
    if not text.lstrip().startswith(HEADER_PREFIX):
        logging.debug("Invalid payload for %s (missing %s header)", filename, HEADER_PREFIX)
        return False

    destination.write_bytes(response.content)
    return True


def find_latest_file_for_date(
    session: requests.Session,
    data_dir: Path,
    target_date: date,
    parents: str,
    prefix: str,
    timeout: int,
    refresh: bool,
) -> RawFileRef | None:
    raw_parent_dir = data_dir / "raw" / parents

    for version in VERSION_CANDIDATES:
        filename = build_filename(prefix, target_date, version)
        local_path = raw_parent_dir / filename
        ok = download_candidate(
            session=session,
            parents=parents,
            filename=filename,
            destination=local_path,
            timeout=timeout,
            refresh=refresh,
        )
        if ok:
            return RawFileRef(
                session_date=target_date,
                parents=parents,
                filename=filename,
                version=version,
                path=local_path,
            )
    return None


def normalize_price(value: str) -> float | None:
    raw = (value or "").strip().replace(",", ".")
    if raw == "":
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    if number <= SENTINEL_THRESHOLD:
        return None
    return number


def parse_raw_file(raw_file: RawFileRef) -> pd.DataFrame:
    content = raw_file.path.read_text(encoding="latin-1", errors="replace")
    rows: list[dict[str, object]] = []
    saw_header = False

    for line_number, raw_line in enumerate(io.StringIO(content), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(";")]
        while parts and parts[-1] == "":
            parts.pop()
        if not parts:
            continue

        if parts[0].upper().startswith(HEADER_PREFIX):
            saw_header = True
            continue

        if len(parts) < 6:
            logging.debug(
                "Skipping short row in %s:%s -> %s", raw_file.filename, line_number, parts
            )
            continue

        try:
            year, month, day, period = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
            session_date = date(year, month, day).isoformat()
        except ValueError:
            logging.debug(
                "Skipping invalid date/period row in %s:%s -> %s",
                raw_file.filename,
                line_number,
                parts,
            )
            continue

        rows.append(
            {
                "session_date": session_date,
                "period": period,
                "price_pt_eur_mwh": normalize_price(parts[4]),
                "price_es_eur_mwh": normalize_price(parts[5]),
                "source_filename": raw_file.filename,
                "_source_version": raw_file.version,
                "_source_parents": raw_file.parents,
            }
        )

    if not saw_header:
        raise ValueError(f"Missing OMIE header in {raw_file.path}")

    return pd.DataFrame.from_records(rows)


def consolidate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "session_date",
                "period",
                "price_pt_eur_mwh",
                "price_es_eur_mwh",
                "source_filename",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df[
            [
                "session_date",
                "period",
                "price_pt_eur_mwh",
                "price_es_eur_mwh",
                "source_filename",
            ]
        ]

    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["price_pt_eur_mwh"] = pd.to_numeric(df["price_pt_eur_mwh"], errors="coerce")
    df["price_es_eur_mwh"] = pd.to_numeric(df["price_es_eur_mwh"], errors="coerce")

    # Keep the newest available version per (session_date, period).
    df = df.sort_values(
        by=["session_date", "period", "_source_version", "_source_parents"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    before = len(df)
    df = df.drop_duplicates(subset=["session_date", "period"], keep="last")
    deduped = before - len(df)
    if deduped:
        logging.info("Dropped %s duplicate rows by (session_date, period)", deduped)

    df = df.sort_values(by=["session_date", "period"], ascending=[True, True]).reset_index(drop=True)
    return df[
        [
            "session_date",
            "period",
            "price_pt_eur_mwh",
            "price_es_eur_mwh",
            "source_filename",
        ]
    ]


def write_outputs(df: pd.DataFrame, data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "omie_dam_prices.parquet"
    csv_path = data_dir / "omie_dam_prices.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    start_date = parse_iso_date(args.start)
    end_date = parse_iso_date(args.end)
    if start_date > end_date:
        raise SystemExit("--start must be on or before --end")

    data_dir = Path(args.data_dir)
    logging.info(
        "Processing OMIE DAM prices from %s to %s (refresh=%s)",
        start_date,
        end_date,
        args.refresh,
    )

    raw_refs: list[RawFileRef] = []
    missing_files = 0

    with build_session(args.retries) as session:
        for target_date in date_range(start_date, end_date):
            for cfg in PARENTS_CONFIG:
                raw_ref = find_latest_file_for_date(
                    session=session,
                    data_dir=data_dir,
                    target_date=target_date,
                    parents=cfg["parents"],
                    prefix=cfg["prefix"],
                    timeout=args.timeout,
                    refresh=args.refresh,
                )
                if raw_ref is None:
                    missing_files += 1
                    logging.warning(
                        "No OMIE file found for %s (%s)", target_date.isoformat(), cfg["parents"]
                    )
                    continue
                raw_refs.append(raw_ref)

    logging.info("Resolved %s raw files (%s missing)", len(raw_refs), missing_files)

    frames: list[pd.DataFrame] = []
    for raw_ref in raw_refs:
        try:
            frames.append(parse_raw_file(raw_ref))
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to parse %s: %s", raw_ref.path, exc)

    df = consolidate_frames(frames)
    parquet_path, csv_path = write_outputs(df, data_dir)
    logging.info("Wrote %s rows to %s", len(df), parquet_path)
    logging.info("Wrote CSV copy to %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
