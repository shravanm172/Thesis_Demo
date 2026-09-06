"""
Pure data ingestion:
Downloads days of Massive flat files (options and/or equity minute
aggregates) from their S3-compatible bucket.

Usage:
    # default: derives the full required date range from
    # demo_earnings_dates_clean.csv via config.get_required_trading_days()
    # (earliest earnings_date - 1mo to latest + 1mo, NYSE trading days only)
    python scripts/download_flat_files.py

    # override: fetch only specific dates instead, e.g. for a quick test
    python scripts/download_flat_files.py --dates-file manifests/required_dates.txt
    where manifests/required_dates.txt is a file with one YYYY-MM-DD date per line
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from config import get_required_trading_days

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

S3_ENDPOINT = "https://files.massive.com"
BUCKET = "flatfiles"

# Key layout for both datasets, confirmed against the live bucket.
DATASET_PREFIXES = {
    "options": "us_options_opra/minute_aggs_v1",
    "equity": "us_stocks_sip/minute_aggs_v1",
}


def s3_key_for(dataset: str, date: datetime) -> str:
    """
    For a given dataset and date, return the exact key to request for that s3 object
    """
    prefix = DATASET_PREFIXES[dataset]
    return f"{prefix}/{date:%Y}/{date:%m}/{date:%Y-%m-%d}.csv.gz"


def local_path_for(dataset: str, date: datetime) -> Path:
    """
    For a given dataset and date, generate a local path for that s3 object on this machine
    """
    return REPO_ROOT / "data" / "raw" / dataset / f"{date:%Y-%m-%d}.csv.gz"


def get_client():
    """
    Get credentials and create client for API calls
    """
    access_key = os.environ.get("MASSIVE_S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("MASSIVE_S3_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        sys.exit(
            "Missing credentials. Set MASSIVE_S3_ACCESS_KEY_ID and "
            "MASSIVE_S3_SECRET_ACCESS_KEY in .env (see .env.example)."
        )

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        config=Config(signature_version="s3v4"),
    )


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def load_dates(args) -> list[datetime]:
    """
    Return the list of dates to fetch. Default: derived from
    input_data.csv via get_required_trading_days() - covers
    both treatment dates and the control-day pool in one pass. If
    --dates-file is given, that explicit list is used instead (an override
    for quick/manual testing against a small set of dates, not the normal
    path).
    """
    if args.dates_file:
        path = Path(args.dates_file)
        if not path.exists():
            sys.exit(f"--dates-file not found: {path}")
        with open(path) as f:
            dates = [parse_date(line.strip()) for line in f if line.strip()]
    else:
        dates = [datetime.combine(d, datetime.min.time()) for d in get_required_trading_days()]

    return sorted(set(dates))


def download_one(client, dataset: str, date: datetime) -> str:
    local_path = local_path_for(dataset, date)
    if local_path.exists():
        return "skipped (already present)"

    key = s3_key_for(dataset, date)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    try:
        client.download_file(BUCKET, key, str(tmp_path))
        tmp_path.rename(local_path)
        return "downloaded"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey"):
            return "missing (no file for this date - holiday, or wrong key prefix)"
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates-file", help="Override: path to a file with one YYYY-MM-DD date per line, instead of deriving the range from the earnings CSV")
    parser.add_argument("--dataset", choices=["options", "equity", "both"], default="both")
    args = parser.parse_args()

    client = get_client()

    dates = load_dates(args)
    datasets = ["options", "equity"] if args.dataset == "both" else [args.dataset]

    results = {"downloaded": [], "skipped": [], "missing": []}
    for date in dates:
        for ds in datasets:
            status = download_one(client, ds, date)
            print(f"{date:%Y-%m-%d} [{ds}]: {status}")
            if status == "downloaded":
                results["downloaded"].append((ds, date))
            elif status.startswith("skipped"):
                results["skipped"].append((ds, date))
            else:
                results["missing"].append((ds, date))

    print()
    print(f"Summary: {len(results['downloaded'])} downloaded, "
          f"{len(results['skipped'])} already present, "
          f"{len(results['missing'])} missing.")
    if results["missing"]:
        print("Missing (investigate before assuming these days just had no data):")
        for ds, date in results["missing"]:
            print(f"  {date:%Y-%m-%d} [{ds}]")


if __name__ == "__main__":
    main()
