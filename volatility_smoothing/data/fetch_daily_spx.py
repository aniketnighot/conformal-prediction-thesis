#!/usr/bin/env python3
"""
Daily SPX Options Data Pipeline

Fetches SPX options data from OpenBB and saves it in WRDS-compatible format.
Designed to run daily via cron/launchd scheduler.

Usage:
    python fetch_daily_spx.py [--output-dir PATH] [--clean-duplicates]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
import sys

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"spx_fetch_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def fetch_spx_data():
    """Fetch SPX options chain from OpenBB"""
    try:
        from openbb import obb

        # Set output type to dataframe
        obb.user.preferences.output_type = "dataframe"
        logger.info("OpenBB initialized successfully")

        # Fetch SPX options chain
        logger.info("Fetching SPX options data from OpenBB (CBOE provider)...")
        df_openbb = obb.derivatives.options.chains(symbol="SPX", provider="cboe")

        logger.info(f"Fetched {len(df_openbb)} option quotes")
        return df_openbb

    except ImportError:
        logger.error("OpenBB not installed. Install with: pip install openbb")
        raise
    except Exception as e:
        logger.error(f"Error fetching data from OpenBB: {e}")
        raise


def transform_to_wrds_format(df_openbb):
    """Transform OpenBB data to WRDS-compatible format"""
    logger.info("Transforming to WRDS format...")

    # Create WRDS-compatible DataFrame
    df_wrds = pd.DataFrame()

    # Map quote date - use today's date
    quote_date = datetime.now().strftime('%Y-%m-%d')
    df_wrds['date'] = [quote_date] * len(df_openbb)

    # Map expiration date
    df_wrds['exdate'] = pd.to_datetime(df_openbb['expiration'])

    # Map strike price - CRITICAL: Multiply by 1000!
    df_wrds['strike_price'] = df_openbb['strike'] * 1000

    # Map option type to 'C' or 'P'
    df_wrds['cp_flag'] = df_openbb['option_type'].str.upper().str[0]

    # Map bid and ask prices
    df_wrds['best_bid'] = df_openbb['bid']
    df_wrds['best_offer'] = df_openbb['ask']

    # Set am_settlement to 1 (required by WRDS format)
    df_wrds['am_settlement'] = 1

    logger.info(f"Transformed {len(df_wrds)} rows to WRDS format")
    return df_wrds


def clean_data(df_wrds):
    """Clean and validate the data"""
    logger.info(f"Cleaning data... Initial rows: {len(df_wrds)}")

    # Remove rows with missing values
    df_clean = df_wrds.dropna()
    logger.info(f"After removing NaN: {len(df_clean)}")

    # Remove expired options (expiry date in the past)
    df_clean['exdate'] = pd.to_datetime(df_clean['exdate'])
    today = pd.Timestamp.now().normalize()
    df_clean = df_clean[df_clean['exdate'] > today]
    logger.info(f"After removing expired options: {len(df_clean)}")

    # Remove zero or negative prices
    df_clean = df_clean[
        (df_clean['best_bid'] > 0) &
        (df_clean['best_offer'] > 0)
    ]
    logger.info(f"After removing zero/negative prices: {len(df_clean)}")

    # Ensure valid option types
    df_clean = df_clean[df_clean['cp_flag'].isin(['C', 'P'])]
    logger.info(f"After validating option types: {len(df_clean)}")

    # Sort by date, expiry, strike
    df_clean = df_clean.sort_values(['date', 'exdate', 'strike_price'])

    logger.info(f"Final clean data: {len(df_clean)} rows")
    logger.info(f"Unique expiries: {df_clean['exdate'].nunique()}")

    return df_clean


def remove_duplicates(df):
    """Remove duplicate entries based on date, exdate, strike_price, cp_flag"""
    duplicate_cols = ['date', 'exdate', 'strike_price', 'cp_flag']
    duplicates = df.duplicated(subset=duplicate_cols, keep='first')
    num_duplicates = duplicates.sum()

    if num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} duplicate rows - removing them")
        df_clean = df[~duplicates].copy()
        logger.info(f"Rows after deduplication: {len(df_clean)}")
        return df_clean
    else:
        logger.info("No duplicates found")
        return df


def save_data(df, output_dir):
    """Save the cleaned data to CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create filename with date
    quote_date = df['date'].iloc[0]
    if isinstance(quote_date, str):
        date_str = quote_date
    else:
        date_str = quote_date.strftime('%Y-%m-%d')

    output_file = output_path / f'spx_options_{date_str}.csv'

    # Save to CSV
    df.to_csv(output_file, index=False)

    logger.info(f"Saved WRDS-compatible CSV to: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    return output_file


def verify_csv(csv_path):
    """Verify the saved CSV file"""
    logger.info("Verifying saved CSV...")

    df_verify = pd.read_csv(csv_path)

    expected_cols = ['date', 'exdate', 'strike_price', 'cp_flag', 'best_bid', 'best_offer', 'am_settlement']

    if list(df_verify.columns) != expected_cols:
        logger.error(f"Column mismatch! Got: {list(df_verify.columns)}")
        return False

    logger.info(f"Verification passed - {len(df_verify)} rows")
    logger.info(f"Columns: {list(df_verify.columns)}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Fetch daily SPX options data from OpenBB')
    parser.add_argument(
        '--output-dir',
        default='volatility_smoothing/data/openbb/spx',
        help='Output directory for CSV files (default: volatility_smoothing/data/openbb/spx)'
    )
    parser.add_argument(
        '--clean-duplicates',
        action='store_true',
        help='Remove duplicate entries from the data'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Starting SPX Options Data Pipeline")
    logger.info("=" * 80)

    try:
        # Fetch data
        df_openbb = fetch_spx_data()

        # Transform to WRDS format
        df_wrds = transform_to_wrds_format(df_openbb)

        # Clean data
        df_clean = clean_data(df_wrds)

        # Remove duplicates if requested
        if args.clean_duplicates:
            df_clean = remove_duplicates(df_clean)

        # Save data
        output_file = save_data(df_clean, args.output_dir)

        # Verify
        if verify_csv(output_file):
            logger.info("=" * 80)
            logger.info("SUCCESS! Data pipeline completed successfully")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("Verification failed!")
            return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
