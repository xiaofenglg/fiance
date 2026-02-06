# -*- coding: utf-8 -*-
"""
V12 Phase 1: Qlib Data Converter

Converts SQLite NAV data to Qlib binary format.

Input: aifinance.sqlite3 (nav_history table)
Output: ~/.qlib/qlib_data/wmp_data/
        ├── calendars/
        │   └── day.txt
        ├── instruments/
        │   └── all.txt
        └── features/
            └── {product_code}/
                ├── nav.day.bin
                ├── return.day.bin
                └── volume.day.bin

Usage:
    python dump_to_qlib.py --db aifinance.sqlite3 --output ~/.qlib/qlib_data/wmp_data
    python dump_to_qlib.py --bank 中信银行 --verify
"""

import argparse
import hashlib
import logging
import os
import sqlite3
import struct
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aifinance.data.unsmoothing import GLMUnsmoothing, IlliquidityException

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find project root directory"""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "aifinance.sqlite3").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


class QlibDataDumper:
    """Qlib Data Format Converter"""

    # Qlib binary format: np.float32, little-endian
    DTYPE = np.float32
    BYTE_ORDER = "<"  # little-endian

    def __init__(
        self,
        db_path: str,
        output_dir: str,
        min_days: int = 180,
        theta0_threshold: float = 0.10,
        filter_money_market: bool = True,
    ):
        """
        Args:
            db_path: Path to SQLite database
            output_dir: Output directory for Qlib data
            min_days: Minimum data days required (filter products with less)
            theta0_threshold: GLM theta_0 threshold for liquidity filter
            filter_money_market: Filter out money market funds (NAV ~= 1.0)
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir).expanduser()
        self.min_days = min_days
        self.theta0_threshold = theta0_threshold
        self.filter_money_market = filter_money_market

        # GLM filter for illiquidity detection
        self.glm = GLMUnsmoothing(
            max_lag=2,
            method="mle",
            theta_0_threshold=theta0_threshold,
            strict_filter=False,
        )

        # Statistics
        self.stats = {
            "total_products": 0,
            "filtered_insufficient_data": 0,
            "filtered_nav_below_one": 0,
            "filtered_money_market": 0,
            "filtered_illiquid": 0,
            "exported_products": 0,
            "total_records": 0,
        }

    def dump(
        self,
        bank_filter: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict:
        """
        Export SQLite data to Qlib format.

        Args:
            bank_filter: Filter by bank name (None = all banks)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            Statistics dict
        """
        logger.info(f"[Qlib Dump] Starting export from {self.db_path}")
        logger.info(f"[Qlib Dump] Output directory: {self.output_dir}")
        logger.info(f"[Qlib Dump] Bank filter: {bank_filter or 'All banks'}")
        logger.info(f"[Qlib Dump] Min days: {self.min_days}, theta0_threshold: {self.theta0_threshold}")

        # Create output directories
        self._create_directories()

        # Load data from SQLite
        products_df, nav_df = self._load_sqlite_data(bank_filter, start_date, end_date)

        if products_df.empty or nav_df.empty:
            logger.warning("[Qlib Dump] No data found")
            return self.stats

        self.stats["total_products"] = len(products_df)
        logger.info(f"[Qlib Dump] Loaded {len(products_df)} products, {len(nav_df)} NAV records")

        # Generate calendar
        all_dates = self._generate_calendar(nav_df)
        self._write_calendar(all_dates)

        # hisensho optimization: Pre-group NAV data to avoid O(N*M) filtering
        nav_groups = nav_df.groupby("product_code")

        # Process each product
        valid_instruments = []
        for _, product in products_df.iterrows():
            product_code = product["product_code"]
            product_name = product.get("product_name", product_code)

            # hisensho optimization: Use pre-grouped data
            if product_code not in nav_groups.groups:
                self.stats["filtered_insufficient_data"] += 1
                continue
            product_nav = nav_groups.get_group(product_code)

            if len(product_nav) < self.min_days:
                self.stats["filtered_insufficient_data"] += 1
                continue

            # hisensho fix: 不在Qlib数据阶段预过滤NAV<1(避免幸存者偏差)
            # NAV<1的产品保留在数据集中，由信号/模型自行处理
            nav_values = product_nav["nav"].values
            nav_min, nav_max = nav_values.min(), nav_values.max()

            # Check for money market fund (NAV stays around 1.0)
            # hisensho: Relaxed threshold from 0.001 to 0.01 for long-term data
            if self.filter_money_market:
                if (nav_max - nav_min) < 0.01 and 0.99 <= nav_min <= 1.01:
                    self.stats["filtered_money_market"] += 1
                    logger.debug(f"[Qlib Dump] Filtered money market: {product_code}")
                    continue

            # GLM liquidity filter
            if not self._check_liquidity(product_nav):
                self.stats["filtered_illiquid"] += 1
                logger.debug(f"[Qlib Dump] Filtered illiquid: {product_code}")
                continue

            # Export product data
            self._export_product(product_code, product_nav, all_dates)
            valid_instruments.append((product_code, product_name))
            self.stats["exported_products"] += 1
            self.stats["total_records"] += len(product_nav)

        # Write instruments list
        self._write_instruments(valid_instruments, all_dates)

        logger.info(f"[Qlib Dump] Export complete. Stats: {self.stats}")
        return self.stats

    def _create_directories(self):
        """Create Qlib directory structure"""
        dirs = [
            self.output_dir / "calendars",
            self.output_dir / "instruments",
            self.output_dir / "features",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _load_sqlite_data(
        self,
        bank_filter: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Build query for products
            products_query = "SELECT product_code, product_name, bank_name FROM products"
            if bank_filter:
                products_query += f" WHERE bank_name = '{bank_filter}'"

            products_df = pd.read_sql_query(products_query, conn)

            if products_df.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Build query for NAV history
            product_codes = products_df["product_code"].tolist()
            placeholders = ",".join(["?" for _ in product_codes])

            nav_query = f"""
                SELECT product_code, date, nav
                FROM nav_history
                WHERE product_code IN ({placeholders})
            """
            params = product_codes

            if start_date:
                nav_query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                nav_query += " AND date <= ?"
                params.append(end_date)

            nav_query += " ORDER BY product_code, date"

            nav_df = pd.read_sql_query(nav_query, conn, params=params)

            return products_df, nav_df

        finally:
            conn.close()

    def _generate_calendar(self, nav_df: pd.DataFrame) -> List[str]:
        """Generate trading calendar from NAV data"""
        all_dates = sorted(nav_df["date"].unique())
        return all_dates

    def _write_calendar(self, dates: List[str]):
        """Write calendar file"""
        calendar_path = self.output_dir / "calendars" / "day.txt"
        with open(calendar_path, "w", encoding="utf-8") as f:
            for date in dates:
                f.write(f"{date}\n")
        logger.info(f"[Qlib Dump] Written calendar: {len(dates)} dates")

    def _write_instruments(
        self,
        instruments: List[Tuple[str, str]],
        dates: List[str],
    ):
        """Write instruments file"""
        if not instruments or not dates:
            return

        instruments_path = self.output_dir / "instruments" / "all.txt"
        start_date = dates[0]
        end_date = dates[-1]

        with open(instruments_path, "w", encoding="utf-8") as f:
            for product_code, product_name in instruments:
                # Qlib format: instrument_name start_date end_date
                f.write(f"{product_code}\t{start_date}\t{end_date}\n")

        logger.info(f"[Qlib Dump] Written instruments: {len(instruments)} products")

    def _check_liquidity(self, product_nav: pd.DataFrame) -> bool:
        """Check product liquidity using GLM theta_0"""
        try:
            nav_values = product_nav["nav"].values

            # Calculate daily returns
            returns = np.diff(nav_values) / nav_values[:-1] * 100 * 365  # Annualized %

            if len(returns) < 30:
                return True  # Not enough data, pass through

            # Estimate GLM theta
            theta = self.glm.estimate_theta(returns)
            theta_0 = theta[0]

            if theta_0 < self.theta0_threshold:
                return False

            return True

        except (IlliquidityException, Exception) as e:
            logger.debug(f"[Qlib Dump] GLM check failed: {e}")
            return True  # Pass through on error

    def _export_product(
        self,
        product_code: str,
        product_nav: pd.DataFrame,
        all_dates: List[str],
    ):
        """Export single product to Qlib binary format"""
        # Create product directory
        # Replace special characters in product_code for filesystem
        safe_code = product_code.replace("/", "_").replace("\\", "_")
        product_dir = self.output_dir / "features" / safe_code
        product_dir.mkdir(parents=True, exist_ok=True)

        # Create date index mapping
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        n_dates = len(all_dates)

        # Initialize arrays with NaN
        nav_array = np.full(n_dates, np.nan, dtype=self.DTYPE)
        return_array = np.full(n_dates, np.nan, dtype=self.DTYPE)
        volume_array = np.ones(n_dates, dtype=self.DTYPE)  # Placeholder

        # Fill in NAV values
        for _, row in product_nav.iterrows():
            date = row["date"]
            if date in date_to_idx:
                idx = date_to_idx[date]
                nav_array[idx] = row["nav"]

        # Calculate daily returns (decimal form, not percentage)
        # hisensho note: Returns stored as decimal (0.015 = 1.5%), consistent with Qlib convention
        for i in range(1, n_dates):
            if not np.isnan(nav_array[i]) and not np.isnan(nav_array[i - 1]) and nav_array[i - 1] > 0:
                return_array[i] = nav_array[i] / nav_array[i - 1] - 1  # Decimal return

        # Write binary files
        self._write_bin_file(product_dir / "nav.day.bin", nav_array)
        self._write_bin_file(product_dir / "return.day.bin", return_array)
        self._write_bin_file(product_dir / "volume.day.bin", volume_array)

    def _write_bin_file(self, path: Path, data: np.ndarray):
        """Write Qlib binary file (np.float32, little-endian)"""
        data = data.astype(self.DTYPE)
        with open(path, "wb") as f:
            f.write(data.tobytes())

    def verify(self) -> bool:
        """Verify exported data integrity"""
        logger.info("[Qlib Dump] Verifying exported data...")

        # Check calendar
        calendar_path = self.output_dir / "calendars" / "day.txt"
        if not calendar_path.exists():
            logger.error(f"[Qlib Dump] Calendar file missing: {calendar_path}")
            return False

        with open(calendar_path, "r") as f:
            dates = [line.strip() for line in f if line.strip()]
        logger.info(f"[Qlib Dump] Calendar: {len(dates)} dates")

        # Check instruments
        instruments_path = self.output_dir / "instruments" / "all.txt"
        if not instruments_path.exists():
            logger.error(f"[Qlib Dump] Instruments file missing: {instruments_path}")
            return False

        with open(instruments_path, "r") as f:
            instruments = [line.strip().split("\t")[0] for line in f if line.strip()]
        logger.info(f"[Qlib Dump] Instruments: {len(instruments)} products")

        # Check features
        features_dir = self.output_dir / "features"
        n_verified = 0
        n_errors = 0

        for product_code in instruments:
            safe_code = product_code.replace("/", "_").replace("\\", "_")
            product_dir = features_dir / safe_code

            if not product_dir.exists():
                logger.warning(f"[Qlib Dump] Missing product dir: {product_dir}")
                n_errors += 1
                continue

            # Verify binary files
            expected_bytes = len(dates) * 4  # float32 = 4 bytes

            for fname in ["nav.day.bin", "return.day.bin", "volume.day.bin"]:
                fpath = product_dir / fname
                if not fpath.exists():
                    logger.warning(f"[Qlib Dump] Missing file: {fpath}")
                    n_errors += 1
                    continue

                actual_bytes = fpath.stat().st_size
                if actual_bytes != expected_bytes:
                    logger.warning(
                        f"[Qlib Dump] Size mismatch {fpath}: "
                        f"expected {expected_bytes}, got {actual_bytes}"
                    )
                    n_errors += 1

            n_verified += 1

        logger.info(f"[Qlib Dump] Verified: {n_verified} products, {n_errors} errors")
        return n_errors == 0


def dump_sqlite_to_qlib(
    db_path: str,
    output_dir: str = "~/.qlib/qlib_data/wmp_data",
    bank_filter: Optional[str] = None,
    min_days: int = 180,
    theta0_threshold: float = 0.10,
    verify: bool = True,
) -> Dict:
    """
    Convenience function to dump SQLite to Qlib format.

    Args:
        db_path: Path to SQLite database
        output_dir: Output directory
        bank_filter: Filter by bank name (None = all banks)
        min_days: Minimum data days required
        theta0_threshold: GLM theta_0 threshold for liquidity filter
        verify: Whether to verify after export

    Returns:
        Statistics dict
    """
    dumper = QlibDataDumper(
        db_path=db_path,
        output_dir=output_dir,
        min_days=min_days,
        theta0_threshold=theta0_threshold,
    )

    stats = dumper.dump(bank_filter=bank_filter)

    if verify:
        success = dumper.verify()
        stats["verification_passed"] = success

    return stats


def main():
    parser = argparse.ArgumentParser(description="Dump SQLite NAV data to Qlib format")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database (default: aifinance.sqlite3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="~/.qlib/qlib_data/wmp_data",
        help="Output directory for Qlib data",
    )
    parser.add_argument(
        "--bank",
        type=str,
        default=None,
        help="Filter by bank name (e.g., 中信银行)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=180,
        help="Minimum data days required (default: 180)",
    )
    parser.add_argument(
        "--theta0",
        type=float,
        default=0.10,
        help="GLM theta_0 threshold for liquidity filter (default: 0.10)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data after export",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Find database
    if args.db:
        db_path = args.db
    else:
        project_root = find_project_root()
        db_path = project_root / "aifinance.sqlite3"

    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Run export
    stats = dump_sqlite_to_qlib(
        db_path=str(db_path),
        output_dir=args.output,
        bank_filter=args.bank,
        min_days=args.min_days,
        theta0_threshold=args.theta0,
        verify=args.verify,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Qlib Data Export Summary")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    if stats.get("verification_passed", True):
        print("Export completed successfully!")
        sys.exit(0)
    else:
        print("Export completed with verification errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
