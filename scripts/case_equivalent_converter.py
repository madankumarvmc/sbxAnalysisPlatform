"""
Case Equivalent Volume Converter Utility

PURPOSE:
This utility provides centralized, standardized volume calculations for the entire warehouse analysis platform.
All volume calculations across the project use case equivalent units for consistency.

BUSINESS RULE:
- Volume = Case Equivalent Volume
- 1 Case = 1 Case Equivalent
- X Eaches = X ÷ Case_Config Case Equivalent
- Total Volume = Cases + (Eaches ÷ Case_Config)

FOR BEGINNERS:
- This ensures all volume metrics are comparable across different SKUs
- Eliminates mixing of cases and eaches in calculations
- Provides accurate warehouse planning metrics
"""

import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


class CaseEquivalentConverter:
    """
    Centralized utility for converting all volume calculations to case equivalent units.

    This class ensures consistent volume calculations across all analysis modules
    by converting mixed case/each quantities to standardized case equivalent volumes.
    """

    def __init__(self, sku_master=None, default_case_config=1, default_pallet_fit=None):
        """
        Initialize the CaseEquivalentConverter.

        Args:
            sku_master (pandas.DataFrame, optional): SKU master data with Case Config
            default_case_config (int): Default eaches per case if SKU master unavailable
            default_pallet_fit (int, optional): Fallback cases-per-pallet when SKU master is missing
                Pallet Fit. Defaults to config.DEFAULT_PALLET_FIT if not provided.
        """
        self.sku_master = sku_master
        self.default_case_config = default_case_config
        # If not provided, fall back to the module-level config constant.
        # Analyzers should pass this from their analysis_config for per-session overrides.
        self.default_pallet_fit = default_pallet_fit if default_pallet_fit is not None else config.DEFAULT_PALLET_FIT

    def add_case_equivalent_columns(self, order_data, sku_master=None, default_case_config=None,
                                    default_pallet_fit=None):
        """
        Add case equivalent volume columns to order data.

        Args:
            order_data (pandas.DataFrame): Order data with Sku Code, Qty in Cases, Qty in Eaches
            sku_master (pandas.DataFrame, optional): SKU master with Case Config data (overrides instance)
            default_case_config (int): Default eaches per case if SKU master unavailable (overrides instance)
            default_pallet_fit (int): Fallback cases per pallet when missing from SKU master.
                                      Defaults to config.DEFAULT_PALLET_FIT.

        Returns:
            pandas.DataFrame: Enhanced order data with case equivalent columns
        """
        if order_data is None or order_data.empty:
            return order_data

        # Resolve parameters: explicit argument > instance default > config constant
        sku_master = sku_master if sku_master is not None else self.sku_master
        default_case_config = default_case_config if default_case_config is not None else self.default_case_config
        default_pallet_fit = default_pallet_fit if default_pallet_fit is not None else self.default_pallet_fit

        # Create copy to avoid modifying original data
        enhanced_data = order_data.copy()

        if sku_master is not None and not sku_master.empty:
            # Build the merge config — only include columns that exist in the master
            merge_columns = ['Sku Code', 'Case Config', 'Pallet Fit']
            if 'Category' in sku_master.columns:
                merge_columns.append('Category')
            sku_config = sku_master[merge_columns].copy()

            # FIX (BUG 6): Drop columns that would cause _x/_y suffixes on merge
            cols_to_drop = [c for c in ['Case Config', 'Pallet Fit', 'Category'] if c in enhanced_data.columns]
            if cols_to_drop:
                enhanced_data = enhanced_data.drop(columns=cols_to_drop)

            # Data is pre-standardized by data_loader, so direct merge is safe
            enhanced_data = enhanced_data.merge(sku_config, on='Sku Code', how='left')

            # Fill missing Case Config with default; track how many were missing
            missing_cc = enhanced_data['Case Config'].isna().sum()
            if missing_cc > 0:
                warnings.warn(
                    f"{missing_cc} row(s) have no Case Config in SKU master — "
                    f"using default of {default_case_config} each(es) per case."
                )
            enhanced_data['Case Config'] = enhanced_data['Case Config'].fillna(default_case_config)

            # Fill missing Pallet Fit with configurable default (not 1)
            missing_pf = enhanced_data['Pallet Fit'].isna().sum()
            if missing_pf > 0:
                warnings.warn(
                    f"{missing_pf} row(s) have no Pallet Fit in SKU master — "
                    f"using fallback of {default_pallet_fit} cases/pallet. "
                    f"Update your SKU master for accurate pallet calculations."
                )
            enhanced_data['Pallet Fit'] = enhanced_data['Pallet Fit'].fillna(default_pallet_fit)
        else:
            # No SKU master available — use defaults for all rows
            enhanced_data['Case Config'] = default_case_config
            enhanced_data['Pallet Fit'] = default_pallet_fit
            warnings.warn(
                f"SKU master data not available. Using default case config of {default_case_config} "
                f"and pallet fit of {default_pallet_fit} (cases/pallet) for all SKUs."
            )

        # FIX (BUG 4): Guard against zero or negative Case Config before division
        invalid_cc = enhanced_data['Case Config'] <= 0
        if invalid_cc.any():
            warnings.warn(
                f"{invalid_cc.sum()} row(s) have invalid Case Config (≤0) — "
                f"replacing with default of {default_case_config}."
            )
            enhanced_data.loc[invalid_cc, 'Case Config'] = default_case_config

        # FIX (BUG 5): Warn about negative quantities (keep them — could be returns/credits)
        negative_qty = (enhanced_data['Qty in Cases'] < 0) | (enhanced_data['Qty in Eaches'] < 0)
        if negative_qty.any():
            warnings.warn(
                f"{negative_qty.sum()} row(s) have negative quantities (likely returns/credits). "
                f"Included as-is in volume calculations."
            )

        # FIX (BUG 3): No intermediate rounding — compute without .round() on intermediates
        enhanced_data['Case_Equivalent_From_Eaches'] = (
            enhanced_data['Qty in Eaches'] / enhanced_data['Case Config']
        )

        enhanced_data['Case_Equivalent_Volume'] = (
            enhanced_data['Qty in Cases'] + enhanced_data['Case_Equivalent_From_Eaches']
        )

        enhanced_data['Pallet_Equivalent_Volume'] = (
            enhanced_data['Case_Equivalent_Volume'] / enhanced_data['Pallet Fit']
        )

        return enhanced_data

    def calculate_total_case_equivalent_volume(self, order_data, sku_master=None, default_case_config=None):
        """
        Calculate total case equivalent volume for entire dataset.

        Args:
            order_data (pandas.DataFrame): Order data
            sku_master (pandas.DataFrame, optional): SKU master data (overrides instance)
            default_case_config (int): Default eaches per case (overrides instance)

        Returns:
            float: Total case equivalent volume
        """
        if order_data is None or order_data.empty:
            return 0.0

        sku_master = sku_master if sku_master is not None else self.sku_master
        default_case_config = default_case_config if default_case_config is not None else self.default_case_config

        enhanced_data = self.add_case_equivalent_columns(order_data, sku_master, default_case_config)
        return enhanced_data['Case_Equivalent_Volume'].sum()

    def calculate_total_pallet_equivalent_volume(self, order_data, sku_master=None, default_case_config=None):
        """
        Calculate total pallet equivalent volume for entire dataset.

        Args:
            order_data (pandas.DataFrame): Order data
            sku_master (pandas.DataFrame, optional): SKU master data (overrides instance)
            default_case_config (int): Default eaches per case (overrides instance)

        Returns:
            float: Total pallet equivalent volume
        """
        if order_data is None or order_data.empty:
            return 0.0

        sku_master = sku_master if sku_master is not None else self.sku_master
        default_case_config = default_case_config if default_case_config is not None else self.default_case_config

        enhanced_data = self.add_case_equivalent_columns(order_data, sku_master, default_case_config)
        return enhanced_data['Pallet_Equivalent_Volume'].sum()

    def convert_daily_aggregation_to_case_equivalent(self, daily_data, order_data, sku_master=None, default_case_config=None):
        """
        Convert daily aggregated data to include case equivalent volumes.

        Args:
            daily_data (pandas.DataFrame): Daily aggregated data
            order_data (pandas.DataFrame): Raw order data for re-aggregation
            sku_master (pandas.DataFrame, optional): SKU master data
            default_case_config (int): Default eaches per case

        Returns:
            pandas.DataFrame: Daily data with case equivalent volume column
        """
        if order_data is None or order_data.empty:
            return daily_data

        # Add case equivalent columns to order data
        enhanced_order_data = self.add_case_equivalent_columns(order_data, sku_master, default_case_config)

        # Re-aggregate by date to get case equivalent daily volumes
        daily_case_equivalent = enhanced_order_data.groupby('Date').agg({
            'Case_Equivalent_Volume': 'sum'
        }).reset_index()
        daily_case_equivalent.columns = ['Date', 'Daily_Case_Equivalent_Volume']

        # Merge with existing daily data
        if daily_data is not None and not daily_data.empty:
            enhanced_daily = daily_data.merge(daily_case_equivalent, on='Date', how='left')
            enhanced_daily['Daily_Case_Equivalent_Volume'] = enhanced_daily['Daily_Case_Equivalent_Volume'].fillna(0)
        else:
            enhanced_daily = daily_case_equivalent

        return enhanced_daily

    def get_sku_case_equivalent_metrics(self, order_data, sku_master=None, default_case_config=None):
        """
        Calculate SKU-level case equivalent metrics.

        Args:
            order_data (pandas.DataFrame): Order data
            sku_master (pandas.DataFrame, optional): SKU master data
            default_case_config (int): Default eaches per case

        Returns:
            pandas.DataFrame: SKU metrics with case equivalent calculations
        """
        if order_data is None or order_data.empty:
            return pd.DataFrame()

        # Add case equivalent columns
        enhanced_data = self.add_case_equivalent_columns(order_data, sku_master, default_case_config)

        # FIX (BUG 3): No .round(4) on groupby aggregation
        sku_metrics = enhanced_data.groupby('Sku Code').agg({
            'Date': ['count', 'nunique'],
            'Order No.': 'nunique',
            'Qty in Cases': ['sum', 'mean'],
            'Qty in Eaches': ['sum', 'mean'],
            'Case_Equivalent_Volume': ['sum', 'mean'],
            'Pallet_Equivalent_Volume': ['sum', 'mean'],
            'Case Config': 'first',
            'Pallet Fit': 'first'
        })

        # Flatten column names
        sku_metrics.columns = ['_'.join(col).strip() for col in sku_metrics.columns]
        sku_metrics = sku_metrics.reset_index()

        # Rename columns for clarity
        column_mapping = {
            'Date_count': 'Total_Order_Lines',
            'Date_nunique': 'Days_Ordered',
            'Order No._nunique': 'Unique_Orders',
            'Qty in Cases_sum': 'Total_Cases',
            'Qty in Cases_mean': 'Avg_Cases_Per_Line',
            'Qty in Eaches_sum': 'Total_Eaches',
            'Qty in Eaches_mean': 'Avg_Eaches_Per_Line',
            'Case_Equivalent_Volume_sum': 'Total_Case_Equivalent_Volume',
            'Case_Equivalent_Volume_mean': 'Avg_Case_Equivalent_Per_Line',
            'Pallet_Equivalent_Volume_sum': 'Total_Pallet_Equivalent_Volume',
            'Pallet_Equivalent_Volume_mean': 'Avg_Pallet_Equivalent_Per_Line',
            'Case Config_first': 'Case_Config',
            'Pallet Fit_first': 'Pallet_Fit'
        }
        sku_metrics = sku_metrics.rename(columns=column_mapping)

        return sku_metrics

    def _classify_pick_types(self, order_data):
        """
        Classify each order line by pick type based on case and each quantities.

        Pick type definitions:
        - Each_Pick  : Eaches only (Cases = 0, Eaches > 0)
        - Case_Pick  : Cases only  (Cases > 0, Eaches = 0)
        - Mixed_Pick : Both cases and eaches on the same line (Cases > 0, Eaches > 0)
        - No_Volume  : Both quantities are zero (data quality issue)

        Args:
            order_data (pandas.DataFrame): Order data

        Returns:
            pandas.DataFrame: Order data with Pick_Type column
        """
        enhanced_data = order_data.copy()

        # FIX (BUG 1): Four explicit, non-overlapping conditions
        conditions = [
            (enhanced_data['Qty in Cases'] == 0) & (enhanced_data['Qty in Eaches'] > 0),   # Each_Pick
            (enhanced_data['Qty in Cases'] > 0)  & (enhanced_data['Qty in Eaches'] == 0),   # Case_Pick
            (enhanced_data['Qty in Cases'] > 0)  & (enhanced_data['Qty in Eaches'] > 0),    # Mixed_Pick
        ]
        choices = ['Each_Pick', 'Case_Pick', 'Mixed_Pick']

        # default='No_Volume' for lines where both quantities are zero
        enhanced_data['Pick_Type'] = np.select(conditions, choices, default='No_Volume')

        return enhanced_data

    def validate_case_equivalent_calculation(self, order_data, sku_master=None, default_case_config=None, sample_size=5):
        """
        Validate case equivalent calculations with sample data for debugging.

        Args:
            order_data (pandas.DataFrame): Order data
            sku_master (pandas.DataFrame, optional): SKU master data
            default_case_config (int): Default eaches per case
            sample_size (int): Number of sample records to show

        Returns:
            dict: Validation results and sample calculations
        """
        if order_data is None or order_data.empty:
            return {'status': 'No data to validate'}

        # Add case equivalent columns
        enhanced_data = self.add_case_equivalent_columns(order_data, sku_master, default_case_config)

        # FIX (IMPROVEMENT): Use random sample to catch mid-dataset edge cases
        n = min(sample_size, len(enhanced_data))
        sample_data = enhanced_data.sample(n, random_state=42)[
            ['Sku Code', 'Qty in Cases', 'Qty in Eaches', 'Case Config', 'Pallet Fit',
             'Case_Equivalent_From_Eaches', 'Case_Equivalent_Volume', 'Pallet_Equivalent_Volume']
        ]

        # Calculate totals
        total_cases = order_data['Qty in Cases'].sum()
        total_eaches = order_data['Qty in Eaches'].sum()
        total_case_equivalent = enhanced_data['Case_Equivalent_Volume'].sum()

        validation_results = {
            'status': 'Success',
            'total_records': len(order_data),
            'total_cases': total_cases,
            'total_eaches': total_eaches,
            'total_case_equivalent_volume': round(total_case_equivalent, 4),
            'sku_master_available': sku_master is not None and not sku_master.empty,
            'default_case_config_used': default_case_config,
            'sample_calculations': sample_data.to_dict('records') if not sample_data.empty else []
        }

        return validation_results


# Test function for standalone execution
if __name__ == "__main__":
    print("CaseEquivalentConverter utility - ready for use")
    print("This utility provides standardized case equivalent volume calculations.")
    print("Use within analysis modules for consistent volume metrics.")
