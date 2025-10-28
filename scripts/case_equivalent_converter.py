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

class CaseEquivalentConverter:
    """
    Centralized utility for converting all volume calculations to case equivalent units.
    
    This class ensures consistent volume calculations across all analysis modules
    by converting mixed case/each quantities to standardized case equivalent volumes.
    """
    
    def __init__(self, sku_master=None, default_case_config=1):
        """
        Initialize the CaseEquivalentConverter.
        
        Args:
            sku_master (pandas.DataFrame, optional): SKU master data with Case Config
            default_case_config (int): Default eaches per case if SKU master unavailable
        """
        self.sku_master = sku_master
        self.default_case_config = default_case_config
    
    def add_case_equivalent_columns(self, order_data, sku_master=None, default_case_config=None):
        """
        Add case equivalent volume columns to order data.
        
        Args:
            order_data (pandas.DataFrame): Order data with Sku Code, Qty in Cases, Qty in Eaches
            sku_master (pandas.DataFrame, optional): SKU master with Case Config data (overrides instance)
            default_case_config (int): Default eaches per case if SKU master unavailable (overrides instance)
            
        Returns:
            pandas.DataFrame: Enhanced order data with case equivalent columns
        """
        if order_data is None or order_data.empty:
            return order_data
        
        # Use instance variables if parameters not provided
        sku_master = sku_master if sku_master is not None else self.sku_master
        default_case_config = default_case_config if default_case_config is not None else self.default_case_config
        
        # Create copy to avoid modifying original data
        enhanced_data = order_data.copy()
        
        if sku_master is not None and not sku_master.empty:
            # Merge with SKU master to get Case Config, Pallet Fit, and Category
            merge_columns = ['Sku Code', 'Case Config', 'Pallet Fit']
            if 'Category' in sku_master.columns:
                merge_columns.append('Category')
            sku_config = sku_master[merge_columns].copy()
            
            # ✅ UPDATED: Data is now pre-standardized by data_loader, so direct merge is safe
            enhanced_data = enhanced_data.merge(sku_config, on='Sku Code', how='left')
            
            # Fill missing Case Config and Pallet Fit with defaults
            enhanced_data['Case Config'] = enhanced_data['Case Config'].fillna(default_case_config)
            enhanced_data['Pallet Fit'] = enhanced_data['Pallet Fit'].fillna(1)  # Default 1 case per pallet if missing
        else:
            # Use default case config for all SKUs
            enhanced_data['Case Config'] = default_case_config
            enhanced_data['Pallet Fit'] = 1  # Default 1 case per pallet
            warnings.warn(f"SKU master data not available. Using default case config of {default_case_config} and pallet fit of 1.")
        
        # Calculate case equivalent from eaches
        enhanced_data['Case_Equivalent_From_Eaches'] = (
            enhanced_data['Qty in Eaches'] / enhanced_data['Case Config']
        ).round(4)
        
        # Calculate total case equivalent volume
        enhanced_data['Case_Equivalent_Volume'] = (
            enhanced_data['Qty in Cases'] + enhanced_data['Case_Equivalent_From_Eaches']
        ).round(4)
        
        # ✅ NEW: Calculate pallet equivalent volume
        enhanced_data['Pallet_Equivalent_Volume'] = (
            enhanced_data['Case_Equivalent_Volume'] / enhanced_data['Pallet Fit']
        ).round(4)
        
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
        
        # Use instance variables if parameters not provided
        sku_master = sku_master if sku_master is not None else self.sku_master
        default_case_config = default_case_config if default_case_config is not None else self.default_case_config
        
        enhanced_data = self.add_case_equivalent_columns(
            order_data, sku_master, default_case_config
        )
        
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
        
        # Use instance variables if parameters not provided
        sku_master = sku_master if sku_master is not None else self.sku_master
        default_case_config = default_case_config if default_case_config is not None else self.default_case_config
        
        enhanced_data = self.add_case_equivalent_columns(
            order_data, sku_master, default_case_config
        )
        
        return enhanced_data['Pallet_Equivalent_Volume'].sum()
    
    def get_case_equivalent_for_picks(self, order_data, sku_master=None, pick_type=None, default_case_config=None):
        """
        Get case equivalent volume for specific pick types.
        
        Args:
            order_data (pandas.DataFrame): Order data with pick type classification
            sku_master (pandas.DataFrame, optional): SKU master data
            pick_type (str, optional): 'Each_Pick', 'Case_Pick', or None for all
            default_case_config (int): Default eaches per case
            
        Returns:
            dict: Case equivalent volumes by category
        """
        if order_data is None or order_data.empty:
            return {'overall': 0.0, 'each_picks': 0.0, 'case_picks': 0.0}
        
        # Add case equivalent columns
        enhanced_data = self.add_case_equivalent_columns(
            order_data, sku_master, default_case_config
        )
        
        # Add pick type classification if not present
        if 'Pick_Type' not in enhanced_data.columns:
            enhanced_data = self._classify_pick_types(enhanced_data)
        
        # Calculate volumes by pick type
        overall_volume = enhanced_data['Case_Equivalent_Volume'].sum()
        
        each_picks = enhanced_data[enhanced_data['Pick_Type'] == 'Each_Pick']
        each_volume = each_picks['Case_Equivalent_Volume'].sum() if len(each_picks) > 0 else 0.0
        
        case_picks = enhanced_data[enhanced_data['Pick_Type'] == 'Case_Pick']
        case_volume = case_picks['Case_Equivalent_Volume'].sum() if len(case_picks) > 0 else 0.0
        
        return {
            'overall': round(overall_volume, 4),
            'each_picks': round(each_volume, 4),
            'case_picks': round(case_volume, 4)
        }
    
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
        enhanced_order_data = self.add_case_equivalent_columns(
            order_data, sku_master, default_case_config
        )
        
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
        enhanced_data = self.add_case_equivalent_columns(
            order_data, sku_master, default_case_config
        )
        
        # Calculate SKU-level metrics
        sku_metrics = enhanced_data.groupby('Sku Code').agg({
            'Date': ['count', 'nunique'],
            'Order No.': 'nunique',
            'Qty in Cases': ['sum', 'mean'],
            'Qty in Eaches': ['sum', 'mean'],
            'Case_Equivalent_Volume': ['sum', 'mean'],  # ← Case equivalent metrics
            'Pallet_Equivalent_Volume': ['sum', 'mean'],  # ← NEW: Pallet equivalent metrics
            'Case Config': 'first',
            'Pallet Fit': 'first'  # ← NEW: Include pallet fit
        }).round(4)
        
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
            'Pallet_Equivalent_Volume_sum': 'Total_Pallet_Equivalent_Volume',  # ← NEW
            'Pallet_Equivalent_Volume_mean': 'Avg_Pallet_Equivalent_Per_Line',  # ← NEW
            'Case Config_first': 'Case_Config',
            'Pallet Fit_first': 'Pallet_Fit'  # ← NEW
        }
        sku_metrics = sku_metrics.rename(columns=column_mapping)
        
        return sku_metrics
    
    def _classify_pick_types(self, order_data):
        """
        Internal method to classify pick types.
        
        Args:
            order_data (pandas.DataFrame): Order data
            
        Returns:
            pandas.DataFrame: Order data with Pick_Type column
        """
        enhanced_data = order_data.copy()
        
        # Classification logic
        conditions = [
            (enhanced_data['Qty in Cases'] == 0) & (enhanced_data['Qty in Eaches'] > 0),  # Each Pick
            (enhanced_data['Qty in Cases'] > 0)  # Case Pick
        ]
        
        choices = ['Each_Pick', 'Case_Pick']
        
        enhanced_data['Pick_Type'] = np.select(conditions, choices, default='Mixed_Pick')
        
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
        enhanced_data = self.add_case_equivalent_columns(
            order_data, sku_master, default_case_config
        )
        
        # Get sample records for validation
        sample_data = enhanced_data.head(sample_size)[
            ['Sku Code', 'Qty in Cases', 'Qty in Eaches', 'Case Config', 'Pallet Fit',
             'Case_Equivalent_From_Eaches', 'Case_Equivalent_Volume', 'Pallet_Equivalent_Volume']
        ]
        
        # Calculate totals
        total_cases = order_data['Qty in Cases'].sum()
        total_eaches = order_data['Qty in Eaches'].sum()
        total_case_equivalent = enhanced_data['Case_Equivalent_Volume'].sum()
        
        # Validation summary
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