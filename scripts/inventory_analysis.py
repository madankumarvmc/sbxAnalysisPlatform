"""
Inventory Analysis Module for Warehouse Analysis Tool

PURPOSE:
This module analyzes inventory data to provide insights on stock levels,
inventory trends, and warehouse space utilization.

FEATURES:
- Daily inventory tracking by SKU
- Statistical analysis (max, 90th percentile)
- Warehouse space utilization metrics
- Integration with ABC classification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Optional, Dict, Any, List

class InventoryAnalyzer:
    """
    Comprehensive inventory analysis for warehouse optimization.
    
    This class analyzes inventory data to provide:
    - SKU-wise daily inventory levels
    - Inventory trends and patterns
    - Statistical measures for capacity planning
    - Warehouse space utilization metrics
    """
    
    def __init__(self, inventory_data: pd.DataFrame, 
                 sku_master: Optional[pd.DataFrame] = None,
                 order_data: Optional[pd.DataFrame] = None,
                 analysis_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the InventoryAnalyzer.
        
        Args:
            inventory_data (pd.DataFrame): Inventory data with daily stock levels
            sku_master (pd.DataFrame, optional): SKU master data for classifications
            order_data (pd.DataFrame, optional): Order data for ABC classification
            analysis_config (dict, optional): Configuration parameters
        """
        self.inventory_data = inventory_data.copy()
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.order_data = order_data.copy() if order_data is not None else None
        self.config = analysis_config or {}
        
        # Extract configuration parameters
        self.date_range = self.config.get('DATE_RANGE', {})
        self.inventory_params = self.config.get('INVENTORY_PARAMS', {})
        
        # Apply date filtering if specified
        if self.date_range.get('START_DATE') or self.date_range.get('END_DATE'):
            self.inventory_data = self._filter_by_date_range(self.inventory_data)
        
        # Analysis results containers
        self.sku_inventory_matrix = None
        self.daily_summary = None
        self.inventory_statistics = None
        
        print(f"InventoryAnalyzer initialized with {len(self.inventory_data)} inventory records")
        if self.sku_master is not None:
            print(f"SKU Master available: {len(self.sku_master)} SKUs")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete inventory analysis pipeline.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        print("🔄 Running complete inventory analysis...")
        
        results = {
            'success': True,
            'analysis_date': datetime.now(),
            'data_summary': self._get_data_summary(),
            'sku_inventory_matrix': self.analyze_sku_inventory_matrix(),
            'daily_summary': self.analyze_daily_summary(),
            'inventory_statistics': self.analyze_inventory_statistics()
        }
        
        print("✅ Inventory analysis completed successfully")
        return results
    
    def analyze_sku_inventory_matrix(self) -> pd.DataFrame:
        """
        Create SKU inventory matrix with daily levels and statistics.
        
        Returns:
            pd.DataFrame: SKU inventory matrix with columns for each date and statistics
        """
        print("📊 Creating SKU inventory matrix...")
        
        # Pivot inventory data to get SKU x Date matrix
        inventory_pivot = self.inventory_data.pivot_table(
            values='Total Stock in Cases (In Base Unit of Measure)',
            index='SKU ID',
            columns='Calendar Day',
            aggfunc='sum',
            fill_value=0
        )
        
        # Create the matrix dataframe
        matrix_df = pd.DataFrame()
        matrix_df['SKU Code'] = inventory_pivot.index
        
        # Add daily columns (format dates as DD.MM.YYYY)
        for date_col in sorted(inventory_pivot.columns):
            date_str = date_col.strftime('%d.%m.%Y')
            matrix_df[date_str] = inventory_pivot[date_col].values
        
        # Calculate statistics
        daily_values = inventory_pivot.values
        matrix_df['Grand Total'] = daily_values.sum(axis=1)
        matrix_df['Max'] = daily_values.max(axis=1)
        matrix_df['90%ile'] = np.percentile(daily_values, 90, axis=1)
        
        # Add SKU master data if available
        if self.sku_master is not None:
            # Merge with SKU master for additional attributes
            sku_info = self.sku_master[['Sku Code', 'Pallet Fit', 'Category']].copy()
            sku_info.rename(columns={'Sku Code': 'SKU Code'}, inplace=True)
            matrix_df = matrix_df.merge(sku_info, on='SKU Code', how='left')
            
            # Calculate pallet-related metrics
            matrix_df['PalletFit'] = matrix_df['Pallet Fit'].fillna(180)  # Default pallet fit
            matrix_df['#Pallets'] = (matrix_df['Max'] / matrix_df['PalletFit']).apply(np.ceil)
        else:
            # Default values if SKU master not available
            matrix_df['PalletFit'] = 180
            matrix_df['#Pallets'] = (matrix_df['Max'] / 180).apply(np.ceil)
            matrix_df['Category'] = 'N/A'
        
        # Add ABC classification if order data is available
        if self.order_data is not None:
            abc_classification = self._calculate_abc_classification()
            matrix_df = matrix_df.merge(
                abc_classification[['SKU Code', 'ABC_Class']], 
                on='SKU Code', 
                how='left'
            )
            matrix_df.rename(columns={'ABC_Class': 'Class'}, inplace=True)
        else:
            matrix_df['Class'] = 'N/A'
        
        # Add placeholder columns for warehouse metrics (to be implemented later)
        matrix_df['BinType'] = 8  # Placeholder
        matrix_df['#Bins'] = (matrix_df['#Pallets'] * 2).astype(int)  # Placeholder calculation
        matrix_df['#Pallet Positions'] = (matrix_df['#Pallets'] * 1.2).astype(int)  # Placeholder
        matrix_df['Area (sq.m)'] = (matrix_df['#Pallets'] * 1.5).round(2)  # Placeholder
        
        # Reorder columns to match the required format
        date_columns = [col for col in matrix_df.columns if '.' in col and len(col.split('.')) == 3]
        fixed_columns = ['SKU Code'] + date_columns + [
            'Grand Total', 'Max', '90%ile', 'PalletFit', '#Pallets', 
            'Class', 'Category', 'BinType', '#Bins', '#Pallet Positions', 'Area (sq.m)'
        ]
        
        # Only include columns that exist
        final_columns = [col for col in fixed_columns if col in matrix_df.columns]
        matrix_df = matrix_df[final_columns]
        
        # Sort by Grand Total descending to show high-inventory SKUs first
        matrix_df = matrix_df.sort_values('Grand Total', ascending=False)
        
        self.sku_inventory_matrix = matrix_df
        print(f"  Created matrix for {len(matrix_df)} SKUs across {len(date_columns)} days")
        
        return matrix_df
    
    def analyze_daily_summary(self) -> pd.DataFrame:
        """
        Create daily inventory summary table.
        
        Returns:
            pd.DataFrame: Daily summary with total cases per date
        """
        print("📊 Creating daily inventory summary...")
        
        # Calculate daily totals
        daily_totals = self.inventory_data.groupby('Calendar Day').agg({
            'Total Stock in Cases (In Base Unit of Measure)': 'sum'
        }).reset_index()
        
        # Rename columns
        daily_totals.columns = ['Date', '#Cases']
        
        # Format date column
        daily_totals['Date'] = daily_totals['Date'].dt.strftime('%d.%m.%Y')
        
        # Calculate statistics
        max_cases = daily_totals['#Cases'].max()
        percentile_90 = np.percentile(daily_totals['#Cases'], 90)
        
        # Add summary rows
        summary_rows = pd.DataFrame([
            {'Date': '', '#Cases': ''},  # Empty row for separation
            {'Date': 'Max', '#Cases': f'{max_cases:,.0f}'},
            {'Date': '90%ile', '#Cases': f'{percentile_90:,.0f}'}
        ])
        
        # Combine daily data with summary
        daily_summary = pd.concat([daily_totals, summary_rows], ignore_index=True)
        
        self.daily_summary = daily_summary
        print(f"  Created daily summary for {len(daily_totals)} days")
        
        return daily_summary
    
    def analyze_inventory_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive inventory statistics.
        
        Returns:
            dict: Inventory statistics and metrics
        """
        print("📊 Calculating inventory statistics...")
        
        stats = {
            'total_skus': self.inventory_data['SKU ID'].nunique(),
            'date_range': {
                'start': self.inventory_data['Calendar Day'].min(),
                'end': self.inventory_data['Calendar Day'].max(),
                'days': (self.inventory_data['Calendar Day'].max() - 
                        self.inventory_data['Calendar Day'].min()).days + 1
            },
            'stock_levels': {
                'total_cases': self.inventory_data['Total Stock in Cases (In Base Unit of Measure)'].sum(),
                'avg_daily_cases': self.inventory_data.groupby('Calendar Day')['Total Stock in Cases (In Base Unit of Measure)'].sum().mean(),
                'max_daily_cases': self.inventory_data.groupby('Calendar Day')['Total Stock in Cases (In Base Unit of Measure)'].sum().max(),
                'min_daily_cases': self.inventory_data.groupby('Calendar Day')['Total Stock in Cases (In Base Unit of Measure)'].sum().min()
            },
            'zero_stock_analysis': {
                'skus_with_zero_stock': (self.inventory_data.groupby('SKU ID')['Total Stock in Cases (In Base Unit of Measure)'].sum() == 0).sum(),
                'days_with_stockouts': self._calculate_stockout_days()
            }
        }
        
        self.inventory_statistics = stats
        return stats
    
    def _calculate_abc_classification(self) -> pd.DataFrame:
        """
        Calculate ABC classification based on order data.
        
        Returns:
            pd.DataFrame: SKU classification data
        """
        if self.order_data is None:
            return pd.DataFrame()
        
        # Calculate total volume per SKU
        sku_volume = self.order_data.groupby('Sku Code').agg({
            'Qty in Cases': 'sum'
        }).reset_index()
        
        sku_volume.columns = ['SKU Code', 'Total_Volume']
        
        # Sort by volume and calculate cumulative percentage
        sku_volume = sku_volume.sort_values('Total_Volume', ascending=False)
        sku_volume['Cumulative_Volume'] = sku_volume['Total_Volume'].cumsum()
        sku_volume['Cumulative_Percentage'] = (sku_volume['Cumulative_Volume'] / 
                                               sku_volume['Total_Volume'].sum() * 100)
        
        # Assign ABC classes based on thresholds
        abc_a_threshold = self.config.get('ABC_THRESHOLDS', {}).get('A_THRESHOLD', 70)
        abc_b_threshold = self.config.get('ABC_THRESHOLDS', {}).get('B_THRESHOLD', 90)
        
        conditions = [
            sku_volume['Cumulative_Percentage'] <= abc_a_threshold,
            sku_volume['Cumulative_Percentage'] <= abc_b_threshold,
            sku_volume['Cumulative_Percentage'] > abc_b_threshold
        ]
        choices = ['A', 'B', 'C']
        
        sku_volume['ABC_Class'] = np.select(conditions, choices, default='C')
        
        return sku_volume[['SKU Code', 'ABC_Class']]
    
    def _calculate_stockout_days(self) -> int:
        """
        Calculate number of days with stockouts.
        
        Returns:
            int: Number of days with at least one SKU stocked out
        """
        # Group by date and check if any SKU has zero stock
        daily_stockouts = self.inventory_data.groupby('Calendar Day').apply(
            lambda x: (x['Total Stock in Cases (In Base Unit of Measure)'] == 0).any()
        )
        
        return daily_stockouts.sum()
    
    def _filter_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe by configured date range.
        
        Args:
            df (pd.DataFrame): Dataframe to filter
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        filtered_df = df.copy()
        
        if self.date_range.get('START_DATE'):
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Calendar Day']) >= self.date_range['START_DATE']]
        
        if self.date_range.get('END_DATE'):
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Calendar Day']) <= self.date_range['END_DATE']]
        
        print(f"Date filter applied: {len(df)} -> {len(filtered_df)} records")
        return filtered_df
    
    def _get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of the inventory data.
        
        Returns:
            dict: Data summary statistics
        """
        return {
            'total_records': len(self.inventory_data),
            'unique_skus': self.inventory_data['SKU ID'].nunique(),
            'date_range': {
                'start': self.inventory_data['Calendar Day'].min().strftime('%Y-%m-%d'),
                'end': self.inventory_data['Calendar Day'].max().strftime('%Y-%m-%d'),
                'days': self.inventory_data['Calendar Day'].nunique()
            },
            'total_stock_cases': self.inventory_data['Total Stock in Cases (In Base Unit of Measure)'].sum(),
            'total_stock_eaches': self.inventory_data['Total Stock in Pieces (In Base Unit of Measue)'].sum()
        }

# Test function for standalone execution
if __name__ == "__main__":
    print("InventoryAnalyzer module - ready for use")
    print("This module requires inventory data to function.")
    print("Use within the main application for proper functionality.")