"""
Receipt Analysis Module for Warehouse Analysis Tool V2

PURPOSE:
This module analyzes receipt (inbound) patterns and performance including:
- Receipt volume trends and patterns
- Supplier performance analysis
- Lead time analysis
- Receiving efficiency metrics
- Seasonal patterns in receipts

FOR BEGINNERS:
- Receipt data shows when goods arrive at the warehouse
- This analysis helps plan receiving capacity and dock scheduling
- It identifies supplier performance issues and seasonal patterns
- Results help optimize inbound operations and staffing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
from pathlib import Path

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
import config
from .case_equivalent_converter import CaseEquivalentConverter

class ReceiptAnalyzer:
    """
    Receipt pattern analysis class.
    
    This class analyzes receipt data to provide:
    - Inbound volume patterns and trends
    - Supplier performance metrics
    - Receiving efficiency analysis
    - Dock utilization patterns
    - Lead time analysis
    """
    
    def __init__(self, receipt_data, order_data=None, sku_master=None, analysis_config=None):
        """
        Initialize the ReceiptAnalyzer.
        
        Args:
            receipt_data (pandas.DataFrame): Cleaned receipt data from DataLoader
            order_data (pandas.DataFrame): Order data for lead time analysis (optional)
            sku_master (pandas.DataFrame): SKU master data for case equivalent calculations (optional)
            analysis_config (dict): Configuration parameters for analysis
        """
        self.receipt_data = receipt_data.copy()
        self.order_data = order_data.copy() if order_data is not None else None
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.config = analysis_config or {}
        
        # Initialize case equivalent converter
        self.converter = CaseEquivalentConverter(self.sku_master)
        
        # Set analysis parameters from config
        self.date_range = self.config.get('DATE_RANGE', {})
        
        # Filter data by date range if specified
        if self.date_range.get('START_DATE') or self.date_range.get('END_DATE'):
            self.receipt_data = self._filter_by_date_range(self.receipt_data)
            if self.order_data is not None:
                self.order_data = self._filter_by_date_range_orders(self.order_data)
        
        # Analysis results containers
        self.daily_patterns = None
        self.supplier_performance = None
        self.efficiency_metrics = None
        
        print(f"ReceiptAnalyzer initialized with {len(self.receipt_data)} receipt records")
        if self.order_data is not None:
            print(f"Order data available for lead time analysis: {len(self.order_data)} records")
    
    def _map_receipt_columns_for_converter(self, receipt_data_subset):
        """
        Map receipt data column names to format expected by CaseEquivalentConverter.
        
        Receipt data uses: 'SKU ID', 'Quantity in Cases', 'Quantity in Eaches'
        Converter expects: 'Sku Code', 'Qty in Cases', 'Qty in Eaches'
        
        Args:
            receipt_data_subset (pandas.DataFrame): Receipt data subset
            
        Returns:
            pandas.DataFrame: Receipt data with column names mapped for converter
        """
        mapped_data = receipt_data_subset.copy()
        
        # Map receipt column names to converter expected names
        column_mapping = {
            'SKU ID': 'Sku Code',
            'Quantity in Cases': 'Qty in Cases', 
            'Quantity in Eaches': 'Qty in Eaches'
        }
        
        mapped_data = mapped_data.rename(columns=column_mapping)
        
        return mapped_data
    
    def _add_case_equivalent_to_receipt_data(self, receipt_data_subset):
        """
        Add case equivalent volume columns to receipt data.
        
        Args:
            receipt_data_subset (pandas.DataFrame): Receipt data subset
            
        Returns:
            pandas.DataFrame: Receipt data enhanced with case equivalent columns
        """
        # Map column names for converter
        mapped_data = self._map_receipt_columns_for_converter(receipt_data_subset)
        
        # Add case equivalent columns using converter
        enhanced_data = self.converter.add_case_equivalent_columns(
            mapped_data, self.sku_master
        )
        
        # Map back to original receipt column names while keeping new case equivalent columns
        enhanced_data = enhanced_data.rename(columns={
            'Sku Code': 'SKU ID',
            'Qty in Cases': 'Quantity in Cases',
            'Qty in Eaches': 'Quantity in Eaches'
        })
        
        return enhanced_data
    
    def run_complete_analysis(self):
        """
        Run complete receipt analysis pipeline.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        print("🔄 Running complete receipt analysis...")
        
        results = {
            'success': True,
            'analysis_date': datetime.now(),
            'data_summary': self._get_data_summary(),
            'daily_patterns': self.analyze_daily_patterns(),
            'volume_analysis': self.analyze_volume_trends(),
            'supplier_performance': self.analyze_supplier_performance(),
            'efficiency_metrics': self.analyze_receiving_efficiency(),
            'dock_utilization': self.analyze_dock_utilization(),
            'sku_receipt_patterns': self.analyze_sku_patterns(),
            'recommendations': self.generate_recommendations()
        }
        
        # Add lead time analysis if order data is available
        if self.order_data is not None:
            results['lead_time_analysis'] = self.analyze_lead_times()
        
        print("✅ Receipt analysis completed successfully")
        return results
    
    def analyze_daily_patterns(self):
        """
        Analyze daily receipt patterns and volumes.
        
        Returns:
            dict: Daily pattern analysis results
        """
        print("📊 Analyzing daily receipt patterns...")
        
        # Convert receipt date to datetime if it's not already
        self.receipt_data['Receipt Date'] = pd.to_datetime(self.receipt_data['Receipt Date'], errors='coerce')
        
        # Add case equivalent volume to receipt data
        enhanced_receipt_data = self._add_case_equivalent_to_receipt_data(self.receipt_data)
        
        # Group by date to get daily summaries
        daily_receipts = enhanced_receipt_data.groupby('Receipt Date').agg({
            'SKU ID': 'nunique',
            'Shipment No': 'nunique',
            'Truck No': 'nunique',
            'Quantity in Cases': 'sum',
            'Quantity in Eaches': 'sum',
            'Case_Equivalent_Volume': 'sum'  # ← NEW: Case equivalent volume
        }).reset_index()
        
        daily_receipts.columns = ['Date', 'Daily_SKUs', 'Daily_Shipments', 'Daily_Trucks', 'Daily_Cases', 'Daily_Eaches', 'Daily_Case_Equivalent_Volume']
        
        # Add day of week analysis
        daily_receipts['Day_of_Week'] = daily_receipts['Date'].dt.day_name()
        daily_receipts['Week_Number'] = daily_receipts['Date'].dt.isocalendar().week
        daily_receipts['Month'] = daily_receipts['Date'].dt.month_name()
        
        # Calculate moving averages
        daily_receipts['Cases_7Day_MA'] = daily_receipts['Daily_Cases'].rolling(window=7, center=True).mean()
        daily_receipts['CaseEquiv_7Day_MA'] = daily_receipts['Daily_Case_Equivalent_Volume'].rolling(window=7, center=True).mean()  # ← NEW
        daily_receipts['Trucks_7Day_MA'] = daily_receipts['Daily_Trucks'].rolling(window=7, center=True).mean()
        
        # Day of week patterns
        dow_patterns = daily_receipts.groupby('Day_of_Week').agg({
            'Daily_Cases': ['mean', 'std', 'min', 'max'],
            'Daily_Case_Equivalent_Volume': ['mean', 'std', 'min', 'max'],  # ← NEW: Primary volume metric
            'Daily_Trucks': ['mean', 'std', 'min', 'max'],
            'Daily_Shipments': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        dow_patterns.columns = ['_'.join(col).strip() for col in dow_patterns.columns]
        dow_patterns = dow_patterns.reset_index()
        
        self.daily_patterns = daily_receipts
        
        return {
            'daily_data': daily_receipts,
            'day_of_week_patterns': dow_patterns,
            'total_receipt_days': len(daily_receipts),
            'avg_daily_cases': daily_receipts['Daily_Cases'].mean(),
            'avg_daily_case_equivalent_volume': daily_receipts['Daily_Case_Equivalent_Volume'].mean(),  # ← NEW: Primary volume metric
            'avg_daily_trucks': daily_receipts['Daily_Trucks'].mean(),
            'busiest_receipt_day': daily_receipts.loc[daily_receipts['Daily_Case_Equivalent_Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d'),  # ← UPDATED: Use case equivalent volume
            'peak_daily_volume': daily_receipts['Daily_Case_Equivalent_Volume'].max(),  # ← UPDATED: Use case equivalent volume
            'peak_daily_cases_legacy': daily_receipts['Daily_Cases'].max()  # ← LEGACY: Keep for backward compatibility
        }
    
    def analyze_volume_trends(self):
        """
        Analyze receipt volume trends over time.
        
        Returns:
            dict: Volume trend analysis results
        """
        print("📈 Analyzing receipt volume trends...")
        
        if self.daily_patterns is None:
            self.analyze_daily_patterns()
        
        daily_cases = self.daily_patterns['Daily_Cases']
        daily_case_equivalent = self.daily_patterns['Daily_Case_Equivalent_Volume']  # ← NEW: Primary volume metric
        daily_trucks = self.daily_patterns['Daily_Trucks']
        
        volume_stats = {
            'case_equivalent_volume': {  # ← NEW: Primary volume metrics
                'total': daily_case_equivalent.sum(),
                'mean': daily_case_equivalent.mean(),
                'median': daily_case_equivalent.median(),
                'std': daily_case_equivalent.std(),
                'min': daily_case_equivalent.min(),
                'max': daily_case_equivalent.max(),
                'cv': daily_case_equivalent.std() / daily_case_equivalent.mean() if daily_case_equivalent.mean() > 0 else 0
            },
            'cases_legacy': {  # ← LEGACY: Keep for backward compatibility
                'total': daily_cases.sum(),
                'mean': daily_cases.mean(),
                'median': daily_cases.median(),
                'std': daily_cases.std(),
                'min': daily_cases.min(),
                'max': daily_cases.max(),
                'cv': daily_cases.std() / daily_cases.mean() if daily_cases.mean() > 0 else 0
            },
            'trucks': {
                'total': daily_trucks.sum(),
                'mean': daily_trucks.mean(),
                'median': daily_trucks.median(),
                'std': daily_trucks.std(),
                'min': daily_trucks.min(),
                'max': daily_trucks.max(),
                'cv': daily_trucks.std() / daily_trucks.mean() if daily_trucks.mean() > 0 else 0
            }
        }
        
        # Calculate monthly trends
        monthly_data = self.daily_patterns.groupby('Month').agg({
            'Daily_Case_Equivalent_Volume': ['sum', 'mean', 'count'],  # ← NEW: Primary volume metric
            'Daily_Cases': ['sum', 'mean'],  # ← LEGACY: Keep for backward compatibility
            'Daily_Trucks': ['sum', 'mean']
        }).round(2)
        
        # Flatten column names
        monthly_data.columns = ['_'.join(col).strip() for col in monthly_data.columns]
        monthly_data = monthly_data.reset_index()
        
        return {
            'volume_statistics': volume_stats,
            'monthly_trends': monthly_data,
            'trend_direction': self._determine_trend_direction(),
            'seasonality_detected': self._detect_seasonality()
        }
    
    def analyze_supplier_performance(self):
        """
        Analyze supplier/truck performance patterns.
        
        Returns:
            dict: Supplier performance analysis results
        """
        print("🚛 Analyzing supplier/truck performance...")
        
        # Add case equivalent volume to receipt data for truck analysis
        enhanced_receipt_data = self._add_case_equivalent_to_receipt_data(self.receipt_data)
        
        # Truck-level analysis
        truck_performance = enhanced_receipt_data.groupby('Truck No').agg({
            'Receipt Date': ['count', 'nunique'],
            'SKU ID': 'nunique',
            'Shipment No': 'nunique',
            'Quantity in Cases': ['sum', 'mean', 'std'],
            'Quantity in Eaches': ['sum', 'mean'],
            'Case_Equivalent_Volume': ['sum', 'mean', 'std']  # ← NEW: Case equivalent volume metrics
        }).round(2)
        
        # Flatten column names
        truck_performance.columns = ['_'.join(col).strip() for col in truck_performance.columns]
        truck_performance = truck_performance.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'Receipt Date_count': 'Total_Deliveries',
            'Receipt Date_nunique': 'Active_Days',
            'SKU ID_nunique': 'Unique_SKUs',
            'Shipment No_nunique': 'Unique_Shipments',
            'Quantity in Cases_sum': 'Total_Cases',
            'Quantity in Cases_mean': 'Avg_Cases_Per_Delivery',
            'Quantity in Cases_std': 'Cases_Std_Dev',
            'Quantity in Eaches_sum': 'Total_Eaches',
            'Quantity in Eaches_mean': 'Avg_Eaches_Per_Delivery',
            'Case_Equivalent_Volume_sum': 'Total_Case_Equivalent_Volume',  # ← NEW: Primary volume metric
            'Case_Equivalent_Volume_mean': 'Avg_Case_Equivalent_Per_Delivery',  # ← NEW
            'Case_Equivalent_Volume_std': 'Case_Equivalent_Std_Dev'  # ← NEW
        }
        truck_performance = truck_performance.rename(columns=column_mapping)
        
        # Calculate performance metrics
        truck_performance['Deliveries_Per_Day'] = (truck_performance['Total_Deliveries'] / truck_performance['Active_Days']).round(2)
        truck_performance['Cases_CV'] = (truck_performance['Cases_Std_Dev'] / truck_performance['Avg_Cases_Per_Delivery']).round(2)
        truck_performance['Cases_CV'] = truck_performance['Cases_CV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        truck_performance['Case_Equivalent_CV'] = (truck_performance['Case_Equivalent_Std_Dev'] / truck_performance['Avg_Case_Equivalent_Per_Delivery']).round(2)  # ← NEW: Primary variability metric
        truck_performance['Case_Equivalent_CV'] = truck_performance['Case_Equivalent_CV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Sort by total case equivalent volume (primary metric)
        truck_performance = truck_performance.sort_values('Total_Case_Equivalent_Volume', ascending=False).reset_index(drop=True)
        
        # Classify truck performance using case equivalent volume (primary metric)
        def classify_truck_performance(row):
            if row['Total_Case_Equivalent_Volume'] >= truck_performance['Total_Case_Equivalent_Volume'].quantile(0.8):
                return 'High Volume'
            elif row['Case_Equivalent_CV'] <= 0.5:
                return 'Consistent'
            elif row['Case_Equivalent_CV'] >= 1.5:
                return 'Variable'
            else:
                return 'Standard'
        
        truck_performance['Performance_Category'] = truck_performance.apply(classify_truck_performance, axis=1)
        
        # Performance summary by category
        performance_summary = truck_performance.groupby('Performance_Category').agg({
            'Truck No': 'count',
            'Total_Case_Equivalent_Volume': 'sum',  # ← NEW: Primary volume metric
            'Avg_Case_Equivalent_Per_Delivery': 'mean',  # ← NEW: Primary delivery metric
            'Case_Equivalent_CV': 'mean',  # ← NEW: Primary variability metric
            'Total_Cases': 'sum',  # ← LEGACY: Keep for backward compatibility
            'Avg_Cases_Per_Delivery': 'mean',  # ← LEGACY
            'Cases_CV': 'mean'  # ← LEGACY
        }).round(2)
        
        performance_summary.columns = ['Truck_Count', 'Total_Case_Equivalent_Volume', 'Avg_Case_Equivalent_Per_Delivery', 'Avg_Case_Equivalent_Variability', 'Total_Cases_Legacy', 'Avg_Cases_Per_Delivery_Legacy', 'Avg_Cases_Variability_Legacy']
        
        return {
            'truck_details': truck_performance,
            'performance_summary': performance_summary,
            'top_performing_trucks': truck_performance.head(10),
            'high_variability_trucks': truck_performance[truck_performance['Case_Equivalent_CV'] > 1.5],  # ← UPDATED: Use case equivalent variability
            'total_trucks': len(truck_performance)
        }
    
    def analyze_receiving_efficiency(self):
        """
        Analyze receiving dock efficiency metrics.
        
        Returns:
            dict: Receiving efficiency analysis results
        """
        print("⚡ Analyzing receiving efficiency...")
        
        # Daily efficiency metrics
        if self.daily_patterns is None:
            self.analyze_daily_patterns()
        
        daily_efficiency = self.daily_patterns.copy()
        
        # Calculate efficiency ratios
        daily_efficiency['Case_Equivalent_Per_Truck'] = (daily_efficiency['Daily_Case_Equivalent_Volume'] / daily_efficiency['Daily_Trucks']).round(2)  # ← NEW: Primary efficiency metric
        daily_efficiency['Cases_Per_Truck'] = (daily_efficiency['Daily_Cases'] / daily_efficiency['Daily_Trucks']).round(2)  # ← LEGACY: Keep for backward compatibility
        daily_efficiency['SKUs_Per_Truck'] = (daily_efficiency['Daily_SKUs'] / daily_efficiency['Daily_Trucks']).round(2)
        daily_efficiency['Shipments_Per_Truck'] = (daily_efficiency['Daily_Shipments'] / daily_efficiency['Daily_Trucks']).round(2)
        
        # Handle division by zero
        daily_efficiency = daily_efficiency.replace([np.inf, -np.inf], np.nan)
        
        # Efficiency statistics
        efficiency_stats = {
            'avg_case_equivalent_per_truck': daily_efficiency['Case_Equivalent_Per_Truck'].mean(),  # ← NEW: Primary efficiency metric
            'avg_cases_per_truck': daily_efficiency['Cases_Per_Truck'].mean(),  # ← LEGACY: Keep for backward compatibility
            'avg_skus_per_truck': daily_efficiency['SKUs_Per_Truck'].mean(),
            'avg_shipments_per_truck': daily_efficiency['Shipments_Per_Truck'].mean(),
            'case_equivalent_per_truck_cv': daily_efficiency['Case_Equivalent_Per_Truck'].std() / daily_efficiency['Case_Equivalent_Per_Truck'].mean(),  # ← NEW: Primary variability metric
            'cases_per_truck_cv': daily_efficiency['Cases_Per_Truck'].std() / daily_efficiency['Cases_Per_Truck'].mean(),  # ← LEGACY
            'most_efficient_day': daily_efficiency.loc[daily_efficiency['Case_Equivalent_Per_Truck'].idxmax(), 'Date'].strftime('%Y-%m-%d'),  # ← UPDATED: Use case equivalent volume
            'peak_efficiency': daily_efficiency['Case_Equivalent_Per_Truck'].max()  # ← UPDATED: Use case equivalent volume
        }
        
        # Identify efficiency trends by day of week
        dow_efficiency = daily_efficiency.groupby('Day_of_Week').agg({
            'Case_Equivalent_Per_Truck': ['mean', 'std'],  # ← NEW: Primary efficiency metric
            'Cases_Per_Truck': ['mean', 'std'],  # ← LEGACY: Keep for backward compatibility
            'SKUs_Per_Truck': 'mean',
            'Daily_Trucks': 'mean'
        }).round(2)
        
        dow_efficiency.columns = ['_'.join(col).strip() for col in dow_efficiency.columns]
        dow_efficiency = dow_efficiency.reset_index()
        
        return {
            'daily_efficiency': daily_efficiency,
            'efficiency_statistics': efficiency_stats,
            'day_of_week_efficiency': dow_efficiency,
            'efficiency_recommendations': self._generate_efficiency_recommendations(efficiency_stats)
        }
    
    def analyze_dock_utilization(self):
        """
        Analyze dock utilization patterns.
        
        Returns:
            dict: Dock utilization analysis results
        """
        print("🚪 Analyzing dock utilization...")
        
        if self.daily_patterns is None:
            self.analyze_daily_patterns()
        
        # Assume standard dock capacity (can be made configurable)
        max_trucks_per_day = 20  # Configurable parameter
        
        dock_utilization = self.daily_patterns.copy()
        dock_utilization['Dock_Utilization_Percent'] = (dock_utilization['Daily_Trucks'] / max_trucks_per_day * 100).round(2)
        
        # Classify utilization levels
        def classify_utilization(utilization):
            if utilization >= 90:
                return 'Over-utilized'
            elif utilization >= 70:
                return 'High'
            elif utilization >= 40:
                return 'Medium'
            else:
                return 'Low'
        
        dock_utilization['Utilization_Category'] = dock_utilization['Dock_Utilization_Percent'].apply(classify_utilization)
        
        # Utilization summary
        utilization_summary = dock_utilization.groupby('Utilization_Category').agg({
            'Date': 'count',
            'Daily_Trucks': 'mean',
            'Daily_Case_Equivalent_Volume': 'mean',  # ← NEW: Primary volume metric
            'Daily_Cases': 'mean'  # ← LEGACY: Keep for backward compatibility
        }).round(2)
        
        utilization_summary.columns = ['Days_Count', 'Avg_Trucks', 'Avg_Case_Equivalent_Volume', 'Avg_Cases_Legacy']
        
        # Peak utilization analysis
        peak_days = dock_utilization[dock_utilization['Dock_Utilization_Percent'] >= 80]
        
        return {
            'dock_utilization_data': dock_utilization,
            'utilization_summary': utilization_summary,
            'avg_utilization': dock_utilization['Dock_Utilization_Percent'].mean(),
            'peak_utilization_days': len(peak_days),
            'over_capacity_days': len(dock_utilization[dock_utilization['Dock_Utilization_Percent'] > 100]),
            'max_capacity_assumption': max_trucks_per_day
        }
    
    def analyze_sku_patterns(self):
        """
        Analyze SKU-specific receipt patterns.
        
        Returns:
            dict: SKU receipt pattern analysis results
        """
        print("🏷️ Analyzing SKU receipt patterns...")
        
        # Add case equivalent volume to receipt data for SKU analysis
        enhanced_receipt_data = self._add_case_equivalent_to_receipt_data(self.receipt_data)
        
        # SKU-level receipt analysis
        sku_receipts = enhanced_receipt_data.groupby('SKU ID').agg({
            'Receipt Date': ['count', 'nunique', 'min', 'max'],
            'Truck No': 'nunique',
            'Quantity in Cases': ['sum', 'mean', 'std'],
            'Quantity in Eaches': ['sum', 'mean'],
            'Case_Equivalent_Volume': ['sum', 'mean', 'std']  # ← NEW: Case equivalent volume metrics
        }).round(2)
        
        # Flatten column names
        sku_receipts.columns = ['_'.join(col).strip() for col in sku_receipts.columns]
        sku_receipts = sku_receipts.reset_index()
        
        # Rename columns
        column_mapping = {
            'Receipt Date_count': 'Total_Receipts',
            'Receipt Date_nunique': 'Active_Days',
            'Receipt Date_min': 'First_Receipt',
            'Receipt Date_max': 'Last_Receipt',
            'Truck No_nunique': 'Unique_Trucks',
            'Quantity in Cases_sum': 'Total_Cases',
            'Quantity in Cases_mean': 'Avg_Cases_Per_Receipt',
            'Quantity in Cases_std': 'Cases_Std_Dev',
            'Quantity in Eaches_sum': 'Total_Eaches',
            'Quantity in Eaches_mean': 'Avg_Eaches_Per_Receipt',
            'Case_Equivalent_Volume_sum': 'Total_Case_Equivalent_Volume',  # ← NEW: Primary volume metric
            'Case_Equivalent_Volume_mean': 'Avg_Case_Equivalent_Per_Receipt',  # ← NEW
            'Case_Equivalent_Volume_std': 'Case_Equivalent_Std_Dev'  # ← NEW
        }
        sku_receipts = sku_receipts.rename(columns=column_mapping)
        
        # Calculate receipt frequency
        total_days = self.receipt_data['Receipt Date'].nunique()
        sku_receipts['Receipt_Frequency_Percent'] = (sku_receipts['Active_Days'] / total_days * 100).round(2)
        sku_receipts['Cases_CV'] = (sku_receipts['Cases_Std_Dev'] / sku_receipts['Avg_Cases_Per_Receipt']).round(2)
        sku_receipts['Cases_CV'] = sku_receipts['Cases_CV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        sku_receipts['Case_Equivalent_CV'] = (sku_receipts['Case_Equivalent_Std_Dev'] / sku_receipts['Avg_Case_Equivalent_Per_Receipt']).round(2)  # ← NEW: Primary variability metric
        sku_receipts['Case_Equivalent_CV'] = sku_receipts['Case_Equivalent_CV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Sort by total case equivalent volume (primary metric)
        sku_receipts = sku_receipts.sort_values('Total_Case_Equivalent_Volume', ascending=False).reset_index(drop=True)
        
        # Classify SKU receipt patterns
        def classify_receipt_pattern(row):
            if row['Receipt_Frequency_Percent'] >= 50:
                return 'Frequent'
            elif row['Receipt_Frequency_Percent'] >= 20:
                return 'Regular'
            elif row['Receipt_Frequency_Percent'] >= 5:
                return 'Occasional'
            else:
                return 'Rare'
        
        sku_receipts['Receipt_Pattern'] = sku_receipts.apply(classify_receipt_pattern, axis=1)
        
        # Pattern summary
        pattern_summary = sku_receipts.groupby('Receipt_Pattern').agg({
            'SKU ID': 'count',
            'Total_Case_Equivalent_Volume': 'sum',  # ← NEW: Primary volume metric
            'Total_Cases': 'sum',  # ← LEGACY: Keep for backward compatibility
            'Receipt_Frequency_Percent': 'mean'
        }).round(2)
        
        pattern_summary.columns = ['SKU_Count', 'Total_Case_Equivalent_Volume', 'Total_Cases_Legacy', 'Avg_Frequency_Percent']
        
        return {
            'sku_receipt_details': sku_receipts,
            'pattern_summary': pattern_summary,
            'frequent_receipts': sku_receipts[sku_receipts['Receipt_Pattern'] == 'Frequent'],
            'rare_receipts': sku_receipts[sku_receipts['Receipt_Pattern'] == 'Rare'],
            'total_skus_received': len(sku_receipts)
        }
    
    def analyze_lead_times(self):
        """
        Analyze lead times between orders and receipts (if order data available).
        
        Returns:
            dict: Lead time analysis results
        """
        if self.order_data is None:
            return {'error': 'Order data not available for lead time analysis'}
        
        print("⏰ Analyzing lead times...")
        
        # This is a simplified lead time analysis
        # In practice, you'd need more sophisticated matching logic
        
        # Calculate average time between order and receipt dates
        order_dates = self.order_data['Date'].unique()
        receipt_dates = self.receipt_data['Receipt Date'].unique()
        
        # Simple analysis: average days between order activity and receipt activity
        avg_order_date = pd.to_datetime(order_dates).mean()
        avg_receipt_date = pd.to_datetime(receipt_dates).mean()
        avg_lead_time_days = (avg_receipt_date - avg_order_date).days
        
        return {
            'estimated_avg_lead_time_days': avg_lead_time_days,
            'note': 'Simplified lead time calculation based on average order and receipt dates',
            'recommendation': 'Implement detailed order-receipt matching for accurate lead time analysis'
        }
    
    def generate_recommendations(self):
        """
        Generate receipt management recommendations.
        
        Returns:
            dict: Recommendations based on analysis
        """
        print("💡 Generating receipt recommendations...")
        
        recommendations = []
        
        # Dock utilization recommendations
        if hasattr(self, 'dock_utilization'):
            dock_util = self.analyze_dock_utilization()
            avg_utilization = dock_util['avg_utilization']
            
            if avg_utilization > 80:
                recommendations.append({
                    'category': 'Dock Capacity',
                    'priority': 'High',
                    'recommendation': f'High dock utilization ({avg_utilization:.1f}%) - consider capacity expansion',
                    'impact': 'Reduced waiting times and improved efficiency'
                })
            elif avg_utilization < 40:
                recommendations.append({
                    'category': 'Dock Efficiency',
                    'priority': 'Medium',
                    'recommendation': f'Low dock utilization ({avg_utilization:.1f}%) - optimize scheduling',
                    'impact': 'Better resource utilization'
                })
        
        # Supplier performance recommendations
        supplier_perf = self.analyze_supplier_performance()
        high_variability = len(supplier_perf['high_variability_trucks'])
        
        if high_variability > 0:
            recommendations.append({
                'category': 'Supplier Management',
                'priority': 'Medium',
                'recommendation': f'{high_variability} trucks show high delivery variability - engage suppliers',
                'impact': 'More predictable receiving workload'
            })
        
        # Efficiency recommendations
        efficiency = self.analyze_receiving_efficiency()
        recommendations.extend(efficiency['efficiency_recommendations'])
        
        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'priority_breakdown': {
                'high': len([r for r in recommendations if r['priority'] == 'High']),
                'medium': len([r for r in recommendations if r['priority'] == 'Medium']),
                'low': len([r for r in recommendations if r['priority'] == 'Low'])
            }
        }
    
    def _generate_efficiency_recommendations(self, efficiency_stats):
        """Generate efficiency-specific recommendations using case equivalent volume"""
        recommendations = []
        
        case_equivalent_per_truck = efficiency_stats['avg_case_equivalent_per_truck']
        
        if case_equivalent_per_truck < 50:
            recommendations.append({
                'category': 'Receiving Efficiency',
                'priority': 'High',
                'recommendation': f'Low case equivalent volume per truck ({case_equivalent_per_truck:.1f}) - optimize load consolidation',
                'impact': 'Reduced receiving labor and dock congestion'
            })
        
        variability = efficiency_stats['case_equivalent_per_truck_cv']
        if variability > 0.5:
            recommendations.append({
                'category': 'Load Consistency',
                'priority': 'Medium',
                'recommendation': f'High case equivalent volume variability (CV={variability:.2f}) - standardize shipment sizes',
                'impact': 'More predictable receiving capacity planning'
            })
        
        return recommendations
    
    def _determine_trend_direction(self):
        """Determine overall trend direction for receipts using case equivalent volume"""
        if self.daily_patterns is None:
            return "Unknown"
        
        # Simple linear trend analysis using case equivalent volume (primary metric)
        x = np.arange(len(self.daily_patterns))
        y = self.daily_patterns['Daily_Case_Equivalent_Volume'].values
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.1:
            return "Increasing"
        elif correlation < -0.1:
            return "Decreasing"
        else:
            return "Stable"
    
    def _detect_seasonality(self):
        """Detect basic seasonality patterns in receipts using case equivalent volume"""
        if self.daily_patterns is None or len(self.daily_patterns) < 14:
            return False
        
        # Check for day-of-week patterns using case equivalent volume (primary metric)
        dow_variance = self.daily_patterns.groupby('Day_of_Week')['Daily_Case_Equivalent_Volume'].var()
        overall_variance = self.daily_patterns['Daily_Case_Equivalent_Volume'].var()
        
        # If day-of-week variance is significantly different, we have seasonality
        return dow_variance.mean() > (overall_variance * 0.1)
    
    def _filter_by_date_range(self, df):
        """Filter receipt data by specified date range"""
        filtered_df = df.copy()
        
        if self.date_range.get('START_DATE'):
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Receipt Date']) >= self.date_range['START_DATE']]
        
        if self.date_range.get('END_DATE'):
            filtered_df = filtered_df[pd.to_datetime(filtered_df['Receipt Date']) <= self.date_range['END_DATE']]
        
        print(f"Receipt date filter applied: {len(df)} -> {len(filtered_df)} records")
        return filtered_df
    
    def _filter_by_date_range_orders(self, df):
        """Filter order data by specified date range"""
        filtered_df = df.copy()
        
        if self.date_range.get('START_DATE'):
            filtered_df = filtered_df[filtered_df['Date'] >= self.date_range['START_DATE']]
        
        if self.date_range.get('END_DATE'):
            filtered_df = filtered_df[filtered_df['Date'] <= self.date_range['END_DATE']]
        
        return filtered_df
    
    def _get_data_summary(self):
        """Get basic data summary"""
        # Add case equivalent volume to receipt data for summary
        enhanced_receipt_data = self._add_case_equivalent_to_receipt_data(self.receipt_data)
        
        summary = {
            'total_receipt_records': len(self.receipt_data),
            'unique_skus_received': self.receipt_data['SKU ID'].nunique(),
            'unique_trucks': self.receipt_data['Truck No'].nunique(),
            'unique_shipments': self.receipt_data['Shipment No'].nunique(),
            'total_case_equivalent_volume_received': enhanced_receipt_data['Case_Equivalent_Volume'].sum(),  # ← NEW: Primary volume metric
            'total_cases_received': self.receipt_data['Quantity in Cases'].sum(),  # ← LEGACY: Keep for backward compatibility
            'total_eaches_received': self.receipt_data['Quantity in Eaches'].sum(),  # ← LEGACY: Keep for backward compatibility
            'date_range': {
                'start': pd.to_datetime(self.receipt_data['Receipt Date']).min().strftime('%Y-%m-%d'),
                'end': pd.to_datetime(self.receipt_data['Receipt Date']).max().strftime('%Y-%m-%d'),
                'days': pd.to_datetime(self.receipt_data['Receipt Date']).nunique()
            }
        }
        
        return summary

# Test function for standalone execution
if __name__ == "__main__":
    print("ReceiptAnalyzer module - ready for use")
    print("This module requires receipt data to function.")
    print("Use within the main analysis pipeline for proper functionality.")