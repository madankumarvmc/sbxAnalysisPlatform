"""
SKU Analysis Module for Warehouse Analysis Tool V2

PURPOSE:
This module analyzes SKU performance and characteristics including:
- SKU profiling and classification
- Volume analysis by product
- Category performance analysis
- SKU ranking and segmentation

FOR BEGINNERS:
- This module analyzes individual products (SKUs) and their performance
- It helps identify which products are most important for the business
- Results are used for inventory planning and category management
- Classification helps prioritize warehouse layout and operations
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

class SKUAnalyzer:
    """
    SKU analysis and classification class.
    
    This class analyzes SKU data to provide:
    - Individual SKU performance metrics
    - Category-level analysis
    - Volume and velocity classifications
    - Profitability and efficiency insights
    """
    
    def __init__(self, order_data, sku_master=None, analysis_config=None):
        """
        Initialize the SKUAnalyzer.
        
        Args:
            order_data (pandas.DataFrame): Cleaned order data from DataLoader
            sku_master (pandas.DataFrame): SKU master data with categories and configs
            analysis_config (dict): Configuration parameters for analysis
        """
        self.order_data = order_data.copy()
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.config = analysis_config or {}
        
        # Initialize case equivalent converter
        self.converter = CaseEquivalentConverter(self.sku_master)
        
        # Set analysis parameters from config
        self.abc_thresholds = self.config.get('ABC_THRESHOLDS', config.DEFAULT_ABC_THRESHOLDS)
        self.date_range = self.config.get('DATE_RANGE', {})
        
        # Filter data by date range if specified
        if self.date_range.get('START_DATE') or self.date_range.get('END_DATE'):
            self.order_data = self._filter_by_date_range(self.order_data)
        
        # Analysis results containers
        self.sku_performance = None
        self.category_analysis = None
        self.abc_classification = None
        
        print(f"SKUAnalyzer initialized with {len(self.order_data)} order records")
        if self.sku_master is not None:
            print(f"SKU Master data available: {len(self.sku_master)} SKUs")
    
    def run_complete_analysis(self):
        """
        Run complete SKU analysis pipeline.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        print("🔄 Running complete SKU analysis...")
        
        results = {
            'success': True,
            'analysis_date': datetime.now(),
            'data_summary': self._get_data_summary(),
            'sku_performance': self.analyze_sku_performance(),
            'abc_classification': self.perform_abc_classification(),
            'category_analysis': self.analyze_categories(),
            'velocity_analysis': self.analyze_velocity(),
            'size_analysis': self.analyze_case_sizes(),
            'recommendations': self.generate_recommendations()
        }
        
        print("✅ SKU analysis completed successfully")
        return results
    
    def analyze_sku_performance(self):
        """
        Analyze individual SKU performance metrics.
        
        Returns:
            dict: SKU performance analysis results
        """
        print("📊 Analyzing SKU performance...")
        
        # Add case equivalent columns to order data
        enhanced_order_data = self.converter.add_case_equivalent_columns(self.order_data)
        
        # Calculate SKU-level performance metrics using case equivalent volume
        sku_metrics = enhanced_order_data.groupby('Sku Code').agg({
            'Date': ['count', 'nunique', 'min', 'max'],
            'Order No.': 'nunique',
            'Shipment No.': 'nunique',
            'Qty in Cases': ['sum', 'mean', 'std', 'min', 'max'],
            'Qty in Eaches': ['sum', 'mean', 'std', 'min', 'max'],
            'Case_Equivalent_Volume': ['sum', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        sku_metrics.columns = ['_'.join(col).strip() for col in sku_metrics.columns]
        sku_metrics = sku_metrics.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'Date_count': 'Total_Order_Lines',
            'Date_nunique': 'Days_Active',
            'Date_min': 'First_Order_Date',
            'Date_max': 'Last_Order_Date',
            'Order No._nunique': 'Unique_Orders',
            'Shipment No._nunique': 'Unique_Shipments',
            'Qty in Cases_sum': 'Total_Cases',
            'Qty in Cases_mean': 'Avg_Cases_Per_Line',
            'Qty in Cases_std': 'Cases_Std_Dev',
            'Qty in Cases_min': 'Min_Cases_Per_Line',
            'Qty in Cases_max': 'Max_Cases_Per_Line',
            'Qty in Eaches_sum': 'Total_Eaches',
            'Qty in Eaches_mean': 'Avg_Eaches_Per_Line',
            'Qty in Eaches_std': 'Eaches_Std_Dev',
            'Qty in Eaches_min': 'Min_Eaches_Per_Line',
            'Qty in Eaches_max': 'Max_Eaches_Per_Line',
            'Case_Equivalent_Volume_sum': 'Total_Case_Equivalent_Volume',
            'Case_Equivalent_Volume_mean': 'Avg_Case_Equivalent_Per_Line',
            'Case_Equivalent_Volume_std': 'Case_Equivalent_Std_Dev',
            'Case_Equivalent_Volume_min': 'Min_Case_Equivalent_Per_Line',
            'Case_Equivalent_Volume_max': 'Max_Case_Equivalent_Per_Line'
        }
        sku_metrics = sku_metrics.rename(columns=column_mapping)
        
        # Calculate additional performance metrics using case equivalent volume as primary
        total_days = self.order_data['Date'].nunique()
        sku_metrics['Activity_Rate'] = (sku_metrics['Days_Active'] / total_days * 100).round(2)
        sku_metrics['Avg_Case_Equivalent_Per_Day'] = (sku_metrics['Total_Case_Equivalent_Volume'] / sku_metrics['Days_Active']).round(2)
        sku_metrics['Avg_Cases_Per_Day'] = (sku_metrics['Total_Cases'] / sku_metrics['Days_Active']).round(2)
        sku_metrics['Case_Equivalent_CV'] = (sku_metrics['Case_Equivalent_Std_Dev'] / sku_metrics['Avg_Case_Equivalent_Per_Line']).round(2)
        sku_metrics['Cases_CV'] = (sku_metrics['Cases_Std_Dev'] / sku_metrics['Avg_Cases_Per_Line']).round(2)
        
        # Handle division by zero
        sku_metrics['Case_Equivalent_CV'] = sku_metrics['Case_Equivalent_CV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        sku_metrics['Cases_CV'] = sku_metrics['Cases_CV'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Add SKU master data if available
        if self.sku_master is not None:
            # Convert Sku Code in both dataframes to same type for proper merging
            sku_metrics['Sku Code'] = sku_metrics['Sku Code'].astype(str)
            sku_master_clean = self.sku_master.copy()
            sku_master_clean['Sku Code'] = sku_master_clean['Sku Code'].astype(str)
            
            sku_metrics = sku_metrics.merge(
                sku_master_clean[['Sku Code', 'Category', 'Case Config', 'Pallet Fit']],
                on='Sku Code',
                how='left'
            )
            
            # Calculate efficiency metrics (using case equivalent volume for primary calculations)
            sku_metrics['Cases_Per_Pallet'] = sku_metrics['Pallet Fit'].fillna(1)
            sku_metrics['Total_Case_Equivalent_Pallets'] = (sku_metrics['Total_Case_Equivalent_Volume'] / sku_metrics['Cases_Per_Pallet']).round(2)
            sku_metrics['Total_Pallets'] = (sku_metrics['Total_Cases'] / sku_metrics['Cases_Per_Pallet']).round(2)
        
        # Sort by case equivalent volume as primary metric
        sku_metrics = sku_metrics.sort_values('Total_Case_Equivalent_Volume', ascending=False).reset_index(drop=True)
        
        self.sku_performance = sku_metrics
        
        # Generate summary statistics using case equivalent volume as primary metric
        performance_summary = {
            'total_skus': len(sku_metrics),
            'avg_case_equivalent_per_sku': sku_metrics['Total_Case_Equivalent_Volume'].mean(),
            'median_case_equivalent_per_sku': sku_metrics['Total_Case_Equivalent_Volume'].median(),
            'avg_cases_per_sku': sku_metrics['Total_Cases'].mean(),
            'median_cases_per_sku': sku_metrics['Total_Cases'].median(),
            'top_10_percent_case_equivalent_volume': sku_metrics.head(int(len(sku_metrics) * 0.1))['Total_Case_Equivalent_Volume'].sum(),
            'bottom_50_percent_case_equivalent_volume': sku_metrics.tail(int(len(sku_metrics) * 0.5))['Total_Case_Equivalent_Volume'].sum(),
            'top_10_percent_volume': sku_metrics.head(int(len(sku_metrics) * 0.1))['Total_Cases'].sum(),
            'bottom_50_percent_volume': sku_metrics.tail(int(len(sku_metrics) * 0.5))['Total_Cases'].sum(),
            'high_case_equivalent_variability_skus': len(sku_metrics[sku_metrics['Case_Equivalent_CV'] > 1.0]),
            'high_variability_skus': len(sku_metrics[sku_metrics['Cases_CV'] > 1.0]),
            'daily_movers': len(sku_metrics[sku_metrics['Activity_Rate'] > 50]),
            'occasional_movers': len(sku_metrics[(sku_metrics['Activity_Rate'] <= 50) & (sku_metrics['Activity_Rate'] > 10)]),
            'slow_movers': len(sku_metrics[sku_metrics['Activity_Rate'] <= 10])
        }
        
        return {
            'sku_details': sku_metrics,
            'performance_summary': performance_summary
        }
    
    def perform_abc_classification(self):
        """
        Perform ABC classification based on volume.
        
        Returns:
            dict: ABC classification results
        """
        print("🎯 Performing ABC classification...")
        
        if self.sku_performance is None:
            self.analyze_sku_performance()
        
        sku_data = self.sku_performance.copy()
        
        # Sort by case equivalent volume (already sorted in sku_performance)
        sku_data['Cumulative_Case_Equivalent_Volume'] = sku_data['Total_Case_Equivalent_Volume'].cumsum()
        total_case_equivalent_volume = sku_data['Total_Case_Equivalent_Volume'].sum()
        sku_data['Cumulative_Case_Equivalent_Percent'] = (sku_data['Cumulative_Case_Equivalent_Volume'] / total_case_equivalent_volume * 100).round(2)
        
        # Keep legacy calculations for backward compatibility
        sku_data['Cumulative_Cases'] = sku_data['Total_Cases'].cumsum()
        total_cases = sku_data['Total_Cases'].sum()
        sku_data['Cumulative_Percent'] = (sku_data['Cumulative_Cases'] / total_cases * 100).round(2)
        
        # Apply ABC classification using case equivalent volume as primary metric
        def classify_abc(cumulative_percent):
            if cumulative_percent <= self.abc_thresholds['A_THRESHOLD']:
                return 'A'
            elif cumulative_percent <= self.abc_thresholds['B_THRESHOLD']:
                return 'B'
            else:
                return 'C'
        
        sku_data['ABC_Category'] = sku_data['Cumulative_Case_Equivalent_Percent'].apply(classify_abc)
        # Keep legacy ABC classification for comparison
        sku_data['ABC_Category_Legacy'] = sku_data['Cumulative_Percent'].apply(classify_abc)
        
        # Generate ABC summary using case equivalent volume as primary metric
        abc_summary = sku_data.groupby('ABC_Category').agg({
            'Sku Code': 'count',
            'Total_Case_Equivalent_Volume': 'sum',
            'Total_Cases': 'sum',
            'Avg_Case_Equivalent_Per_Day': 'mean',
            'Avg_Cases_Per_Day': 'mean',
            'Activity_Rate': 'mean'
        }).round(2)
        
        abc_summary.columns = ['SKU_Count', 'Total_Case_Equivalent_Volume', 'Total_Cases', 'Avg_Daily_Case_Equivalent', 'Avg_Daily_Cases', 'Avg_Activity_Rate']
        abc_summary['Case_Equivalent_Volume_Percent'] = (abc_summary['Total_Case_Equivalent_Volume'] / abc_summary['Total_Case_Equivalent_Volume'].sum() * 100).round(2)
        abc_summary['Volume_Percent'] = (abc_summary['Total_Cases'] / abc_summary['Total_Cases'].sum() * 100).round(2)
        abc_summary['SKU_Percent'] = (abc_summary['SKU_Count'] / abc_summary['SKU_Count'].sum() * 100).round(2)
        
        self.abc_classification = sku_data
        
        return {
            'sku_with_abc': sku_data,
            'abc_summary': abc_summary,
            'classification_thresholds': self.abc_thresholds
        }
    
    def analyze_categories(self):
        """
        Analyze performance by product category.
        
        Returns:
            dict: Category analysis results
        """
        print("📂 Analyzing category performance...")
        
        if self.sku_master is None:
            return {
                'error': 'SKU Master data not available for category analysis',
                'category_summary': None
            }
        
        if self.sku_performance is None:
            self.analyze_sku_performance()
        
        # Category-level analysis using case equivalent volume as primary metric
        category_stats = self.sku_performance.groupby('Category').agg({
            'Sku Code': 'count',
            'Total_Case_Equivalent_Volume': ['sum', 'mean', 'std'],
            'Total_Cases': ['sum', 'mean', 'std'],
            'Days_Active': 'mean',
            'Activity_Rate': 'mean',
            'Avg_Case_Equivalent_Per_Day': 'mean',
            'Avg_Cases_Per_Day': 'mean',
            'Case_Equivalent_CV': 'mean',
            'Cases_CV': 'mean'
        }).round(2)
        
        # Flatten column names
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.reset_index()
        
        # Rename columns with case equivalent volume as primary
        column_mapping = {
            'Sku Code_count': 'SKU_Count',
            'Total_Case_Equivalent_Volume_sum': 'Category_Total_Case_Equivalent_Volume',
            'Total_Case_Equivalent_Volume_mean': 'Avg_Case_Equivalent_Per_SKU',
            'Total_Case_Equivalent_Volume_std': 'Case_Equivalent_Std_Dev',
            'Total_Cases_sum': 'Category_Total_Cases',
            'Total_Cases_mean': 'Avg_Cases_Per_SKU',
            'Total_Cases_std': 'Cases_Std_Dev',
            'Days_Active_mean': 'Avg_Days_Active',
            'Activity_Rate_mean': 'Avg_Activity_Rate',
            'Avg_Case_Equivalent_Per_Day_mean': 'Avg_Case_Equivalent_Per_Day',
            'Avg_Cases_Per_Day_mean': 'Avg_Cases_Per_Day',
            'Case_Equivalent_CV_mean': 'Avg_Case_Equivalent_Variability',
            'Cases_CV_mean': 'Avg_Variability'
        }
        category_stats = category_stats.rename(columns=column_mapping)
        
        # Calculate category performance metrics using case equivalent volume as primary
        total_case_equivalent_all_categories = category_stats['Category_Total_Case_Equivalent_Volume'].sum()
        total_cases_all_categories = category_stats['Category_Total_Cases'].sum()
        category_stats['Case_Equivalent_Volume_Share'] = (category_stats['Category_Total_Case_Equivalent_Volume'] / total_case_equivalent_all_categories * 100).round(2)
        category_stats['Volume_Share'] = (category_stats['Category_Total_Cases'] / total_cases_all_categories * 100).round(2)
        category_stats['Case_Equivalent_Per_SKU_Rank'] = category_stats['Avg_Case_Equivalent_Per_SKU'].rank(ascending=False)
        category_stats['Cases_Per_SKU_Rank'] = category_stats['Avg_Cases_Per_SKU'].rank(ascending=False)
        
        # Sort by case equivalent volume as primary metric
        category_stats = category_stats.sort_values('Category_Total_Case_Equivalent_Volume', ascending=False).reset_index(drop=True)
        
        self.category_analysis = category_stats
        
        return {
            'category_summary': category_stats,
            'top_category': category_stats.iloc[0]['Category'] if len(category_stats) > 0 else None,
            'most_diverse_category': category_stats.loc[category_stats['SKU_Count'].idxmax(), 'Category'] if len(category_stats) > 0 else None,
            'highest_case_equivalent_velocity_category': category_stats.loc[category_stats['Avg_Case_Equivalent_Per_Day'].idxmax(), 'Category'] if len(category_stats) > 0 else None,
            'highest_velocity_category': category_stats.loc[category_stats['Avg_Cases_Per_Day'].idxmax(), 'Category'] if len(category_stats) > 0 else None
        }
    
    def analyze_velocity(self):
        """
        Analyze SKU velocity and movement patterns.
        
        Returns:
            dict: Velocity analysis results
        """
        print("🏃 Analyzing SKU velocity...")
        
        if self.sku_performance is None:
            self.analyze_sku_performance()
        
        sku_data = self.sku_performance.copy()
        
        # Define velocity categories based on activity rate
        def classify_velocity(activity_rate):
            if activity_rate >= 70:
                return 'Fast'
            elif activity_rate >= 30:
                return 'Medium'
            elif activity_rate >= 5:
                return 'Slow'
            else:
                return 'Dead'
        
        sku_data['Velocity_Category'] = sku_data['Activity_Rate'].apply(classify_velocity)
        
        # Calculate velocity metrics using case equivalent volume as primary metric
        velocity_summary = sku_data.groupby('Velocity_Category').agg({
            'Sku Code': 'count',
            'Total_Case_Equivalent_Volume': 'sum',
            'Total_Cases': 'sum',
            'Activity_Rate': 'mean',
            'Avg_Case_Equivalent_Per_Day': 'mean',
            'Avg_Cases_Per_Day': 'mean'
        }).round(2)
        
        velocity_summary.columns = ['SKU_Count', 'Total_Case_Equivalent_Volume', 'Total_Cases', 'Avg_Activity_Rate', 'Avg_Case_Equivalent_Per_Day', 'Avg_Cases_Per_Day']
        velocity_summary['Case_Equivalent_Volume_Percent'] = (velocity_summary['Total_Case_Equivalent_Volume'] / velocity_summary['Total_Case_Equivalent_Volume'].sum() * 100).round(2)
        velocity_summary['Volume_Percent'] = (velocity_summary['Total_Cases'] / velocity_summary['Total_Cases'].sum() * 100).round(2)
        velocity_summary['SKU_Percent'] = (velocity_summary['SKU_Count'] / velocity_summary['SKU_Count'].sum() * 100).round(2)
        
        return {
            'sku_with_velocity': sku_data,
            'velocity_summary': velocity_summary,
            'fast_movers': len(sku_data[sku_data['Velocity_Category'] == 'Fast']),
            'dead_stock': len(sku_data[sku_data['Velocity_Category'] == 'Dead']),
            'velocity_distribution': sku_data['Velocity_Category'].value_counts().to_dict()
        }
    
    def analyze_case_sizes(self):
        """
        Analyze case configurations and pallet efficiency.
        
        Returns:
            dict: Case size analysis results
        """
        print("📦 Analyzing case sizes and configurations...")
        
        if self.sku_master is None:
            return {
                'error': 'SKU Master data not available for case size analysis',
                'case_analysis': None
            }
        
        if self.sku_performance is None:
            self.analyze_sku_performance()
        
        # Analyze case configurations using case equivalent volume as primary metric
        case_config_stats = self.sku_performance.groupby('Case Config').agg({
            'Sku Code': 'count',
            'Total_Case_Equivalent_Volume': 'sum',
            'Total_Cases': 'sum',
            'Total_Case_Equivalent_Pallets': 'sum',
            'Total_Pallets': 'sum'
        }).round(2)
        
        case_config_stats.columns = ['SKU_Count', 'Total_Case_Equivalent_Volume', 'Total_Cases', 'Total_Case_Equivalent_Pallets', 'Total_Pallets']
        case_config_stats['Avg_Case_Equivalent_Per_SKU'] = (case_config_stats['Total_Case_Equivalent_Volume'] / case_config_stats['SKU_Count']).round(2)
        case_config_stats['Avg_Cases_Per_SKU'] = (case_config_stats['Total_Cases'] / case_config_stats['SKU_Count']).round(2)
        case_config_stats = case_config_stats.reset_index()
        
        # Pallet efficiency analysis using case equivalent volume as primary metric
        pallet_efficiency = self.sku_performance.groupby('Pallet Fit').agg({
            'Sku Code': 'count',
            'Total_Case_Equivalent_Volume': 'sum',
            'Total_Cases': 'sum',
            'Total_Case_Equivalent_Pallets': 'sum',
            'Total_Pallets': 'sum'
        }).round(2)
        
        pallet_efficiency.columns = ['SKU_Count', 'Total_Case_Equivalent_Volume', 'Total_Cases', 'Total_Case_Equivalent_Pallets', 'Total_Pallets']
        pallet_efficiency['Case_Equivalent_Utilization_Rate'] = ((pallet_efficiency['Total_Case_Equivalent_Volume'] / pallet_efficiency['Total_Case_Equivalent_Pallets']) / pallet_efficiency.index * 100).round(2)
        pallet_efficiency['Utilization_Rate'] = ((pallet_efficiency['Total_Cases'] / pallet_efficiency['Total_Pallets']) / pallet_efficiency.index * 100).round(2)
        pallet_efficiency = pallet_efficiency.reset_index()
        
        return {
            'case_config_analysis': case_config_stats,
            'pallet_efficiency': pallet_efficiency,
            'avg_case_config': self.sku_performance['Case Config'].mean(),
            'avg_pallet_fit': self.sku_performance['Pallet Fit'].mean(),
            'total_case_equivalent_pallets_moved': self.sku_performance['Total_Case_Equivalent_Pallets'].sum(),
            'total_pallets_moved': self.sku_performance['Total_Pallets'].sum()
        }
    
    def generate_recommendations(self):
        """
        Generate SKU management recommendations.
        
        Returns:
            dict: Recommendations based on analysis
        """
        print("💡 Generating SKU recommendations...")
        
        if self.sku_performance is None:
            self.analyze_sku_performance()
        
        recommendations = []
        
        # ABC-based recommendations
        if self.abc_classification is not None:
            a_skus = len(self.abc_classification[self.abc_classification['ABC_Category'] == 'A'])
            total_skus = len(self.abc_classification)
            
            if a_skus / total_skus > 0.25:
                recommendations.append({
                    'category': 'ABC Classification',
                    'priority': 'High',
                    'recommendation': 'Consider tightening A-category criteria - too many SKUs in A category',
                    'impact': 'Improved focus on truly critical items'
                })
        
        # Slow mover recommendations
        slow_movers = len(self.sku_performance[self.sku_performance['Activity_Rate'] < 10])
        if slow_movers > 0:
            recommendations.append({
                'category': 'Slow Movers',
                'priority': 'Medium',
                'recommendation': f'Review {slow_movers} slow-moving SKUs for potential discontinuation',
                'impact': 'Reduced inventory holding costs'
            })
        
        # High variability recommendations (using case equivalent volume as primary)
        high_case_equiv_cv_skus = len(self.sku_performance[self.sku_performance['Case_Equivalent_CV'] > 2.0])
        high_cv_skus = len(self.sku_performance[self.sku_performance['Cases_CV'] > 2.0])
        if high_case_equiv_cv_skus > 0:
            recommendations.append({
                'category': 'Case Equivalent Demand Variability',
                'priority': 'Medium',
                'recommendation': f'{high_case_equiv_cv_skus} SKUs show high case equivalent demand variability - consider safety stock adjustments',
                'impact': 'Better service levels and inventory optimization based on standardized volume'
            })
        if high_cv_skus > 0 and high_cv_skus != high_case_equiv_cv_skus:
            recommendations.append({
                'category': 'Demand Variability',
                'priority': 'Medium',
                'recommendation': f'{high_cv_skus} SKUs show high raw case demand variability - compare with case equivalent analysis',
                'impact': 'Better service levels and inventory optimization'
            })
        
        # Category concentration recommendations (using case equivalent volume as primary)
        if self.category_analysis is not None and len(self.category_analysis) > 0:
            top_category_case_equiv_share = self.category_analysis.iloc[0]['Case_Equivalent_Volume_Share']
            top_category_share = self.category_analysis.iloc[0]['Volume_Share']
            if top_category_case_equiv_share > 60:
                recommendations.append({
                    'category': 'Portfolio Diversification (Case Equivalent)',
                    'priority': 'Low',
                    'recommendation': f'High case equivalent volume concentration in {self.category_analysis.iloc[0]["Category"]} category ({top_category_case_equiv_share:.1f}%)',
                    'impact': 'Consider portfolio diversification to reduce risk based on standardized volume'
                })
            if top_category_share > 60 and abs(top_category_share - top_category_case_equiv_share) > 5:
                recommendations.append({
                    'category': 'Portfolio Diversification',
                    'priority': 'Low',
                    'recommendation': f'High raw case concentration in {self.category_analysis.iloc[0]["Category"]} category ({top_category_share:.1f}%) differs from case equivalent analysis',
                    'impact': 'Consider portfolio diversification to reduce risk'
                })
        
        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'priority_breakdown': {
                'high': len([r for r in recommendations if r['priority'] == 'High']),
                'medium': len([r for r in recommendations if r['priority'] == 'Medium']),
                'low': len([r for r in recommendations if r['priority'] == 'Low'])
            }
        }
    
    def _filter_by_date_range(self, df):
        """Filter data by specified date range"""
        filtered_df = df.copy()
        
        if self.date_range.get('START_DATE'):
            filtered_df = filtered_df[filtered_df['Date'] >= self.date_range['START_DATE']]
        
        if self.date_range.get('END_DATE'):
            filtered_df = filtered_df[filtered_df['Date'] <= self.date_range['END_DATE']]
        
        print(f"Date filter applied: {len(df)} -> {len(filtered_df)} records")
        return filtered_df
    
    def _get_data_summary(self):
        """Get basic data summary including case equivalent volume"""
        # Add case equivalent columns to order data for summary
        enhanced_order_data = self.converter.add_case_equivalent_columns(self.order_data)
        
        summary = {
            'total_order_records': len(self.order_data),
            'unique_skus_in_orders': self.order_data['Sku Code'].nunique(),
            'total_case_equivalent_volume': enhanced_order_data['Case_Equivalent_Volume'].sum(),
            'total_volume_cases': self.order_data['Qty in Cases'].sum(),
            'total_volume_eaches': self.order_data['Qty in Eaches'].sum(),
            'case_equivalent_to_cases_ratio': round(enhanced_order_data['Case_Equivalent_Volume'].sum() / self.order_data['Qty in Cases'].sum(), 2) if self.order_data['Qty in Cases'].sum() > 0 else 0,
            'date_range': {
                'start': self.order_data['Date'].min().strftime('%Y-%m-%d'),
                'end': self.order_data['Date'].max().strftime('%Y-%m-%d')
            }
        }
        
        if self.sku_master is not None:
            summary['sku_master_records'] = len(self.sku_master)
            summary['categories_available'] = self.sku_master['Category'].nunique()
            summary['sku_coverage'] = round(self.order_data['Sku Code'].nunique() / len(self.sku_master) * 100, 2)
        
        return summary

# Test function for standalone execution
if __name__ == "__main__":
    print("SKUAnalyzer module - ready for use")
    print("This module requires order data and optionally SKU master data to function.")
    print("Use within the main analysis pipeline for proper functionality.")