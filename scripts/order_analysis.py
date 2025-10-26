"""
Order Analysis Module for Warehouse Analysis Tool V2

PURPOSE:
This module analyzes order patterns and trends including:
- Daily order volumes and statistics
- Percentile analysis for capacity planning
- Customer analysis and trends
- Peak period identification

FOR BEGINNERS:
- This module takes order data and calculates various statistics
- It identifies patterns like busy days, seasonal trends, etc.
- The results help with capacity planning and staffing decisions
- All calculations are clearly documented with business explanations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
from pathlib import Path

# Import configuration and utilities
sys.path.append(str(Path(__file__).parent.parent))
import config
from .case_equivalent_converter import CaseEquivalentConverter

class OrderAnalyzer:
    """
    Order pattern analysis class.
    
    This class analyzes order data to identify:
    - Daily volume patterns
    - Statistical distributions
    - Percentile analysis for capacity planning
    - Customer behavior patterns
    - Seasonal trends
    """
    
    def __init__(self, order_data, sku_master=None, analysis_config=None):
        """
        Initialize the OrderAnalyzer.
        
        Args:
            order_data (pandas.DataFrame): Cleaned order data from DataLoader
            sku_master (pandas.DataFrame, optional): SKU master data with Case Config
            analysis_config (dict): Configuration parameters for analysis
        """
        self.order_data = order_data.copy()
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.config = analysis_config or {}
        
        # Initialize case equivalent converter
        self.converter = CaseEquivalentConverter(self.sku_master)
        
        # Set analysis parameters from config
        self.percentile_levels = self.config.get('PERCENTILE_LEVELS', config.DEFAULT_PERCENTILE_LEVELS)
        self.date_range = self.config.get('DATE_RANGE', {})
        self.fms_thresholds = self.config.get('FMS_THRESHOLDS', config.DEFAULT_FMS_THRESHOLDS)
        
        # Filter data by date range if specified
        if self.date_range.get('START_DATE') or self.date_range.get('END_DATE'):
            self.order_data = self._filter_by_date_range(self.order_data)
        
        # Analysis results containers
        self.daily_summary = None
        self.volume_analysis = None
        self.percentile_analysis = None
        self.sku_frequency_analysis = None
        self.customer_analysis = None
        
        print(f"OrderAnalyzer initialized with {len(self.order_data)} order records")
    
    def run_complete_analysis(self):
        """
        Run complete order analysis pipeline.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        print("🔄 Running complete order analysis...")
        
        results = {
            'success': True,
            'analysis_date': datetime.now(),
            'data_summary': self._get_data_summary(),
            'daily_analysis': self.analyze_daily_patterns(),
            'volume_analysis': self.analyze_volume_statistics(),
            'percentile_analysis': self.analyze_percentiles(),
            'sku_frequency': self.analyze_sku_frequency(),
            'peak_analysis': self.analyze_peak_periods(),
            'trends': self.analyze_trends(),
            'outbound_summary': self.analyze_outbound_summary_statistics(),
            'order_profiles': self.analyze_order_profiles(),
            'monthly_volumes': self.analyze_monthly_volumes(),
            'enhanced_weekday_patterns': self.analyze_enhanced_weekday_patterns()
        }
        
        print("✅ Order analysis completed successfully")
        return results
    
    def analyze_daily_patterns(self):
        """
        Analyze daily order patterns and volumes.
        
        Returns:
            dict: Daily pattern analysis results
        """
        print("📊 Analyzing daily order patterns...")
        
        # Add case equivalent volume calculation
        order_data_enhanced = self.converter.add_case_equivalent_columns(
            self.order_data
        )
        
        # Group by date to get daily summaries (including case equivalent volume)
        daily_data = order_data_enhanced.groupby('Date').agg({
            'Order No.': 'nunique',
            'Shipment No.': 'nunique', 
            'Sku Code': 'nunique',
            'Qty in Cases': 'sum',
            'Qty in Eaches': 'sum',
            'Case_Equivalent_Volume': 'sum'  # ← NEW: Case equivalent daily volume
        }).reset_index()
        
        daily_data.columns = ['Date', 'Daily_Orders', 'Daily_Shipments', 'Daily_SKUs', 'Daily_Cases', 'Daily_Eaches', 'Daily_Case_Equivalent_Volume']
        
        # Add day of week analysis
        daily_data['Day_of_Week'] = daily_data['Date'].dt.day_name()
        daily_data['Week_Number'] = daily_data['Date'].dt.isocalendar().week
        daily_data['Month'] = daily_data['Date'].dt.month_name()
        
        # Calculate moving averages (including case equivalent volume)
        daily_data['Cases_7Day_MA'] = daily_data['Daily_Cases'].rolling(window=7, center=True).mean()
        daily_data['Orders_7Day_MA'] = daily_data['Daily_Orders'].rolling(window=7, center=True).mean()
        daily_data['Volume_7Day_MA'] = daily_data['Daily_Case_Equivalent_Volume'].rolling(window=7, center=True).mean()  # ← NEW
        
        # Day of week patterns (including case equivalent volume)
        dow_patterns = daily_data.groupby('Day_of_Week').agg({
            'Daily_Orders': ['mean', 'std', 'min', 'max'],
            'Daily_Cases': ['mean', 'std', 'min', 'max'],
            'Daily_Shipments': ['mean', 'std', 'min', 'max'],
            'Daily_Case_Equivalent_Volume': ['mean', 'std', 'min', 'max']  # ← NEW
        }).round(2)
        
        # Flatten column names
        dow_patterns.columns = ['_'.join(col).strip() for col in dow_patterns.columns]
        dow_patterns = dow_patterns.reset_index()
        
        self.daily_summary = daily_data
        
        return {
            'daily_data': daily_data,
            'day_of_week_patterns': dow_patterns,
            'total_days': len(daily_data),
            'avg_daily_orders': daily_data['Daily_Orders'].mean(),
            'avg_daily_cases': daily_data['Daily_Cases'].mean(),
            'avg_daily_case_equivalent_volume': daily_data['Daily_Case_Equivalent_Volume'].mean(),  # ← NEW
            'busiest_day': daily_data.loc[daily_data['Daily_Case_Equivalent_Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d'),  # ← UPDATED to use case equivalent
            'busiest_day_volume': daily_data['Daily_Case_Equivalent_Volume'].max()  # ← UPDATED to use case equivalent
        }
    
    def analyze_volume_statistics(self):
        """
        Analyze volume statistics and distributions.
        
        Returns:
            dict: Volume analysis results
        """
        print("📈 Analyzing volume statistics...")
        
        if self.daily_summary is None:
            self.analyze_daily_patterns()
        
        daily_cases = self.daily_summary['Daily_Cases']
        daily_orders = self.daily_summary['Daily_Orders']
        
        volume_stats = {
            'cases': {
                'total': daily_cases.sum(),
                'mean': daily_cases.mean(),
                'median': daily_cases.median(),
                'std': daily_cases.std(),
                'min': daily_cases.min(),
                'max': daily_cases.max(),
                'cv': daily_cases.std() / daily_cases.mean() if daily_cases.mean() > 0 else 0
            },
            'orders': {
                'total': daily_orders.sum(),
                'mean': daily_orders.mean(),
                'median': daily_orders.median(),
                'std': daily_orders.std(),
                'min': daily_orders.min(),
                'max': daily_orders.max(),
                'cv': daily_orders.std() / daily_orders.mean() if daily_orders.mean() > 0 else 0
            }
        }
        
        # Calculate distribution characteristics
        volume_stats['distribution'] = {
            'cases_skewness': daily_cases.skew(),
            'cases_kurtosis': daily_cases.kurtosis(),
            'orders_skewness': daily_orders.skew(),
            'orders_kurtosis': daily_orders.kurtosis()
        }
        
        self.volume_analysis = volume_stats
        return volume_stats
    
    def analyze_percentiles(self):
        """
        Analyze percentiles for capacity planning.
        
        Returns:
            dict: Percentile analysis results
        """
        print("📊 Analyzing percentiles for capacity planning...")
        
        if self.daily_summary is None:
            self.analyze_daily_patterns()
        
        daily_cases = self.daily_summary['Daily_Cases']
        daily_orders = self.daily_summary['Daily_Orders']
        
        percentile_results = {}
        
        for percentile in self.percentile_levels:
            percentile_results[f'p{percentile}'] = {
                'cases': np.percentile(daily_cases, percentile),
                'orders': np.percentile(daily_orders, percentile),
                'interpretation': f"{percentile}% of days have volume below this level"
            }
        
        # Add capacity planning recommendations
        p95_cases = percentile_results['p95']['cases']
        p90_cases = percentile_results['p90']['cases']
        mean_cases = daily_cases.mean()
        
        capacity_recommendations = {
            'normal_capacity': mean_cases * 1.1,  # 10% buffer above average
            'peak_capacity': p95_cases * 1.05,    # 5% buffer above 95th percentile
            'surge_capacity': daily_cases.max() * 1.1,  # 10% buffer above historical max
            'utilization_at_normal': (mean_cases / (mean_cases * 1.1)) * 100,
            'utilization_at_peak': (p95_cases / (p95_cases * 1.05)) * 100
        }
        
        # ✅ NEW: Calculate comprehensive daily metrics for horizontal percentile table
        order_data_enhanced = self.converter.add_case_equivalent_columns(
            self.order_data, self.sku_master
        )
        
        # Create comprehensive daily summary with all required metrics
        daily_comprehensive = order_data_enhanced.groupby('Date').agg({
            'Order No.': 'nunique',           # Distinct_Orders
            'Shipment No.': 'nunique',        # Distinct_Shipments  
            'Sku Code': 'nunique',            # Distinct_SKUs
            'Qty in Cases': 'sum',            # Qty_Ordered_Cases
            'Qty in Eaches': 'sum',           # Qty_Ordered_Eaches
            'Case_Equivalent_Volume': 'sum',  # Total_Case_Equiv
            'Pallet_Equivalent_Volume': 'sum' # Total_Pallet_Equiv
        }).reset_index()
        
        # Add distinct customers calculation (using Order No. as proxy for now)
        daily_comprehensive['Distinct_Customers'] = daily_comprehensive['Order No.']
        
        # Rename columns for clarity
        daily_comprehensive = daily_comprehensive.rename(columns={
            'Order No.': 'Distinct_Orders',
            'Shipment No.': 'Distinct_Shipments',
            'Sku Code': 'Distinct_SKUs',
            'Qty in Cases': 'Qty_Ordered_Cases',
            'Qty in Eaches': 'Qty_Ordered_Eaches',
            'Case_Equivalent_Volume': 'Total_Case_Equiv',
            'Pallet_Equivalent_Volume': 'Total_Pallet_Equiv'
        })
        
        # Calculate percentiles for all metrics (for horizontal table)
        metrics = ['Distinct_Customers', 'Distinct_Shipments', 'Distinct_Orders', 'Distinct_SKUs', 
                  'Qty_Ordered_Cases', 'Qty_Ordered_Eaches', 'Total_Case_Equiv', 'Total_Pallet_Equiv']
        
        horizontal_percentiles = {}
        
        # Add max values
        horizontal_percentiles['Max'] = {}
        for metric in metrics:
            horizontal_percentiles['Max'][metric] = daily_comprehensive[metric].max()
        
        # Add percentile calculations
        for percentile in [95, 90, 85]:  # Key percentiles for capacity planning
            horizontal_percentiles[f'{percentile}.0%ile'] = {}
            for metric in metrics:
                horizontal_percentiles[f'{percentile}.0%ile'][metric] = np.percentile(daily_comprehensive[metric], percentile)
        
        # Add average values
        horizontal_percentiles['Average'] = {}
        for metric in metrics:
            horizontal_percentiles['Average'][metric] = daily_comprehensive[metric].mean()
        
        self.percentile_analysis = {
            'percentiles': percentile_results,
            'capacity_planning': capacity_recommendations,
            'horizontal_percentiles': horizontal_percentiles,  # ← NEW: For horizontal table
            'daily_comprehensive': daily_comprehensive        # ← NEW: Store comprehensive daily data
        }
        
        return self.percentile_analysis
    
    def analyze_sku_frequency(self):
        """
        Analyze SKU ordering frequency for FMS classification.
        
        Returns:
            dict: SKU frequency analysis results
        """
        print("🏷️ Analyzing SKU frequency patterns...")
        
        # Calculate SKU-level statistics
        sku_stats = self.order_data.groupby('Sku Code').agg({
            'Date': ['count', 'nunique'],
            'Order No.': 'nunique',
            'Qty in Cases': ['sum', 'mean'],
            'Qty in Eaches': ['sum', 'mean']
        }).round(2)
        
        # Flatten column names
        sku_stats.columns = ['_'.join(col).strip() for col in sku_stats.columns]
        sku_stats = sku_stats.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'Date_count': 'Total_Order_Lines',
            'Date_nunique': 'Days_Ordered',
            'Order No._nunique': 'Unique_Orders',
            'Qty in Cases_sum': 'Total_Cases',
            'Qty in Cases_mean': 'Avg_Cases_Per_Line',
            'Qty in Eaches_sum': 'Total_Eaches',
            'Qty in Eaches_mean': 'Avg_Eaches_Per_Line'
        }
        sku_stats = sku_stats.rename(columns=column_mapping)
        
        # Calculate total days in dataset for frequency calculation
        total_days = self.order_data['Date'].nunique()
        sku_stats['Order_Frequency_Percent'] = (sku_stats['Days_Ordered'] / total_days * 100).round(2)
        
        # Sort by frequency for FMS classification
        sku_stats = sku_stats.sort_values('Order_Frequency_Percent', ascending=False).reset_index(drop=True)
        
        # Apply FMS classification
        sku_stats['Cumulative_Frequency_Percent'] = sku_stats['Order_Frequency_Percent'].cumsum() / sku_stats['Order_Frequency_Percent'].sum() * 100
        
        def classify_fms(cumulative_freq):
            if cumulative_freq <= self.fms_thresholds['F_THRESHOLD']:
                return 'Fast'
            elif cumulative_freq <= self.fms_thresholds['M_THRESHOLD']:
                return 'Medium'
            else:
                return 'Slow'
        
        sku_stats['FMS_Category'] = sku_stats['Cumulative_Frequency_Percent'].apply(classify_fms)
        
        # Summary by FMS category
        fms_summary = sku_stats.groupby('FMS_Category').agg({
            'Sku Code': 'count',
            'Total_Cases': 'sum',
            'Days_Ordered': 'mean',
            'Order_Frequency_Percent': 'mean'
        }).round(2)
        fms_summary.columns = ['SKU_Count', 'Total_Cases', 'Avg_Days_Ordered', 'Avg_Frequency_Percent']
        
        self.sku_frequency_analysis = {
            'sku_details': sku_stats,
            'fms_summary': fms_summary,
            'total_skus': len(sku_stats),
            'analysis_period_days': total_days
        }
        
        return self.sku_frequency_analysis
    
    def analyze_peak_periods(self):
        """
        Identify peak periods and patterns.
        
        Returns:
            dict: Peak period analysis results
        """
        print("⛰️ Analyzing peak periods...")
        
        if self.daily_summary is None:
            self.analyze_daily_patterns()
        
        daily_cases = self.daily_summary['Daily_Cases']
        
        # Define peak thresholds
        mean_volume = daily_cases.mean()
        std_volume = daily_cases.std()
        
        peak_threshold = mean_volume + std_volume  # 1 standard deviation above mean
        high_peak_threshold = mean_volume + (2 * std_volume)  # 2 standard deviations above mean
        
        # Identify peak days
        peak_days = self.daily_summary[self.daily_summary['Daily_Cases'] >= peak_threshold].copy()
        high_peak_days = self.daily_summary[self.daily_summary['Daily_Cases'] >= high_peak_threshold].copy()
        
        # Peak day patterns
        peak_day_patterns = peak_days['Day_of_Week'].value_counts().to_dict()
        
        # Peak period analysis
        peak_analysis = {
            'peak_threshold': peak_threshold,
            'high_peak_threshold': high_peak_threshold,
            'peak_days_count': len(peak_days),
            'high_peak_days_count': len(high_peak_days),
            'peak_day_percentage': (len(peak_days) / len(self.daily_summary)) * 100,
            'peak_day_patterns': peak_day_patterns,
            'avg_peak_volume': peak_days['Daily_Cases'].mean() if len(peak_days) > 0 else 0,
            'max_peak_volume': peak_days['Daily_Cases'].max() if len(peak_days) > 0 else 0
        }
        
        return peak_analysis
    
    def analyze_trends(self):
        """
        Analyze volume trends over time.
        
        Returns:
            dict: Trend analysis results
        """
        print("📈 Analyzing volume trends...")
        
        if self.daily_summary is None:
            self.analyze_daily_patterns()
        
        # Calculate monthly trends
        monthly_data = self.daily_summary.groupby('Month').agg({
            'Daily_Cases': ['sum', 'mean', 'count'],
            'Daily_Orders': ['sum', 'mean']
        }).round(2)
        
        # Flatten column names
        monthly_data.columns = ['_'.join(col).strip() for col in monthly_data.columns]
        monthly_data = monthly_data.reset_index()
        
        # Calculate growth rates (if multiple months available)
        growth_analysis = {}
        if len(monthly_data) > 1:
            monthly_data = monthly_data.sort_values('Month')
            monthly_data['Cases_Growth'] = monthly_data['Daily_Cases_sum'].pct_change() * 100
            monthly_data['Orders_Growth'] = monthly_data['Daily_Orders_sum'].pct_change() * 100
            
            growth_analysis = {
                'avg_monthly_cases_growth': monthly_data['Cases_Growth'].mean(),
                'avg_monthly_orders_growth': monthly_data['Orders_Growth'].mean(),
                'months_with_positive_growth': (monthly_data['Cases_Growth'] > 0).sum(),
                'months_with_negative_growth': (monthly_data['Cases_Growth'] < 0).sum()
            }
        
        trend_results = {
            'monthly_summary': monthly_data,
            'growth_analysis': growth_analysis,
            'trend_direction': self._determine_trend_direction(),
            'seasonality_detected': self._detect_seasonality()
        }
        
        return trend_results
    
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
        """Get basic data summary"""
        return {
            'total_records': len(self.order_data),
            'date_range': {
                'start': self.order_data['Date'].min().strftime('%Y-%m-%d'),
                'end': self.order_data['Date'].max().strftime('%Y-%m-%d'),
                'days': (self.order_data['Date'].max() - self.order_data['Date'].min()).days
            },
            'unique_orders': self.order_data['Order No.'].nunique(),
            'unique_shipments': self.order_data['Shipment No.'].nunique(),
            'unique_skus': self.order_data['Sku Code'].nunique(),
            'total_cases': self.order_data['Qty in Cases'].sum(),
            'total_eaches': self.order_data['Qty in Eaches'].sum()
        }
    
    def _determine_trend_direction(self):
        """Determine overall trend direction"""
        if self.daily_summary is None:
            return "Unknown"
        
        # Simple linear trend analysis
        x = np.arange(len(self.daily_summary))
        y = self.daily_summary['Daily_Cases'].values
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.1:
            return "Increasing"
        elif correlation < -0.1:
            return "Decreasing"
        else:
            return "Stable"
    
    def _detect_seasonality(self):
        """Detect basic seasonality patterns"""
        if self.daily_summary is None or len(self.daily_summary) < 14:
            return False
        
        # Check for day-of-week patterns
        dow_variance = self.daily_summary.groupby('Day_of_Week')['Daily_Cases'].var()
        overall_variance = self.daily_summary['Daily_Cases'].var()
        
        # If day-of-week variance is significantly different, we have seasonality
        return dow_variance.mean() > (overall_variance * 0.1)
    
    def analyze_outbound_summary_statistics(self):
        """
        Analyze outbound summary statistics for warehouse analytics.
        
        Returns:
            dict: Comprehensive outbound statistics with pick type breakdown
        """
        print("📊 Analyzing outbound summary statistics...")
        
        # Ensure we have daily data
        if self.daily_summary is None:
            self.analyze_daily_patterns()
        
        # ✅ FIXED: Add case equivalent volume and pick type classification
        order_data_enhanced = self.converter.add_case_equivalent_columns(
            self.order_data
        )
        order_data_enhanced = self._classify_pick_types_enhanced(order_data_enhanced)
        
        # ✅ FIXED: Overall statistics including case equivalent volume
        total_orders = self.order_data['Order No.'].nunique()
        total_lines = len(self.order_data)
        total_eaches = self.order_data['Qty in Eaches'].sum()
        total_cases = self.order_data['Qty in Cases'].sum()
        total_case_equivalent_volume = order_data_enhanced['Case_Equivalent_Volume'].sum()  # ← NEW
        total_skus = self.order_data['Sku Code'].nunique()
        
        # Calculate time periods
        date_range = (self.order_data['Date'].max() - self.order_data['Date'].min()).days
        months_in_data = max(1, date_range / 30.44)  # Average days per month
        days_in_data = self.order_data['Date'].nunique()
        
        # ✅ FIXED: Each picks statistics with case equivalent volume
        each_picks = order_data_enhanced[order_data_enhanced['Pick_Type'] == 'Each_Pick']
        each_orders = each_picks['Order No.'].nunique() if len(each_picks) > 0 else 0
        each_lines = len(each_picks)
        each_eaches = each_picks['Qty in Eaches'].sum() if len(each_picks) > 0 else 0
        each_case_equivalent_volume = each_picks['Case_Equivalent_Volume'].sum() if len(each_picks) > 0 else 0  # ← NEW
        
        # ✅ FIXED: Case picks statistics with case equivalent volume
        case_picks = order_data_enhanced[order_data_enhanced['Pick_Type'] == 'Case_Pick']
        case_orders = case_picks['Order No.'].nunique() if len(case_picks) > 0 else 0
        case_lines = len(case_picks)
        case_cases = case_picks['Qty in Cases'].sum() if len(case_picks) > 0 else 0
        case_eaches = case_picks['Qty in Eaches'].sum() if len(case_picks) > 0 else 0
        case_case_equivalent_volume = case_picks['Case_Equivalent_Volume'].sum() if len(case_picks) > 0 else 0  # ← NEW
        
        # Build comprehensive statistics
        outbound_stats = {
            'overall': {
                'annual_total': {
                    'orders': round(total_orders * (365.25 / date_range), 0) if date_range > 0 else total_orders,
                    'lines': round(total_lines * (365.25 / date_range), 0) if date_range > 0 else total_lines,
                    'volume': round(total_case_equivalent_volume * (365.25 / date_range), 2) if date_range > 0 else total_case_equivalent_volume,  # ← FIXED
                    'skus': total_skus
                },
                'monthly_average': {
                    'orders': round(total_orders / months_in_data, 0),
                    'lines': round(total_lines / months_in_data, 0),
                    'volume': round(total_case_equivalent_volume / months_in_data, 2),  # ← FIXED
                    'skus': total_skus
                },
                'monthly_peak': {
                    'orders': round(self.daily_summary['Daily_Orders'].max() * 30, 0),
                    'lines': round(total_lines / months_in_data * 1.2, 0),  # Estimate 20% above average
                    'volume': round(self.daily_summary['Daily_Case_Equivalent_Volume'].max() * 30, 2),  # ← FIXED
                    'skus': total_skus
                },
                'daily_average': {
                    'orders': round(total_orders / days_in_data, 0),
                    'lines': round(total_lines / days_in_data, 0),
                    'volume': round(total_case_equivalent_volume / days_in_data, 2),  # ← FIXED
                    'skus': round(total_skus / days_in_data * 7, 0)  # Weekly SKU average
                },
                'absolute_peak': {
                    'orders': self.daily_summary['Daily_Orders'].max(),
                    'lines': round(total_lines / days_in_data * 1.5, 0),  # Estimate
                    'volume': self.daily_summary['Daily_Case_Equivalent_Volume'].max(),  # ← FIXED
                    'skus': total_skus
                },
                'design_peak': {
                    'orders': round(self.daily_summary['Daily_Orders'].quantile(0.95), 0),
                    'lines': round(total_lines / days_in_data * 1.3, 0),
                    'volume': round(self.daily_summary['Daily_Case_Equivalent_Volume'].quantile(0.95), 2),  # ← FIXED
                    'skus': total_skus
                }
            },
            'each_picks': {
                'annual_total': {
                    'orders': round(each_orders * (365.25 / date_range), 0) if date_range > 0 else each_orders,
                    'lines': round(each_lines * (365.25 / date_range), 0) if date_range > 0 else each_lines,
                    'volume': round(each_case_equivalent_volume * (365.25 / date_range), 2) if date_range > 0 else each_case_equivalent_volume  # ← FIXED
                },
                'daily_average': {
                    'orders': round(each_orders / days_in_data, 0),
                    'lines': round(each_lines / days_in_data, 0),
                    'volume': round(each_case_equivalent_volume / days_in_data, 2)  # ← FIXED
                }
            },
            'case_picks': {
                'annual_total': {
                    'orders': round(case_orders * (365.25 / date_range), 0) if date_range > 0 else case_orders,
                    'lines': round(case_lines * (365.25 / date_range), 0) if date_range > 0 else case_lines,
                    'volume': round(case_case_equivalent_volume * (365.25 / date_range), 2) if date_range > 0 else case_case_equivalent_volume  # ← FIXED
                },
                'daily_average': {
                    'orders': round(case_orders / days_in_data, 0),
                    'lines': round(case_lines / days_in_data, 0),
                    'volume': round(case_case_equivalent_volume / days_in_data, 2)  # ← FIXED
                }
            }
        }
        
        # Calculate design P/A ratios
        for category in ['overall', 'each_picks', 'case_picks']:
            if category in outbound_stats:
                daily_avg = outbound_stats[category].get('daily_average', {})
                design_peak = outbound_stats[category].get('design_peak', daily_avg)
                
                ratios = {}
                for metric in daily_avg.keys():
                    avg_val = daily_avg.get(metric, 0)
                    peak_val = design_peak.get(metric, avg_val)
                    ratios[f'{metric}_ratio'] = round(peak_val / avg_val, 2) if avg_val > 0 else 0
                
                outbound_stats[category]['design_pa_ratios'] = ratios
        
        return outbound_stats
    
    def analyze_order_profiles(self):
        """
        Analyze order profile statistics and ratios.
        
        Returns:
            dict: Order profile analysis with key ratios
        """
        print("📈 Analyzing order profiles...")
        
        # ✅ FIXED: Add case equivalent volume calculation
        order_data_enhanced = self.converter.add_case_equivalent_columns(
            self.order_data
        )
        
        # Calculate order-level statistics
        order_stats = order_data_enhanced.groupby('Order No.').agg({
            'Order No.': 'count',  # Lines per order
            'Qty in Cases': 'sum',
            'Qty in Eaches': 'sum',
            'Case_Equivalent_Volume': 'sum',  # ← NEW: Case equivalent per order
            'Sku Code': 'nunique'
        }).rename(columns={'Order No.': 'Lines_Per_Order'})
        
        # ✅ FIXED: Calculate case equivalent volume per order (not raw addition)
        order_stats['Case_Equivalent_Per_Order'] = order_stats['Case_Equivalent_Volume']
        
        # ✅ FIXED: Calculate line-level case equivalent statistics
        line_stats = order_data_enhanced.copy()
        line_stats['Case_Equivalent_Per_Line'] = line_stats['Case_Equivalent_Volume']
        
        # ✅ FIXED: Calculate key ratios using case equivalent volume
        profiles = {
            'lines_per_order': round(order_stats['Lines_Per_Order'].mean(), 2),
            'case_equivalent_per_line': round(line_stats['Case_Equivalent_Per_Line'].mean(), 2),  # ← FIXED
            'case_equivalent_per_order': round(order_stats['Case_Equivalent_Per_Order'].mean(), 2),  # ← FIXED
            'eaches_per_line': round(line_stats['Qty in Eaches'].mean(), 2),
            'eaches_per_order': round(order_stats['Qty in Eaches'].mean(), 2),
            'cases_per_line': round(line_stats['Qty in Cases'].mean(), 2),
            'cases_per_order': round(order_stats['Qty in Cases'].mean(), 2),
            'eaches_per_case': round(line_stats['Qty in Eaches'].sum() / max(line_stats['Qty in Cases'].sum(), 1), 2),
            'pick_lines_per_order': round(order_stats['Lines_Per_Order'].mean(), 2)  # Same as lines per order
        }
        
        # Calculate each vs case line ratios
        each_lines = len(line_stats[line_stats['Qty in Eaches'] > 0])
        case_lines = len(line_stats[line_stats['Qty in Cases'] > 0])
        total_lines = len(line_stats)
        
        profiles['each_lines_per_order'] = round(each_lines / order_stats.shape[0], 2) if order_stats.shape[0] > 0 else 0
        profiles['case_lines_per_order'] = round(case_lines / order_stats.shape[0], 2) if order_stats.shape[0] > 0 else 0
        
        return {
            'statistical_values': profiles,
            'order_level_stats': order_stats.describe(),
            'line_level_stats': line_stats[['Case_Equivalent_Per_Line', 'Qty in Cases', 'Qty in Eaches']].describe()
        }
    
    def analyze_monthly_volumes(self):
        """
        Analyze volume patterns by month with pick type breakdown using case equivalent volumes.
        
        Returns:
            dict: Monthly volume analysis with case equivalent volumes
        """
        print("📅 Analyzing monthly volume patterns...")
        
        # ✅ FIXED: Use case equivalent volumes for monthly analysis
        # Get enhanced data with case equivalent volumes
        order_data_enhanced = self.converter.add_case_equivalent_columns(
            self.order_data, self.sku_master
        )
        
        # Classify pick types on enhanced data
        order_data_enhanced = self._classify_pick_types_enhanced(order_data_enhanced)
        order_data_enhanced['Month_Year'] = order_data_enhanced['Date'].dt.strftime('%b-%y')
        order_data_enhanced['Year_Month'] = order_data_enhanced['Date'].dt.to_period('M')
        
        # Overall monthly analysis using case equivalent volumes
        monthly_overall = order_data_enhanced.groupby('Month_Year').agg({
            'Order No.': 'nunique',
            'Date': 'count',  # Lines
            'Case_Equivalent_Volume': 'sum',  # ← NEW: Case equivalent volume
            'Sku Code': 'nunique'
        }).rename(columns={'Order No.': 'Orders', 'Date': 'Lines', 'Case_Equivalent_Volume': 'Volume', 'Sku Code': 'SKUs'})
        
        # Each picks monthly analysis using case equivalent volumes
        each_picks_monthly = order_data_enhanced[order_data_enhanced['Pick_Type'] == 'Each_Pick'].groupby('Month_Year').agg({
            'Order No.': 'nunique',
            'Date': 'count',
            'Case_Equivalent_Volume': 'sum'  # ← NEW: Case equivalent volume for each picks
        }).rename(columns={'Order No.': 'Orders', 'Date': 'Lines', 'Case_Equivalent_Volume': 'Volume'})
        
        # Case picks monthly analysis using case equivalent volumes
        case_picks_monthly = order_data_enhanced[order_data_enhanced['Pick_Type'] == 'Case_Pick'].groupby('Month_Year').agg({
            'Order No.': 'nunique',
            'Date': 'count',
            'Case_Equivalent_Volume': 'sum'  # ← NEW: Case equivalent volume for case picks
        }).rename(columns={'Order No.': 'Orders', 'Date': 'Lines', 'Case_Equivalent_Volume': 'Volume'})
        
        # Sort by actual date order
        monthly_overall = monthly_overall.reset_index()
        monthly_overall['Sort_Date'] = pd.to_datetime(monthly_overall['Month_Year'], format='%b-%y')
        monthly_overall = monthly_overall.sort_values('Sort_Date').drop('Sort_Date', axis=1)
        
        return {
            'monthly_totals': {
                'overall': monthly_overall,
                'each_picks': each_picks_monthly.reset_index(),
                'case_picks': case_picks_monthly.reset_index()
            },
            'summary': {
                'total_months': len(monthly_overall),
                'avg_monthly_orders': monthly_overall['Orders'].mean(),
                'peak_month': monthly_overall.loc[monthly_overall['Orders'].idxmax(), 'Month_Year'],
                'peak_orders': monthly_overall['Orders'].max()
            }
        }
    
    def analyze_enhanced_weekday_patterns(self):
        """
        Enhanced weekday analysis with averages and pick type breakdown.
        
        Returns:
            dict: Enhanced weekday patterns analysis
        """
        print("📊 Analyzing enhanced weekday patterns...")
        
        # Get existing day-of-week patterns
        existing_patterns = self.daily_summary.groupby('Day_of_Week').agg({
            'Daily_Orders': ['mean', 'std', 'min', 'max'],
            'Daily_Cases': ['mean', 'std', 'min', 'max'],
            'Daily_Shipments': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        existing_patterns.columns = ['_'.join(col).strip() for col in existing_patterns.columns]
        existing_patterns = existing_patterns.reset_index()
        
        # Enhanced weekday analysis with pick types
        order_data_enhanced = self._classify_pick_types()
        order_data_enhanced['Day_of_Week'] = order_data_enhanced['Date'].dt.day_name()
        
        # Overall weekday averages
        weekday_overall = order_data_enhanced.groupby('Day_of_Week').agg({
            'Order No.': 'nunique',
            'Date': 'count',  # Lines
            'Qty in Eaches': 'sum',
            'Sku Code': 'nunique'
        }).rename(columns={'Order No.': 'Orders', 'Date': 'Lines', 'Qty in Eaches': 'Eaches', 'Sku Code': 'SKUs'})
        
        # Calculate averages (divide by number of weeks)
        total_weeks = max(1, (order_data_enhanced['Date'].max() - order_data_enhanced['Date'].min()).days / 7)
        for col in weekday_overall.columns:
            weekday_overall[col] = weekday_overall[col] / total_weeks
        
        weekday_overall = weekday_overall.round(0)
        
        # Each picks weekday averages
        each_picks = order_data_enhanced[order_data_enhanced['Pick_Type'] == 'Each_Pick']
        weekday_each = each_picks.groupby('Day_of_Week').agg({
            'Order No.': 'nunique',
            'Date': 'count',
            'Qty in Eaches': 'sum'
        }).rename(columns={'Order No.': 'Orders', 'Date': 'Lines', 'Qty in Eaches': 'Eaches'})
        
        for col in weekday_each.columns:
            weekday_each[col] = weekday_each[col] / total_weeks
        weekday_each = weekday_each.round(0)
        
        # Case picks weekday averages
        case_picks = order_data_enhanced[order_data_enhanced['Pick_Type'] == 'Case_Pick']
        weekday_case = case_picks.groupby('Day_of_Week').agg({
            'Order No.': 'nunique',
            'Date': 'count',
            'Qty in Cases': 'sum',
            'Qty in Eaches': 'sum'
        }).rename(columns={'Order No.': 'Orders', 'Date': 'Lines', 'Qty in Cases': 'Cases', 'Qty in Eaches': 'Eaches'})
        
        for col in weekday_case.columns:
            weekday_case[col] = weekday_case[col] / total_weeks
        weekday_case = weekday_case.round(0)
        
        # Reorder by weekday
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_overall = weekday_overall.reindex(day_order).reset_index()
        weekday_each = weekday_each.reindex(day_order).reset_index() 
        weekday_case = weekday_case.reindex(day_order).reset_index()
        
        return {
            'existing_patterns': existing_patterns,
            'weekday_averages': {
                'overall': weekday_overall,
                'each_picks': weekday_each,
                'case_picks': weekday_case
            },
            'summary': {
                'busiest_day': weekday_overall.loc[weekday_overall['Orders'].idxmax(), 'Day_of_Week'],
                'quietest_day': weekday_overall.loc[weekday_overall['Orders'].idxmin(), 'Day_of_Week'],
                'total_weeks_analyzed': round(total_weeks, 1)
            }
        }
    
    def _classify_pick_types(self):
        """
        Classify orders into pick types (Each Pick vs Case Pick).
        
        Returns:
            DataFrame: Enhanced order data with pick type classification
        """
        order_data_copy = self.order_data.copy()
        
        # Simple classification based on quantities
        # Each Pick: Has eaches but no cases (or very few cases)
        # Case Pick: Has cases (may also have eaches)
        conditions = [
            (order_data_copy['Qty in Cases'] == 0) & (order_data_copy['Qty in Eaches'] > 0),
            (order_data_copy['Qty in Cases'] > 0)
        ]
        
        choices = ['Each_Pick', 'Case_Pick']
        
        order_data_copy['Pick_Type'] = np.select(conditions, choices, default='Mixed_Pick')
        
        return order_data_copy
    
    def _classify_pick_types_enhanced(self, order_data_enhanced):
        """
        Classify orders into pick types for data that already has case equivalent columns.
        
        Args:
            order_data_enhanced (pandas.DataFrame): Order data with case equivalent columns
            
        Returns:
            pandas.DataFrame: Enhanced order data with pick type classification
        """
        # Simple classification based on quantities
        conditions = [
            (order_data_enhanced['Qty in Cases'] == 0) & (order_data_enhanced['Qty in Eaches'] > 0),
            (order_data_enhanced['Qty in Cases'] > 0)
        ]
        
        choices = ['Each_Pick', 'Case_Pick']
        
        order_data_enhanced['Pick_Type'] = np.select(conditions, choices, default='Mixed_Pick')
        
        return order_data_enhanced

# Test function for standalone execution
if __name__ == "__main__":
    print("OrderAnalyzer module - ready for use")
    print("This module requires order data from DataLoader to function.")
    print("Use within the main analysis pipeline for proper functionality.")