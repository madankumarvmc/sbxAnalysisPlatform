"""
ABC-FMS Analysis Module for Warehouse Analysis Tool V2

PURPOSE:
This module performs cross-tabulation analysis combining ABC (volume) and FMS (frequency)
classifications to create a comprehensive SKU segmentation matrix that drives warehouse
layout, picking strategies, and inventory management decisions.

FOR BEGINNERS:
- ABC Classification: Based on case equivalent volume (standardized volume metric)
  A = High volume items (typically 70% of total case equivalent volume)
  B = Medium volume items (typically 20% of total case equivalent volume)  
  C = Low volume items (typically 10% of total case equivalent volume)
  
- Case Equivalent Volume = Cases + (Eaches ÷ Case_Config) for standardized comparison

- FMS Classification: Based on order frequency
  Fast = Frequently ordered items (typically 70% of orders)
  Medium = Moderately ordered items (typically 20% of orders)
  Slow = Infrequently ordered items (typically 10% of orders)

- Cross-tabulation creates 9 segments (A-Fast, A-Medium, A-Slow, etc.)
- Each segment requires different warehouse management strategies
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

class ABCFMSAnalyzer:
    """
    ABC-FMS cross-tabulation analysis class.
    
    This class performs advanced SKU segmentation by combining:
    - ABC analysis (volume-based classification)
    - FMS analysis (frequency-based classification)
    - Cross-tabulation matrix creation
    - Strategic recommendations for each segment
    """
    
    def __init__(self, order_data, sku_master=None, analysis_config=None):
        """
        Initialize the ABCFMSAnalyzer.
        
        Args:
            order_data (pandas.DataFrame): Cleaned order data from DataLoader
            sku_master (pandas.DataFrame): SKU master data with categories
            analysis_config (dict): Configuration parameters for analysis
        """
        self.order_data = order_data.copy()
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.config = analysis_config or {}
        
        # Initialize case equivalent converter
        self.converter = CaseEquivalentConverter(self.sku_master)
        
        # Set analysis parameters from config
        self.abc_thresholds = self.config.get('ABC_THRESHOLDS', config.DEFAULT_ABC_THRESHOLDS)
        self.fms_thresholds = self.config.get('FMS_THRESHOLDS', config.DEFAULT_FMS_THRESHOLDS)
        self.date_range = self.config.get('DATE_RANGE', {})
        
        # Filter data by date range if specified
        if self.date_range.get('START_DATE') or self.date_range.get('END_DATE'):
            self.order_data = self._filter_by_date_range(self.order_data)
        
        # Analysis results containers
        self.sku_classifications = None
        self.cross_tabulation = None
        self.segment_strategies = None
        
        print(f"ABCFMSAnalyzer initialized with {len(self.order_data)} order records")
        print(f"ABC Thresholds: A={self.abc_thresholds['A_THRESHOLD']}%, B={self.abc_thresholds['B_THRESHOLD']}%")
        print(f"FMS Thresholds: F={self.fms_thresholds['F_THRESHOLD']}%, M={self.fms_thresholds['M_THRESHOLD']}%")
    
    def run_complete_analysis(self):
        """
        Run complete ABC-FMS analysis pipeline.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        print("🔄 Running complete ABC-FMS analysis...")
        
        results = {
            'success': True,
            'analysis_date': datetime.now(),
            'data_summary': self._get_data_summary(),
            'sku_classifications': self.perform_dual_classification(),
            'cross_tabulation': self.create_cross_tabulation_matrix(),
            'segment_analysis': self.analyze_segments(),
            'strategic_recommendations': self.generate_strategic_recommendations(),
            'warehouse_layout': self.recommend_warehouse_layout(),
            'inventory_strategies': self.recommend_inventory_strategies()
        }
        
        print("✅ ABC-FMS analysis completed successfully")
        return results
    
    def perform_dual_classification(self):
        """
        Perform both ABC and FMS classifications on SKU data.
        
        Returns:
            dict: Dual classification results
        """
        print("🎯 Performing ABC and FMS classifications...")
        
        # Add case equivalent columns to order data
        enhanced_order_data = self.converter.add_case_equivalent_columns(self.order_data, self.sku_master)
        
        # Calculate SKU-level metrics for classification
        sku_metrics = enhanced_order_data.groupby('Sku Code').agg({
            'Date': ['count', 'nunique'],
            'Order No.': 'nunique',
            'Qty in Cases': 'sum',
            'Qty in Eaches': 'sum',
            'Case_Equivalent_Volume': 'sum'
        })
        
        # Flatten column names
        sku_metrics.columns = ['_'.join(col).strip() for col in sku_metrics.columns]
        sku_metrics = sku_metrics.reset_index()
        
        # Rename for clarity
        column_mapping = {
            'Date_count': 'Total_Order_Lines',
            'Date_nunique': 'Days_Ordered',
            'Order No._nunique': 'Unique_Orders',
            'Qty in Cases_sum': 'Total_Cases',
            'Qty in Eaches_sum': 'Total_Eaches',
            'Case_Equivalent_Volume_sum': 'Total_Case_Equivalent_Volume'
        }
        sku_metrics = sku_metrics.rename(columns=column_mapping)
        
        # Add SKU master data if available
        if self.sku_master is not None:
            sku_metrics['Sku Code'] = sku_metrics['Sku Code'].astype(str)
            sku_master_clean = self.sku_master.copy()
            sku_master_clean['Sku Code'] = sku_master_clean['Sku Code'].astype(str)
            
            sku_metrics = sku_metrics.merge(
                sku_master_clean[['Sku Code', 'Category', 'Case Config', 'Pallet Fit']],
                on='Sku Code',
                how='left'
            )
        
        # ABC Classification (Volume-based) - Using Case Equivalent Volume as primary metric
        sku_metrics = sku_metrics.sort_values('Total_Case_Equivalent_Volume', ascending=False).reset_index(drop=True)
        sku_metrics['Cumulative_Case_Equivalent_Volume'] = sku_metrics['Total_Case_Equivalent_Volume'].cumsum()
        total_case_equivalent_volume = sku_metrics['Total_Case_Equivalent_Volume'].sum()
        sku_metrics['Volume_Cumulative_Percent'] = (sku_metrics['Cumulative_Case_Equivalent_Volume'] / total_case_equivalent_volume * 100)
        
        # Keep legacy calculations for backward compatibility
        sku_metrics['Cumulative_Cases'] = sku_metrics['Total_Cases'].cumsum()
        total_cases = sku_metrics['Total_Cases'].sum()
        sku_metrics['Volume_Cumulative_Percent_Legacy'] = (sku_metrics['Cumulative_Cases'] / total_cases * 100)
        
        def classify_abc(cumulative_percent):
            if cumulative_percent <= self.abc_thresholds['A_THRESHOLD']:
                return 'A'
            elif cumulative_percent <= self.abc_thresholds['B_THRESHOLD']:
                return 'B'
            else:
                return 'C'
        
        sku_metrics['ABC_Category'] = sku_metrics['Volume_Cumulative_Percent'].apply(classify_abc)
        # Keep legacy ABC classification for comparison
        sku_metrics['ABC_Category_Legacy'] = sku_metrics['Volume_Cumulative_Percent_Legacy'].apply(classify_abc)
        
        # FMS Classification (Frequency-based)
        # Calculate total days and order frequency
        total_days = self.order_data['Date'].nunique()
        sku_metrics['Order_Frequency_Percent'] = (sku_metrics['Days_Ordered'] / total_days * 100)
        
        # Sort by frequency for FMS classification
        sku_metrics_freq = sku_metrics.sort_values('Order_Frequency_Percent', ascending=False).reset_index(drop=True)
        sku_metrics_freq['Cumulative_Frequency'] = sku_metrics_freq['Order_Frequency_Percent'].cumsum()
        total_frequency = sku_metrics_freq['Order_Frequency_Percent'].sum()
        sku_metrics_freq['Frequency_Cumulative_Percent'] = (sku_metrics_freq['Cumulative_Frequency'] / total_frequency * 100)
        
        def classify_fms(cumulative_percent):
            if cumulative_percent <= self.fms_thresholds['F_THRESHOLD']:
                return 'Fast'
            elif cumulative_percent <= self.fms_thresholds['M_THRESHOLD']:
                return 'Medium'
            else:
                return 'Slow'
        
        sku_metrics_freq['FMS_Category'] = sku_metrics_freq['Frequency_Cumulative_Percent'].apply(classify_fms)
        
        # Merge FMS classification back to main dataset
        sku_metrics = sku_metrics.merge(
            sku_metrics_freq[['Sku Code', 'FMS_Category', 'Frequency_Cumulative_Percent']],
            on='Sku Code',
            how='left'
        )
        
        # Create combined ABC-FMS classification
        sku_metrics['ABC_FMS_Segment'] = sku_metrics['ABC_Category'] + '-' + sku_metrics['FMS_Category']
        
        # Calculate additional performance metrics using case equivalent volume as primary
        sku_metrics['Case_Equivalent_Per_Order_Line'] = (sku_metrics['Total_Case_Equivalent_Volume'] / sku_metrics['Total_Order_Lines']).round(2)
        sku_metrics['Case_Equivalent_Per_Day'] = (sku_metrics['Total_Case_Equivalent_Volume'] / sku_metrics['Days_Ordered']).round(2)
        sku_metrics['Cases_Per_Order_Line'] = (sku_metrics['Total_Cases'] / sku_metrics['Total_Order_Lines']).round(2)
        sku_metrics['Cases_Per_Day'] = (sku_metrics['Total_Cases'] / sku_metrics['Days_Ordered']).round(2)
        sku_metrics['Order_Lines_Per_Day'] = (sku_metrics['Total_Order_Lines'] / sku_metrics['Days_Ordered']).round(2)
        
        self.sku_classifications = sku_metrics
        
        return {
            'sku_with_classifications': sku_metrics,
            'total_skus_classified': len(sku_metrics),
            'abc_thresholds_used': self.abc_thresholds,
            'fms_thresholds_used': self.fms_thresholds
        }
    
    def create_cross_tabulation_matrix(self):
        """
        Create ABC-FMS cross-tabulation matrix.
        
        Returns:
            dict: Cross-tabulation analysis results
        """
        print("📊 Creating ABC-FMS cross-tabulation matrix...")
        
        if self.sku_classifications is None:
            self.perform_dual_classification()
        
        # Create cross-tabulation matrix
        cross_tab = pd.crosstab(
            self.sku_classifications['ABC_Category'],
            self.sku_classifications['FMS_Category'],
            values=self.sku_classifications['Sku Code'],
            aggfunc='count'
        ).fillna(0)
        
        # Create volume cross-tabulation using case equivalent volume as primary metric
        volume_cross_tab = pd.crosstab(
            self.sku_classifications['ABC_Category'],
            self.sku_classifications['FMS_Category'],
            values=self.sku_classifications['Total_Case_Equivalent_Volume'],
            aggfunc='sum'
        ).fillna(0)
        
        # Keep legacy volume cross-tabulation for comparison
        volume_cross_tab_legacy = pd.crosstab(
            self.sku_classifications['ABC_Category'],
            self.sku_classifications['FMS_Category'],
            values=self.sku_classifications['Total_Cases'],
            aggfunc='sum'
        ).fillna(0)
        
        # Create detailed segment analysis using case equivalent volume as primary metric
        segment_details = self.sku_classifications.groupby('ABC_FMS_Segment').agg({
            'Sku Code': 'count',
            'Total_Case_Equivalent_Volume': ['sum', 'mean'],
            'Total_Cases': ['sum', 'mean'],
            'Days_Ordered': 'mean',
            'Order_Frequency_Percent': 'mean',
            'Case_Equivalent_Per_Day': 'mean',
            'Cases_Per_Day': 'mean',
            'Order_Lines_Per_Day': 'mean'
        }).round(2)
        
        # Flatten column names
        segment_details.columns = ['_'.join(col).strip() for col in segment_details.columns]
        segment_details = segment_details.reset_index()
        
        # Rename columns with case equivalent volume as primary
        column_mapping = {
            'Sku Code_count': 'SKU_Count',
            'Total_Case_Equivalent_Volume_sum': 'Total_Case_Equivalent_Volume',
            'Total_Case_Equivalent_Volume_mean': 'Avg_Case_Equivalent_Volume_Per_SKU',
            'Total_Cases_sum': 'Total_Volume',
            'Total_Cases_mean': 'Avg_Volume_Per_SKU',
            'Days_Ordered_mean': 'Avg_Days_Ordered',
            'Order_Frequency_Percent_mean': 'Avg_Frequency_Percent',
            'Case_Equivalent_Per_Day_mean': 'Avg_Case_Equivalent_Per_Day',
            'Cases_Per_Day_mean': 'Avg_Cases_Per_Day',
            'Order_Lines_Per_Day_mean': 'Avg_Order_Lines_Per_Day'
        }
        segment_details = segment_details.rename(columns=column_mapping)
        
        # Calculate percentages using case equivalent volume as primary metric
        total_skus = segment_details['SKU_Count'].sum()
        total_case_equivalent_volume = segment_details['Total_Case_Equivalent_Volume'].sum()
        total_volume = segment_details['Total_Volume'].sum()
        
        segment_details['SKU_Percent'] = (segment_details['SKU_Count'] / total_skus * 100).round(2)
        segment_details['Case_Equivalent_Volume_Percent'] = (segment_details['Total_Case_Equivalent_Volume'] / total_case_equivalent_volume * 100).round(2)
        segment_details['Volume_Percent'] = (segment_details['Total_Volume'] / total_volume * 100).round(2)
        
        # Sort by importance (case equivalent volume descending)
        segment_details = segment_details.sort_values('Total_Case_Equivalent_Volume', ascending=False)
        
        self.cross_tabulation = {
            'sku_matrix': cross_tab,
            'volume_matrix': volume_cross_tab,
            'volume_matrix_legacy': volume_cross_tab_legacy,
            'segment_details': segment_details
        }
        
        return self.cross_tabulation
    
    def analyze_segments(self):
        """
        Analyze characteristics of each ABC-FMS segment.
        
        Returns:
            dict: Segment analysis results
        """
        print("🔍 Analyzing segment characteristics...")
        
        if self.cross_tabulation is None:
            self.create_cross_tabulation_matrix()
        
        segment_details = self.cross_tabulation['segment_details']
        
        # Define segment characteristics and priorities
        segment_profiles = {
            'A-Fast': {
                'priority': 'Critical',
                'characteristics': 'High volume, high frequency - core business drivers',
                'warehouse_priority': 1,
                'pick_zone': 'Golden Zone',
                'inventory_strategy': 'High service level, frequent replenishment'
            },
            'A-Medium': {
                'priority': 'High',
                'characteristics': 'High volume, medium frequency - important steady items',
                'warehouse_priority': 2,
                'pick_zone': 'Primary Zone',
                'inventory_strategy': 'High service level, regular replenishment'
            },
            'A-Slow': {
                'priority': 'High',
                'characteristics': 'High volume, low frequency - bulk or seasonal items',
                'warehouse_priority': 3,
                'pick_zone': 'Bulk Zone',
                'inventory_strategy': 'Demand forecasting critical'
            },
            'B-Fast': {
                'priority': 'High',
                'characteristics': 'Medium volume, high frequency - frequent small orders',
                'warehouse_priority': 4,
                'pick_zone': 'Primary Zone',
                'inventory_strategy': 'Moderate service level, frequent replenishment'
            },
            'B-Medium': {
                'priority': 'Medium',
                'characteristics': 'Medium volume, medium frequency - standard items',
                'warehouse_priority': 5,
                'pick_zone': 'Secondary Zone',
                'inventory_strategy': 'Standard service level and replenishment'
            },
            'B-Slow': {
                'priority': 'Medium',
                'characteristics': 'Medium volume, low frequency - occasional bulk orders',
                'warehouse_priority': 6,
                'pick_zone': 'Secondary Zone',
                'inventory_strategy': 'Lower service level acceptable'
            },
            'C-Fast': {
                'priority': 'Medium',
                'characteristics': 'Low volume, high frequency - small regular items',
                'warehouse_priority': 7,
                'pick_zone': 'Secondary Zone',
                'inventory_strategy': 'Minimize inventory, frequent orders'
            },
            'C-Medium': {
                'priority': 'Low',
                'characteristics': 'Low volume, medium frequency - occasional items',
                'warehouse_priority': 8,
                'pick_zone': 'Tertiary Zone',
                'inventory_strategy': 'Minimize inventory'
            },
            'C-Slow': {
                'priority': 'Low',
                'characteristics': 'Low volume, low frequency - dead stock candidates',
                'warehouse_priority': 9,
                'pick_zone': 'Remote Zone',
                'inventory_strategy': 'Consider discontinuation'
            }
        }
        
        # Add profile information to segment details
        segment_analysis = segment_details.copy()
        segment_analysis['Priority'] = segment_analysis['ABC_FMS_Segment'].map(
            lambda x: segment_profiles.get(x, {}).get('priority', 'Unknown')
        )
        segment_analysis['Characteristics'] = segment_analysis['ABC_FMS_Segment'].map(
            lambda x: segment_profiles.get(x, {}).get('characteristics', 'Unknown')
        )
        segment_analysis['Warehouse_Priority'] = segment_analysis['ABC_FMS_Segment'].map(
            lambda x: segment_profiles.get(x, {}).get('warehouse_priority', 999)
        )
        segment_analysis['Pick_Zone'] = segment_analysis['ABC_FMS_Segment'].map(
            lambda x: segment_profiles.get(x, {}).get('pick_zone', 'Unknown')
        )
        segment_analysis['Inventory_Strategy'] = segment_analysis['ABC_FMS_Segment'].map(
            lambda x: segment_profiles.get(x, {}).get('inventory_strategy', 'Unknown')
        )
        
        # Sort by warehouse priority
        segment_analysis = segment_analysis.sort_values('Warehouse_Priority')
        
        return {
            'segment_profiles': segment_analysis,
            'critical_segments': segment_analysis[segment_analysis['Priority'] == 'Critical'],
            'high_priority_segments': segment_analysis[segment_analysis['Priority'] == 'High'],
            'total_segments': len(segment_analysis),
            'segment_definitions': segment_profiles
        }
    
    def generate_strategic_recommendations(self):
        """
        Generate strategic recommendations based on ABC-FMS analysis.
        
        Returns:
            dict: Strategic recommendations
        """
        print("💡 Generating strategic recommendations...")
        
        if self.cross_tabulation is None:
            self.create_cross_tabulation_matrix()
        
        segment_details = self.cross_tabulation['segment_details']
        recommendations = []
        
        # Analyze A-Fast segment (most critical) using case equivalent volume as primary metric
        a_fast = segment_details[segment_details['ABC_FMS_Segment'] == 'A-Fast']
        if len(a_fast) > 0:
            a_fast_case_equiv_volume_percent = a_fast['Case_Equivalent_Volume_Percent'].iloc[0]
            a_fast_volume_percent = a_fast['Volume_Percent'].iloc[0]
            a_fast_sku_count = a_fast['SKU_Count'].iloc[0]
            
            recommendations.append({
                'segment': 'A-Fast',
                'priority': 'Critical',
                'recommendation': f'Focus maximum attention on {a_fast_sku_count} A-Fast SKUs ({a_fast_case_equiv_volume_percent:.1f}% of case equivalent volume, {a_fast_volume_percent:.1f}% of raw volume)',
                'actions': [
                    'Place in golden zone (eye level, closest to shipping)',
                    'Implement 99.5%+ service level target',
                    'Use daily replenishment cycles',
                    'Consider automated picking systems'
                ]
            })
        
        # Analyze C-Slow segment (potential dead stock) using case equivalent volume as primary metric
        c_slow = segment_details[segment_details['ABC_FMS_Segment'] == 'C-Slow']
        if len(c_slow) > 0:
            c_slow_sku_count = c_slow['SKU_Count'].iloc[0]
            c_slow_case_equiv_volume_percent = c_slow['Case_Equivalent_Volume_Percent'].iloc[0]
            c_slow_volume_percent = c_slow['Volume_Percent'].iloc[0]
            
            recommendations.append({
                'segment': 'C-Slow',
                'priority': 'High',
                'recommendation': f'Review {c_slow_sku_count} C-Slow SKUs for potential elimination ({c_slow_case_equiv_volume_percent:.1f}% of case equivalent volume, {c_slow_volume_percent:.1f}% of raw volume)',
                'actions': [
                    'Conduct dead stock analysis',
                    'Consider discontinuation or liquidation',
                    'Move to remote storage locations',
                    'Implement strict inventory controls'
                ]
            })
        
        # Analyze segment balance
        a_category_skus = segment_details[segment_details['ABC_FMS_Segment'].str.startswith('A')]['SKU_Count'].sum()
        total_skus = segment_details['SKU_Count'].sum()
        a_percentage = (a_category_skus / total_skus * 100)
        
        if a_percentage > 30:
            recommendations.append({
                'segment': 'ABC Balance',
                'priority': 'Medium',
                'recommendation': f'A-category represents {a_percentage:.1f}% of SKUs - consider tightening criteria',
                'actions': [
                    'Review ABC threshold settings',
                    'Focus on true high-impact items',
                    'Consider sub-categorization within A items'
                ]
            })
        
        # Pick zone optimization
        recommendations.append({
            'segment': 'Pick Zone Optimization',
            'priority': 'High',
            'recommendation': 'Optimize warehouse layout based on ABC-FMS segments',
            'actions': [
                'Golden Zone: A-Fast items only',
                'Primary Zone: A-Medium and B-Fast items',
                'Secondary Zone: B-Medium and C-Fast items',
                'Remote Zone: C-Slow items'
            ]
        })
        
        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'implementation_priority': sorted(recommendations, key=lambda x: {'Critical': 1, 'High': 2, 'Medium': 3, 'Low': 4}[x['priority']])
        }
    
    def recommend_warehouse_layout(self):
        """
        Recommend warehouse layout based on ABC-FMS analysis.
        
        Returns:
            dict: Warehouse layout recommendations
        """
        print("🏭 Generating warehouse layout recommendations...")
        
        if self.cross_tabulation is None:
            self.create_cross_tabulation_matrix()
        
        segment_details = self.cross_tabulation['segment_details']
        
        # Define pick zones with space allocation
        pick_zones = {
            'Golden Zone (A-Fast)': {
                'segments': ['A-Fast'],
                'space_allocation_percent': 15,
                'characteristics': 'Eye level, closest to shipping dock, easiest access',
                'picking_method': 'Automated or highly efficient manual'
            },
            'Primary Zone (High Priority)': {
                'segments': ['A-Medium', 'B-Fast'],
                'space_allocation_percent': 35,
                'characteristics': 'Easy access, good visibility, near golden zone',
                'picking_method': 'Manual picking with RF scanners'
            },
            'Secondary Zone (Standard)': {
                'segments': ['B-Medium', 'B-Slow', 'C-Fast'],
                'space_allocation_percent': 35,
                'characteristics': 'Standard racking, normal access',
                'picking_method': 'Batch picking acceptable'
            },
            'Remote Zone (Low Priority)': {
                'segments': ['C-Medium', 'C-Slow'],
                'space_allocation_percent': 15,
                'characteristics': 'Higher racks, further from shipping, bulk storage',
                'picking_method': 'Forklift picking, consolidation area'
            }
        }
        
        # Calculate actual space requirements using case equivalent volume as primary metric
        zone_requirements = {}
        for zone_name, zone_info in pick_zones.items():
            zone_skus = segment_details[segment_details['ABC_FMS_Segment'].isin(zone_info['segments'])]
            zone_requirements[zone_name] = {
                'sku_count': zone_skus['SKU_Count'].sum(),
                'case_equivalent_volume_percent': zone_skus['Case_Equivalent_Volume_Percent'].sum(),
                'volume_percent': zone_skus['Volume_Percent'].sum(),
                'recommended_space_percent': zone_info['space_allocation_percent'],
                'characteristics': zone_info['characteristics'],
                'picking_method': zone_info['picking_method']
            }
        
        return {
            'pick_zone_layout': zone_requirements,
            'layout_principles': [
                'Minimize travel time for high-frequency items',
                'Place A-Fast items in golden zone (eye level)',
                'Co-locate complementary products',
                'Ensure adequate space for future growth',
                'Consider seasonal variation in placements'
            ],
            'implementation_steps': [
                '1. Audit current warehouse layout',
                '2. Identify golden zone areas',
                '3. Relocate A-Fast items to optimal positions',
                '4. Reorganize remaining items by ABC-FMS segments',
                '5. Update warehouse management system',
                '6. Train picking staff on new layout'
            ]
        }
    
    def recommend_inventory_strategies(self):
        """
        Recommend inventory management strategies by segment.
        
        Returns:
            dict: Inventory strategy recommendations
        """
        print("📦 Generating inventory strategy recommendations...")
        
        if self.cross_tabulation is None:
            self.create_cross_tabulation_matrix()
        
        segment_details = self.cross_tabulation['segment_details']
        
        inventory_strategies = {}
        
        for _, segment in segment_details.iterrows():
            segment_name = segment['ABC_FMS_Segment']
            
            # Determine service level target
            if segment_name in ['A-Fast', 'A-Medium']:
                service_level = 99.5
                safety_stock_days = 3
                review_frequency = 'Daily'
            elif segment_name in ['B-Fast', 'A-Slow']:
                service_level = 98.0
                safety_stock_days = 5
                review_frequency = 'Weekly'
            elif segment_name in ['B-Medium', 'C-Fast']:
                service_level = 95.0
                safety_stock_days = 7
                review_frequency = 'Bi-weekly'
            else:  # C-Medium, C-Slow
                service_level = 90.0
                safety_stock_days = 14
                review_frequency = 'Monthly'
            
            inventory_strategies[segment_name] = {
                'sku_count': segment['SKU_Count'],
                'case_equivalent_volume_percent': segment['Case_Equivalent_Volume_Percent'],
                'volume_percent': segment['Volume_Percent'],
                'service_level_target': service_level,
                'safety_stock_days': safety_stock_days,
                'review_frequency': review_frequency,
                'replenishment_priority': self._get_replenishment_priority(segment_name),
                'forecasting_method': self._get_forecasting_method(segment_name),
                'inventory_investment': self._calculate_inventory_investment_case_equivalent(segment),
                'inventory_investment_legacy': self._calculate_inventory_investment(segment)
            }
        
        return {
            'segment_strategies': inventory_strategies,
            'overall_recommendations': [
                'Implement differentiated service levels by segment',
                'Use advanced forecasting for A-items',
                'Consider vendor-managed inventory for A-Fast items',
                'Implement regular dead stock reviews for C-Slow items',
                'Use ABC-FMS segments for budgeting and planning'
            ]
        }
    
    def _get_replenishment_priority(self, segment):
        """Get replenishment priority for segment"""
        priority_map = {
            'A-Fast': 1, 'A-Medium': 2, 'B-Fast': 3,
            'A-Slow': 4, 'B-Medium': 5, 'C-Fast': 6,
            'B-Slow': 7, 'C-Medium': 8, 'C-Slow': 9
        }
        return priority_map.get(segment, 9)
    
    def _get_forecasting_method(self, segment):
        """Get recommended forecasting method for segment"""
        if segment in ['A-Fast', 'A-Medium', 'B-Fast']:
            return 'Advanced (ML/AI)'
        elif segment in ['A-Slow', 'B-Medium']:
            return 'Statistical'
        else:
            return 'Simple moving average'
    
    def _calculate_inventory_investment_case_equivalent(self, segment_data):
        """Calculate relative inventory investment recommendation using case equivalent volume"""
        case_equiv_volume_percent = segment_data['Case_Equivalent_Volume_Percent']
        if case_equiv_volume_percent > 20:
            return 'High'
        elif case_equiv_volume_percent > 5:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_inventory_investment(self, segment_data):
        """Calculate relative inventory investment recommendation (legacy)"""
        volume_percent = segment_data['Volume_Percent']
        if volume_percent > 20:
            return 'High'
        elif volume_percent > 5:
            return 'Medium'
        else:
            return 'Low'
    
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
        """Get basic data summary including case equivalent volume metrics"""
        # Add case equivalent columns to order data for summary
        enhanced_order_data = self.converter.add_case_equivalent_columns(self.order_data, self.sku_master)
        
        summary = {
            'total_order_records': len(self.order_data),
            'unique_skus': self.order_data['Sku Code'].nunique(),
            'total_case_equivalent_volume': enhanced_order_data['Case_Equivalent_Volume'].sum(),
            'total_volume_cases': self.order_data['Qty in Cases'].sum(),
            'total_volume_eaches': self.order_data['Qty in Eaches'].sum(),
            'case_equivalent_to_cases_ratio': round(enhanced_order_data['Case_Equivalent_Volume'].sum() / self.order_data['Qty in Cases'].sum(), 2) if self.order_data['Qty in Cases'].sum() > 0 else 0,
            'analysis_period': {
                'start': self.order_data['Date'].min().strftime('%Y-%m-%d'),
                'end': self.order_data['Date'].max().strftime('%Y-%m-%d'),
                'days': self.order_data['Date'].nunique()
            },
            'classification_thresholds': {
                'abc_thresholds': self.abc_thresholds,
                'fms_thresholds': self.fms_thresholds
            }
        }
        
        return summary

# Test function for standalone execution
if __name__ == "__main__":
    print("ABCFMSAnalyzer module - ready for use")
    print("This module requires order data and optionally SKU master data to function.")
    print("Use within the main analysis pipeline for proper functionality.")