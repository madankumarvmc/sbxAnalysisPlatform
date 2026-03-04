"""
Manpower Analysis Module for Warehouse Analysis Tool

PURPOSE:
This module analyzes manpower requirements and efficiency for warehouse operations
including picking, receiving, putaway, and loading activities.

FEATURES:
- Picking manpower analysis and time studies
- Receiving and putaway efficiency calculations
- Loading manpower requirements
- Resource optimization recommendations

NOTE: This is a placeholder module. The actual implementation will be added later.
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import warnings
from typing import Optional, Dict, Any, List

class ManpowerAnalyzer:
    """
    Comprehensive manpower analysis for warehouse operations.
    
    This class analyzes manpower requirements and efficiency for:
    - Picking operations
    - Receiving and putaway operations  
    - Loading operations
    - Overall workforce optimization
    
    NOTE: This is a placeholder implementation.
    """
    
    def __init__(self, order_data: Optional[pd.DataFrame] = None,
                 receipt_data: Optional[pd.DataFrame] = None,
                 sku_master: Optional[pd.DataFrame] = None,
                 analysis_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ManpowerAnalyzer.
        
        Args:
            order_data (pd.DataFrame, optional): Order data for picking analysis
            receipt_data (pd.DataFrame, optional): Receipt data for receiving analysis
            sku_master (pd.DataFrame, optional): SKU master data
            analysis_config (dict, optional): Configuration parameters
        """
        self.order_data = order_data.copy() if order_data is not None else None
        self.receipt_data = receipt_data.copy() if receipt_data is not None else None
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.config = analysis_config or {}
        
        # Extract configuration parameters
        self.manpower_params = self.config.get('MANPOWER_PARAMS', {})
        self.date_range = self.config.get('DATE_RANGE', {})
        
        # Extract timing parameters
        self.picking_params = self.manpower_params.get('picking', {})
        self.receiving_params = self.manpower_params.get('receiving_putaway', {})
        self.loading_params = self.manpower_params.get('loading', {})
        
        # Analysis results containers (placeholders)
        self.picking_analysis = None
        self.receiving_analysis = None
        self.loading_analysis = None
        self.efficiency_summary = None
        
        data_sources = []
        if self.order_data is not None:
            data_sources.append(f"{len(self.order_data)} orders")
        if self.receipt_data is not None:
            data_sources.append(f"{len(self.receipt_data)} receipts")
            
        print(f"ManpowerAnalyzer initialized with: {', '.join(data_sources) if data_sources else 'no data'}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete manpower analysis pipeline.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        print("🔄 Running complete manpower analysis...")
        
        results = {
            'success': True,
            'analysis_date': datetime.now(),
            'data_summary': self._get_data_summary(),
            'picking_analysis': self.analyze_picking_manpower(),
            'receiving_analysis': self.analyze_receiving_manpower(),
            'loading_analysis': self.analyze_loading_manpower(),
            'efficiency_summary': self.analyze_efficiency_summary()
        }
        
        print("✅ Manpower analysis completed successfully")
        return results
    
    def analyze_picking_manpower(self) -> Dict[str, Any]:
        """
        Analyze picking manpower requirements and efficiency.
        Uses Simple or Detailed mode based on configuration.
        
        Returns:
            dict: Picking manpower analysis results
        """
        print("📊 Analyzing picking manpower requirements...")
        
        # Get complexity mode from config
        complexity_mode = self.manpower_params.get('complexity_mode', 'Simple')
        
        if complexity_mode == 'Simple':
            picking_analysis = self._analyze_picking_simple()
        else:
            picking_analysis = self._analyze_picking_detailed()
            
        self.picking_analysis = picking_analysis
        return picking_analysis
    
    def _analyze_picking_simple(self) -> Dict[str, Any]:
        """
        Simple picking analysis using single time per pallet and percentile volume.
        
        Returns:
            dict: Simple picking analysis results
        """
        # Get simplified picking parameters
        picking_config = self.manpower_params.get('picking_simplified', {})
        
        # Basic analysis structure
        analysis = {
            'complexity_mode': 'Simple',
            'total_orders_analyzed': len(self.order_data) if self.order_data is not None else 0,
            'calculation_method': '95th percentile daily volume → pallets → time → staff',
            'parameters_used': picking_config.copy(),
            'daily_summary': {},
            'shift_breakdown': [],
            'hourly_requirements': [],
            'notes': 'Simple calculation using average time per pallet'
        }
        
        if self.order_data is None or len(self.order_data) == 0:
            analysis['error'] = 'No order data available for picking analysis'
            return analysis
        
        try:
            # Step 1: Calculate daily volumes and get percentile
            daily_cases = self._calculate_daily_volumes()
            percentile = picking_config.get('percentile_for_planning', 95)
            peak_daily_cases = daily_cases.quantile(percentile / 100)

            # Order line analysis — each line = one pick trip (primary warehousing KPI)
            daily_order_lines = self.order_data.groupby('Date').size()
            peak_daily_lines = daily_order_lines.quantile(percentile / 100)
            avg_daily_lines = daily_order_lines.mean()

            # Pick type breakdown (inline classification from Qty in Cases / Qty in Eaches)
            _has_cases = (self.order_data['Qty in Cases'] > 0) if 'Qty in Cases' in self.order_data.columns else pd.Series(False, index=self.order_data.index)
            _has_eaches = (self.order_data['Qty in Eaches'] > 0) if 'Qty in Eaches' in self.order_data.columns else pd.Series(False, index=self.order_data.index)
            _pick_type = pd.Series('No_Volume', index=self.order_data.index)
            _pick_type[_has_cases & ~_has_eaches] = 'Case_Pick'
            _pick_type[~_has_cases & _has_eaches] = 'Each_Pick'
            _pick_type[_has_cases & _has_eaches] = 'Mixed_Pick'
            # Aggregate by pick type
            _line_counts = _pick_type.value_counts().rename('Order_Lines')
            _case_sums = self.order_data.groupby(_pick_type)['Qty in Cases'].sum().rename('Total_Cases') if 'Qty in Cases' in self.order_data.columns else pd.Series(dtype=float)
            _each_sums = self.order_data.groupby(_pick_type)['Qty in Eaches'].sum().rename('Total_Eaches') if 'Qty in Eaches' in self.order_data.columns else pd.Series(dtype=float)
            pick_type_summary = pd.concat([_line_counts, _case_sums, _each_sums], axis=1).fillna(0).to_dict('index')
            
            # Step 2: Convert to pallets
            cases_per_pallet = picking_config.get('average_cases_per_pallet', 100)
            total_pallets = peak_daily_cases / cases_per_pallet
            
            # Step 3: Calculate time required
            time_per_pallet_min = picking_config.get('average_time_per_pallet', 30.0)
            theoretical_time_hours = (total_pallets * time_per_pallet_min) / 60
            
            # Step 4: Adjust for efficiency
            work_efficiency = max(picking_config.get('work_efficiency', 85.0) / 100, 0.01)
            actual_time_hours = theoretical_time_hours / work_efficiency

            # Step 5: Calculate staff requirements
            shift_hours = picking_config.get('shift_hours', 8.0)
            break_time_hours = picking_config.get('break_time_minutes', 30) / 60
            net_hours_per_person = shift_hours - break_time_hours
            
            pickers_required = math.ceil(actual_time_hours / net_hours_per_person)
            
            # Multiple shifts
            shifts_per_day = picking_config.get('shifts_per_day', 1)
            pickers_per_shift = math.ceil(pickers_required / shifts_per_day)
            
            # Daily Summary
            analysis['daily_summary'] = {
                'planning_percentile': f"{percentile}th percentile",
                'peak_daily_cases': round(peak_daily_cases, 0),
                'peak_daily_order_lines': round(peak_daily_lines, 0),
                'avg_daily_order_lines': round(avg_daily_lines, 1),
                'total_pallets': round(total_pallets, 1),
                'theoretical_time_hours': round(theoretical_time_hours, 2),
                'actual_time_hours': round(actual_time_hours, 2),
                'work_efficiency_percent': picking_config.get('work_efficiency', 85.0),
                'total_pickers_required': pickers_required,
                'pickers_per_shift': pickers_per_shift,
                'capacity_utilization': round((actual_time_hours / (pickers_required * net_hours_per_person)) * 100, 1),
                'productivity_benchmark_lines_per_person_day': round(
                    peak_daily_lines / pickers_required, 1
                ) if pickers_required > 0 else 0,
            }
            analysis['pick_type_breakdown'] = pick_type_summary
            
            # Shift Breakdown
            for shift in range(shifts_per_day):
                shift_start = 8 + (shift * shift_hours)  # Starting at 8 AM
                shift_end = shift_start + shift_hours
                
                analysis['shift_breakdown'].append({
                    'shift_number': shift + 1,
                    'shift_time': f"{shift_start:02.0f}:00-{shift_end:02.0f}:00",
                    'pickers_required': pickers_per_shift,
                    'productive_hours_per_person': net_hours_per_person,
                    'total_capacity_hours': pickers_per_shift * net_hours_per_person,
                    'planned_workload_hours': actual_time_hours / shifts_per_day,
                    'utilization_percent': round((actual_time_hours / shifts_per_day) / (pickers_per_shift * net_hours_per_person) * 100, 1)
                })
            
            # Hourly Requirements (spread evenly across working hours)
            hours_per_day = shifts_per_day * shift_hours
            pallets_per_hour = total_pallets / hours_per_day
            
            for hour in range(int(hours_per_day)):
                hour_start = 8 + hour  # Starting at 8 AM
                
                analysis['hourly_requirements'].append({
                    'hour': f"{hour_start:02.0f}:00",
                    'pallets_to_pick': round(pallets_per_hour, 1),
                    'time_required_minutes': round(pallets_per_hour * time_per_pallet_min, 1),
                    'staff_required': pickers_per_shift
                })
                
        except Exception as e:
            analysis['error'] = f"Calculation error: {str(e)}"
            print(f"⚠️ Error in picking analysis: {str(e)}")
        
        return analysis
    
    def _analyze_picking_detailed(self) -> Dict[str, Any]:
        """
        Detailed picking analysis (placeholder for future implementation).
        
        Returns:
            dict: Detailed picking analysis results (placeholder)
        """
        return {
            'complexity_mode': 'Detailed',
            'status': 'Not implemented',
            'notes': 'Detailed picking analysis will be implemented later'
        }
    
    def _calculate_daily_volumes(self) -> pd.Series:
        """
        Calculate daily case volumes from order data.
        
        Returns:
            pd.Series: Daily case volumes indexed by date
        """
        if 'Date' not in self.order_data.columns or 'Qty in Cases' not in self.order_data.columns:
            raise ValueError("Required columns 'Date' and 'Qty in Cases' not found in order data")
        
        # Group by date and sum cases
        daily_cases = self.order_data.groupby('Date')['Qty in Cases'].sum()
        
        print(f"📊 Daily volume analysis: {len(daily_cases)} days, avg {daily_cases.mean():.0f} cases/day")
        
        return daily_cases
    
    def analyze_receiving_manpower(self) -> Dict[str, Any]:
        """
        Analyze receiving and putaway manpower requirements.
        Uses Simple or Detailed mode based on configuration.
        
        Returns:
            dict: Receiving manpower analysis results
        """
        print("📊 Analyzing receiving and putaway manpower...")
        
        # Get complexity mode from config
        complexity_mode = self.manpower_params.get('complexity_mode', 'Simple')
        
        if complexity_mode == 'Simple':
            receiving_analysis = self._analyze_receiving_simple()
        else:
            receiving_analysis = self._analyze_receiving_detailed()
            
        self.receiving_analysis = receiving_analysis
        return receiving_analysis
    
    def _analyze_receiving_simple(self) -> Dict[str, Any]:
        """
        Simple receiving analysis using single time per pallet and percentile volume.
        
        Returns:
            dict: Simple receiving analysis results
        """
        # Get simplified receiving parameters
        receiving_config = self.manpower_params.get('receiving_simplified', {})
        
        # Basic analysis structure
        analysis = {
            'complexity_mode': 'Simple',
            'total_receipts_analyzed': len(self.receipt_data) if self.receipt_data is not None else 0,
            'calculation_method': '95th percentile daily volume → pallets → time → staff',
            'parameters_used': receiving_config.copy(),
            'daily_summary': {},
            'shift_breakdown': [],
            'hourly_requirements': [],
            'notes': 'Simple calculation using average time per pallet for unloading + putaway'
        }
        
        if self.receipt_data is None or len(self.receipt_data) == 0:
            analysis['error'] = 'No receipt data available for receiving analysis'
            return analysis
        
        try:
            # Step 1: Calculate daily receipt volumes and get percentile
            daily_cases = self._calculate_daily_receipt_volumes()
            percentile = receiving_config.get('percentile_for_planning', 95)
            peak_daily_cases = daily_cases.quantile(percentile / 100)
            
            # Step 2: Convert to pallets
            cases_per_pallet = receiving_config.get('average_cases_per_pallet', 100)
            total_pallets = peak_daily_cases / cases_per_pallet
            
            # Step 3: Calculate time required
            time_per_pallet_min = receiving_config.get('unloading_putaway_time_per_pallet', 45.0)
            theoretical_time_hours = (total_pallets * time_per_pallet_min) / 60
            
            # Step 4: Adjust for efficiency
            work_efficiency = max(receiving_config.get('work_efficiency', 85.0) / 100, 0.01)
            actual_time_hours = theoretical_time_hours / work_efficiency

            # Step 5: Calculate staff requirements
            shift_hours = receiving_config.get('shift_hours', 8.0)
            break_time_hours = receiving_config.get('break_time_minutes', 30) / 60
            net_hours_per_person = shift_hours - break_time_hours
            
            receivers_required = math.ceil(actual_time_hours / net_hours_per_person)
            
            # Multiple shifts
            shifts_per_day = receiving_config.get('shifts_per_day', 1)
            receivers_per_shift = math.ceil(receivers_required / shifts_per_day)
            
            # Daily Summary
            analysis['daily_summary'] = {
                'planning_percentile': f"{percentile}th percentile",
                'peak_daily_cases': round(peak_daily_cases, 0),
                'total_pallets': round(total_pallets, 1),
                'theoretical_time_hours': round(theoretical_time_hours, 2),
                'actual_time_hours': round(actual_time_hours, 2),
                'work_efficiency_percent': receiving_config.get('work_efficiency', 85.0),
                'total_receivers_required': receivers_required,
                'receivers_per_shift': receivers_per_shift,
                'capacity_utilization': round((actual_time_hours / (receivers_required * net_hours_per_person)) * 100, 1)
            }
            
            # Shift Breakdown
            for shift in range(shifts_per_day):
                shift_start = 8 + (shift * shift_hours)  # Starting at 8 AM
                shift_end = shift_start + shift_hours
                
                analysis['shift_breakdown'].append({
                    'shift_number': shift + 1,
                    'shift_time': f"{shift_start:02.0f}:00-{shift_end:02.0f}:00",
                    'receivers_required': receivers_per_shift,
                    'productive_hours_per_person': net_hours_per_person,
                    'total_capacity_hours': receivers_per_shift * net_hours_per_person,
                    'planned_workload_hours': actual_time_hours / shifts_per_day,
                    'utilization_percent': round((actual_time_hours / shifts_per_day) / (receivers_per_shift * net_hours_per_person) * 100, 1)
                })
            
            # Hourly Requirements (spread evenly across working hours)
            hours_per_day = shifts_per_day * shift_hours
            pallets_per_hour = total_pallets / hours_per_day
            
            for hour in range(int(hours_per_day)):
                hour_start = 8 + hour  # Starting at 8 AM
                
                analysis['hourly_requirements'].append({
                    'hour': f"{hour_start:02.0f}:00",
                    'pallets_to_receive': round(pallets_per_hour, 1),
                    'time_required_minutes': round(pallets_per_hour * time_per_pallet_min, 1),
                    'staff_required': receivers_per_shift
                })

            # Truck-level analysis (dock door scheduling metric)
            if 'Truck No' in self.receipt_data.columns:
                daily_trucks = self.receipt_data.groupby('Receipt Date')['Truck No'].nunique()
                shipment_cases = self.receipt_data.groupby('Truck No')['Quantity in Cases'].sum()
                analysis['truck_analysis'] = {
                    'avg_trucks_per_day': round(daily_trucks.mean(), 1),
                    'peak_trucks_per_day': int(daily_trucks.quantile(percentile / 100)),
                    'max_trucks_single_day': int(daily_trucks.max()),
                    'avg_cases_per_truck': round(shipment_cases.mean(), 0),
                    'min_cases_per_truck': round(shipment_cases.min(), 0),
                    'max_cases_per_truck': round(shipment_cases.max(), 0),
                }

        except Exception as e:
            analysis['error'] = f"Calculation error: {str(e)}"
            print(f"⚠️ Error in receiving analysis: {str(e)}")

        return analysis

    def _analyze_receiving_detailed(self) -> Dict[str, Any]:
        """
        Detailed receiving analysis (placeholder for future implementation).
        
        Returns:
            dict: Detailed receiving analysis results (placeholder)
        """
        return {
            'complexity_mode': 'Detailed',
            'status': 'Not implemented',
            'notes': 'Detailed receiving analysis will be implemented later'
        }
    
    def _calculate_daily_receipt_volumes(self) -> pd.Series:
        """
        Calculate daily case volumes from receipt data.
        
        Returns:
            pd.Series: Daily case volumes indexed by receipt date
        """
        if 'Receipt Date' not in self.receipt_data.columns or 'Quantity in Cases' not in self.receipt_data.columns:
            raise ValueError("Required columns 'Receipt Date' and 'Quantity in Cases' not found in receipt data")
        
        # Group by receipt date and sum cases
        daily_cases = self.receipt_data.groupby('Receipt Date')['Quantity in Cases'].sum()
        
        print(f"📊 Daily receipt volume analysis: {len(daily_cases)} days, avg {daily_cases.mean():.0f} cases/day")
        
        return daily_cases
    
    def analyze_loading_manpower(self) -> Dict[str, Any]:
        """
        Analyze loading manpower requirements.
        
        Returns:
            dict: Loading manpower analysis results
        """
        print("📊 Analyzing loading manpower requirements...")
        
        # Placeholder implementation
        loading_analysis = {
            'loading_time_per_case': self.loading_params.get('loading_time_per_case', 4.0),
            'estimated_daily_loading_hours': 0,  # Placeholder
            'recommended_loaders': 0,  # Placeholder
            'efficiency_percentage': 90.0,  # Placeholder
            'notes': 'Placeholder analysis - actual implementation pending'
        }
        
        # Calculate values from order data using percentile peak day (consistent with picking/receiving)
        if self.order_data is not None and len(self.order_data) > 0 and 'Qty in Cases' in self.order_data.columns:
            daily_cases = (self.order_data.groupby('Date')['Qty in Cases'].sum()
                           if 'Date' in self.order_data.columns
                           else pd.Series([self.order_data['Qty in Cases'].sum()]))
            percentile_level = self.manpower_params.get('picking_simplified', {}).get('percentile_for_planning', 95)
            peak_daily_cases = float(np.percentile(daily_cases, percentile_level))
            avg_daily_cases  = float(daily_cases.mean())

            loading_time_per_case_sec = self.loading_params.get('loading_time_per_case', 4.0)
            # Staff sizing uses peak-day cases; reporting includes avg for context
            peak_daily_loading_hours = (peak_daily_cases * loading_time_per_case_sec) / 3600

            # Use picking shift config as proxy for loading (loading config has no shift params)
            shift_hours = self.manpower_params.get('picking_simplified', {}).get('shift_hours', 8.0)
            efficiency  = max(
                self.manpower_params.get('picking_simplified', {}).get('work_efficiency', 85.0) / 100, 0.01
            )
            net_hours = shift_hours * efficiency

            loading_analysis.update({
                'planning_percentile':           f'{percentile_level}th percentile',
                'total_cases_to_load':           int(self.order_data['Qty in Cases'].sum()),
                'average_daily_cases':           round(avg_daily_cases, 0),
                'peak_daily_cases':              round(peak_daily_cases, 0),
                'estimated_daily_loading_hours': round(peak_daily_loading_hours, 2),
                'recommended_loaders':           max(1, math.ceil(peak_daily_loading_hours / net_hours))
            })

        # Per-truck loading time (dock scheduling metric) — uses receipt data as proxy for truck size
        if self.receipt_data is not None and 'Truck No' in self.receipt_data.columns and 'Quantity in Cases' in self.receipt_data.columns:
            avg_cases_per_truck = self.receipt_data.groupby('Truck No')['Quantity in Cases'].sum().mean()
            loading_time_per_case_sec = self.loading_params.get('loading_time_per_case', 4.0)
            loading_analysis['avg_loading_time_per_truck_minutes'] = round(
                (avg_cases_per_truck * loading_time_per_case_sec) / 60, 1
            )
        
        self.loading_analysis = loading_analysis
        return loading_analysis
    
    def analyze_efficiency_summary(self) -> Dict[str, Any]:
        """
        Create overall efficiency summary and recommendations.
        
        Returns:
            dict: Efficiency summary and recommendations
        """
        print("📊 Creating efficiency summary...")
        
        # Calculate totals from sub-analyses
        total_staff = 0
        total_hours = 0

        if self.picking_analysis and 'daily_summary' in self.picking_analysis:
            total_staff += self.picking_analysis['daily_summary'].get('total_pickers_required', 0)
            total_hours += self.picking_analysis['daily_summary'].get('actual_time_hours', 0)

        if self.receiving_analysis and 'daily_summary' in self.receiving_analysis:
            total_staff += self.receiving_analysis['daily_summary'].get('total_receivers_required', 0)
            total_hours += self.receiving_analysis['daily_summary'].get('actual_time_hours', 0)

        if self.loading_analysis:
            total_staff += self.loading_analysis.get('recommended_loaders', 0)
            total_hours += self.loading_analysis.get('estimated_daily_loading_hours', 0)

        # Build opportunities list from actual utilization values
        opportunities = []
        if self.picking_analysis and 'daily_summary' in self.picking_analysis:
            pick_util = self.picking_analysis['daily_summary'].get('capacity_utilization', 0)
            if pick_util > 90:
                opportunities.append(f'Picking capacity critically high ({pick_util:.0f}%): add a shift or increase headcount')
            elif pick_util < 40:
                opportunities.append(f'Picking capacity underutilised ({pick_util:.0f}%): consider consolidating shifts')
        if self.receiving_analysis and 'daily_summary' in self.receiving_analysis:
            recv_util = self.receiving_analysis['daily_summary'].get('capacity_utilization', 0)
            if recv_util > 90:
                opportunities.append(f'Receiving capacity critically high ({recv_util:.0f}%): consider staggered truck bookings')
            elif recv_util < 40:
                opportunities.append(f'Receiving capacity underutilised ({recv_util:.0f}%): review scheduling')
        opportunities += [
            'Implement pick-path optimisation to reduce travel time',
            'Review putaway strategies to minimise congestion at peak inbound times',
        ]

        efficiency_summary = {
            'overall_efficiency':        85.0,
            'total_recommended_staff':   0,
            'peak_hours_staff_multiplier': 1.5,
            'optimization_opportunities': opportunities,
            'cost_analysis': {
                'estimated_daily_labor_hours':  0,
                'estimated_monthly_labor_cost': 0,
                'potential_savings': 'To be calculated in full implementation'
            },
        }
        
        recommended_pickers   = (self.picking_analysis['daily_summary'].get('total_pickers_required', 0)
                                  if self.picking_analysis and 'daily_summary' in self.picking_analysis else 0)
        recommended_receivers = (self.receiving_analysis['daily_summary'].get('total_receivers_required', 0)
                                  if self.receiving_analysis and 'daily_summary' in self.receiving_analysis else 0)
        recommended_loaders   = self.loading_analysis.get('recommended_loaders', 0) if self.loading_analysis else 0

        efficiency_summary.update({
            'total_recommended_staff':     recommended_pickers + recommended_receivers + recommended_loaders,
            'recommended_pickers':         recommended_pickers,
            'recommended_receivers':       recommended_receivers,
            'recommended_loaders':         recommended_loaders,
            'estimated_daily_labor_hours': round(total_hours, 2)
        })
        
        self.efficiency_summary = efficiency_summary
        return efficiency_summary
    
    def _get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of the available data for manpower analysis.
        
        Returns:
            dict: Data summary statistics
        """
        summary = {
            'analysis_type': 'manpower_analysis',
            'configuration_loaded': bool(self.manpower_params)
        }
        
        if self.order_data is not None:
            summary.update({
                'order_records': len(self.order_data),
                'order_date_range': {
                    'start': self.order_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in self.order_data.columns else 'Unknown',
                    'end': self.order_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in self.order_data.columns else 'Unknown'
                }
            })
        
        if self.receipt_data is not None:
            summary.update({
                'receipt_records': len(self.receipt_data),
                'receipt_date_range': {
                    'start': self.receipt_data['Receipt Date'].min().strftime('%Y-%m-%d') if 'Receipt Date' in self.receipt_data.columns else 'Unknown',
                    'end': self.receipt_data['Receipt Date'].max().strftime('%Y-%m-%d') if 'Receipt Date' in self.receipt_data.columns else 'Unknown'
                }
            })
        
        if self.sku_master is not None:
            summary['sku_master_records'] = len(self.sku_master)
        
        return summary

# Test function for standalone execution
if __name__ == "__main__":
    print("ManpowerAnalyzer module - placeholder implementation")
    print("This module requires order/receipt data to function.")
    print("Use within the main application for proper functionality.")
    print("Actual manpower analysis implementation will be added later.")