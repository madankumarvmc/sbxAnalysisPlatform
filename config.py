#!/usr/bin/env python3
"""
Warehouse Analysis Tool V2 - Configuration Settings for Streamlit

This configuration module provides default settings and mappings for the
warehouse analysis tool, adapted to work with Streamlit session state.
"""

import os
from pathlib import Path

# =============================================================================
# SHEET NAMES MAPPING
# =============================================================================
# Expected sheet names in the Excel file - matches ITC_WMS_RAW_DATA.xlsx structure
SHEET_NAMES = {
    'ORDER_DATA': 'OrderData',
    'SKU_MASTER': 'SkuMaster', 
    'RECEIPT_DATA': 'ReceiptData',
    'INVENTORY_DATA': 'InventoryData'
}

# =============================================================================
# COLUMN MAPPINGS - Exact matches to ITC_WMS_RAW_DATA.xlsx
# =============================================================================

# OrderData sheet columns
ORDER_COLUMNS = {
    'date': 'Date',
    'order_no': 'Order No.',
    'shipment_no': 'Shipment No.',
    'sku_code': 'Sku Code',
    'qty_cases': 'Qty in Cases',
    'qty_eaches': 'Qty in Eaches'
}

# SkuMaster sheet columns  
SKU_COLUMNS = {
    'category': 'Category',
    'sku_code': 'Sku Code',
    'case_config': 'Case Config',
    'pallet_fit': 'Pallet Fit'
}

# ReceiptData sheet columns
RECEIPT_COLUMNS = {
    'receipt_date': 'Receipt Date',
    'sku_id': 'SKU ID',
    'shipment_no': 'Shipment No',
    'truck_no': 'Truck No',
    'batch': 'Batch',
    'qty_cases': 'Quantity in Cases',
    'qty_eaches': 'Quantity in Eaches'
}

# InventoryData sheet columns
INVENTORY_COLUMNS = {
    'calendar_day': 'Calendar Day',
    'sku_id': 'SKU ID',
    'sku_name': 'SKU Name',
    'stock_cases': 'Total Stock in Cases (In Base Unit of Measure)',
    'stock_eaches': 'Total Stock in Pieces (In Base Unit of Measue)'  # Note: keeping original typo
}

# =============================================================================
# DEFAULT ANALYSIS PARAMETERS
# =============================================================================

# ABC Classification Thresholds (cumulative percentage)
DEFAULT_ABC_THRESHOLDS = {
    'A_THRESHOLD': 70,    # Items contributing to first 70% of volume = A
    'B_THRESHOLD': 90     # Items contributing to 70-90% of volume = B
                         # Items contributing to 90-100% of volume = C
}

# FMS (Fast/Medium/Slow) Classification Thresholds (cumulative percentage)  
DEFAULT_FMS_THRESHOLDS = {
    'F_THRESHOLD': 70,    # Items with top 70% order frequency = Fast
    'M_THRESHOLD': 90     # Items with 70-90% order frequency = Medium
                         # Items with 90-100% order frequency = Slow
}

# Default percentile levels for capacity planning
DEFAULT_PERCENTILE_LEVELS = [95, 90, 85, 80, 75]

# Default inventory parameters
DEFAULT_INVENTORY_PARAMS = {
    'SAFETY_STOCK_DAYS': 7,
    'REORDER_POINT_DAYS': 14,
    'DEFAULT_LEAD_TIME': 7,
    'STORAGE_COST_PERCENT': 15.0
}

# Default manpower parameters
DEFAULT_MANPOWER_PARAMS = {
    'TARGET_EFFICIENCY': 85,
    'STANDARD_PICK_RATE': 60,
    'SHIFTS_PER_DAY': 1,
    'BREAK_TIME_MINUTES': 30,
    'WORKING_HOURS_PER_DAY': 8,
    'WORKING_DAYS_PER_WEEK': 5
}

# Default manpower analysis parameters for timing studies
DEFAULT_MANPOWER_ANALYSIS_PARAMS = {
    'picking': {
        'avg_walk_distance_per_pallet': 50.0,    # meters
        'scan_time': 3.0,                        # seconds
        'qty_pick_time': 2.0,                    # seconds
        'misc_time_per_pallet': 30.0             # seconds
    },
    'receiving_putaway': {
        'unloading_time_per_case': 5.0,          # seconds
        'avg_walk_distance_per_pallet': 40.0,    # meters
        'scan_time': 3.0,                        # seconds
        'misc_time': 20.0                        # seconds
    },
    'loading': {
        'loading_time_per_case': 4.0             # seconds
    }
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Minimum data requirements for analysis
VALIDATION_SETTINGS = {
    'MIN_ORDER_ROWS': 50,
    'MIN_SKU_ROWS': 10,
    'MIN_DATE_RANGE_DAYS': 7,
    'MAX_MISSING_PERCENTAGE': 10.0
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Excel sheet names for output reports
OUTPUT_SHEET_NAMES = {
    'EXECUTIVE_SUMMARY': 'Executive_Summary',
    'ORDER_ANALYSIS': 'Order_Analysis', 
    'SKU_ANALYSIS': 'SKU_Analysis',
    'ABC_FMS_ANALYSIS': 'ABC_FMS_Analysis',
    'INVENTORY_ANALYSIS': 'Inventory_Analysis',
    'RECEIPT_ANALYSIS': 'Receipt_Analysis',
    'MANPOWER_ANALYSIS': 'Manpower_Analysis',
    'VARIABLE_CONFIG': 'Variable_Configuration',
    'RAW_DATA_SUMMARY': 'Raw_Data_Summary'
}

# =============================================================================
# HELPER FUNCTIONS FOR STREAMLIT INTEGRATION
# =============================================================================

def get_variable_defaults():
    """
    Get default variable configuration for Streamlit session state.
    
    Returns:
        dict: Default configuration variables
    """
    return {
        'global': {
            'abc_a_threshold': DEFAULT_ABC_THRESHOLDS['A_THRESHOLD'],
            'abc_b_threshold': DEFAULT_ABC_THRESHOLDS['B_THRESHOLD'],
            'currency_symbol': '$',
            'decimal_places': 2,
            'working_days_per_week': DEFAULT_MANPOWER_PARAMS['WORKING_DAYS_PER_WEEK'],
            'working_hours_per_day': DEFAULT_MANPOWER_PARAMS['WORKING_HOURS_PER_DAY'],
        },
        'order_analysis': {
            'fms_fast_threshold': DEFAULT_FMS_THRESHOLDS['F_THRESHOLD'],
            'fms_medium_threshold': DEFAULT_FMS_THRESHOLDS['M_THRESHOLD'],
            'percentile_levels': DEFAULT_PERCENTILE_LEVELS.copy(),
            'vip_customer_threshold': 50,
            'seasonal_adjustment': False,
        },
        'receipt_analysis': {
            'percentile_levels': DEFAULT_PERCENTILE_LEVELS.copy(),
        },
        'inventory_analysis': {
            'safety_stock_days': DEFAULT_INVENTORY_PARAMS['SAFETY_STOCK_DAYS'],
            'reorder_point_days': DEFAULT_INVENTORY_PARAMS['REORDER_POINT_DAYS'],
            'default_lead_time': DEFAULT_INVENTORY_PARAMS['DEFAULT_LEAD_TIME'],
            'storage_cost_percent': DEFAULT_INVENTORY_PARAMS['STORAGE_COST_PERCENT'],
        },
        'manpower_analysis': {
            'target_efficiency': DEFAULT_MANPOWER_PARAMS['TARGET_EFFICIENCY'],
            'standard_pick_rate': DEFAULT_MANPOWER_PARAMS['STANDARD_PICK_RATE'],
            'shifts_per_day': DEFAULT_MANPOWER_PARAMS['SHIFTS_PER_DAY'],
            'break_time_minutes': DEFAULT_MANPOWER_PARAMS['BREAK_TIME_MINUTES'],
            'picking': DEFAULT_MANPOWER_ANALYSIS_PARAMS['picking'].copy(),
            'receiving_putaway': DEFAULT_MANPOWER_ANALYSIS_PARAMS['receiving_putaway'].copy(),
            'loading': DEFAULT_MANPOWER_ANALYSIS_PARAMS['loading'].copy()
        }
    }

def validate_sheet_availability(file_sheets):
    """
    Check which analysis modules can be run based on available sheets.
    
    Args:
        file_sheets (dict): Dictionary of available sheets from uploaded file
        
    Returns:
        dict: Available analysis modules and their status
    """
    available_analyses = {
        'order_analysis': SHEET_NAMES['ORDER_DATA'] in file_sheets,
        'sku_analysis': SHEET_NAMES['SKU_MASTER'] in file_sheets,
        'abc_fms_analysis': (
            SHEET_NAMES['ORDER_DATA'] in file_sheets and 
            SHEET_NAMES['SKU_MASTER'] in file_sheets
        ),
        'inventory_analysis': SHEET_NAMES['INVENTORY_DATA'] in file_sheets,
        'receipt_analysis': SHEET_NAMES['RECEIPT_DATA'] in file_sheets
    }
    
    return available_analyses

def get_required_columns(sheet_name):
    """
    Get required columns for a specific sheet.
    
    Args:
        sheet_name (str): Name of the sheet
        
    Returns:
        dict: Required column mappings for the sheet
    """
    column_mappings = {
        SHEET_NAMES['ORDER_DATA']: ORDER_COLUMNS,
        SHEET_NAMES['SKU_MASTER']: SKU_COLUMNS,
        SHEET_NAMES['RECEIPT_DATA']: RECEIPT_COLUMNS,
        SHEET_NAMES['INVENTORY_DATA']: INVENTORY_COLUMNS
    }
    
    return column_mappings.get(sheet_name, {})

def create_analysis_config(streamlit_variables):
    """
    Convert Streamlit session state variables to analysis configuration.
    
    Args:
        streamlit_variables (dict): Variables from Streamlit session state
        
    Returns:
        dict: Configuration object for analysis modules
    """
    config = {
        'ABC_THRESHOLDS': {
            'A_THRESHOLD': streamlit_variables.get('global', {}).get('abc_a_threshold', 70.0),
            'B_THRESHOLD': streamlit_variables.get('global', {}).get('abc_b_threshold', 90.0)
        },
        'FMS_THRESHOLDS': {
            'F_THRESHOLD': streamlit_variables.get('order_analysis', {}).get('fms_fast_threshold', 70.0),
            'M_THRESHOLD': streamlit_variables.get('order_analysis', {}).get('fms_medium_threshold', 90.0)
        },
        'PERCENTILE_LEVELS': streamlit_variables.get('order_analysis', {}).get('percentile_levels', [95, 90, 85, 80, 75]),
        'RECEIPT_PERCENTILE_LEVELS': streamlit_variables.get('receipt_analysis', {}).get('percentile_levels', [95, 90, 85, 80, 75]),
        'DATE_RANGE': {
            'START_DATE': None,
            'END_DATE': None
        },
        'INVENTORY_PARAMS': streamlit_variables.get('inventory_analysis', DEFAULT_INVENTORY_PARAMS),
        'MANPOWER_PARAMS': streamlit_variables.get('manpower_analysis', DEFAULT_MANPOWER_PARAMS.copy()),
        'OUTPUT_SETTINGS': {
            'CURRENCY_SYMBOL': streamlit_variables.get('global', {}).get('currency_symbol', '$'),
            'DECIMAL_PLACES': streamlit_variables.get('global', {}).get('decimal_places', 2),
            'VERBOSE_OUTPUT': False  # Controlled by Streamlit progress indicators
        }
    }
    
    return config

# =============================================================================
# BUSINESS RULES AND CONSTANTS
# =============================================================================

# Standard business calendar
BUSINESS_CALENDAR = {
    'WORKING_DAYS_PER_WEEK': 5,
    'WORKING_HOURS_PER_DAY': 8,
    'WEEKS_PER_MONTH': 4.33,
    'MONTHS_PER_YEAR': 12
}

# Unit conversion factors
CONVERSION_FACTORS = {
    'HOURS_TO_MINUTES': 60,
    'DAYS_TO_HOURS': 24,
    'WEEKS_TO_DAYS': 7
}

# Analysis categories
ANALYSIS_CATEGORIES = {
    'HIGH_PRIORITY': ['order_analysis', 'abc_fms_analysis'],
    'MEDIUM_PRIORITY': ['sku_analysis', 'inventory_analysis'],
    'LOW_PRIORITY': ['receipt_analysis']
}