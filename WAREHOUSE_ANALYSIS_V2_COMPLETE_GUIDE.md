# Warehouse Analysis Tool V2 - Complete Development Guide

## Project Overview

This guide provides complete documentation for building a clean, simple warehouse analysis tool that processes Excel data and generates comprehensive Excel reports. The tool is designed to be beginner-friendly with clear documentation and step-by-step execution.

## Table of Contents

1. [Project Requirements](#project-requirements)
2. [Architecture & Design](#architecture--design)
3. [Directory Structure](#directory-structure)
4. [Complete Script Collection](#complete-script-collection)
5. [Configuration Documentation](#configuration-documentation)
6. [Usage Instructions](#usage-instructions)
7. [Development Guidelines](#development-guidelines)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)
10. [Integration with New Claude Project](#integration-with-new-claude-project)

## Project Requirements

### Functional Requirements

1. **Data Loading & Validation**
   - Load order data from Excel files
   - Support optional SKU master, receipt, and inventory data
   - Validate data quality and completeness
   - Handle missing columns and data gracefully

2. **Core Analysis Modules**
   - Order pattern analysis (daily trends, volume statistics)
   - SKU profiling and classification
   - ABC-FMS cross-tabulation analysis
   - Receipt pattern analysis (if data available)
   - Percentile analysis for capacity planning

3. **Excel Report Generation**
   - Multi-sheet formatted Excel workbook
   - Professional formatting and charts
   - Summary and detailed analysis sheets
   - Raw data preservation

4. **User Experience**
   - Configuration-driven setup
   - Clear error messages and guidance
   - Progress indicators and status updates
   - Beginner-friendly documentation

### Technical Requirements

1. **Dependencies**
   - Python 3.8+
   - pandas >= 2.0.0
   - openpyxl >= 3.1.0
   - numpy >= 1.24.0

2. **Data Format Support**
   - Excel files (.xlsx)
   - Multiple sheet support
   - Flexible column naming
   - Date format handling

3. **Performance**
   - Handle datasets up to 1M rows
   - Memory-efficient processing
   - Progress tracking for large datasets

## Architecture & Design

### Design Principles

1. **Simplicity First**: Each script has a single, clear purpose
2. **Excel-Centric**: Always generate working Excel outputs
3. **Configuration-Driven**: All settings in one config file
4. **Error Resilient**: Graceful handling of missing data and errors
5. **Beginner-Friendly**: Comprehensive documentation and clear error messages

### Module Architecture

```
Main Execution (run_analysis.py)
├── Configuration (config.py)
├── Data Loading (data_loader.py)
├── Analysis Modules
│   ├── Order Analysis (order_analysis.py)
│   ├── SKU Analysis (sku_analysis.py)
│   ├── ABC-FMS Analysis (abc_fms_analysis.py)
│   └── Receipt Analysis (receipt_analysis.py)
└── Excel Generation (excel_generator.py)
```

### Data Flow

```
Excel Input → Data Loader → Analysis Modules → Excel Generator → Excel Output
     ↑              ↑             ↑               ↑              ↑
Configuration → Validation → Processing → Formatting → Final Report
```

## Directory Structure

```
warehouse_analysis_v2/
├── README.md                 # Setup and usage instructions
├── requirements.txt          # Python dependencies
├── config.py                # User configuration settings
├── run_analysis.py          # Main execution script
├── scripts/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and validation
│   ├── order_analysis.py    # Order pattern analysis
│   ├── sku_analysis.py      # SKU profiling and classification
│   ├── abc_fms_analysis.py  # ABC-FMS cross-tabulation
│   ├── receipt_analysis.py  # Receipt pattern analysis
│   └── excel_generator.py   # Excel report creation
├── data/
│   └── sample_data.xlsx     # Sample input file
├── outputs/
│   └── excel_reports/       # Generated Excel reports
└── docs/
    ├── user_guide.md        # User documentation
    ├── developer_guide.md   # Developer documentation
    └── api_reference.md     # API documentation
```

## Complete Script Collection

### 1. requirements.txt
```txt
pandas>=2.0.0
openpyxl>=3.1.0
numpy>=1.24.0
```

### 2. README.md
```markdown
# Warehouse Analysis Tool V2

A simple, powerful tool for analyzing warehouse order data and generating comprehensive Excel reports.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Your Data**
   - Edit `config.py`
   - Update `DATA_FILE_PATH` to point to your Excel file
   - Adjust column names to match your data

3. **Run Analysis**
   ```bash
   python run_analysis.py
   ```

4. **View Results**
   - Check `outputs/excel_reports/` for your report
   - Open Excel file to review analysis

## Features

- ✅ Order pattern analysis
- ✅ SKU profiling and ABC-FMS classification
- ✅ Percentile analysis for capacity planning
- ✅ Receipt analysis (optional)
- ✅ Professional Excel reports
- ✅ Beginner-friendly setup

## Support

If you encounter issues:
1. Check the configuration in `config.py`
2. Verify your Excel file format
3. Review error messages for guidance
4. Enable verbose output for debugging

For detailed documentation, see the complete development guide.
```

### 3. scripts/__init__.py
```python
"""
Warehouse Analysis Tool V2 - Analysis Scripts Package

This package contains all the analysis modules for the warehouse analysis tool.
Each module performs a specific type of analysis and returns structured results.
"""

__version__ = "2.0.0"
__author__ = "Warehouse Analysis Team"

# Import all analysis modules for easy access
from .data_loader import DataLoader
from .order_analysis import OrderAnalyzer
from .sku_analysis import SKUAnalyzer
from .abc_fms_analysis import ABCFMSAnalyzer
from .receipt_analysis import ReceiptAnalyzer
from .excel_generator import ExcelGenerator

__all__ = [
    'DataLoader',
    'OrderAnalyzer', 
    'SKUAnalyzer',
    'ABCFMSAnalyzer',
    'ReceiptAnalyzer',
    'ExcelGenerator'
]
```

### 4. config.py - User Configuration
```python
#!/usr/bin/env python3
"""
Warehouse Analysis Tool V2 - Configuration Settings

INSTRUCTIONS FOR BEGINNERS:
1. Update the file paths below to point to your Excel data file
2. Adjust analysis parameters if needed (default values work for most cases)
3. Run: python run_analysis.py

This file contains all the settings you need to customize for your analysis.
"""

import os
from pathlib import Path

# =============================================================================
# DATA FILE CONFIGURATION
# =============================================================================

# IMPORTANT: Update this path to your Excel data file
# Example: r"C:\Users\YourName\Documents\warehouse_data.xlsx"
DATA_FILE_PATH = r"data/sample_data.xlsx"

# Sheet names in your Excel file - update these to match your file
SHEET_NAMES = {
    'ORDER_DATA': 'OrderData',          # Sheet containing order line data
    'SKU_MASTER': 'SkuMaster',          # Sheet containing SKU details
    'RECEIPT_DATA': 'ReceiptData',      # Sheet containing receipt data (optional)
    'INVENTORY_DATA': 'InventoryData'   # Sheet containing inventory data (optional)
}

# =============================================================================
# EXPECTED COLUMN NAMES
# =============================================================================

# Order Data Columns - update these to match your column names
ORDER_COLUMNS = {
    'date': 'Date',                     # Date column (YYYY-MM-DD format)
    'shipment_no': 'Shipment No.',      # Shipment identifier
    'order_no': 'Order No.',            # Order identifier
    'customer_id': 'Customer ID',       # Customer identifier (optional)
    'sku_code': 'Sku Code',            # SKU/Product code
    'qty_cases': 'Qty in Cases',        # Quantity in cases
    'qty_eaches': 'Qty in Eaches'       # Quantity in pieces/eaches
}

# SKU Master Columns - update these to match your column names
SKU_COLUMNS = {
    'sku_code': 'Sku Code',            # SKU/Product code
    'category': 'Category',             # Product category
    'case_config': 'Case Config',       # Cases per pallet/unit
    'pallet_fit': 'Pallet Fit'         # Units per pallet
}

# Receipt Data Columns (optional) - update if you have receipt data
RECEIPT_COLUMNS = {
    'receipt_date': 'Receipt Date',     # Receipt date
    'sku_id': 'SKU ID',               # SKU identifier
    'shipment_no': 'Shipment No',      # Shipment number
    'truck_no': 'Truck No',           # Truck identifier
    'qty_cases': 'Quantity in Cases',  # Received quantity in cases
    'qty_eaches': 'Quantity in Eaches' # Received quantity in pieces
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# ABC Classification Thresholds (based on volume percentage)
ABC_THRESHOLDS = {
    'A_THRESHOLD': 70.0,    # Items contributing to first 70% of volume = A
    'B_THRESHOLD': 90.0     # Items contributing to 70-90% of volume = B
                           # Items contributing to 90-100% of volume = C
}

# FMS (Fast/Medium/Slow) Classification Thresholds (based on order frequency)
FMS_THRESHOLDS = {
    'F_THRESHOLD': 70.0,    # Items with top 70% order frequency = Fast
    'M_THRESHOLD': 90.0     # Items with 70-90% order frequency = Medium
                           # Items with 90-100% order frequency = Slow
}

# Percentile Levels for Analysis
PERCENTILE_LEVELS = [95, 90, 85, 80, 75]  # Which percentiles to calculate

# Date Range for Analysis (leave None to use all data)
DATE_RANGE = {
    'START_DATE': None,     # Format: 'YYYY-MM-DD' or None for all data
    'END_DATE': None        # Format: 'YYYY-MM-DD' or None for all data
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output file settings
OUTPUT_SETTINGS = {
    'EXCEL_FILENAME': 'Warehouse_Analysis_Report.xlsx',
    'OUTPUT_DIR': 'outputs/excel_reports/',
    'INCLUDE_RAW_DATA': True,           # Include raw data sheets in output
    'MAX_ROWS_PER_SHEET': 1000000,     # Excel row limit
    'DATE_FORMAT': '%Y-%m-%d'          # Date format in output
}

# Excel Sheet Names for Output
OUTPUT_SHEETS = {
    'SUMMARY': 'Executive Summary',
    'DATE_ANALYSIS': 'Daily Order Analysis',
    'SKU_PROFILE': 'SKU Profile & Classification',
    'ABC_FMS_SUMMARY': 'ABC-FMS Analysis',
    'ABC_FMS_DETAIL': 'ABC-FMS Detailed',
    'PERCENTILES': 'Volume Percentiles',
    'RECEIPT_ANALYSIS': 'Receipt Analysis',
    'RAW_ORDER_DATA': 'Raw Order Data',
    'RAW_SKU_DATA': 'Raw SKU Data'
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Minimum data requirements for analysis
VALIDATION_RULES = {
    'MIN_ROWS': 100,           # Minimum rows required for analysis
    'MIN_SKUS': 10,            # Minimum unique SKUs required
    'MIN_DATES': 7,            # Minimum unique dates required
    'REQUIRED_COLUMNS': list(ORDER_COLUMNS.values())[:3]  # Essential columns
}

# =============================================================================
# BUSINESS RULES
# =============================================================================

# Business logic settings
BUSINESS_RULES = {
    'CASES_TO_EACHES_DEFAULT': 12,     # Default case size if not specified
    'WORKING_DAYS_PER_MONTH': 22,      # For monthly calculations
    'WORKING_DAYS_PER_YEAR': 250,      # For annual calculations
    'PEAK_FACTOR': 1.3,                # Peak season multiplier
    'SAFETY_STOCK_FACTOR': 1.2         # Safety stock multiplier
}

# =============================================================================
# DEBUG AND LOGGING
# =============================================================================

# Debug settings
DEBUG_SETTINGS = {
    'VERBOSE_OUTPUT': True,             # Print detailed progress messages
    'SAVE_INTERMEDIATE_FILES': False,   # Save intermediate processing files
    'VALIDATE_DATA_QUALITY': True,     # Run data quality checks
    'PRINT_SAMPLE_DATA': True          # Print sample data for verification
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_config():
    """
    Validate configuration settings before running analysis.
    This function checks if all required settings are properly configured.
    """
    errors = []
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE_PATH):
        errors.append(f"Data file not found: {DATA_FILE_PATH}")
    
    # Check output directory
    output_dir = Path(OUTPUT_SETTINGS['OUTPUT_DIR'])
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
    
    # Validate thresholds
    if ABC_THRESHOLDS['A_THRESHOLD'] >= ABC_THRESHOLDS['B_THRESHOLD']:
        errors.append("ABC A_THRESHOLD must be less than B_THRESHOLD")
    
    if FMS_THRESHOLDS['F_THRESHOLD'] >= FMS_THRESHOLDS['M_THRESHOLD']:
        errors.append("FMS F_THRESHOLD must be less than M_THRESHOLD")
    
    return errors

def print_config_summary():
    """Print a summary of current configuration for user verification."""
    print("=" * 60)
    print("WAREHOUSE ANALYSIS TOOL V2 - CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Data File: {DATA_FILE_PATH}")
    print(f"Output Directory: {OUTPUT_SETTINGS['OUTPUT_DIR']}")
    print(f"ABC Thresholds: A<{ABC_THRESHOLDS['A_THRESHOLD']}%, B<{ABC_THRESHOLDS['B_THRESHOLD']}%")
    print(f"FMS Thresholds: F<{FMS_THRESHOLDS['F_THRESHOLD']}%, M<{FMS_THRESHOLDS['M_THRESHOLD']}%")
    print(f"Analysis Percentiles: {PERCENTILE_LEVELS}")
    print("=" * 60)

if __name__ == "__main__":
    # Test configuration when run directly
    print_config_summary()
    errors = validate_config()
    if errors:
        print("CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid!")
```

### 5. run_analysis.py - Main Execution Script
```python
#!/usr/bin/env python3
"""
Warehouse Analysis Tool V2 - Main Execution Script

WHAT THIS SCRIPT DOES:
This is the main script that runs the complete warehouse analysis.
It coordinates all the analysis modules and generates the final Excel report.

HOW TO USE:
1. Update config.py with your data file path and settings
2. Run this script: python run_analysis.py
3. Check the outputs/excel_reports/ folder for your Excel report

FOR BEGINNERS:
- This script will guide you through the entire process
- It will validate your data and configuration first
- Any errors will be clearly explained
- The Excel report will contain multiple sheets with different analyses
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent / 'scripts'))

# Import configuration and analysis modules
try:
    import config
    from scripts.data_loader import DataLoader
    from scripts.order_analysis import OrderAnalyzer
    from scripts.sku_analysis import SKUAnalyzer
    from scripts.abc_fms_analysis import ABCFMSAnalyzer
    from scripts.receipt_analysis import ReceiptAnalyzer
    from scripts.excel_generator import ExcelGenerator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required files are in the correct directories.")
    sys.exit(1)

def print_header():
    """Print welcome header with instructions."""
    print("\n" + "=" * 80)
    print("🏭 WAREHOUSE ANALYSIS TOOL V2")
    print("📊 Excel Report Generation System")
    print("=" * 80)
    print("This tool will analyze your warehouse data and generate comprehensive Excel reports.")
    print("The analysis includes:")
    print("  • Order patterns and trends")
    print("  • SKU profiling and classification")
    print("  • ABC-FMS analysis")
    print("  • Receipt analysis (if data available)")
    print("  • Volume percentiles and statistics")
    print("=" * 80)

def validate_setup():
    """
    Validate that everything is set up correctly before starting analysis.
    Returns True if setup is valid, False otherwise.
    """
    print("\n🔍 STEP 1: Validating Setup...")
    
    # Check configuration
    config_errors = config.validate_config()
    if config_errors:
        print("❌ Configuration Errors Found:")
        for error in config_errors:
            print(f"   • {error}")
        print("\n💡 Please fix these errors in config.py and try again.")
        return False
    
    # Print configuration summary
    config.print_config_summary()
    print("✅ Configuration validated successfully!")
    return True

def load_and_validate_data():
    """
    Load data from Excel file and validate it.
    Returns DataLoader instance if successful, None otherwise.
    """
    print("\n📂 STEP 2: Loading and Validating Data...")
    
    try:
        loader = DataLoader()
        
        # Load main order data
        print("   Loading order data...")
        order_data = loader.load_order_data()
        if order_data is None or order_data.empty:
            print("❌ Failed to load order data. Check your file path and sheet names.")
            return None
        
        print(f"   ✅ Loaded {len(order_data):,} order records")
        
        # Load SKU master data
        print("   Loading SKU master data...")
        sku_data = loader.load_sku_master()
        if sku_data is not None:
            print(f"   ✅ Loaded {len(sku_data):,} SKU records")
        else:
            print("   ⚠️  SKU master data not available (analysis will continue)")
        
        # Load optional data
        receipt_data = loader.load_receipt_data()
        if receipt_data is not None:
            print(f"   ✅ Loaded {len(receipt_data):,} receipt records")
        
        inventory_data = loader.load_inventory_data()
        if inventory_data is not None:
            print(f"   ✅ Loaded {len(inventory_data):,} inventory records")
        
        # Validate data quality
        print("   Validating data quality...")
        validation_results = loader.validate_data_quality()
        
        if not validation_results['is_valid']:
            print("❌ Data validation failed:")
            for error in validation_results['errors']:
                print(f"   • {error}")
            return None
        
        print("✅ Data validation completed successfully!")
        return loader
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            traceback.print_exc()
        return None

def run_analysis(loader):
    """
    Run all analysis modules and collect results.
    Returns dictionary of analysis results.
    """
    print("\n📊 STEP 3: Running Analysis Modules...")
    
    results = {}
    
    try:
        # 1. Order Analysis
        print("   📈 Running order analysis...")
        order_analyzer = OrderAnalyzer(loader.order_data)
        results['order_analysis'] = order_analyzer.run_analysis()
        results['date_summary'] = order_analyzer.get_date_summary()
        results['percentiles'] = order_analyzer.get_percentile_analysis()
        print("   ✅ Order analysis completed")
        
        # 2. SKU Analysis
        print("   🏷️  Running SKU analysis...")
        sku_analyzer = SKUAnalyzer(loader.order_data, loader.sku_master)
        results['sku_analysis'] = sku_analyzer.run_analysis()
        results['sku_profile'] = sku_analyzer.get_sku_profile()
        print("   ✅ SKU analysis completed")
        
        # 3. ABC-FMS Analysis
        print("   🔤 Running ABC-FMS classification...")
        abc_fms_analyzer = ABCFMSAnalyzer(loader.order_data, loader.sku_master)
        results['abc_fms_analysis'] = abc_fms_analyzer.run_analysis()
        results['abc_fms_summary'] = abc_fms_analyzer.get_summary()
        results['abc_fms_detail'] = abc_fms_analyzer.get_detailed_classification()
        print("   ✅ ABC-FMS analysis completed")
        
        # 4. Receipt Analysis (if data available)
        if loader.receipt_data is not None:
            print("   🚛 Running receipt analysis...")
            receipt_analyzer = ReceiptAnalyzer(loader.receipt_data)
            results['receipt_analysis'] = receipt_analyzer.run_analysis()
            print("   ✅ Receipt analysis completed")
        else:
            print("   ⚠️  Receipt analysis skipped (no receipt data)")
        
        print("✅ All analysis modules completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            traceback.print_exc()
        return None

def generate_excel_report(results, loader):
    """
    Generate comprehensive Excel report with all analysis results.
    Returns path to generated file.
    """
    print("\n📄 STEP 4: Generating Excel Report...")
    
    try:
        generator = ExcelGenerator()
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = config.OUTPUT_SETTINGS['EXCEL_FILENAME']
        filename_parts = base_filename.split('.')
        timestamped_filename = f"{filename_parts[0]}_{timestamp}.{filename_parts[1]}"
        
        output_path = generator.create_comprehensive_report(
            results, 
            loader,
            filename=timestamped_filename
        )
        
        if output_path:
            print(f"✅ Excel report generated successfully!")
            print(f"📁 File location: {output_path}")
            return output_path
        else:
            print("❌ Failed to generate Excel report")
            return None
            
    except Exception as e:
        print(f"❌ Error generating Excel report: {e}")
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            traceback.print_exc()
        return None

def print_summary(output_path, results):
    """Print analysis summary and next steps."""
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    if output_path:
        print(f"📊 Excel Report: {output_path}")
    
    print("\n📈 Analysis Summary:")
    if 'order_analysis' in results:
        order_stats = results['order_analysis'].get('basic_stats', {})
        print(f"   • Total Records: {order_stats.get('total_records', 'N/A'):,}")
        print(f"   • Unique SKUs: {order_stats.get('unique_skus', 'N/A'):,}")
        print(f"   • Unique Dates: {order_stats.get('unique_dates', 'N/A'):,}")
        print(f"   • Total Volume: {order_stats.get('total_volume', 'N/A'):,} cases")
    
    print("\n📋 Excel Report Contains:")
    for sheet_name in config.OUTPUT_SHEETS.values():
        print(f"   • {sheet_name}")
    
    print("\n🎯 Next Steps:")
    print("   1. Open the Excel file to review detailed analysis")
    print("   2. Verify the data and calculations")
    print("   3. Use the insights for warehouse planning")
    print("   4. Run analysis again with updated data as needed")
    print("=" * 80)

def main():
    """Main execution function."""
    try:
        # Print welcome header
        print_header()
        
        # Step 1: Validate setup
        if not validate_setup():
            return False
        
        # Step 2: Load and validate data
        loader = load_and_validate_data()
        if loader is None:
            return False
        
        # Step 3: Run analysis
        results = run_analysis(loader)
        if results is None:
            return False
        
        # Step 4: Generate Excel report
        output_path = generate_excel_report(results, loader)
        if output_path is None:
            return False
        
        # Print summary
        print_summary(output_path, results)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user.")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Warehouse Analysis Tool V2...")
    success = main()
    
    if success:
        print("\n✅ Analysis completed successfully!")
        input("\nPress Enter to exit...")
    else:
        print("\n❌ Analysis failed. Please check the errors above.")
        input("\nPress Enter to exit...")
```

### 6. scripts/data_loader.py
```python
"""
Data Loader Module for Warehouse Analysis Tool V2

PURPOSE:
This module handles loading data from Excel files and validating data quality.
It provides a clean interface for accessing order data, SKU master data,
receipt data, and inventory data.

FOR BEGINNERS:
- This module reads Excel files and converts them to pandas DataFrames
- It validates that required columns exist and data types are correct
- It handles missing data and provides helpful error messages
- All data loading goes through this module for consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

class DataLoader:
    """
    Data loading and validation class for warehouse analysis.
    
    This class handles all data loading operations including:
    - Loading order data from Excel
    - Loading SKU master data
    - Loading receipt and inventory data (optional)
    - Validating data quality and completeness
    - Cleaning and preprocessing data
    """
    
    def __init__(self):
        """
        Initialize the DataLoader.
        
        Sets up the data loader with configuration settings and
        prepares for data loading operations.
        """
        self.data_file_path = config.DATA_FILE_PATH
        self.sheet_names = config.SHEET_NAMES
        self.order_columns = config.ORDER_COLUMNS
        self.sku_columns = config.SKU_COLUMNS
        self.receipt_columns = config.RECEIPT_COLUMNS
        
        # Initialize data containers
        self.order_data = None
        self.sku_master = None
        self.receipt_data = None
        self.inventory_data = None
        
        # Validation results
        self.validation_results = {}
        
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            print(f"DataLoader initialized with file: {self.data_file_path}")
    
    def load_order_data(self):
        """
        Load order data from Excel file.
        
        Returns:
            pandas.DataFrame: Order data with standardized column names
            None: If loading fails
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Loading order data from sheet: {self.sheet_names['ORDER_DATA']}")
            
            # Read Excel sheet
            self.order_data = pd.read_excel(
                self.data_file_path,
                sheet_name=self.sheet_names['ORDER_DATA']
            )
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Raw order data shape: {self.order_data.shape}")
            
            # Validate required columns exist
            missing_columns = []
            for config_col, excel_col in self.order_columns.items():
                if excel_col not in self.order_data.columns:
                    missing_columns.append(excel_col)
            
            if missing_columns:
                print(f"❌ Missing required columns in order data: {missing_columns}")
                print(f"Available columns: {list(self.order_data.columns)}")
                return None
            
            # Clean and process order data
            self.order_data = self._clean_order_data(self.order_data)
            
            if config.DEBUG_SETTINGS['PRINT_SAMPLE_DATA']:
                print("Sample order data:")
                print(self.order_data.head())
            
            return self.order_data
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.data_file_path}")
            return None
        except ValueError as e:
            print(f"❌ Sheet '{self.sheet_names['ORDER_DATA']}' not found in Excel file")
            print(f"Available sheets: {pd.ExcelFile(self.data_file_path).sheet_names}")
            return None
        except Exception as e:
            print(f"❌ Error loading order data: {e}")
            return None
    
    def _clean_order_data(self, data):
        """
        Clean and standardize order data.
        
        Args:
            data (pandas.DataFrame): Raw order data
            
        Returns:
            pandas.DataFrame: Cleaned order data
        """
        cleaned_data = data.copy()
        
        try:
            # Standardize column names
            column_mapping = {v: k for k, v in self.order_columns.items()}
            cleaned_data = cleaned_data.rename(columns=column_mapping)
            
            # Convert date column to datetime
            if 'date' in cleaned_data.columns:
                cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')
                
                # Remove rows with invalid dates
                before_count = len(cleaned_data)
                cleaned_data = cleaned_data.dropna(subset=['date'])
                after_count = len(cleaned_data)
                
                if before_count != after_count:
                    print(f"⚠️  Removed {before_count - after_count} rows with invalid dates")
            
            # Convert quantity columns to numeric
            for qty_col in ['qty_cases', 'qty_eaches']:
                if qty_col in cleaned_data.columns:
                    cleaned_data[qty_col] = pd.to_numeric(cleaned_data[qty_col], errors='coerce')
                    cleaned_data[qty_col] = cleaned_data[qty_col].fillna(0)
            
            # Calculate total case equivalent
            if 'qty_cases' in cleaned_data.columns and 'qty_eaches' in cleaned_data.columns:
                # Use case configuration from SKU master if available, otherwise use default
                case_size = config.BUSINESS_RULES['CASES_TO_EACHES_DEFAULT']
                cleaned_data['total_case_equivalent'] = (
                    cleaned_data['qty_cases'] + 
                    cleaned_data['qty_eaches'] / case_size
                )
            
            # Remove rows with zero quantities
            if 'total_case_equivalent' in cleaned_data.columns:
                before_count = len(cleaned_data)
                cleaned_data = cleaned_data[cleaned_data['total_case_equivalent'] > 0]
                after_count = len(cleaned_data)
                
                if before_count != after_count:
                    print(f"⚠️  Removed {before_count - after_count} rows with zero quantities")
            
            # Apply date range filter if specified
            if config.DATE_RANGE['START_DATE'] or config.DATE_RANGE['END_DATE']:
                before_count = len(cleaned_data)
                
                if config.DATE_RANGE['START_DATE']:
                    start_date = pd.to_datetime(config.DATE_RANGE['START_DATE'])
                    cleaned_data = cleaned_data[cleaned_data['date'] >= start_date]
                
                if config.DATE_RANGE['END_DATE']:
                    end_date = pd.to_datetime(config.DATE_RANGE['END_DATE'])
                    cleaned_data = cleaned_data[cleaned_data['date'] <= end_date]
                
                after_count = len(cleaned_data)
                if before_count != after_count:
                    print(f"📅 Applied date filter: {before_count - after_count} rows removed")
            
            # Sort by date
            if 'date' in cleaned_data.columns:
                cleaned_data = cleaned_data.sort_values('date')
            
            # Reset index
            cleaned_data = cleaned_data.reset_index(drop=True)
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Cleaned order data shape: {cleaned_data.shape}")
            
            return cleaned_data
            
        except Exception as e:
            print(f"❌ Error cleaning order data: {e}")
            return data
    
    def load_sku_master(self):
        """
        Load SKU master data from Excel file.
        
        Returns:
            pandas.DataFrame: SKU master data
            None: If loading fails or sheet doesn't exist
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Loading SKU master from sheet: {self.sheet_names['SKU_MASTER']}")
            
            self.sku_master = pd.read_excel(
                self.data_file_path,
                sheet_name=self.sheet_names['SKU_MASTER']
            )
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"SKU master data shape: {self.sku_master.shape}")
            
            # Clean SKU master data
            self.sku_master = self._clean_sku_master(self.sku_master)
            
            return self.sku_master
            
        except ValueError:
            print(f"⚠️  SKU master sheet '{self.sheet_names['SKU_MASTER']}' not found")
            return None
        except Exception as e:
            print(f"⚠️  Error loading SKU master: {e}")
            return None
    
    def _clean_sku_master(self, data):
        """
        Clean and standardize SKU master data.
        
        Args:
            data (pandas.DataFrame): Raw SKU master data
            
        Returns:
            pandas.DataFrame: Cleaned SKU master data
        """
        try:
            cleaned_data = data.copy()
            
            # Standardize column names if they exist
            available_mappings = {}
            for config_col, excel_col in self.sku_columns.items():
                if excel_col in cleaned_data.columns:
                    available_mappings[excel_col] = config_col
            
            cleaned_data = cleaned_data.rename(columns=available_mappings)
            
            # Convert numeric columns
            for col in ['case_config', 'pallet_fit']:
                if col in cleaned_data.columns:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            if config.DEBUG_SETTINGS['PRINT_SAMPLE_DATA']:
                print("Sample SKU master data:")
                print(cleaned_data.head())
            
            return cleaned_data
            
        except Exception as e:
            print(f"⚠️  Error cleaning SKU master: {e}")
            return data
    
    def load_receipt_data(self):
        """
        Load receipt data from Excel file (optional).
        
        Returns:
            pandas.DataFrame: Receipt data
            None: If loading fails or sheet doesn't exist
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Loading receipt data from sheet: {self.sheet_names['RECEIPT_DATA']}")
            
            self.receipt_data = pd.read_excel(
                self.data_file_path,
                sheet_name=self.sheet_names['RECEIPT_DATA']
            )
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Receipt data shape: {self.receipt_data.shape}")
            
            return self.receipt_data
            
        except ValueError:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Receipt data sheet '{self.sheet_names['RECEIPT_DATA']}' not found")
            return None
        except Exception as e:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Error loading receipt data: {e}")
            return None
    
    def load_inventory_data(self):
        """
        Load inventory data from Excel file (optional).
        
        Returns:
            pandas.DataFrame: Inventory data
            None: If loading fails or sheet doesn't exist
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Loading inventory data from sheet: {self.sheet_names['INVENTORY_DATA']}")
            
            self.inventory_data = pd.read_excel(
                self.data_file_path,
                sheet_name=self.sheet_names['INVENTORY_DATA']
            )
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Inventory data shape: {self.inventory_data.shape}")
            
            return self.inventory_data
            
        except ValueError:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Inventory data sheet '{self.sheet_names['INVENTORY_DATA']}' not found")
            return None
        except Exception as e:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Error loading inventory data: {e}")
            return None
    
    def validate_data_quality(self):
        """
        Validate the quality and completeness of loaded data.
        
        Returns:
            dict: Validation results with is_valid flag and list of errors
        """
        errors = []
        warnings = []
        
        try:
            # Check if main order data is loaded
            if self.order_data is None or self.order_data.empty:
                errors.append("No order data loaded")
                return {'is_valid': False, 'errors': errors, 'warnings': warnings}
            
            # Check minimum row requirements
            if len(self.order_data) < config.VALIDATION_RULES['MIN_ROWS']:
                errors.append(f"Insufficient data: {len(self.order_data)} rows (minimum: {config.VALIDATION_RULES['MIN_ROWS']})")
            
            # Check unique SKUs
            if 'sku_code' in self.order_data.columns:
                unique_skus = self.order_data['sku_code'].nunique()
                if unique_skus < config.VALIDATION_RULES['MIN_SKUS']:
                    errors.append(f"Insufficient SKU variety: {unique_skus} SKUs (minimum: {config.VALIDATION_RULES['MIN_SKUS']})")
            
            # Check date range
            if 'date' in self.order_data.columns:
                unique_dates = self.order_data['date'].nunique()
                if unique_dates < config.VALIDATION_RULES['MIN_DATES']:
                    errors.append(f"Insufficient date range: {unique_dates} days (minimum: {config.VALIDATION_RULES['MIN_DATES']})")
                
                # Check for future dates
                max_date = self.order_data['date'].max()
                if max_date > datetime.now() + timedelta(days=1):
                    warnings.append(f"Data contains future dates up to {max_date.strftime('%Y-%m-%d')}")
            
            # Check for missing values in critical columns
            critical_columns = ['date', 'sku_code']
            for col in critical_columns:
                if col in self.order_data.columns:
                    missing_count = self.order_data[col].isna().sum()
                    if missing_count > 0:
                        warnings.append(f"Missing values in {col}: {missing_count} records")
            
            # Check data types
            if 'total_case_equivalent' in self.order_data.columns:
                if not pd.api.types.is_numeric_dtype(self.order_data['total_case_equivalent']):
                    errors.append("Total case equivalent column is not numeric")
            
            # Summary statistics
            if 'total_case_equivalent' in self.order_data.columns:
                total_volume = self.order_data['total_case_equivalent'].sum()
                if total_volume <= 0:
                    errors.append("Total volume is zero or negative")
            
            self.validation_results = {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'summary': {
                    'total_rows': len(self.order_data),
                    'unique_skus': self.order_data['sku_code'].nunique() if 'sku_code' in self.order_data.columns else 0,
                    'unique_dates': self.order_data['date'].nunique() if 'date' in self.order_data.columns else 0,
                    'date_range': {
                        'start': self.order_data['date'].min() if 'date' in self.order_data.columns else None,
                        'end': self.order_data['date'].max() if 'date' in self.order_data.columns else None
                    },
                    'total_volume': self.order_data['total_case_equivalent'].sum() if 'total_case_equivalent' in self.order_data.columns else 0
                }
            }
            
            # Print warnings
            for warning in warnings:
                print(f"⚠️  Warning: {warning}")
            
            return self.validation_results
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
    
    def get_data_summary(self):
        """
        Get a summary of all loaded data.
        
        Returns:
            dict: Summary of loaded datasets
        """
        summary = {
            'order_data': {
                'loaded': self.order_data is not None,
                'rows': len(self.order_data) if self.order_data is not None else 0,
                'columns': list(self.order_data.columns) if self.order_data is not None else []
            },
            'sku_master': {
                'loaded': self.sku_master is not None,
                'rows': len(self.sku_master) if self.sku_master is not None else 0,
                'columns': list(self.sku_master.columns) if self.sku_master is not None else []
            },
            'receipt_data': {
                'loaded': self.receipt_data is not None,
                'rows': len(self.receipt_data) if self.receipt_data is not None else 0,
                'columns': list(self.receipt_data.columns) if self.receipt_data is not None else []
            },
            'inventory_data': {
                'loaded': self.inventory_data is not None,
                'rows': len(self.inventory_data) if self.inventory_data is not None else 0,
                'columns': list(self.inventory_data.columns) if self.inventory_data is not None else []
            }
        }
        
        return summary

# Test function for standalone execution
if __name__ == "__main__":
    print("Testing DataLoader...")
    loader = DataLoader()
    
    # Test loading order data
    order_data = loader.load_order_data()
    if order_data is not None:
        print(f"✅ Loaded {len(order_data)} order records")
    
    # Test validation
    validation = loader.validate_data_quality()
    print(f"Validation result: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
    
    # Print summary
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for dataset, info in summary.items():
        status = "✅ Loaded" if info['loaded'] else "❌ Not loaded"
        print(f"  {dataset}: {status} ({info['rows']} rows)")
```

### 7. scripts/order_analysis.py
```python
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

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

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
    
    def __init__(self, order_data):
        """
        Initialize the OrderAnalyzer.
        
        Args:
            order_data (pandas.DataFrame): Cleaned order data from DataLoader
        """
        self.order_data = order_data.copy()
        self.analysis_results = {}
        
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            print(f"OrderAnalyzer initialized with {len(self.order_data)} records")
    
    def run_analysis(self):
        """
        Run complete order analysis.
        
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("Running order analysis...")
            
            # Basic statistics
            self.analysis_results['basic_stats'] = self._calculate_basic_statistics()
            
            # Date range analysis
            self.analysis_results['date_range'] = self._analyze_date_range()
            
            # Volume analysis
            self.analysis_results['volume_stats'] = self._analyze_volume_patterns()
            
            # Customer analysis
            if 'customer_id' in self.order_data.columns:
                self.analysis_results['customer_stats'] = self._analyze_customer_patterns()
            
            # Peak period analysis
            self.analysis_results['peak_analysis'] = self._identify_peak_periods()
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("✅ Order analysis completed")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Error in order analysis: {e}")
            return {}
    
    def _calculate_basic_statistics(self):
        """
        Calculate basic order statistics.
        
        Returns:
            dict: Basic statistics including counts, totals, and averages
        """
        try:
            stats = {}
            
            # Record counts
            stats['total_records'] = len(self.order_data)
            stats['unique_orders'] = self.order_data['order_no'].nunique() if 'order_no' in self.order_data.columns else None
            stats['unique_shipments'] = self.order_data['shipment_no'].nunique() if 'shipment_no' in self.order_data.columns else None
            stats['unique_skus'] = self.order_data['sku_code'].nunique() if 'sku_code' in self.order_data.columns else None
            stats['unique_dates'] = self.order_data['date'].nunique() if 'date' in self.order_data.columns else None
            stats['unique_customers'] = self.order_data['customer_id'].nunique() if 'customer_id' in self.order_data.columns else None
            
            # Volume statistics
            if 'total_case_equivalent' in self.order_data.columns:
                stats['total_volume'] = self.order_data['total_case_equivalent'].sum()
                stats['average_line_volume'] = self.order_data['total_case_equivalent'].mean()
                stats['median_line_volume'] = self.order_data['total_case_equivalent'].median()
                stats['max_line_volume'] = self.order_data['total_case_equivalent'].max()
                stats['min_line_volume'] = self.order_data['total_case_equivalent'].min()
            
            # Quantity statistics
            if 'qty_cases' in self.order_data.columns:
                stats['total_cases'] = self.order_data['qty_cases'].sum()
            if 'qty_eaches' in self.order_data.columns:
                stats['total_eaches'] = self.order_data['qty_eaches'].sum()
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Error calculating basic statistics: {e}")
            return {}
    
    def _analyze_date_range(self):
        """
        Analyze date range and patterns.
        
        Returns:
            dict: Date range analysis including start, end, and duration
        """
        try:
            if 'date' not in self.order_data.columns:
                return {}
            
            date_stats = {}
            
            # Date range
            min_date = self.order_data['date'].min()
            max_date = self.order_data['date'].max()
            date_range_days = (max_date - min_date).days + 1
            
            date_stats['start_date'] = min_date.strftime('%Y-%m-%d')
            date_stats['end_date'] = max_date.strftime('%Y-%m-%d')
            date_stats['total_days'] = date_range_days
            date_stats['active_days'] = self.order_data['date'].nunique()
            date_stats['coverage_percentage'] = (date_stats['active_days'] / date_range_days) * 100
            
            # Day of week patterns
            self.order_data['day_of_week'] = self.order_data['date'].dt.day_name()
            self.order_data['day_of_week_num'] = self.order_data['date'].dt.dayofweek
            
            if 'total_case_equivalent' in self.order_data.columns:
                dow_volume = self.order_data.groupby('day_of_week')['total_case_equivalent'].sum().to_dict()
                date_stats['volume_by_day_of_week'] = dow_volume
            
            # Month patterns
            self.order_data['month'] = self.order_data['date'].dt.month_name()
            self.order_data['year_month'] = self.order_data['date'].dt.to_period('M').astype(str)
            
            if 'total_case_equivalent' in self.order_data.columns:
                month_volume = self.order_data.groupby('month')['total_case_equivalent'].sum().to_dict()
                date_stats['volume_by_month'] = month_volume
            
            return date_stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing date range: {e}")
            return {}
    
    def _analyze_volume_patterns(self):
        """
        Analyze volume patterns and distributions.
        
        Returns:
            dict: Volume pattern analysis
        """
        try:
            if 'total_case_equivalent' not in self.order_data.columns:
                return {}
            
            volume_stats = {}
            
            # Daily volume aggregation
            daily_volume = self.order_data.groupby('date')['total_case_equivalent'].agg([
                'sum', 'count', 'mean', 'std'
            ]).reset_index()
            daily_volume.columns = ['date', 'total_volume', 'order_lines', 'avg_line_volume', 'volume_std']
            
            # Daily volume statistics
            volume_stats['daily_volume_stats'] = {
                'average_daily_volume': daily_volume['total_volume'].mean(),
                'median_daily_volume': daily_volume['total_volume'].median(),
                'max_daily_volume': daily_volume['total_volume'].max(),
                'min_daily_volume': daily_volume['total_volume'].min(),
                'std_daily_volume': daily_volume['total_volume'].std(),
                'coefficient_of_variation': daily_volume['total_volume'].std() / daily_volume['total_volume'].mean()
            }
            
            # Volume distribution percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            volume_percentiles = {}
            for p in percentiles:
                volume_percentiles[f'p{p}'] = np.percentile(daily_volume['total_volume'], p)
            volume_stats['daily_volume_percentiles'] = volume_percentiles
            
            # Peak capacity requirements
            peak_factor = config.BUSINESS_RULES['PEAK_FACTOR']
            volume_stats['capacity_requirements'] = {
                'average_capacity_needed': volume_stats['daily_volume_stats']['average_daily_volume'],
                'peak_capacity_needed': volume_stats['daily_volume_stats']['max_daily_volume'],
                'recommended_capacity': volume_stats['daily_volume_stats']['average_daily_volume'] * peak_factor,
                'p95_capacity': volume_percentiles['p95'],
                'p99_capacity': volume_percentiles['p99']
            }
            
            return volume_stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing volume patterns: {e}")
            return {}
    
    def _analyze_customer_patterns(self):
        """
        Analyze customer behavior patterns.
        
        Returns:
            dict: Customer analysis results
        """
        try:
            if 'customer_id' not in self.order_data.columns:
                return {}
            
            customer_stats = {}
            
            # Customer volume analysis
            customer_volume = self.order_data.groupby('customer_id')['total_case_equivalent'].agg([
                'sum', 'count', 'mean'
            ]).reset_index()
            customer_volume.columns = ['customer_id', 'total_volume', 'order_lines', 'avg_line_volume']
            customer_volume = customer_volume.sort_values('total_volume', ascending=False)
            
            # Customer statistics
            customer_stats['total_customers'] = len(customer_volume)
            customer_stats['top_10_customers_volume_share'] = (
                customer_volume.head(10)['total_volume'].sum() / 
                customer_volume['total_volume'].sum() * 100
            )
            
            # Customer frequency analysis
            customer_frequency = self.order_data.groupby('customer_id')['date'].nunique().reset_index()
            customer_frequency.columns = ['customer_id', 'order_days']
            
            customer_stats['customer_frequency_stats'] = {
                'avg_order_days_per_customer': customer_frequency['order_days'].mean(),
                'max_order_days': customer_frequency['order_days'].max(),
                'customers_ordering_daily': len(customer_frequency[customer_frequency['order_days'] >= self.order_data['date'].nunique() * 0.8])
            }
            
            return customer_stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing customer patterns: {e}")
            return {}
    
    def _identify_peak_periods(self):
        """
        Identify peak periods and patterns.
        
        Returns:
            dict: Peak period analysis
        """
        try:
            if 'date' not in self.order_data.columns or 'total_case_equivalent' not in self.order_data.columns:
                return {}
            
            peak_stats = {}
            
            # Daily volume for peak analysis
            daily_volume = self.order_data.groupby('date')['total_case_equivalent'].sum().reset_index()
            daily_volume.columns = ['date', 'volume']
            
            # Calculate thresholds for peak identification
            mean_volume = daily_volume['volume'].mean()
            std_volume = daily_volume['volume'].std()
            
            # Define peak thresholds
            peak_threshold = mean_volume + std_volume
            high_peak_threshold = mean_volume + 2 * std_volume
            
            # Identify peak days
            peak_days = daily_volume[daily_volume['volume'] >= peak_threshold]
            high_peak_days = daily_volume[daily_volume['volume'] >= high_peak_threshold]
            
            peak_stats['peak_analysis'] = {
                'total_days_analyzed': len(daily_volume),
                'peak_days_count': len(peak_days),
                'high_peak_days_count': len(high_peak_days),
                'peak_days_percentage': len(peak_days) / len(daily_volume) * 100,
                'peak_threshold': peak_threshold,
                'high_peak_threshold': high_peak_threshold
            }
            
            # Peak day details
            if len(peak_days) > 0:
                peak_stats['peak_days_details'] = peak_days.sort_values('volume', ascending=False).head(10).to_dict('records')
            
            # Monthly peak analysis
            if len(daily_volume) > 30:  # Only if we have enough data
                daily_volume['month'] = pd.to_datetime(daily_volume['date']).dt.month_name()
                monthly_avg = daily_volume.groupby('month')['volume'].mean()
                peak_stats['peak_months'] = monthly_avg.sort_values(ascending=False).to_dict()
            
            return peak_stats
            
        except Exception as e:
            print(f"⚠️  Error identifying peak periods: {e}")
            return {}
    
    def get_date_summary(self):
        """
        Get daily order summary for Excel export.
        
        Returns:
            pandas.DataFrame: Daily summary with all key metrics
        """
        try:
            if 'date' not in self.order_data.columns:
                return pd.DataFrame()
            
            # Group by date and calculate metrics
            date_summary = self.order_data.groupby('date').agg({
                'order_no': 'nunique',
                'shipment_no': 'nunique',
                'sku_code': 'nunique',
                'customer_id': 'nunique' if 'customer_id' in self.order_data.columns else lambda x: None,
                'total_case_equivalent': ['sum', 'count', 'mean'],
                'qty_cases': 'sum',
                'qty_eaches': 'sum'
            }).reset_index()
            
            # Flatten column names
            date_summary.columns = [
                'Date',
                'Distinct_Orders',
                'Distinct_Shipments', 
                'Distinct_SKUs',
                'Distinct_Customers',
                'Total_Case_Equiv',
                'Total_Order_Lines',
                'Avg_Line_Volume',
                'Total_Cases',
                'Total_Eaches'
            ]
            
            # Remove customer column if not available
            if 'customer_id' not in self.order_data.columns:
                date_summary = date_summary.drop('Distinct_Customers', axis=1)
            
            # Add calculated fields
            date_summary['Day_of_Week'] = pd.to_datetime(date_summary['Date']).dt.day_name()
            date_summary['Month'] = pd.to_datetime(date_summary['Date']).dt.month_name()
            date_summary['Week_of_Year'] = pd.to_datetime(date_summary['Date']).dt.isocalendar().week
            
            # Add cumulative metrics
            date_summary['Cumulative_Volume'] = date_summary['Total_Case_Equiv'].cumsum()
            date_summary['Volume_Rank'] = date_summary['Total_Case_Equiv'].rank(ascending=False, method='dense')
            
            # Sort by date
            date_summary = date_summary.sort_values('Date')
            
            return date_summary
            
        except Exception as e:
            print(f"⚠️  Error creating date summary: {e}")
            return pd.DataFrame()
    
    def get_percentile_analysis(self):
        """
        Get percentile analysis for capacity planning.
        
        Returns:
            pandas.DataFrame: Percentile analysis results
        """
        try:
            if 'date' not in self.order_data.columns or 'total_case_equivalent' not in self.order_data.columns:
                return pd.DataFrame()
            
            # Daily volume data
            daily_volume = self.order_data.groupby('date')['total_case_equivalent'].sum()
            
            # Calculate percentiles
            percentile_data = []
            percentiles = config.PERCENTILE_LEVELS
            
            for p in percentiles:
                value = np.percentile(daily_volume, p)
                percentile_data.append({
                    'Percentile': f'{p}th',
                    'Volume': value,
                    'Description': f'{p}% of days have volume ≤ {value:.0f} cases'
                })
            
            # Add summary statistics
            percentile_data.extend([
                {
                    'Percentile': 'Maximum',
                    'Volume': daily_volume.max(),
                    'Description': f'Highest single day volume'
                },
                {
                    'Percentile': 'Average',
                    'Volume': daily_volume.mean(),
                    'Description': f'Average daily volume'
                },
                {
                    'Percentile': 'Minimum',
                    'Volume': daily_volume.min(),
                    'Description': f'Lowest single day volume'
                }
            ])
            
            percentile_df = pd.DataFrame(percentile_data)
            
            # Add capacity planning insights
            percentile_df['Capacity_Buffer'] = percentile_df['Volume'] * config.BUSINESS_RULES['PEAK_FACTOR']
            percentile_df['Capacity_Recommendation'] = percentile_df.apply(
                lambda row: f"Plan for {row['Capacity_Buffer']:.0f} cases capacity" if pd.notnull(row['Capacity_Buffer']) else "",
                axis=1
            )
            
            return percentile_df
            
        except Exception as e:
            print(f"⚠️  Error creating percentile analysis: {e}")
            return pd.DataFrame()

# Test function for standalone execution
if __name__ == "__main__":
    print("Testing OrderAnalyzer...")
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    sample_data = pd.DataFrame({
        'date': np.random.choice(dates, 1000),
        'order_no': ['ORD' + str(i) for i in range(1000)],
        'shipment_no': ['SHP' + str(i//5) for i in range(1000)],
        'sku_code': ['SKU' + str(i%100) for i in range(1000)],
        'customer_id': ['CUST' + str(i%50) for i in range(1000)],
        'qty_cases': np.random.randint(1, 20, 1000),
        'qty_eaches': np.random.randint(0, 12, 1000),
        'total_case_equivalent': np.random.randint(1, 25, 1000)
    })
    
    analyzer = OrderAnalyzer(sample_data)
    results = analyzer.run_analysis()
    
    print(f"✅ Analysis completed with {len(results)} result categories")
    
    # Test summary functions
    date_summary = analyzer.get_date_summary()
    print(f"✅ Date summary: {len(date_summary)} days analyzed")
    
    percentiles = analyzer.get_percentile_analysis()
    print(f"✅ Percentile analysis: {len(percentiles)} percentiles calculated")
```

### 8. scripts/sku_analysis.py
```python
"""
SKU Analysis Module for Warehouse Analysis Tool V2

PURPOSE:
This module analyzes SKU (product) patterns and characteristics including:
- SKU volume and movement patterns
- Product performance analysis
- Category analysis (if available)
- Movement frequency classification
- Product lifecycle insights

FOR BEGINNERS:
- This module takes order data and analyzes it at the product (SKU) level
- It identifies which products are high-volume, high-frequency, etc.
- The results help with inventory management and slotting decisions
- All calculations help optimize warehouse layout and operations
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class SKUAnalyzer:
    """
    SKU analysis class for product-level insights.
    
    This class analyzes SKU data to identify:
    - Volume patterns per SKU
    - Movement frequency patterns
    - Product performance metrics
    - Category analysis (if available)
    - SKU lifecycle insights
    """
    
    def __init__(self, order_data, sku_master=None):
        """
        Initialize the SKUAnalyzer.
        
        Args:
            order_data (pandas.DataFrame): Cleaned order data from DataLoader
            sku_master (pandas.DataFrame): SKU master data (optional)
        """
        self.order_data = order_data.copy()
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.analysis_results = {}
        
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            print(f"SKUAnalyzer initialized with {len(self.order_data)} order records")
            if self.sku_master is not None:
                print(f"SKU master data available with {len(self.sku_master)} SKU records")
    
    def run_analysis(self):
        """
        Run complete SKU analysis.
        
        Returns:
            dict: Comprehensive SKU analysis results
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("Running SKU analysis...")
            
            # Basic SKU statistics
            self.analysis_results['basic_stats'] = self._calculate_basic_sku_stats()
            
            # SKU performance analysis
            self.analysis_results['performance_stats'] = self._analyze_sku_performance()
            
            # Movement frequency analysis
            self.analysis_results['movement_analysis'] = self._analyze_movement_patterns()
            
            # Category analysis (if SKU master available)
            if self.sku_master is not None:
                self.analysis_results['category_analysis'] = self._analyze_categories()
            
            # SKU lifecycle analysis
            self.analysis_results['lifecycle_analysis'] = self._analyze_sku_lifecycle()
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("✅ SKU analysis completed")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Error in SKU analysis: {e}")
            return {}
    
    def _calculate_basic_sku_stats(self):
        """
        Calculate basic SKU statistics.
        
        Returns:
            dict: Basic SKU statistics
        """
        try:
            if 'sku_code' not in self.order_data.columns:
                return {}
            
            stats = {}
            
            # Basic counts
            stats['total_skus'] = self.order_data['sku_code'].nunique()
            stats['total_order_lines'] = len(self.order_data)
            
            # Volume statistics
            if 'total_case_equivalent' in self.order_data.columns:
                stats['total_volume'] = self.order_data['total_case_equivalent'].sum()
                stats['avg_volume_per_sku'] = stats['total_volume'] / stats['total_skus']
            
            # Order line statistics
            stats['avg_lines_per_sku'] = stats['total_order_lines'] / stats['total_skus']
            
            # Date range for analysis
            if 'date' in self.order_data.columns:
                stats['analysis_start_date'] = self.order_data['date'].min().strftime('%Y-%m-%d')
                stats['analysis_end_date'] = self.order_data['date'].max().strftime('%Y-%m-%d')
                stats['analysis_days'] = self.order_data['date'].nunique()
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Error calculating basic SKU statistics: {e}")
            return {}
    
    def _analyze_sku_performance(self):
        """
        Analyze SKU performance metrics.
        
        Returns:
            dict: SKU performance analysis
        """
        try:
            if 'sku_code' not in self.order_data.columns:
                return {}
            
            performance_stats = {}
            
            # Aggregate by SKU
            sku_agg = self.order_data.groupby('sku_code').agg({
                'total_case_equivalent': ['sum', 'count', 'mean', 'std'],
                'date': ['nunique', 'min', 'max'],
                'order_no': 'nunique',
                'shipment_no': 'nunique'
            }).reset_index()
            
            # Flatten column names
            sku_agg.columns = [
                'sku_code', 'total_volume', 'order_lines', 'avg_line_volume', 'volume_std',
                'order_days', 'first_order_date', 'last_order_date',
                'unique_orders', 'unique_shipments'
            ]
            
            # Calculate additional metrics
            sku_agg['volume_coefficient_of_variation'] = sku_agg['volume_std'] / sku_agg['avg_line_volume']
            sku_agg['volume_percentage'] = (sku_agg['total_volume'] / sku_agg['total_volume'].sum()) * 100
            sku_agg['lines_percentage'] = (sku_agg['order_lines'] / sku_agg['order_lines'].sum()) * 100
            
            # Performance statistics
            performance_stats['top_10_volume_skus'] = sku_agg.nlargest(10, 'total_volume')[['sku_code', 'total_volume', 'volume_percentage']].to_dict('records')
            performance_stats['top_10_frequency_skus'] = sku_agg.nlargest(10, 'order_lines')[['sku_code', 'order_lines', 'lines_percentage']].to_dict('records')
            
            # Volume concentration
            performance_stats['volume_concentration'] = {
                'top_10_skus_volume_share': sku_agg.nlargest(10, 'total_volume')['volume_percentage'].sum(),
                'top_20_skus_volume_share': sku_agg.nlargest(20, 'total_volume')['volume_percentage'].sum(),
                'top_50_skus_volume_share': sku_agg.nlargest(50, 'total_volume')['volume_percentage'].sum()
            }
            
            # Movement patterns
            total_days = self.order_data['date'].nunique() if 'date' in self.order_data.columns else 1
            sku_agg['movement_frequency'] = sku_agg['order_days'] / total_days
            
            performance_stats['movement_distribution'] = {
                'daily_movers': len(sku_agg[sku_agg['movement_frequency'] >= 0.8]),
                'frequent_movers': len(sku_agg[(sku_agg['movement_frequency'] >= 0.5) & (sku_agg['movement_frequency'] < 0.8)]),
                'occasional_movers': len(sku_agg[(sku_agg['movement_frequency'] >= 0.2) & (sku_agg['movement_frequency'] < 0.5)]),
                'rare_movers': len(sku_agg[sku_agg['movement_frequency'] < 0.2])
            }
            
            return performance_stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing SKU performance: {e}")
            return {}
    
    def _analyze_movement_patterns(self):
        """
        Analyze SKU movement patterns and frequency.
        
        Returns:
            dict: Movement pattern analysis
        """
        try:
            if 'sku_code' not in self.order_data.columns or 'date' not in self.order_data.columns:
                return {}
            
            movement_stats = {}
            
            # Calculate movement metrics per SKU
            sku_movement = self.order_data.groupby('sku_code').agg({
                'date': ['nunique', 'count'],
                'total_case_equivalent': 'sum'
            }).reset_index()
            
            sku_movement.columns = ['sku_code', 'movement_days', 'total_lines', 'total_volume']
            
            # Calculate total analysis period
            total_days = self.order_data['date'].nunique()
            
            # Movement frequency metrics
            sku_movement['movement_frequency'] = sku_movement['movement_days'] / total_days
            sku_movement['lines_per_day'] = sku_movement['total_lines'] / sku_movement['movement_days']
            sku_movement['volume_per_day'] = sku_movement['total_volume'] / sku_movement['movement_days']
            
            # Movement pattern classification
            def classify_movement(freq):
                if freq >= 0.8:
                    return 'Daily'
                elif freq >= 0.5:
                    return 'Frequent'
                elif freq >= 0.2:
                    return 'Occasional'
                else:
                    return 'Rare'
            
            sku_movement['movement_pattern'] = sku_movement['movement_frequency'].apply(classify_movement)
            
            # Movement pattern statistics
            pattern_stats = sku_movement['movement_pattern'].value_counts().to_dict()
            movement_stats['pattern_distribution'] = pattern_stats
            
            # Velocity analysis
            movement_stats['velocity_analysis'] = {
                'high_velocity_skus': len(sku_movement[sku_movement['lines_per_day'] >= 5]),
                'medium_velocity_skus': len(sku_movement[(sku_movement['lines_per_day'] >= 1) & (sku_movement['lines_per_day'] < 5)]),
                'low_velocity_skus': len(sku_movement[sku_movement['lines_per_day'] < 1])
            }
            
            # Days between orders analysis
            sku_gaps = []
            for sku in self.order_data['sku_code'].unique():
                sku_dates = self.order_data[self.order_data['sku_code'] == sku]['date'].sort_values().unique()
                if len(sku_dates) > 1:
                    gaps = pd.Series(sku_dates).diff().dt.days.dropna()
                    sku_gaps.extend(gaps.tolist())
            
            if sku_gaps:
                movement_stats['order_gap_analysis'] = {
                    'avg_days_between_orders': np.mean(sku_gaps),
                    'median_days_between_orders': np.median(sku_gaps),
                    'max_gap_days': max(sku_gaps),
                    'min_gap_days': min(sku_gaps)
                }
            
            return movement_stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing movement patterns: {e}")
            return {}
    
    def _analyze_categories(self):
        """
        Analyze SKU categories if master data is available.
        
        Returns:
            dict: Category analysis results
        """
        try:
            if self.sku_master is None or 'category' not in self.sku_master.columns:
                return {}
            
            category_stats = {}
            
            # Merge order data with SKU master
            merged_data = self.order_data.merge(
                self.sku_master[['sku_code', 'category']], 
                on='sku_code', 
                how='left'
            )
            
            # Category performance analysis
            category_performance = merged_data.groupby('category').agg({
                'sku_code': 'nunique',
                'total_case_equivalent': ['sum', 'count', 'mean'],
                'date': 'nunique'
            }).reset_index()
            
            category_performance.columns = [
                'category', 'unique_skus', 'total_volume', 'order_lines', 
                'avg_line_volume', 'order_days'
            ]
            
            # Calculate percentages
            category_performance['volume_percentage'] = (
                category_performance['total_volume'] / category_performance['total_volume'].sum() * 100
            )
            category_performance['sku_percentage'] = (
                category_performance['unique_skus'] / category_performance['unique_skus'].sum() * 100
            )
            
            # Sort by volume
            category_performance = category_performance.sort_values('total_volume', ascending=False)
            
            category_stats['category_performance'] = category_performance.to_dict('records')
            category_stats['total_categories'] = len(category_performance)
            category_stats['top_category_volume_share'] = category_performance.iloc[0]['volume_percentage'] if len(category_performance) > 0 else 0
            
            return category_stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing categories: {e}")
            return {}
    
    def _analyze_sku_lifecycle(self):
        """
        Analyze SKU lifecycle patterns.
        
        Returns:
            dict: SKU lifecycle analysis
        """
        try:
            if 'sku_code' not in self.order_data.columns or 'date' not in self.order_data.columns:
                return {}
            
            lifecycle_stats = {}
            
            # Calculate lifecycle metrics per SKU
            sku_lifecycle = self.order_data.groupby('sku_code').agg({
                'date': ['min', 'max', 'nunique'],
                'total_case_equivalent': ['sum', 'count']
            }).reset_index()
            
            sku_lifecycle.columns = [
                'sku_code', 'first_order', 'last_order', 'active_days',
                'total_volume', 'order_lines'
            ]
            
            # Calculate lifecycle duration
            sku_lifecycle['lifecycle_days'] = (sku_lifecycle['last_order'] - sku_lifecycle['first_order']).dt.days + 1
            sku_lifecycle['activity_ratio'] = sku_lifecycle['active_days'] / sku_lifecycle['lifecycle_days']
            
            # Analysis period
            analysis_start = self.order_data['date'].min()
            analysis_end = self.order_data['date'].max()
            
            # Classify SKUs by lifecycle stage
            def classify_lifecycle(row):
                if row['first_order'] == analysis_start and row['last_order'] == analysis_end:
                    return 'Established'
                elif row['first_order'] == analysis_start:
                    return 'Declining'
                elif row['last_order'] == analysis_end:
                    return 'Growing'
                else:
                    return 'Discontinued'
            
            sku_lifecycle['lifecycle_stage'] = sku_lifecycle.apply(classify_lifecycle, axis=1)
            
            # Lifecycle statistics
            stage_counts = sku_lifecycle['lifecycle_stage'].value_counts().to_dict()
            lifecycle_stats['lifecycle_distribution'] = stage_counts
            
            # Activity patterns
            lifecycle_stats['activity_patterns'] = {
                'highly_active_skus': len(sku_lifecycle[sku_lifecycle['activity_ratio'] >= 0.8]),
                'moderately_active_skus': len(sku_lifecycle[(sku_lifecycle['activity_ratio'] >= 0.5) & (sku_lifecycle['activity_ratio'] < 0.8)]),
                'sporadically_active_skus': len(sku_lifecycle[(sku_lifecycle['activity_ratio'] >= 0.2) & (sku_lifecycle['activity_ratio'] < 0.5)]),
                'rarely_active_skus': len(sku_lifecycle[sku_lifecycle['activity_ratio'] < 0.2])
            }
            
            # New and discontinued SKUs
            recent_threshold = analysis_end - pd.Timedelta(days=30)
            old_threshold = analysis_start + pd.Timedelta(days=30)
            
            lifecycle_stats['recent_activity'] = {
                'new_skus_last_30_days': len(sku_lifecycle[sku_lifecycle['first_order'] >= recent_threshold]),
                'discontinued_skus_last_30_days': len(sku_lifecycle[sku_lifecycle['last_order'] <= old_threshold])
            }
            
            return lifecycle_stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing SKU lifecycle: {e}")
            return {}
    
    def get_sku_profile(self):
        """
        Get detailed SKU profile for Excel export.
        
        Returns:
            pandas.DataFrame: Detailed SKU profile with all metrics
        """
        try:
            if 'sku_code' not in self.order_data.columns:
                return pd.DataFrame()
            
            # Base SKU aggregation
            sku_profile = self.order_data.groupby('sku_code').agg({
                'total_case_equivalent': ['sum', 'count', 'mean', 'std'],
                'date': ['nunique', 'min', 'max'],
                'order_no': 'nunique',
                'shipment_no': 'nunique',
                'qty_cases': 'sum',
                'qty_eaches': 'sum'
            }).reset_index()
            
            # Flatten column names
            sku_profile.columns = [
                'Sku_Code', 'Total_Volume', 'Order_Lines', 'Avg_Line_Volume', 'Volume_Std',
                'Order_Days', 'First_Order_Date', 'Last_Order_Date',
                'Unique_Orders', 'Unique_Shipments', 'Total_Cases', 'Total_Eaches'
            ]
            
            # Calculate additional metrics
            total_analysis_days = self.order_data['date'].nunique() if 'date' in self.order_data.columns else 1
            
            sku_profile['Movement_Frequency'] = sku_profile['Order_Days'] / total_analysis_days
            sku_profile['Avg_Volume_Per_Day'] = sku_profile['Total_Volume'] / sku_profile['Order_Days']
            sku_profile['Avg_Lines_Per_Day'] = sku_profile['Order_Lines'] / sku_profile['Order_Days']
            
            # Volume and line percentages
            sku_profile['Volume_Percentage'] = (sku_profile['Total_Volume'] / sku_profile['Total_Volume'].sum()) * 100
            sku_profile['Lines_Percentage'] = (sku_profile['Order_Lines'] / sku_profile['Order_Lines'].sum()) * 100
            
            # Rankings
            sku_profile['Volume_Rank'] = sku_profile['Total_Volume'].rank(ascending=False, method='dense')
            sku_profile['Frequency_Rank'] = sku_profile['Order_Lines'].rank(ascending=False, method='dense')
            
            # Movement pattern classification
            def classify_movement(freq):
                if freq >= 0.8:
                    return 'Daily'
                elif freq >= 0.5:
                    return 'Frequent'
                elif freq >= 0.2:
                    return 'Occasional'
                else:
                    return 'Rare'
            
            sku_profile['Movement_Pattern'] = sku_profile['Movement_Frequency'].apply(classify_movement)
            
            # Merge with SKU master if available
            if self.sku_master is not None:
                # Ensure consistent column names
                master_cols = ['sku_code']
                if 'category' in self.sku_master.columns:
                    master_cols.append('category')
                if 'case_config' in self.sku_master.columns:
                    master_cols.append('case_config')
                if 'pallet_fit' in self.sku_master.columns:
                    master_cols.append('pallet_fit')
                
                sku_master_subset = self.sku_master[master_cols].copy()
                sku_master_subset.columns = [col.title().replace('_', '_') for col in sku_master_subset.columns]
                sku_master_subset = sku_master_subset.rename(columns={'Sku_Code': 'Sku_Code'})
                
                sku_profile = sku_profile.merge(sku_master_subset, on='Sku_Code', how='left')
            
            # Sort by volume rank
            sku_profile = sku_profile.sort_values('Volume_Rank')
            
            return sku_profile
            
        except Exception as e:
            print(f"⚠️  Error creating SKU profile: {e}")
            return pd.DataFrame()

# Test function for standalone execution
if __name__ == "__main__":
    print("Testing SKUAnalyzer...")
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    sample_order_data = pd.DataFrame({
        'date': np.random.choice(dates, 1000),
        'order_no': ['ORD' + str(i) for i in range(1000)],
        'shipment_no': ['SHP' + str(i//5) for i in range(1000)],
        'sku_code': ['SKU' + str(i%100) for i in range(1000)],
        'qty_cases': np.random.randint(1, 20, 1000),
        'qty_eaches': np.random.randint(0, 12, 1000),
        'total_case_equivalent': np.random.randint(1, 25, 1000)
    })
    
    sample_sku_master = pd.DataFrame({
        'sku_code': ['SKU' + str(i) for i in range(100)],
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'case_config': np.random.randint(6, 24, 100),
        'pallet_fit': np.random.randint(20, 50, 100)
    })
    
    analyzer = SKUAnalyzer(sample_order_data, sample_sku_master)
    results = analyzer.run_analysis()
    
    print(f"✅ Analysis completed with {len(results)} result categories")
    
    # Test profile function
    sku_profile = analyzer.get_sku_profile()
    print(f"✅ SKU profile: {len(sku_profile)} SKUs profiled")
```

Due to length constraints, I'll continue with the remaining scripts (ABC-FMS Analysis, Receipt Analysis, and Excel Generator) in the next part of the documentation file. The complete documentation structure is established and the remaining scripts follow the same pattern of comprehensive documentation and beginner-friendly explanations.

### 9. scripts/abc_fms_analysis.py
```python
"""
ABC-FMS Analysis Module for Warehouse Analysis Tool V2

PURPOSE:
This module performs ABC-FMS classification analysis:
- ABC Classification: Based on volume contribution (A=high volume, B=medium, C=low)
- FMS Classification: Based on movement frequency (F=fast moving, M=medium, S=slow)
- Cross-tabulation analysis: Combines both classifications
- Strategic slotting recommendations

FOR BEGINNERS:
- ABC analysis helps identify which products contribute most to volume
- FMS analysis helps identify which products move most frequently
- The combination helps optimize warehouse layout and picking strategies
- A-Fast items should be closest to shipping, C-Slow items can be furthest away
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class ABCFMSAnalyzer:
    """
    ABC-FMS classification analysis class.
    
    This class performs dual classification:
    - ABC: Based on cumulative volume contribution
    - FMS: Based on cumulative movement frequency
    - Cross-tabulation: Combining both for strategic insights
    """
    
    def __init__(self, order_data, sku_master=None):
        """
        Initialize the ABCFMSAnalyzer.
        
        Args:
            order_data (pandas.DataFrame): Cleaned order data from DataLoader
            sku_master (pandas.DataFrame): SKU master data (optional)
        """
        self.order_data = order_data.copy()
        self.sku_master = sku_master.copy() if sku_master is not None else None
        self.analysis_results = {}
        
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            print(f"ABCFMSAnalyzer initialized with {len(self.order_data)} order records")
    
    def run_analysis(self):
        """
        Run complete ABC-FMS classification analysis.
        
        Returns:
            dict: Comprehensive ABC-FMS analysis results
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("Running ABC-FMS classification analysis...")
            
            # Prepare SKU-level data for classification
            sku_data = self._prepare_sku_data()
            
            if sku_data.empty:
                print("⚠️  No SKU data available for ABC-FMS analysis")
                return {}
            
            # Perform ABC classification
            sku_data = self._perform_abc_classification(sku_data)
            
            # Perform FMS classification
            sku_data = self._perform_fms_classification(sku_data)
            
            # Cross-tabulation analysis
            self.analysis_results['cross_tabulation'] = self._perform_cross_tabulation(sku_data)
            
            # Classification summary
            self.analysis_results['classification_summary'] = self._generate_classification_summary(sku_data)
            
            # Strategic insights
            self.analysis_results['strategic_insights'] = self._generate_strategic_insights(sku_data)
            
            # Store detailed classification
            self.classified_data = sku_data
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("✅ ABC-FMS analysis completed")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Error in ABC-FMS analysis: {e}")
            return {}
    
    def _prepare_sku_data(self):
        """
        Prepare SKU-level data for classification.
        
        Returns:
            pandas.DataFrame: SKU-level aggregated data
        """
        try:
            if 'sku_code' not in self.order_data.columns:
                return pd.DataFrame()
            
            # Aggregate order data by SKU
            sku_aggregation = {
                'total_case_equivalent': 'sum',
                'date': 'nunique',  # Number of unique order dates (frequency indicator)
                'order_no': 'nunique',  # Number of unique orders
                'shipment_no': 'nunique'  # Number of unique shipments
            }
            
            # Add quantity columns if available
            if 'qty_cases' in self.order_data.columns:
                sku_aggregation['qty_cases'] = 'sum'
            if 'qty_eaches' in self.order_data.columns:
                sku_aggregation['qty_eaches'] = 'sum'
            
            sku_data = self.order_data.groupby('sku_code').agg(sku_aggregation).reset_index()
            
            # Rename columns for clarity
            column_mapping = {
                'total_case_equivalent': 'Total_Volume',
                'date': 'Order_Days',
                'order_no': 'Unique_Orders',
                'shipment_no': 'Unique_Shipments'
            }
            
            sku_data = sku_data.rename(columns=column_mapping)
            
            # Calculate additional metrics
            total_analysis_days = self.order_data['date'].nunique() if 'date' in self.order_data.columns else 1
            sku_data['Movement_Frequency'] = sku_data['Order_Days'] / total_analysis_days
            sku_data['Avg_Volume_Per_Day'] = sku_data['Total_Volume'] / sku_data['Order_Days']
            
            # Add order line count
            order_lines = self.order_data.groupby('sku_code').size().reset_index(name='Total_Order_Lines')
            sku_data = sku_data.merge(order_lines, on='sku_code', how='left')
            
            # Merge with SKU master if available
            if self.sku_master is not None:
                sku_data = sku_data.merge(self.sku_master, on='sku_code', how='left')
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Prepared SKU data for {len(sku_data)} SKUs")
            
            return sku_data
            
        except Exception as e:
            print(f"⚠️  Error preparing SKU data: {e}")
            return pd.DataFrame()
    
    def _perform_abc_classification(self, sku_data):
        """
        Perform ABC classification based on volume.
        
        Args:
            sku_data (pandas.DataFrame): SKU-level data
            
        Returns:
            pandas.DataFrame: SKU data with ABC classification
        """
        try:
            # Sort by volume in descending order
            sku_data = sku_data.sort_values('Total_Volume', ascending=False).reset_index(drop=True)
            
            # Calculate cumulative volume percentage
            sku_data['Cumulative_Volume'] = sku_data['Total_Volume'].cumsum()
            total_volume = sku_data['Total_Volume'].sum()
            sku_data['Cumulative_Volume_Pct'] = (sku_data['Cumulative_Volume'] / total_volume) * 100
            
            # Assign ABC classification
            def assign_abc(cum_pct):
                if cum_pct <= config.ABC_THRESHOLDS['A_THRESHOLD']:
                    return 'A'
                elif cum_pct <= config.ABC_THRESHOLDS['B_THRESHOLD']:
                    return 'B'
                else:
                    return 'C'
            
            sku_data['ABC'] = sku_data['Cumulative_Volume_Pct'].apply(assign_abc)
            
            # Calculate volume percentage for each SKU
            sku_data['Volume_Percentage'] = (sku_data['Total_Volume'] / total_volume) * 100
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                abc_counts = sku_data['ABC'].value_counts()
                print(f"ABC Classification: A={abc_counts.get('A', 0)}, B={abc_counts.get('B', 0)}, C={abc_counts.get('C', 0)}")
            
            return sku_data
            
        except Exception as e:
            print(f"⚠️  Error performing ABC classification: {e}")
            return sku_data
    
    def _perform_fms_classification(self, sku_data):
        """
        Perform FMS classification based on movement frequency.
        
        Args:
            sku_data (pandas.DataFrame): SKU data with ABC classification
            
        Returns:
            pandas.DataFrame: SKU data with ABC and FMS classification
        """
        try:
            # Sort by order lines (frequency indicator) in descending order for FMS
            sku_data_fms = sku_data.sort_values('Total_Order_Lines', ascending=False).reset_index(drop=True)
            
            # Calculate cumulative order lines percentage
            sku_data_fms['Cumulative_Lines'] = sku_data_fms['Total_Order_Lines'].cumsum()
            total_lines = sku_data_fms['Total_Order_Lines'].sum()
            sku_data_fms['Cumulative_Lines_Pct'] = (sku_data_fms['Cumulative_Lines'] / total_lines) * 100
            
            # Assign FMS classification
            def assign_fms(cum_pct):
                if cum_pct <= config.FMS_THRESHOLDS['F_THRESHOLD']:
                    return 'F'
                elif cum_pct <= config.FMS_THRESHOLDS['M_THRESHOLD']:
                    return 'M'
                else:
                    return 'S'
            
            sku_data_fms['FMS'] = sku_data_fms['Cumulative_Lines_Pct'].apply(assign_fms)
            
            # Calculate order lines percentage for each SKU
            sku_data_fms['Lines_Percentage'] = (sku_data_fms['Total_Order_Lines'] / total_lines) * 100
            
            # Merge FMS classification back to original data (preserving ABC sort order)
            fms_classification = sku_data_fms[['sku_code', 'FMS', 'Cumulative_Lines_Pct', 'Lines_Percentage']].copy()
            sku_data = sku_data.merge(fms_classification, on='sku_code', how='left')
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                fms_counts = sku_data['FMS'].value_counts()
                print(f"FMS Classification: F={fms_counts.get('F', 0)}, M={fms_counts.get('M', 0)}, S={fms_counts.get('S', 0)}")
            
            return sku_data
            
        except Exception as e:
            print(f"⚠️  Error performing FMS classification: {e}")
            return sku_data
    
    def _perform_cross_tabulation(self, sku_data):
        """
        Perform cross-tabulation analysis of ABC and FMS classifications.
        
        Args:
            sku_data (pandas.DataFrame): SKU data with both classifications
            
        Returns:
            dict: Cross-tabulation analysis results
        """
        try:
            cross_tab_results = {}
            
            # SKU count cross-tabulation
            sku_cross_tab = pd.crosstab(sku_data['ABC'], sku_data['FMS'], margins=True)
            cross_tab_results['sku_count_matrix'] = sku_cross_tab.to_dict()
            
            # Volume cross-tabulation
            volume_cross_tab = pd.crosstab(
                sku_data['ABC'], 
                sku_data['FMS'], 
                values=sku_data['Total_Volume'], 
                aggfunc='sum', 
                margins=True,
                fill_value=0
            )
            cross_tab_results['volume_matrix'] = volume_cross_tab.to_dict()
            
            # Order lines cross-tabulation
            lines_cross_tab = pd.crosstab(
                sku_data['ABC'], 
                sku_data['FMS'], 
                values=sku_data['Total_Order_Lines'], 
                aggfunc='sum', 
                margins=True,
                fill_value=0
            )
            cross_tab_results['lines_matrix'] = lines_cross_tab.to_dict()
            
            # Calculate percentages
            total_skus = len(sku_data)
            sku_percentage_matrix = (sku_cross_tab / total_skus * 100).round(2)
            cross_tab_results['sku_percentage_matrix'] = sku_percentage_matrix.to_dict()
            
            total_volume = sku_data['Total_Volume'].sum()
            volume_percentage_matrix = (volume_cross_tab / total_volume * 100).round(2)
            cross_tab_results['volume_percentage_matrix'] = volume_percentage_matrix.to_dict()
            
            # Strategic combinations analysis
            strategic_combinations = []
            for abc in ['A', 'B', 'C']:
                for fms in ['F', 'M', 'S']:
                    combo_data = sku_data[(sku_data['ABC'] == abc) & (sku_data['FMS'] == fms)]
                    if len(combo_data) > 0:
                        strategic_combinations.append({
                            'classification': f'{abc}-{fms}',
                            'sku_count': len(combo_data),
                            'volume_contribution': combo_data['Total_Volume'].sum(),
                            'lines_contribution': combo_data['Total_Order_Lines'].sum(),
                            'strategic_priority': self._get_strategic_priority(abc, fms)
                        })
            
            cross_tab_results['strategic_combinations'] = strategic_combinations
            
            return cross_tab_results
            
        except Exception as e:
            print(f"⚠️  Error performing cross-tabulation: {e}")
            return {}
    
    def _get_strategic_priority(self, abc, fms):
        """
        Determine strategic priority for ABC-FMS combination.
        
        Args:
            abc (str): ABC classification
            fms (str): FMS classification
            
        Returns:
            str: Strategic priority level
        """
        priority_matrix = {
            'A-F': 'Critical',
            'A-M': 'High',
            'A-S': 'Medium-High',
            'B-F': 'High',
            'B-M': 'Medium',
            'B-S': 'Medium-Low',
            'C-F': 'Medium',
            'C-M': 'Low',
            'C-S': 'Very Low'
        }
        return priority_matrix.get(f'{abc}-{fms}', 'Unknown')
    
    def _generate_classification_summary(self, sku_data):
        """
        Generate classification summary statistics.
        
        Args:
            sku_data (pandas.DataFrame): Classified SKU data
            
        Returns:
            dict: Classification summary
        """
        try:
            summary = {}
            
            # ABC summary
            abc_summary = sku_data.groupby('ABC').agg({
                'sku_code': 'count',
                'Total_Volume': 'sum',
                'Total_Order_Lines': 'sum',
                'Volume_Percentage': 'sum'
            }).reset_index()
            abc_summary.columns = ['ABC', 'SKU_Count', 'Total_Volume', 'Total_Lines', 'Volume_Percentage']
            summary['abc_summary'] = abc_summary.to_dict('records')
            
            # FMS summary
            fms_summary = sku_data.groupby('FMS').agg({
                'sku_code': 'count',
                'Total_Volume': 'sum',
                'Total_Order_Lines': 'sum',
                'Lines_Percentage': 'sum'
            }).reset_index()
            fms_summary.columns = ['FMS', 'SKU_Count', 'Total_Volume', 'Total_Lines', 'Lines_Percentage']
            summary['fms_summary'] = fms_summary.to_dict('records')
            
            # Overall statistics
            summary['overall_stats'] = {
                'total_skus': len(sku_data),
                'total_volume': sku_data['Total_Volume'].sum(),
                'total_order_lines': sku_data['Total_Order_Lines'].sum(),
                'a_class_skus': len(sku_data[sku_data['ABC'] == 'A']),
                'fast_moving_skus': len(sku_data[sku_data['FMS'] == 'F']),
                'a_fast_combination': len(sku_data[(sku_data['ABC'] == 'A') & (sku_data['FMS'] == 'F')])
            }
            
            return summary
            
        except Exception as e:
            print(f"⚠️  Error generating classification summary: {e}")
            return {}
    
    def _generate_strategic_insights(self, sku_data):
        """
        Generate strategic insights from ABC-FMS analysis.
        
        Args:
            sku_data (pandas.DataFrame): Classified SKU data
            
        Returns:
            dict: Strategic insights and recommendations
        """
        try:
            insights = {}
            
            # Critical SKUs (A-F combination)
            critical_skus = sku_data[(sku_data['ABC'] == 'A') & (sku_data['FMS'] == 'F')]
            insights['critical_skus'] = {
                'count': len(critical_skus),
                'volume_contribution': critical_skus['Total_Volume'].sum() if len(critical_skus) > 0 else 0,
                'recommendation': 'Place in golden zone - closest to shipping dock'
            }
            
            # Problem SKUs (C-F combination - low volume but high frequency)
            problem_skus = sku_data[(sku_data['ABC'] == 'C') & (sku_data['FMS'] == 'F')]
            insights['problem_skus'] = {
                'count': len(problem_skus),
                'recommendation': 'Investigate: Low volume but high frequency - possible data issues or small pack sizes'
            }
            
            # Opportunity SKUs (A-S combination - high volume but low frequency)
            opportunity_skus = sku_data[(sku_data['ABC'] == 'A') & (sku_data['FMS'] == 'S')]
            insights['opportunity_skus'] = {
                'count': len(opportunity_skus),
                'recommendation': 'Large order quantities - ensure adequate bulk storage and material handling capacity'
            }
            
            # Slotting recommendations
            slotting_priority = sku_data.copy()
            slotting_priority['Slotting_Score'] = 0
            
            # Assign slotting scores
            slotting_priority.loc[slotting_priority['ABC'] == 'A', 'Slotting_Score'] += 3
            slotting_priority.loc[slotting_priority['ABC'] == 'B', 'Slotting_Score'] += 2
            slotting_priority.loc[slotting_priority['ABC'] == 'C', 'Slotting_Score'] += 1
            
            slotting_priority.loc[slotting_priority['FMS'] == 'F', 'Slotting_Score'] += 3
            slotting_priority.loc[slotting_priority['FMS'] == 'M', 'Slotting_Score'] += 2
            slotting_priority.loc[slotting_priority['FMS'] == 'S', 'Slotting_Score'] += 1
            
            # Top slotting priorities
            top_priority_skus = slotting_priority.nlargest(10, 'Slotting_Score')[['sku_code', 'ABC', 'FMS', 'Slotting_Score']].to_dict('records')
            insights['top_slotting_priorities'] = top_priority_skus
            
            # Classification balance analysis
            abc_balance = sku_data['ABC'].value_counts(normalize=True) * 100
            fms_balance = sku_data['FMS'].value_counts(normalize=True) * 100
            
            insights['classification_balance'] = {
                'abc_distribution': abc_balance.to_dict(),
                'fms_distribution': fms_balance.to_dict(),
                'is_abc_balanced': abs(abc_balance.get('A', 0) - 20) < 10,  # Ideal A class ~20%
                'is_fms_balanced': abs(fms_balance.get('F', 0) - 30) < 15   # Ideal fast movers ~30%
            }
            
            return insights
            
        except Exception as e:
            print(f"⚠️  Error generating strategic insights: {e}")
            return {}
    
    def get_summary(self):
        """
        Get ABC-FMS summary for Excel export.
        
        Returns:
            pandas.DataFrame: ABC-FMS summary table
        """
        try:
            if not hasattr(self, 'classified_data') or self.classified_data.empty:
                return pd.DataFrame()
            
            # Create summary cross-tabulation
            summary_data = []
            
            for abc in ['A', 'B', 'C']:
                row_data = {'ABC': abc}
                
                abc_skus = self.classified_data[self.classified_data['ABC'] == abc]
                
                # Overall totals for this ABC class
                row_data['Total_SKUs'] = len(abc_skus)
                row_data['Total_Volume'] = abc_skus['Total_Volume'].sum()
                row_data['Total_Lines'] = abc_skus['Total_Order_Lines'].sum()
                
                # Break down by FMS
                for fms in ['F', 'M', 'S']:
                    combo_skus = abc_skus[abc_skus['FMS'] == fms]
                    row_data[f'SKUs_{fms}'] = len(combo_skus)
                    row_data[f'Volume_{fms}'] = combo_skus['Total_Volume'].sum()
                    row_data[f'Lines_{fms}'] = combo_skus['Total_Order_Lines'].sum()
                
                summary_data.append(row_data)
            
            # Add totals row
            total_row = {'ABC': 'Total'}
            total_row['Total_SKUs'] = self.classified_data['sku_code'].nunique()
            total_row['Total_Volume'] = self.classified_data['Total_Volume'].sum()
            total_row['Total_Lines'] = self.classified_data['Total_Order_Lines'].sum()
            
            for fms in ['F', 'M', 'S']:
                fms_skus = self.classified_data[self.classified_data['FMS'] == fms]
                total_row[f'SKUs_{fms}'] = len(fms_skus)
                total_row[f'Volume_{fms}'] = fms_skus['Total_Volume'].sum()
                total_row[f'Lines_{fms}'] = fms_skus['Total_Order_Lines'].sum()
            
            summary_data.append(total_row)
            
            summary_df = pd.DataFrame(summary_data)
            
            return summary_df
            
        except Exception as e:
            print(f"⚠️  Error creating ABC-FMS summary: {e}")
            return pd.DataFrame()
    
    def get_detailed_classification(self):
        """
        Get detailed ABC-FMS classification for Excel export.
        
        Returns:
            pandas.DataFrame: Detailed classification with all SKUs
        """
        try:
            if not hasattr(self, 'classified_data') or self.classified_data.empty:
                return pd.DataFrame()
            
            # Select and order columns for export
            export_columns = [
                'sku_code', 'ABC', 'FMS', 'Total_Volume', 'Total_Order_Lines',
                'Volume_Percentage', 'Lines_Percentage', 'Order_Days', 'Movement_Frequency'
            ]
            
            # Add additional columns if available
            available_columns = [col for col in export_columns if col in self.classified_data.columns]
            
            # Add category if available from SKU master
            if 'category' in self.classified_data.columns:
                available_columns.append('category')
            
            detailed_data = self.classified_data[available_columns].copy()
            
            # Rename columns for clarity
            column_mapping = {
                'sku_code': 'SKU_Code',
                'category': 'Category'
            }
            detailed_data = detailed_data.rename(columns=column_mapping)
            
            # Add strategic classification
            detailed_data['ABC_FMS_Combination'] = detailed_data['ABC'] + '-' + detailed_data['FMS']
            detailed_data['Strategic_Priority'] = detailed_data.apply(
                lambda row: self._get_strategic_priority(row['ABC'], row['FMS']), axis=1
            )
            
            # Sort by volume rank (A class first, then by volume within class)
            detailed_data = detailed_data.sort_values(['ABC', 'Total_Volume'], ascending=[True, False])
            
            return detailed_data
            
        except Exception as e:
            print(f"⚠️  Error creating detailed classification: {e}")
            return pd.DataFrame()

# Test function for standalone execution
if __name__ == "__main__":
    print("Testing ABCFMSAnalyzer...")
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    sample_order_data = pd.DataFrame({
        'date': np.random.choice(dates, 1000),
        'order_no': ['ORD' + str(i) for i in range(1000)],
        'shipment_no': ['SHP' + str(i//5) for i in range(1000)],
        'sku_code': ['SKU' + str(i%100) for i in range(1000)],
        'qty_cases': np.random.randint(1, 20, 1000),
        'qty_eaches': np.random.randint(0, 12, 1000),
        'total_case_equivalent': np.random.randint(1, 25, 1000)
    })
    
    analyzer = ABCFMSAnalyzer(sample_order_data)
    results = analyzer.run_analysis()
    
    print(f"✅ Analysis completed with {len(results)} result categories")
    
    # Test summary functions
    summary = analyzer.get_summary()
    print(f"✅ ABC-FMS summary: {len(summary)} classification groups")
    
    detailed = analyzer.get_detailed_classification()
    print(f"✅ Detailed classification: {len(detailed)} SKUs classified")
```

### 10. scripts/receipt_analysis.py
```python
"""
Receipt Analysis Module for Warehouse Analysis Tool V2

PURPOSE:
This module analyzes receiving patterns and capacity requirements:
- Daily receipt volumes and patterns
- Truck utilization and efficiency
- Receiving capacity planning
- Peak period identification for receiving
- Dock utilization analysis

FOR BEGINNERS:
- This module helps plan receiving operations
- It identifies busy receiving days and capacity needs
- Results help with dock scheduling and labor planning
- Truck utilization metrics help optimize delivery scheduling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

class ReceiptAnalyzer:
    """
    Receipt analysis class for receiving operations insights.
    
    This class analyzes receipt data to identify:
    - Daily receipt patterns
    - Truck utilization and efficiency
    - Receiving capacity requirements
    - Peak periods and seasonality
    - Dock utilization patterns
    """
    
    def __init__(self, receipt_data):
        """
        Initialize the ReceiptAnalyzer.
        
        Args:
            receipt_data (pandas.DataFrame): Cleaned receipt data from DataLoader
        """
        self.receipt_data = receipt_data.copy()
        self.analysis_results = {}
        
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            print(f"ReceiptAnalyzer initialized with {len(self.receipt_data)} receipt records")
    
    def run_analysis(self):
        """
        Run complete receipt analysis.
        
        Returns:
            dict: Comprehensive receipt analysis results
        """
        try:
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("Running receipt analysis...")
            
            # Clean and prepare receipt data
            self.receipt_data = self._clean_receipt_data()
            
            if self.receipt_data.empty:
                print("⚠️  No valid receipt data available for analysis")
                return {}
            
            # Basic receipt statistics
            self.analysis_results['basic_stats'] = self._calculate_basic_receipt_stats()
            
            # Daily receipt patterns
            self.analysis_results['daily_patterns'] = self._analyze_daily_patterns()
            
            # Truck utilization analysis
            self.analysis_results['truck_analysis'] = self._analyze_truck_utilization()
            
            # Capacity planning analysis
            self.analysis_results['capacity_planning'] = self._analyze_capacity_requirements()
            
            # Peak period analysis
            self.analysis_results['peak_analysis'] = self._identify_receiving_peaks()
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print("✅ Receipt analysis completed")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Error in receipt analysis: {e}")
            return {}
    
    def _clean_receipt_data(self):
        """
        Clean and standardize receipt data.
        
        Returns:
            pandas.DataFrame: Cleaned receipt data
        """
        try:
            cleaned_data = self.receipt_data.copy()
            
            # Standardize column names (assuming they match config.RECEIPT_COLUMNS)
            receipt_columns = config.RECEIPT_COLUMNS
            
            # Map to standard names if needed
            column_mapping = {}
            for config_col, excel_col in receipt_columns.items():
                if excel_col in cleaned_data.columns:
                    column_mapping[excel_col] = config_col
            
            cleaned_data = cleaned_data.rename(columns=column_mapping)
            
            # Convert date column to datetime
            if 'receipt_date' in cleaned_data.columns:
                cleaned_data['receipt_date'] = pd.to_datetime(cleaned_data['receipt_date'], errors='coerce')
                
                # Remove rows with invalid dates
                before_count = len(cleaned_data)
                cleaned_data = cleaned_data.dropna(subset=['receipt_date'])
                after_count = len(cleaned_data)
                
                if before_count != after_count:
                    print(f"⚠️  Removed {before_count - after_count} rows with invalid receipt dates")
            
            # Convert quantity columns to numeric
            for qty_col in ['qty_cases', 'qty_eaches']:
                if qty_col in cleaned_data.columns:
                    cleaned_data[qty_col] = pd.to_numeric(cleaned_data[qty_col], errors='coerce')
                    cleaned_data[qty_col] = cleaned_data[qty_col].fillna(0)
            
            # Calculate total case equivalent for receipts
            if 'qty_cases' in cleaned_data.columns and 'qty_eaches' in cleaned_data.columns:
                case_size = config.BUSINESS_RULES['CASES_TO_EACHES_DEFAULT']
                cleaned_data['total_case_equivalent'] = (
                    cleaned_data['qty_cases'] + 
                    cleaned_data['qty_eaches'] / case_size
                )
            
            # Remove rows with zero quantities
            if 'total_case_equivalent' in cleaned_data.columns:
                before_count = len(cleaned_data)
                cleaned_data = cleaned_data[cleaned_data['total_case_equivalent'] > 0]
                after_count = len(cleaned_data)
                
                if before_count != after_count:
                    print(f"⚠️  Removed {before_count - after_count} rows with zero receipt quantities")
            
            # Sort by date
            if 'receipt_date' in cleaned_data.columns:
                cleaned_data = cleaned_data.sort_values('receipt_date')
            
            # Reset index
            cleaned_data = cleaned_data.reset_index(drop=True)
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Cleaned receipt data shape: {cleaned_data.shape}")
            
            return cleaned_data
            
        except Exception as e:
            print(f"❌ Error cleaning receipt data: {e}")
            return self.receipt_data
    
    def _calculate_basic_receipt_stats(self):
        """
        Calculate basic receipt statistics.
        
        Returns:
            dict: Basic receipt statistics
        """
        try:
            stats = {}
            
            # Record counts
            stats['total_receipt_lines'] = len(self.receipt_data)
            stats['unique_receipt_dates'] = self.receipt_data['receipt_date'].nunique() if 'receipt_date' in self.receipt_data.columns else None
            stats['unique_trucks'] = self.receipt_data['truck_no'].nunique() if 'truck_no' in self.receipt_data.columns else None
            stats['unique_shipments'] = self.receipt_data['shipment_no'].nunique() if 'shipment_no' in self.receipt_data.columns else None
            stats['unique_skus'] = self.receipt_data['sku_id'].nunique() if 'sku_id' in self.receipt_data.columns else None
            
            # Volume statistics
            if 'total_case_equivalent' in self.receipt_data.columns:
                stats['total_received_volume'] = self.receipt_data['total_case_equivalent'].sum()
                stats['average_receipt_line_volume'] = self.receipt_data['total_case_equivalent'].mean()
                stats['max_receipt_line_volume'] = self.receipt_data['total_case_equivalent'].max()
            
            # Date range
            if 'receipt_date' in self.receipt_data.columns:
                stats['receipt_period_start'] = self.receipt_data['receipt_date'].min().strftime('%Y-%m-%d')
                stats['receipt_period_end'] = self.receipt_data['receipt_date'].max().strftime('%Y-%m-%d')
                stats['receipt_period_days'] = (self.receipt_data['receipt_date'].max() - self.receipt_data['receipt_date'].min()).days + 1
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Error calculating basic receipt statistics: {e}")
            return {}
    
    def _analyze_daily_patterns(self):
        """
        Analyze daily receipt patterns.
        
        Returns:
            dict: Daily receipt pattern analysis
        """
        try:
            if 'receipt_date' not in self.receipt_data.columns or 'total_case_equivalent' not in self.receipt_data.columns:
                return {}
            
            daily_patterns = {}
            
            # Daily receipt aggregation
            daily_receipts = self.receipt_data.groupby('receipt_date').agg({
                'total_case_equivalent': 'sum',
                'truck_no': 'nunique' if 'truck_no' in self.receipt_data.columns else lambda x: None,
                'shipment_no': 'nunique' if 'shipment_no' in self.receipt_data.columns else None,
                'sku_id': 'nunique' if 'sku_id' in self.receipt_data.columns else None
            }).reset_index()
            
            # Rename columns
            column_names = ['receipt_date', 'daily_volume', 'daily_trucks', 'daily_shipments', 'daily_skus']
            daily_receipts.columns = column_names[:len(daily_receipts.columns)]
            
            # Daily statistics
            daily_patterns['daily_volume_stats'] = {
                'average_daily_volume': daily_receipts['daily_volume'].mean(),
                'median_daily_volume': daily_receipts['daily_volume'].median(),
                'max_daily_volume': daily_receipts['daily_volume'].max(),
                'min_daily_volume': daily_receipts['daily_volume'].min(),
                'std_daily_volume': daily_receipts['daily_volume'].std()
            }
            
            # Day of week patterns
            daily_receipts['day_of_week'] = daily_receipts['receipt_date'].dt.day_name()
            dow_volume = daily_receipts.groupby('day_of_week')['daily_volume'].mean().to_dict()
            daily_patterns['day_of_week_patterns'] = dow_volume
            
            # Monthly patterns
            daily_receipts['month'] = daily_receipts['receipt_date'].dt.month_name()
            monthly_volume = daily_receipts.groupby('month')['daily_volume'].mean().to_dict()
            daily_patterns['monthly_patterns'] = monthly_volume
            
            return daily_patterns
            
        except Exception as e:
            print(f"⚠️  Error analyzing daily patterns: {e}")
            return {}
    
    def _analyze_truck_utilization(self):
        """
        Analyze truck utilization and efficiency.
        
        Returns:
            dict: Truck utilization analysis
        """
        try:
            if 'truck_no' not in self.receipt_data.columns:
                return {}
            
            truck_analysis = {}
            
            # Truck-level aggregation
            truck_utilization = self.receipt_data.groupby('truck_no').agg({
                'total_case_equivalent': 'sum',
                'receipt_date': 'nunique',
                'shipment_no': 'nunique' if 'shipment_no' in self.receipt_data.columns else lambda x: 1,
                'sku_id': 'nunique' if 'sku_id' in self.receipt_data.columns else lambda x: None
            }).reset_index()
            
            truck_utilization.columns = ['truck_no', 'total_volume', 'delivery_days', 'shipments_per_truck', 'unique_skus']
            
            # Truck efficiency metrics
            truck_analysis['truck_efficiency'] = {
                'average_volume_per_truck': truck_utilization['total_volume'].mean(),
                'max_volume_per_truck': truck_utilization['total_volume'].max(),
                'min_volume_per_truck': truck_utilization['total_volume'].min(),
                'average_shipments_per_truck': truck_utilization['shipments_per_truck'].mean()
            }
            
            # Utilization distribution
            target_volume = config.RECEIVING_THRESHOLDS.get('TARGET_CASES_PER_TRUCK', 1000)
            truck_utilization['utilization_percentage'] = (truck_utilization['total_volume'] / target_volume) * 100
            
            utilization_categories = {
                'over_utilized': len(truck_utilization[truck_utilization['utilization_percentage'] > 120]),
                'well_utilized': len(truck_utilization[(truck_utilization['utilization_percentage'] >= 80) & (truck_utilization['utilization_percentage'] <= 120)]),
                'under_utilized': len(truck_utilization[truck_utilization['utilization_percentage'] < 80])
            }
            
            truck_analysis['utilization_distribution'] = utilization_categories
            truck_analysis['average_truck_utilization'] = truck_utilization['utilization_percentage'].mean()
            
            return truck_analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing truck utilization: {e}")
            return {}
    
    def _analyze_capacity_requirements(self):
        """
        Analyze receiving capacity requirements.
        
        Returns:
            dict: Capacity planning analysis
        """
        try:
            if 'receipt_date' not in self.receipt_data.columns or 'total_case_equivalent' not in self.receipt_data.columns:
                return {}
            
            capacity_analysis = {}
            
            # Daily capacity requirements
            daily_receipts = self.receipt_data.groupby('receipt_date')['total_case_equivalent'].sum()
            
            # Calculate capacity percentiles
            percentiles = [50, 75, 90, 95, 99]
            capacity_percentiles = {}
            for p in percentiles:
                capacity_percentiles[f'p{p}'] = np.percentile(daily_receipts, p)
            
            capacity_analysis['capacity_percentiles'] = capacity_percentiles
            
            # Dock utilization analysis
            if 'truck_no' in self.receipt_data.columns:
                daily_trucks = self.receipt_data.groupby('receipt_date')['truck_no'].nunique()
                
                # Assume dock configuration from config
                max_docks = 10  # Default assumption - could be added to config
                dock_utilization = (daily_trucks / max_docks) * 100
                
                capacity_analysis['dock_utilization'] = {
                    'average_trucks_per_day': daily_trucks.mean(),
                    'max_trucks_per_day': daily_trucks.max(),
                    'average_dock_utilization': dock_utilization.mean(),
                    'max_dock_utilization': dock_utilization.max()
                }
            
            # Labor requirements estimation
            cases_per_hour_per_fte = config.RECEIVING_LABOR_STANDARDS.get('CASES_PER_HOUR_PER_FTE', 120)
            working_hours_per_day = 8  # Standard work day
            
            daily_fte_requirements = daily_receipts / (cases_per_hour_per_fte * working_hours_per_day)
            
            capacity_analysis['labor_requirements'] = {
                'average_daily_fte': daily_fte_requirements.mean(),
                'peak_daily_fte': daily_fte_requirements.max(),
                'p95_daily_fte': np.percentile(daily_fte_requirements, 95)
            }
            
            return capacity_analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing capacity requirements: {e}")
            return {}
    
    def _identify_receiving_peaks(self):
        """
        Identify peak receiving periods.
        
        Returns:
            dict: Peak period analysis
        """
        try:
            if 'receipt_date' not in self.receipt_data.columns or 'total_case_equivalent' not in self.receipt_data.columns:
                return {}
            
            peak_analysis = {}
            
            # Daily volume for peak analysis
            daily_volume = self.receipt_data.groupby('receipt_date')['total_case_equivalent'].sum().reset_index()
            daily_volume.columns = ['receipt_date', 'volume']
            
            # Calculate peak thresholds
            mean_volume = daily_volume['volume'].mean()
            std_volume = daily_volume['volume'].std()
            
            peak_threshold = mean_volume + std_volume
            high_peak_threshold = mean_volume + 2 * std_volume
            
            # Identify peak days
            peak_days = daily_volume[daily_volume['volume'] >= peak_threshold]
            high_peak_days = daily_volume[daily_volume['volume'] >= high_peak_threshold]
            
            peak_analysis['peak_days_analysis'] = {
                'total_days_analyzed': len(daily_volume),
                'peak_days_count': len(peak_days),
                'high_peak_days_count': len(high_peak_days),
                'peak_days_percentage': len(peak_days) / len(daily_volume) * 100,
                'peak_threshold_volume': peak_threshold,
                'high_peak_threshold_volume': high_peak_threshold
            }
            
            # Peak day details
            if len(peak_days) > 0:
                peak_details = peak_days.sort_values('volume', ascending=False).head(10)
                peak_analysis['top_peak_days'] = peak_details.to_dict('records')
            
            # Seasonal analysis (if enough data)
            if len(daily_volume) > 60:  # At least 2 months of data
                daily_volume['month'] = daily_volume['receipt_date'].dt.month_name()
                monthly_avg = daily_volume.groupby('month')['volume'].mean()
                peak_analysis['seasonal_patterns'] = monthly_avg.sort_values(ascending=False).to_dict()
            
            return peak_analysis
            
        except Exception as e:
            print(f"⚠️  Error identifying receiving peaks: {e}")
            return {}
    
    def get_receipt_summary(self):
        """
        Get daily receipt summary for Excel export.
        
        Returns:
            pandas.DataFrame: Daily receipt summary
        """
        try:
            if 'receipt_date' not in self.receipt_data.columns:
                return pd.DataFrame()
            
            # Group by date and calculate metrics
            receipt_summary = self.receipt_data.groupby('receipt_date').agg({
                'total_case_equivalent': 'sum',
                'truck_no': 'nunique' if 'truck_no' in self.receipt_data.columns else lambda x: None,
                'shipment_no': 'nunique' if 'shipment_no' in self.receipt_data.columns else None,
                'sku_id': 'nunique' if 'sku_id' in self.receipt_data.columns else None
            }).reset_index()
            
            # Rename columns
            receipt_summary.columns = [
                'Receipt_Date',
                'Total_Cases_Received',
                'Trucks_Received',
                'Shipments_Received',
                'SKUs_Received'
            ]
            
            # Add calculated fields
            receipt_summary['Day_of_Week'] = pd.to_datetime(receipt_summary['Receipt_Date']).dt.day_name()
            receipt_summary['Month'] = pd.to_datetime(receipt_summary['Receipt_Date']).dt.month_name()
            
            # Add performance metrics
            if 'Trucks_Received' in receipt_summary.columns:
                receipt_summary['Cases_Per_Truck'] = receipt_summary['Total_Cases_Received'] / receipt_summary['Trucks_Received']
                receipt_summary['Cases_Per_Truck'] = receipt_summary['Cases_Per_Truck'].fillna(0)
            
            # Add cumulative metrics
            receipt_summary['Cumulative_Cases'] = receipt_summary['Total_Cases_Received'].cumsum()
            
            # Sort by date
            receipt_summary = receipt_summary.sort_values('Receipt_Date')
            
            return receipt_summary
            
        except Exception as e:
            print(f"⚠️  Error creating receipt summary: {e}")
            return pd.DataFrame()

# Test function for standalone execution
if __name__ == "__main__":
    print("Testing ReceiptAnalyzer...")
    
    # Create sample receipt data for testing
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    sample_receipt_data = pd.DataFrame({
        'receipt_date': np.random.choice(dates, 500),
        'sku_id': ['SKU' + str(i%50) for i in range(500)],
        'shipment_no': ['SHP' + str(i//10) for i in range(500)],
        'truck_no': ['TRK' + str(i//20) for i in range(500)],
        'qty_cases': np.random.randint(10, 100, 500),
        'qty_eaches': np.random.randint(0, 50, 500)
    })
    
    analyzer = ReceiptAnalyzer(sample_receipt_data)
    results = analyzer.run_analysis()
    
    print(f"✅ Analysis completed with {len(results)} result categories")
    
    # Test summary function
    receipt_summary = analyzer.get_receipt_summary()
    print(f"✅ Receipt summary: {len(receipt_summary)} days analyzed")
```

### 11. scripts/excel_generator.py
```python
"""
Excel Report Generator for Warehouse Analysis Tool V2

PURPOSE:
This module creates comprehensive Excel reports with multiple sheets:
- Executive summary with key metrics
- Detailed analysis results from all modules
- Professional formatting and charts
- Raw data preservation for reference

FOR BEGINNERS:
- This module takes all analysis results and creates a single Excel file
- Each analysis gets its own sheet in the workbook
- Professional formatting makes the reports presentation-ready
- Charts and conditional formatting highlight key insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule
from openpyxl.utils.dataframe import dataframe_to_rows

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

class ExcelGenerator:
    """
    Excel report generation class.
    
    This class creates comprehensive Excel reports containing:
    - Executive summary sheet
    - Individual analysis sheets
    - Raw data sheets
    - Professional formatting
    - Charts and conditional formatting
    """
    
    def __init__(self):
        """Initialize the ExcelGenerator."""
        self.output_dir = Path(config.OUTPUT_SETTINGS['OUTPUT_DIR'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
            print(f"ExcelGenerator initialized, output directory: {self.output_dir}")
    
    def create_comprehensive_report(self, analysis_results, data_loader, filename=None):
        """
        Create comprehensive Excel report with all analysis results.
        
        Args:
            analysis_results (dict): Results from all analysis modules
            data_loader (DataLoader): Data loader instance with original data
            filename (str): Output filename (optional)
            
        Returns:
            str: Path to generated Excel file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Warehouse_Analysis_Report_{timestamp}.xlsx"
            
            output_path = self.output_dir / filename
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"Creating Excel report: {output_path}")
            
            # Create workbook
            workbook = Workbook()
            
            # Remove default sheet
            workbook.remove(workbook.active)
            
            # Create executive summary sheet
            self._create_executive_summary(workbook, analysis_results)
            
            # Create analysis sheets
            if 'date_summary' in analysis_results:
                self._create_date_analysis_sheet(workbook, analysis_results['date_summary'])
            
            if 'sku_profile' in analysis_results:
                self._create_sku_analysis_sheet(workbook, analysis_results['sku_profile'])
            
            if 'abc_fms_summary' in analysis_results:
                self._create_abc_fms_summary_sheet(workbook, analysis_results['abc_fms_summary'])
            
            if 'abc_fms_detail' in analysis_results:
                self._create_abc_fms_detail_sheet(workbook, analysis_results['abc_fms_detail'])
            
            if 'percentiles' in analysis_results:
                self._create_percentiles_sheet(workbook, analysis_results['percentiles'])
            
            if 'receipt_analysis' in analysis_results:
                self._create_receipt_analysis_sheet(workbook, analysis_results['receipt_analysis'])
            
            # Add raw data sheets if requested
            if config.OUTPUT_SETTINGS['INCLUDE_RAW_DATA']:
                self._add_raw_data_sheets(workbook, data_loader)
            
            # Save workbook
            workbook.save(output_path)
            
            if config.DEBUG_SETTINGS['VERBOSE_OUTPUT']:
                print(f"✅ Excel report saved: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error creating Excel report: {e}")
            return None
    
    def _create_executive_summary(self, workbook, analysis_results):
        """
        Create executive summary sheet.
        
        Args:
            workbook: Excel workbook object
            analysis_results (dict): Analysis results
        """
        try:
            sheet = workbook.create_sheet(config.OUTPUT_SHEETS['SUMMARY'])
            
            # Header
            sheet['A1'] = 'Warehouse Analysis Executive Summary'
            sheet['A1'].font = Font(size=16, bold=True)
            sheet['A2'] = f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            
            row = 4
            
            # Overall statistics
            if 'order_analysis' in analysis_results:
                order_stats = analysis_results['order_analysis'].get('basic_stats', {})
                
                sheet[f'A{row}'] = 'OVERALL STATISTICS'
                sheet[f'A{row}'].font = Font(bold=True, size=12)
                row += 1
                
                stats_data = [
                    ['Total Order Records', order_stats.get('total_records', 'N/A')],
                    ['Unique SKUs', order_stats.get('unique_skus', 'N/A')],
                    ['Analysis Period (Days)', order_stats.get('unique_dates', 'N/A')],
                    ['Total Volume (Cases)', order_stats.get('total_volume', 'N/A')],
                    ['Average Daily Volume', f"{order_stats.get('total_volume', 0) / max(order_stats.get('unique_dates', 1), 1):.0f}" if order_stats.get('total_volume') else 'N/A']
                ]
                
                for stat_name, stat_value in stats_data:
                    sheet[f'A{row}'] = stat_name
                    sheet[f'B{row}'] = stat_value
                    row += 1
                
                row += 1
            
            # ABC-FMS Summary
            if 'abc_fms_analysis' in analysis_results:
                abc_summary = analysis_results['abc_fms_analysis'].get('classification_summary', {})
                
                sheet[f'A{row}'] = 'ABC-FMS CLASSIFICATION SUMMARY'
                sheet[f'A{row}'].font = Font(bold=True, size=12)
                row += 1
                
                # ABC breakdown
                if 'abc_summary' in abc_summary:
                    sheet[f'A{row}'] = 'ABC Classification:'
                    sheet[f'A{row}'].font = Font(bold=True)
                    row += 1
                    
                    for abc_data in abc_summary['abc_summary']:
                        sheet[f'A{row}'] = f"Class {abc_data['ABC']}"
                        sheet[f'B{row}'] = f"{abc_data['SKU_Count']} SKUs"
                        sheet[f'C{row}'] = f"{abc_data.get('Volume_Percentage', 0):.1f}% of volume"
                        row += 1
                    row += 1
                
                # FMS breakdown
                if 'fms_summary' in abc_summary:
                    sheet[f'A{row}'] = 'FMS Classification:'
                    sheet[f'A{row}'].font = Font(bold=True)
                    row += 1
                    
                    for fms_data in abc_summary['fms_summary']:
                        sheet[f'A{row}'] = f"Class {fms_data['FMS']}"
                        sheet[f'B{row}'] = f"{fms_data['SKU_Count']} SKUs"
                        sheet[f'C{row}'] = f"{fms_data.get('Lines_Percentage', 0):.1f}% of order lines"
                        row += 1
                    row += 1
            
            # Capacity Planning Summary
            if 'percentiles' in analysis_results and not analysis_results['percentiles'].empty:
                sheet[f'A{row}'] = 'CAPACITY PLANNING HIGHLIGHTS'
                sheet[f'A{row}'].font = Font(bold=True, size=12)
                row += 1
                
                percentiles_df = analysis_results['percentiles']
                
                # Key percentiles
                key_percentiles = ['95th', 'Average', 'Maximum']
                for percentile in key_percentiles:
                    percentile_row = percentiles_df[percentiles_df['Percentile'] == percentile]
                    if not percentile_row.empty:
                        volume = percentile_row.iloc[0]['Volume']
                        sheet[f'A{row}'] = f'{percentile} Daily Volume'
                        sheet[f'B{row}'] = f'{volume:.0f} cases'
                        row += 1
                
                row += 1
            
            # Receipt Analysis Summary (if available)
            if 'receipt_analysis' in analysis_results:
                receipt_stats = analysis_results['receipt_analysis'].get('basic_stats', {})
                
                sheet[f'A{row}'] = 'RECEIVING OPERATIONS SUMMARY'
                sheet[f'A{row}'].font = Font(bold=True, size=12)
                row += 1
                
                receipt_data = [
                    ['Total Receipt Records', receipt_stats.get('total_receipt_lines', 'N/A')],
                    ['Unique Receipt Days', receipt_stats.get('unique_receipt_dates', 'N/A')],
                    ['Total Trucks', receipt_stats.get('unique_trucks', 'N/A')],
                    ['Total Received Volume', receipt_stats.get('total_received_volume', 'N/A')]
                ]
                
                for stat_name, stat_value in receipt_data:
                    sheet[f'A{row}'] = stat_name
                    sheet[f'B{row}'] = stat_value
                    row += 1
            
            # Apply formatting
            self._apply_header_formatting(sheet)
            
        except Exception as e:
            print(f"⚠️  Error creating executive summary: {e}")
    
    def _create_date_analysis_sheet(self, workbook, date_summary):
        """
        Create date analysis sheet.
        
        Args:
            workbook: Excel workbook object
            date_summary (pandas.DataFrame): Date analysis results
        """
        try:
            if date_summary.empty:
                return
            
            sheet = workbook.create_sheet(config.OUTPUT_SHEETS['DATE_ANALYSIS'])
            
            # Add header
            sheet['A1'] = 'Daily Order Analysis'
            sheet['A1'].font = Font(size=14, bold=True)
            sheet['A2'] = f'Analysis covers {len(date_summary)} days'
            
            # Write data starting from row 4
            start_row = 4
            
            # Write column headers
            for col_idx, column in enumerate(date_summary.columns, 1):
                cell = sheet.cell(row=start_row, column=col_idx, value=column)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="E6E6FA", end_color="E6E6FA", fill_type="solid")
            
            # Write data
            for row_idx, (_, row) in enumerate(date_summary.iterrows(), start_row + 1):
                for col_idx, value in enumerate(row, 1):
                    sheet.cell(row=row_idx, column=col_idx, value=value)
            
            # Apply formatting
            self._apply_data_formatting(sheet, len(date_summary) + start_row, len(date_summary.columns))
            
        except Exception as e:
            print(f"⚠️  Error creating date analysis sheet: {e}")
    
    def _create_sku_analysis_sheet(self, workbook, sku_profile):
        """
        Create SKU analysis sheet.
        
        Args:
            workbook: Excel workbook object
            sku_profile (pandas.DataFrame): SKU analysis results
        """
        try:
            if sku_profile.empty:
                return
            
            sheet = workbook.create_sheet(config.OUTPUT_SHEETS['SKU_PROFILE'])
            
            # Add header
            sheet['A1'] = 'SKU Profile & Classification'
            sheet['A1'].font = Font(size=14, bold=True)
            sheet['A2'] = f'Analysis covers {len(sku_profile)} SKUs'
            
            # Write data starting from row 4
            start_row = 4
            
            # Write column headers
            for col_idx, column in enumerate(sku_profile.columns, 1):
                cell = sheet.cell(row=start_row, column=col_idx, value=column)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="E6E6FA", end_color="E6E6FA", fill_type="solid")
            
            # Write data
            for row_idx, (_, row) in enumerate(sku_profile.iterrows(), start_row + 1):
                for col_idx, value in enumerate(row, 1):
                    sheet.cell(row=row_idx, column=col_idx, value=value)
            
            # Apply conditional formatting for volume rankings
            self._apply_sku_conditional_formatting(sheet, len(sku_profile) + start_row, sku_profile.columns)
            
        except Exception as e:
            print(f"⚠️  Error creating SKU analysis sheet: {e}")
    
    def _create_abc_fms_summary_sheet(self, workbook, abc_fms_summary):
        """
        Create ABC-FMS summary sheet.
        
        Args:
            workbook: Excel workbook object
            abc_fms_summary (pandas.DataFrame): ABC-FMS summary results
        """
        try:
            if abc_fms_summary.empty:
                return
            
            sheet = workbook.create_sheet(config.OUTPUT_SHEETS['ABC_FMS_SUMMARY'])
            
            # Add header
            sheet['A1'] = 'ABC-FMS Cross-Tabulation Summary'
            sheet['A1'].font = Font(size=14, bold=True)
            
            # Write data starting from row 4
            start_row = 4
            
            # Write column headers
            for col_idx, column in enumerate(abc_fms_summary.columns, 1):
                cell = sheet.cell(row=start_row, column=col_idx, value=column)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="E6E6FA", end_color="E6E6FA", fill_type="solid")
            
            # Write data
            for row_idx, (_, row) in enumerate(abc_fms_summary.iterrows(), start_row + 1):
                for col_idx, value in enumerate(row, 1):
                    sheet.cell(row=row_idx, column=col_idx, value=value)
            
            # Apply formatting
            self._apply_data_formatting(sheet, len(abc_fms_summary) + start_row, len(abc_fms_summary.columns))
            
        except Exception as e:
            print(f"⚠️  Error creating ABC-FMS summary sheet: {e}")
    
    def _create_abc_fms_detail_sheet(self, workbook, abc_fms_detail):
        """
        Create detailed ABC-FMS classification sheet.
        
        Args:
            workbook: Excel workbook object
            abc_fms_detail (pandas.DataFrame): Detailed ABC-FMS results
        """
        try:
            if abc_fms_detail.empty:
                return
            
            sheet = workbook.create_sheet(config.OUTPUT_SHEETS['ABC_FMS_DETAIL'])
            
            # Add header
            sheet['A1'] = 'Detailed ABC-FMS Classification'
            sheet['A1'].font = Font(size=14, bold=True)
            sheet['A2'] = f'Individual SKU classifications for {len(abc_fms_detail)} SKUs'
            
            # Write data starting from row 4
            start_row = 4
            
            # Write column headers
            for col_idx, column in enumerate(abc_fms_detail.columns, 1):
                cell = sheet.cell(row=start_row, column=col_idx, value=column)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="E6E6FA", end_color="E6E6FA", fill_type="solid")
            
            # Write data
            for row_idx, (_, row) in enumerate(abc_fms_detail.iterrows(), start_row + 1):
                for col_idx, value in enumerate(row, 1):
                    sheet.cell(row=row_idx, column=col_idx, value=value)
            
            # Apply ABC-FMS conditional formatting
            self._apply_abc_fms_conditional_formatting(sheet, len(abc_fms_detail) + start_row, abc_fms_detail.columns)
            
        except Exception as e:
            print(f"⚠️  Error creating ABC-FMS detail sheet: {e}")
    
    def _create_percentiles_sheet(self, workbook, percentiles):
        """
        Create percentiles analysis sheet.
        
        Args:
            workbook: Excel workbook object
            percentiles (pandas.DataFrame): Percentiles analysis results
        """
        try:
            if percentiles.empty:
                return
            
            sheet = workbook.create_sheet(config.OUTPUT_SHEETS['PERCENTILES'])
            
            # Add header
            sheet['A1'] = 'Volume Percentiles for Capacity Planning'
            sheet['A1'].font = Font(size=14, bold=True)
            sheet['A2'] = 'Use these percentiles to plan warehouse capacity and staffing'
            
            # Write data starting from row 4
            start_row = 4
            
            # Write column headers
            for col_idx, column in enumerate(percentiles.columns, 1):
                cell = sheet.cell(row=start_row, column=col_idx, value=column)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="E6E6FA", end_color="E6E6FA", fill_type="solid")
            
            # Write data
            for row_idx, (_, row) in enumerate(percentiles.iterrows(), start_row + 1):
                for col_idx, value in enumerate(row, 1):
                    sheet.cell(row=row_idx, column=col_idx, value=value)
            
            # Apply formatting
            self._apply_data_formatting(sheet, len(percentiles) + start_row, len(percentiles.columns))
            
        except Exception as e:
            print(f"⚠️  Error creating percentiles sheet: {e}")
    
    def _create_receipt_analysis_sheet(self, workbook, receipt_analysis):
        """
        Create receipt analysis sheet.
        
        Args:
            workbook: Excel workbook object
            receipt_analysis (dict): Receipt analysis results
        """
        try:
            sheet = workbook.create_sheet(config.OUTPUT_SHEETS['RECEIPT_ANALYSIS'])
            
            # Add header
            sheet['A1'] = 'Receipt Analysis Summary'
            sheet['A1'].font = Font(size=14, bold=True)
            sheet['A2'] = 'Receiving operations analysis and capacity planning'
            
            row = 4
            
            # Basic statistics
            if 'basic_stats' in receipt_analysis:
                basic_stats = receipt_analysis['basic_stats']
                
                sheet[f'A{row}'] = 'BASIC RECEIPT STATISTICS'
                sheet[f'A{row}'].font = Font(bold=True, size=12)
                row += 1
                
                stats_data = [
                    ['Total Receipt Lines', basic_stats.get('total_receipt_lines', 'N/A')],
                    ['Unique Receipt Days', basic_stats.get('unique_receipt_dates', 'N/A')],
                    ['Unique Trucks', basic_stats.get('unique_trucks', 'N/A')],
                    ['Total Received Volume', basic_stats.get('total_received_volume', 'N/A')]
                ]
                
                for stat_name, stat_value in stats_data:
                    sheet[f'A{row}'] = stat_name
                    sheet[f'B{row}'] = stat_value
                    row += 1
                
                row += 2
            
            # Capacity analysis
            if 'capacity_planning' in receipt_analysis:
                capacity_data = receipt_analysis['capacity_planning']
                
                sheet[f'A{row}'] = 'CAPACITY PLANNING'
                sheet[f'A{row}'].font = Font(bold=True, size=12)
                row += 1
                
                if 'capacity_percentiles' in capacity_data:
                    for percentile, value in capacity_data['capacity_percentiles'].items():
                        sheet[f'A{row}'] = f'{percentile.upper()} Daily Volume'
                        sheet[f'B{row}'] = f'{value:.0f} cases'
                        row += 1
                
                row += 1
                
                if 'labor_requirements' in capacity_data:
                    labor_data = capacity_data['labor_requirements']
                    labor_stats = [
                        ['Average Daily FTE', f"{labor_data.get('average_daily_fte', 0):.1f}"],
                        ['Peak Daily FTE', f"{labor_data.get('peak_daily_fte', 0):.1f}"],
                        ['95th Percentile FTE', f"{labor_data.get('p95_daily_fte', 0):.1f}"]
                    ]
                    
                    for stat_name, stat_value in labor_stats:
                        sheet[f'A{row}'] = stat_name
                        sheet[f'B{row}'] = stat_value
                        row += 1
            
            # Apply formatting
            self._apply_header_formatting(sheet)
            
        except Exception as e:
            print(f"⚠️  Error creating receipt analysis sheet: {e}")
    
    def _add_raw_data_sheets(self, workbook, data_loader):
        """
        Add raw data sheets to workbook.
        
        Args:
            workbook: Excel workbook object
            data_loader (DataLoader): Data loader with original data
        """
        try:
            # Add order data
            if data_loader.order_data is not None and not data_loader.order_data.empty:
                sheet = workbook.create_sheet(config.OUTPUT_SHEETS['RAW_ORDER_DATA'])
                self._write_dataframe_to_sheet(sheet, data_loader.order_data, 'Raw Order Data')
            
            # Add SKU master data
            if data_loader.sku_master is not None and not data_loader.sku_master.empty:
                sheet = workbook.create_sheet(config.OUTPUT_SHEETS['RAW_SKU_DATA'])
                self._write_dataframe_to_sheet(sheet, data_loader.sku_master, 'Raw SKU Master Data')
            
        except Exception as e:
            print(f"⚠️  Error adding raw data sheets: {e}")
    
    def _write_dataframe_to_sheet(self, sheet, dataframe, title):
        """
        Write DataFrame to Excel sheet with formatting.
        
        Args:
            sheet: Excel sheet object
            dataframe (pandas.DataFrame): Data to write
            title (str): Sheet title
        """
        try:
            # Add title
            sheet['A1'] = title
            sheet['A1'].font = Font(size=14, bold=True)
            sheet['A2'] = f'Contains {len(dataframe)} records'
            
            # Write data starting from row 4
            start_row = 4
            
            # Write column headers
            for col_idx, column in enumerate(dataframe.columns, 1):
                cell = sheet.cell(row=start_row, column=col_idx, value=str(column))
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="E6E6FA", end_color="E6E6FA", fill_type="solid")
            
            # Write data (limit to max rows to prevent Excel issues)
            max_rows = min(len(dataframe), config.OUTPUT_SETTINGS['MAX_ROWS_PER_SHEET'])
            
            for row_idx in range(max_rows):
                for col_idx, value in enumerate(dataframe.iloc[row_idx], 1):
                    sheet.cell(row=start_row + row_idx + 1, column=col_idx, value=value)
            
            # Apply basic formatting
            self._apply_data_formatting(sheet, start_row + max_rows, len(dataframe.columns))
            
        except Exception as e:
            print(f"⚠️  Error writing DataFrame to sheet: {e}")
    
    def _apply_header_formatting(self, sheet):
        """Apply formatting to sheet headers."""
        try:
            # Auto-adjust column widths
            for column in sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                sheet.column_dimensions[column_letter].width = adjusted_width
                
        except Exception as e:
            print(f"⚠️  Error applying header formatting: {e}")
    
    def _apply_data_formatting(self, sheet, max_row, max_col):
        """Apply data formatting to sheet."""
        try:
            # Add borders
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            for row in sheet.iter_rows(min_row=4, max_row=max_row, min_col=1, max_col=max_col):
                for cell in row:
                    cell.border = thin_border
            
            # Auto-adjust column widths
            for column in sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                sheet.column_dimensions[column_letter].width = adjusted_width
                
        except Exception as e:
            print(f"⚠️  Error applying data formatting: {e}")
    
    def _apply_sku_conditional_formatting(self, sheet, max_row, columns):
        """Apply conditional formatting for SKU analysis."""
        try:
            # Find volume percentage column
            volume_col = None
            for idx, col in enumerate(columns, 1):
                if 'volume_percentage' in str(col).lower():
                    volume_col = idx
                    break
            
            if volume_col:
                # Apply color scale for volume percentage
                volume_range = f"{chr(64 + volume_col)}5:{chr(64 + volume_col)}{max_row}"
                rule = ColorScaleRule(
                    start_type='min', start_color='FFFFFF',
                    mid_type='percentile', mid_value=50, mid_color='FFFF99',
                    end_type='max', end_color='FF6B6B'
                )
                sheet.conditional_formatting.add(volume_range, rule)
                
        except Exception as e:
            print(f"⚠️  Error applying SKU conditional formatting: {e}")
    
    def _apply_abc_fms_conditional_formatting(self, sheet, max_row, columns):
        """Apply conditional formatting for ABC-FMS classification."""
        try:
            # Find ABC and FMS columns
            abc_col = None
            fms_col = None
            
            for idx, col in enumerate(columns, 1):
                if str(col).upper() == 'ABC':
                    abc_col = idx
                elif str(col).upper() == 'FMS':
                    fms_col = idx
            
            # Apply ABC classification formatting
            if abc_col:
                abc_range = f"{chr(64 + abc_col)}5:{chr(64 + abc_col)}{max_row}"
                
                # A class - green
                rule_a = CellIsRule(operator='equal', formula=['"A"'], fill=PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid'))
                sheet.conditional_formatting.add(abc_range, rule_a)
                
                # B class - yellow
                rule_b = CellIsRule(operator='equal', formula=['"B"'], fill=PatternFill(start_color='FFFF99', end_color='FFFF99', fill_type='solid'))
                sheet.conditional_formatting.add(abc_range, rule_b)
                
                # C class - red
                rule_c = CellIsRule(operator='equal', formula=['"C"'], fill=PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid'))
                sheet.conditional_formatting.add(abc_range, rule_c)
            
            # Apply FMS classification formatting
            if fms_col:
                fms_range = f"{chr(64 + fms_col)}5:{chr(64 + fms_col)}{max_row}"
                
                # F class - blue
                rule_f = CellIsRule(operator='equal', formula=['"F"'], fill=PatternFill(start_color='ADD8E6', end_color='ADD8E6', fill_type='solid'))
                sheet.conditional_formatting.add(fms_range, rule_f)
                
                # M class - orange
                rule_m = CellIsRule(operator='equal', formula=['"M"'], fill=PatternFill(start_color='FFA500', end_color='FFA500', fill_type='solid'))
                sheet.conditional_formatting.add(fms_range, rule_m)
                
                # S class - gray
                rule_s = CellIsRule(operator='equal', formula=['"S"'], fill=PatternFill(start_color='D3D3D3', end_color='D3D3D3', fill_type='solid'))
                sheet.conditional_formatting.add(fms_range, rule_s)
                
        except Exception as e:
            print(f"⚠️  Error applying ABC-FMS conditional formatting: {e}")

# Test function for standalone execution
if __name__ == "__main__":
    print("Testing ExcelGenerator...")
    
    # Create sample data for testing
    sample_results = {
        'order_analysis': {
            'basic_stats': {
                'total_records': 1000,
                'unique_skus': 100,
                'unique_dates': 30,
                'total_volume': 50000
            }
        },
        'date_summary': pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30),
            'Total_Case_Equiv': np.random.normal(1500, 300, 30),
            'Distinct_Orders': np.random.poisson(50, 30)
        }),
        'percentiles': pd.DataFrame({
            'Percentile': ['95th', '90th', 'Average'],
            'Volume': [2000, 1800, 1500],
            'Description': ['95% of days ≤ 2000', '90% of days ≤ 1800', 'Average daily volume']
        })
    }
    
    # Mock data loader
    class MockDataLoader:
        def __init__(self):
            self.order_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100),
                'sku_code': ['SKU' + str(i%20) for i in range(100)],
                'total_case_equivalent': np.random.randint(1, 50, 100)
            })
            self.sku_master = None
    
    generator = ExcelGenerator()
    output_path = generator.create_comprehensive_report(sample_results, MockDataLoader())
    
    if output_path:
        print(f"✅ Test Excel report generated: {output_path}")
    else:
        print("❌ Failed to generate test report")
```

## Configuration Documentation

### Data File Configuration

The tool expects Excel files with specific sheet structure:

1. **OrderData Sheet** (Required)
   - Date column (YYYY-MM-DD format)
   - SKU/Product codes
   - Quantities (cases and/or eaches)
   - Order/Shipment identifiers

2. **SkuMaster Sheet** (Optional but recommended)
   - SKU codes matching order data
   - Product categories
   - Case configurations
   - Pallet fit information

3. **ReceiptData Sheet** (Optional)
   - Receipt dates
   - SKU codes
   - Received quantities
   - Truck/shipment information

4. **InventoryData Sheet** (Optional)
   - Snapshot dates
   - SKU codes
   - Inventory quantities
   - Location information

### Analysis Parameters

1. **ABC Classification**
   - Based on volume contribution
   - A items: Top 70% of volume
   - B items: Next 20% of volume (70-90%)
   - C items: Bottom 10% of volume (90-100%)

2. **FMS Classification**
   - Based on order frequency
   - Fast: Top 70% of order frequency
   - Medium: Next 20% (70-90%)
   - Slow: Bottom 10% (90-100%)

3. **Percentile Analysis**
   - Configurable percentile levels
   - Default: 95th, 90th, 85th, 80th, 75th
   - Used for capacity planning

## Usage Instructions

### First-Time Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Data File**
   - Edit `config.py`
   - Update `DATA_FILE_PATH` to your Excel file
   - Adjust column names to match your data
   - Set analysis parameters if needed

3. **Prepare Data Directory**
   ```bash
   mkdir -p data outputs/excel_reports
   ```

4. **Place Your Excel File**
   - Copy your warehouse data Excel file to the data directory
   - Ensure sheet names match configuration

### Running Analysis

1. **Basic Execution**
   ```bash
   python run_analysis.py
   ```

2. **The tool will automatically:**
   - Validate configuration
   - Load and validate data
   - Run all analysis modules
   - Generate timestamped Excel report

3. **Monitor Progress**
   - Watch console output for progress updates
   - Check for any warnings or errors
   - Review validation results

### Output Files

The tool generates a comprehensive Excel report with multiple sheets:

1. **Executive Summary** - Key metrics and insights
2. **Daily Order Analysis** - Date-wise trends and patterns
3. **SKU Profile & Classification** - Product-level analysis
4. **ABC-FMS Analysis** - Classification matrices
5. **ABC-FMS Detailed** - Item-level classifications
6. **Volume Percentiles** - Capacity planning data
7. **Receipt Analysis** - Receiving patterns (if data available)
8. **Raw Data Sheets** - Original data for reference

## Development Guidelines

### Code Structure

1. **Modular Design**
   - Each analysis in separate module
   - Clear interfaces between modules
   - Minimal dependencies between components

2. **Error Handling**
   - Graceful handling of missing data
   - Clear error messages for users
   - Detailed logging for debugging

3. **Documentation**
   - Comprehensive docstrings
   - Inline comments for complex logic
   - User-friendly error messages

### Adding New Analysis Modules

1. **Create New Module**
   ```python
   class NewAnalyzer:
       def __init__(self, data):
           self.data = data
       
       def run_analysis(self):
           # Implementation
           pass
       
       def get_results(self):
           # Return formatted results
           pass
   ```

2. **Update Main Script**
   - Import new analyzer
   - Add to analysis pipeline
   - Include in Excel generation

3. **Update Configuration**
   - Add any new parameters
   - Update output sheet definitions
   - Document new features

### Testing Guidelines

1. **Data Validation Testing**
   - Test with various data formats
   - Validate error handling
   - Check edge cases

2. **Analysis Testing**
   - Verify calculations
   - Test with known datasets
   - Compare with manual calculations

3. **Output Testing**
   - Verify Excel formatting
   - Check all sheets generated
   - Validate formulas and charts

## Testing & Validation

### Data Quality Checks

The tool performs comprehensive data validation:

1. **File and Sheet Validation**
   - Excel file exists and is readable
   - Required sheets are present
   - Column names match configuration

2. **Data Completeness**
   - Minimum row requirements
   - Required columns present
   - Date ranges valid

3. **Data Quality**
   - Numeric columns are properly formatted
   - Dates are valid and reasonable
   - No critical missing values

### Calculation Validation

1. **Cross-Check Results**
   - Verify totals match across sheets
   - Check percentile calculations
   - Validate classification logic

2. **Business Rule Validation**
   - ABC/FMS thresholds applied correctly
   - Capacity calculations reasonable
   - Date range handling accurate

### Output Validation

1. **Excel Report Checks**
   - All sheets generated successfully
   - Formatting applied correctly
   - Charts and formulas working

2. **Data Integrity**
   - Raw data preserved accurately
   - Calculations traceable
   - Timestamps and metadata correct

## Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Check `DATA_FILE_PATH` in config.py
   - Ensure Excel file exists
   - Verify file permissions

2. **Sheet Not Found Errors**
   - Check sheet names in Excel file
   - Update `SHEET_NAMES` in config.py
   - Ensure spelling matches exactly

3. **Column Not Found Errors**
   - Check column names in Excel sheets
   - Update column mappings in config.py
   - Handle optional columns gracefully

4. **Data Validation Failures**
   - Review minimum data requirements
   - Check date formats and ranges
   - Ensure numeric columns are properly formatted

5. **Memory Issues**
   - Reduce data size or date range
   - Check available system memory
   - Consider processing in chunks

### Debugging Tips

1. **Enable Verbose Output**
   ```python
   DEBUG_SETTINGS = {
       'VERBOSE_OUTPUT': True,
       'PRINT_SAMPLE_DATA': True
   }
   ```

2. **Check Sample Data**
   - Review printed sample data
   - Verify column names and formats
   - Check for unexpected values

3. **Validate Configuration**
   ```bash
   python config.py
   ```

4. **Test Individual Modules**
   ```bash
   python scripts/data_loader.py
   python scripts/order_analysis.py
   ```

## Integration with New Claude Project

When providing this documentation to a new Claude project, include:

1. **Context**: "Build a warehouse analysis tool following this complete specification"

2. **Requirements**: "Implement exactly as documented, maintaining all error handling and user guidance"

3. **Priority**: "Focus on Excel output first, ensure beginners can use it easily"

4. **Files to Create**:
   - All scripts as documented above
   - Complete README.md with setup instructions
   - Sample data file structure
   - Requirements.txt

5. **Key Points**:
   - Keep it simple and focused
   - Extensive error handling and user guidance
   - Configuration-driven approach
   - Excel-first output strategy
   - Comprehensive documentation

## Summary

This documentation provides everything needed to recreate the warehouse analysis tool in a new environment while maintaining the focus on simplicity, reliability, and user-friendliness. The tool follows a clean, step-by-step approach:

1. **Phase 1**: Excel report generation (this documentation)
2. **Phase 2**: Dashboard and charts (future enhancement)
3. **Phase 3**: Word reports and LLM integration (future enhancement)

Each phase builds on the previous one, ensuring you always have working Excel outputs while progressively adding more features. The beginner-friendly approach, comprehensive error handling, and detailed documentation make this tool accessible to users of all technical levels.

**Next Steps After Implementation:**
1. Test with your actual warehouse data
2. Customize configuration for your specific needs
3. Validate analysis results against known benchmarks
4. Use Excel reports for warehouse planning and optimization
5. Plan for Phase 2 dashboard development when ready