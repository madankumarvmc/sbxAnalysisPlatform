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
import sys
import os

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
import config
from .case_equivalent_converter import CaseEquivalentConverter

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
    
    def __init__(self, uploaded_file=None, verbose=False):
        """
        Initialize the DataLoader.
        
        Args:
            uploaded_file: Streamlit uploaded file object or file path
            verbose (bool): Enable verbose output for debugging
        """
        self.uploaded_file = uploaded_file
        self.verbose = verbose
        
        # Configuration settings
        self.sheet_names = config.SHEET_NAMES
        self.order_columns = config.ORDER_COLUMNS
        self.sku_columns = config.SKU_COLUMNS
        self.receipt_columns = config.RECEIPT_COLUMNS
        self.inventory_columns = config.INVENTORY_COLUMNS
        
        # Initialize data containers
        self.order_data = None
        self.sku_master = None
        self.receipt_data = None
        self.inventory_data = None
        
        # Case equivalent converter (initialized after SKU master is loaded)
        self.converter = None
        
        # Validation results
        self.validation_results = {}
        
        if self.verbose:
            print(f"DataLoader initialized")
    
    def load_all_data(self):
        """
        Load all available data from the Excel file.
        
        Returns:
            dict: Dictionary containing loaded data and validation results
        """
        if self.uploaded_file is None:
            raise ValueError("No file provided for data loading")
        
        results = {
            'success': True,
            'data': {},
            'validation': {},
            'errors': []
        }
        
        try:
            # Try to load each sheet
            sheets_to_load = [
                ('order_data', self.sheet_names['ORDER_DATA'], self.load_order_data),
                ('sku_master', self.sheet_names['SKU_MASTER'], self.load_sku_master),
                ('receipt_data', self.sheet_names['RECEIPT_DATA'], self.load_receipt_data),
                ('inventory_data', self.sheet_names['INVENTORY_DATA'], self.load_inventory_data)
            ]
            
            for data_key, sheet_name, load_function in sheets_to_load:
                try:
                    data = load_function()
                    if data is not None:
                        results['data'][data_key] = data
                        results['validation'][data_key] = self.validation_results.get(data_key, {})
                        if self.verbose:
                            print(f"✅ Successfully loaded {sheet_name}: {len(data)} rows")
                    else:
                        if self.verbose:
                            print(f"⚠️ Could not load {sheet_name} (sheet may not exist)")
                        
                except Exception as e:
                    error_msg = f"Error loading {sheet_name}: {str(e)}"
                    results['errors'].append(error_msg)
                    if self.verbose:
                        print(f"❌ {error_msg}")
            
            # Check if we have minimum required data
            if 'order_data' not in results['data']:
                results['success'] = False
                results['errors'].append("Order data is required but could not be loaded")
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"General loading error: {str(e)}")
        
        return results
    
    def load_order_data(self):
        """
        Load order data from Excel file.
        
        Returns:
            pandas.DataFrame: Order data with standardized column names
            None: If loading fails
        """
        try:
            if self.verbose:
                print(f"Loading order data from sheet: {self.sheet_names['ORDER_DATA']}")
            
            # Read Excel sheet
            self.order_data = pd.read_excel(
                self.uploaded_file,
                sheet_name=self.sheet_names['ORDER_DATA']
            )
            
            if self.verbose:
                print(f"Raw order data shape: {self.order_data.shape}")
            
            # Validate required columns exist
            missing_columns = []
            for config_col, excel_col in self.order_columns.items():
                if excel_col not in self.order_data.columns:
                    missing_columns.append(excel_col)
            
            if missing_columns:
                error_msg = f"Missing required columns in order data: {missing_columns}"
                print(f"❌ {error_msg}")
                print(f"Available columns: {list(self.order_data.columns)}")
                return None
            
            # Clean and process order data
            self.order_data = self._clean_order_data(self.order_data)
            
            # Validate data quality (converter should be available if SKU master was loaded)
            self.validation_results['order_data'] = self._validate_order_data(self.order_data)
            
            if self.verbose:
                print(f"Cleaned order data shape: {self.order_data.shape}")
                print(f"Date range: {self.order_data['Date'].min()} to {self.order_data['Date'].max()}")
            
            return self.order_data
            
        except Exception as e:
            print(f"❌ Error loading order data: {str(e)}")
            return None
    
    def load_sku_master(self):
        """
        Load SKU master data from Excel file.
        
        Returns:
            pandas.DataFrame: SKU master data with standardized column names
            None: If loading fails
        """
        try:
            if self.verbose:
                print(f"Loading SKU master from sheet: {self.sheet_names['SKU_MASTER']}")
            
            # Read Excel sheet
            self.sku_master = pd.read_excel(
                self.uploaded_file,
                sheet_name=self.sheet_names['SKU_MASTER']
            )
            
            if self.verbose:
                print(f"Raw SKU master shape: {self.sku_master.shape}")
            
            # Validate required columns exist
            missing_columns = []
            for config_col, excel_col in self.sku_columns.items():
                if excel_col not in self.sku_master.columns:
                    missing_columns.append(excel_col)
            
            if missing_columns:
                print(f"❌ Missing required columns in SKU master: {missing_columns}")
                return None
            
            # Clean and process SKU master data
            self.sku_master = self._clean_sku_master(self.sku_master)
            
            # Initialize case equivalent converter with SKU master
            self.converter = CaseEquivalentConverter(self.sku_master)
            
            # Validate data quality
            self.validation_results['sku_master'] = self._validate_sku_master(self.sku_master)
            
            if self.verbose:
                print(f"Cleaned SKU master shape: {self.sku_master.shape}")
                print(f"Categories: {self.sku_master['Category'].unique()}")
            
            return self.sku_master
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading SKU master: {str(e)}")
            return None
    
    def load_receipt_data(self):
        """
        Load receipt data from Excel file.
        
        Returns:
            pandas.DataFrame: Receipt data with standardized column names
            None: If loading fails
        """
        try:
            if self.verbose:
                print(f"Loading receipt data from sheet: {self.sheet_names['RECEIPT_DATA']}")
            
            # Read Excel sheet
            self.receipt_data = pd.read_excel(
                self.uploaded_file,
                sheet_name=self.sheet_names['RECEIPT_DATA']
            )
            
            if self.verbose:
                print(f"Raw receipt data shape: {self.receipt_data.shape}")
            
            # Validate required columns exist
            missing_columns = []
            for config_col, excel_col in self.receipt_columns.items():
                if excel_col not in self.receipt_data.columns:
                    missing_columns.append(excel_col)
            
            if missing_columns:
                print(f"❌ Missing required columns in receipt data: {missing_columns}")
                return None
            
            # Clean and process receipt data
            self.receipt_data = self._clean_receipt_data(self.receipt_data)
            
            # Validate data quality (converter should be available if SKU master was loaded)
            self.validation_results['receipt_data'] = self._validate_receipt_data(self.receipt_data)
            
            if self.verbose:
                print(f"Cleaned receipt data shape: {self.receipt_data.shape}")
            
            return self.receipt_data
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading receipt data: {str(e)}")
            return None
    
    def load_inventory_data(self):
        """
        Load inventory data from Excel file.
        
        Returns:
            pandas.DataFrame: Inventory data with standardized column names
            None: If loading fails
        """
        try:
            if self.verbose:
                print(f"Loading inventory data from sheet: {self.sheet_names['INVENTORY_DATA']}")
            
            # Read Excel sheet
            self.inventory_data = pd.read_excel(
                self.uploaded_file,
                sheet_name=self.sheet_names['INVENTORY_DATA']
            )
            
            if self.verbose:
                print(f"Raw inventory data shape: {self.inventory_data.shape}")
            
            # Validate required columns exist
            missing_columns = []
            for config_col, excel_col in self.inventory_columns.items():
                if excel_col not in self.inventory_data.columns:
                    missing_columns.append(excel_col)
            
            if missing_columns:
                print(f"❌ Missing required columns in inventory data: {missing_columns}")
                return None
            
            # Clean and process inventory data
            self.inventory_data = self._clean_inventory_data(self.inventory_data)
            
            # Validate data quality
            self.validation_results['inventory_data'] = self._validate_inventory_data(self.inventory_data)
            
            if self.verbose:
                print(f"Cleaned inventory data shape: {self.inventory_data.shape}")
            
            return self.inventory_data
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading inventory data: {str(e)}")
            return None
    
    def _standardize_data_types(self, df, sheet_type):
        """
        Standardize data types for consistent processing across all sheets.
        This solves merge issues caused by mixed data types (especially SKU codes).
        
        Args:
            df: pandas DataFrame to standardize
            sheet_type: Type of sheet ('order', 'sku_master', 'receipt', 'inventory')
            
        Returns:
            pandas DataFrame with standardized data types
        """
        df = df.copy()
        
        if self.verbose:
            print(f"🔧 Standardizing data types for {sheet_type} data...")
        
        # Define standardization rules for each sheet type
        standardization_rules = {
            'order': {
                'Date': 'datetime',
                'Order No.': 'string',
                'Shipment No.': 'string', 
                'Sku Code': 'sku_code',
                'Qty in Cases': 'float',
                'Qty in Eaches': 'float'
            },
            'sku_master': {
                'Category': 'string',
                'Sku Code': 'sku_code', 
                'Case Config': 'float',
                'Pallet Fit': 'float'
            },
            'receipt': {
                'Receipt Date': 'datetime',
                'SKU ID': 'sku_code',
                'Shipment No': 'string',
                'Truck No': 'string', 
                'Batch': 'string',
                'Quantity in Cases': 'float',
                'Quantity in Eaches': 'float'
            },
            'inventory': {
                'Calendar Day': 'datetime',
                'SKU ID': 'sku_code',
                'SKU Name': 'string',
                'Total Stock in Cases (In Base Unit of Measure)': 'float',
                'Total Stock in Pieces (In Base Unit of Measue)': 'float'
            }
        }
        
        if sheet_type not in standardization_rules:
            if self.verbose:
                print(f"⚠️ No standardization rules for {sheet_type}. Skipping.")
            return df
        
        rules = standardization_rules[sheet_type]
        
        # Apply standardization for each column
        for column, data_type in rules.items():
            if column not in df.columns:
                continue
                
            try:
                if data_type == 'sku_code':
                    # Standardize SKU codes to string format (critical for merging)
                    df[column] = df[column].astype(str)
                    # Remove .0 suffix that appears from numeric conversions
                    df[column] = df[column].str.replace(r'\.0$', '', regex=True)
                    # Trim whitespace
                    df[column] = df[column].str.strip()
                    # Handle NaN/null values
                    df[column] = df[column].replace('nan', pd.NA)
                    df[column] = df[column].replace('', pd.NA)
                    
                elif data_type == 'datetime':
                    # Handle different datetime formats
                    if column == 'Calendar Day':
                        # Handle DD.MM.YYYY format for inventory data - strip whitespace first
                        df[column] = df[column].astype(str).str.strip()
                        df[column] = pd.to_datetime(df[column], format='%d.%m.%Y', errors='coerce')
                    elif column == 'Receipt Date':
                        # Handle DD.MM.YYYY format for receipt data - strip whitespace first
                        df[column] = df[column].astype(str).str.strip()
                        df[column] = pd.to_datetime(df[column], format='%d.%m.%Y', errors='coerce')
                    else:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                        
                elif data_type == 'float':
                    # Convert to float for consistent calculations
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    
                elif data_type == 'string':
                    # Standardize strings
                    df[column] = df[column].astype(str).str.strip()
                    df[column] = df[column].replace('nan', pd.NA)
                    df[column] = df[column].replace('', pd.NA)
                    
                if self.verbose:
                    print(f"  ✅ {column} → {data_type}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ Failed to convert {column} to {data_type}: {e}")
        
        if self.verbose:
            print(f"✅ Data type standardization complete for {sheet_type}")
            
        return df
    
    def _validate_standardized_data(self, df, sheet_type):
        """
        Validate that data has been properly standardized.
        
        Args:
            df: pandas DataFrame to validate
            sheet_type: Type of sheet ('order', 'sku_master', 'receipt', 'inventory')
            
        Returns:
            dict: Validation results with warnings and errors
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Expected data types after standardization
        expected_types = {
            'order': {
                'Date': 'datetime64[ns]',
                'Order No.': 'object',  # String
                'Shipment No.': 'object',  # String
                'Sku Code': 'object',  # String
                'Qty in Cases': ['float64', 'int64'],  # Numeric
                'Qty in Eaches': ['float64', 'int64']  # Numeric
            },
            'sku_master': {
                'Category': 'object',  # String
                'Sku Code': 'object',  # String
                'Case Config': ['float64', 'int64'],  # Numeric
                'Pallet Fit': ['float64', 'int64']  # Numeric
            }
        }
        
        if sheet_type not in expected_types:
            return validation_results  # Skip validation for unknown sheet types
        
        expected = expected_types[sheet_type]
        
        for column, expected_dtype in expected.items():
            if column not in df.columns:
                continue
                
            actual_dtype = str(df[column].dtype)
            
            # Handle multiple acceptable types
            if isinstance(expected_dtype, list):
                if actual_dtype not in expected_dtype:
                    validation_results['warnings'].append(
                        f"{column}: Expected {expected_dtype}, got {actual_dtype}"
                    )
            else:
                if actual_dtype != expected_dtype:
                    validation_results['warnings'].append(
                        f"{column}: Expected {expected_dtype}, got {actual_dtype}"
                    )
        
        # Validate SKU codes specifically (critical for merging)
        sku_col = 'Sku Code' if 'Sku Code' in df.columns else ('SKU ID' if 'SKU ID' in df.columns else None)
        if sku_col:
            # Check for mixed data types in SKU column
            sample_skus = df[sku_col].dropna().head(100)
            if len(sample_skus) > 0:
                # All should be strings after standardization
                non_string_count = sum(1 for sku in sample_skus if not isinstance(sku, str))
                if non_string_count > 0:
                    validation_results['errors'].append(
                        f"{sku_col}: Found {non_string_count} non-string values after standardization"
                    )
                    validation_results['valid'] = False
                
                # Check for .0 suffixes (should be removed)
                dot_zero_count = sum(1 for sku in sample_skus if str(sku).endswith('.0'))
                if dot_zero_count > 0:
                    validation_results['warnings'].append(
                        f"{sku_col}: Found {dot_zero_count} SKUs with .0 suffix (may cause merge issues)"
                    )
        
        if self.verbose and (validation_results['warnings'] or validation_results['errors']):
            print(f"🔍 Validation results for {sheet_type}:")
            for warning in validation_results['warnings']:
                print(f"  ⚠️ {warning}")
            for error in validation_results['errors']:
                print(f"  ❌ {error}")
        
        return validation_results
    
    def _clean_order_data(self, df):
        """Clean and standardize order data"""
        df = df.copy()
        
        # Apply data type standardization first
        df = self._standardize_data_types(df, 'order')
        
        # Validate standardization
        validation = self._validate_standardized_data(df, 'order')
        if not validation['valid']:
            warnings.warn(f"Order data validation failed: {validation['errors']}")
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Date', 'Sku Code'])
        
        # Fill missing quantities with 0
        df['Qty in Cases'] = df['Qty in Cases'].fillna(0)
        df['Qty in Eaches'] = df['Qty in Eaches'].fillna(0)
        
        return df
    
    def _clean_sku_master(self, df):
        """Clean and standardize SKU master data"""
        df = df.copy()
        
        # Apply data type standardization first
        df = self._standardize_data_types(df, 'sku_master')
        
        # Validate standardization
        validation = self._validate_standardized_data(df, 'sku_master')
        if not validation['valid']:
            warnings.warn(f"SKU master validation failed: {validation['errors']}")
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Sku Code'])
        
        # Fill missing values with defaults
        df['Case Config'] = df['Case Config'].fillna(1)
        df['Pallet Fit'] = df['Pallet Fit'].fillna(1)
        
        return df
    
    def _clean_receipt_data(self, df):
        """Clean and standardize receipt data"""
        df = df.copy()
        
        # Apply data type standardization first
        df = self._standardize_data_types(df, 'receipt')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Receipt Date', 'SKU ID'])
        
        # Fill missing quantities with 0
        df['Quantity in Cases'] = df['Quantity in Cases'].fillna(0)
        df['Quantity in Eaches'] = df['Quantity in Eaches'].fillna(0)
        
        return df
    
    def _clean_inventory_data(self, df):
        """Clean and standardize inventory data"""
        df = df.copy()
        
        # Apply data type standardization first
        df = self._standardize_data_types(df, 'inventory')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Calendar Day', 'SKU ID'])
        
        # Fill missing stock with 0
        df['Total Stock in Cases (In Base Unit of Measure)'] = df['Total Stock in Cases (In Base Unit of Measure)'].fillna(0)
        df['Total Stock in Pieces (In Base Unit of Measue)'] = df['Total Stock in Pieces (In Base Unit of Measue)'].fillna(0)
        
        return df
    
    def _validate_order_data(self, df):
        """Validate order data quality with case equivalent volume metrics"""
        validation = {
            'total_rows': len(df),
            'date_range_days': (df['Date'].max() - df['Date'].min()).days,
            'unique_orders': df['Order No.'].nunique(),
            'unique_skus': df['Sku Code'].nunique(),
            'total_volume_cases': df['Qty in Cases'].sum(),
            'total_volume_eaches': df['Qty in Eaches'].sum(),
            'missing_dates': df['Date'].isna().sum(),
            'missing_skus': df['Sku Code'].isna().sum()
        }
        
        # Add case equivalent volume metrics if converter is available
        if self.converter is not None:
            try:
                total_case_equivalent = self.converter.calculate_total_case_equivalent_volume(df)
                validation['total_case_equivalent_volume'] = total_case_equivalent
                validation['case_equivalent_to_cases_ratio'] = total_case_equivalent / validation['total_volume_cases'] if validation['total_volume_cases'] > 0 else 0
                validation['mixed_case_each_lines'] = len(df[(df['Qty in Cases'] > 0) & (df['Qty in Eaches'] > 0)])
                validation['case_only_lines'] = len(df[(df['Qty in Cases'] > 0) & (df['Qty in Eaches'] == 0)])
                validation['each_only_lines'] = len(df[(df['Qty in Cases'] == 0) & (df['Qty in Eaches'] > 0)])
            except Exception as e:
                validation['case_equivalent_error'] = str(e)
        else:
            validation['case_equivalent_volume'] = 'Not available - SKU master required'
        
        # Check minimum requirements
        validation['meets_minimum_rows'] = validation['total_rows'] >= config.VALIDATION_SETTINGS['MIN_ORDER_ROWS']
        validation['meets_date_range'] = validation['date_range_days'] >= config.VALIDATION_SETTINGS['MIN_DATE_RANGE_DAYS']
        
        return validation
    
    def _validate_sku_master(self, df):
        """Validate SKU master data quality with case equivalent validation"""
        validation = {
            'total_rows': len(df),
            'unique_skus': df['Sku Code'].nunique(),
            'unique_categories': df['Category'].nunique(),
            'missing_categories': df['Category'].isna().sum(),
            'categories': df['Category'].unique().tolist(),
            'missing_case_config': df['Case Config'].isna().sum(),
            'invalid_case_config': len(df[df['Case Config'] <= 0]),
            'case_config_range': {
                'min': df['Case Config'].min(),
                'max': df['Case Config'].max(),
                'mean': df['Case Config'].mean()
            },
            'missing_pallet_fit': df['Pallet Fit'].isna().sum()
        }
        
        # Check case equivalent conversion readiness
        validation['case_equivalent_ready'] = (
            validation['missing_case_config'] == 0 and 
            validation['invalid_case_config'] == 0
        )
        
        validation['meets_minimum_rows'] = validation['total_rows'] >= config.VALIDATION_SETTINGS['MIN_SKU_ROWS']
        
        return validation
    
    def _validate_receipt_data(self, df):
        """Validate receipt data quality with case equivalent metrics"""
        validation = {
            'total_rows': len(df),
            'date_range_days': (df['Receipt Date'].max() - df['Receipt Date'].min()).days,
            'unique_skus': df['SKU ID'].nunique(),
            'total_received_cases': df['Quantity in Cases'].sum(),
            'total_received_eaches': df['Quantity in Eaches'].sum(),
            'missing_dates': df['Receipt Date'].isna().sum(),
            'mixed_receipt_lines': len(df[(df['Quantity in Cases'] > 0) & (df['Quantity in Eaches'] > 0)]),
            'case_only_receipts': len(df[(df['Quantity in Cases'] > 0) & (df['Quantity in Eaches'] == 0)]),
            'each_only_receipts': len(df[(df['Quantity in Cases'] == 0) & (df['Quantity in Eaches'] > 0)])
        }
        
        # Add case equivalent metrics if converter is available
        if self.converter is not None:
            try:
                # Map receipt columns to converter format
                mapped_df = df.copy()
                mapped_df['Sku Code'] = mapped_df['SKU ID']
                mapped_df['Qty in Cases'] = mapped_df['Quantity in Cases'] 
                mapped_df['Qty in Eaches'] = mapped_df['Quantity in Eaches']
                
                total_case_equivalent = self.converter.calculate_total_case_equivalent_volume(mapped_df)
                validation['total_case_equivalent_received'] = total_case_equivalent
                validation['case_equivalent_to_cases_ratio'] = total_case_equivalent / validation['total_received_cases'] if validation['total_received_cases'] > 0 else 0
            except Exception as e:
                validation['case_equivalent_error'] = str(e)
        else:
            validation['case_equivalent_received'] = 'Not available - SKU master required'
        
        return validation
    
    def _validate_inventory_data(self, df):
        """Validate inventory data quality"""
        validation = {
            'total_rows': len(df),
            'unique_skus': df['SKU ID'].nunique(),
            'total_stock_cases': df['Total Stock in Cases (In Base Unit of Measure)'].sum(),
            'total_stock_eaches': df['Total Stock in Pieces (In Base Unit of Measue)'].sum(),
            'date_range_days': (df['Calendar Day'].max() - df['Calendar Day'].min()).days,
            'unique_dates': df['Calendar Day'].nunique()
        }
        
        return validation

# Test function for standalone execution
if __name__ == "__main__":
    print("DataLoader module - ready for use")
    print("This module requires an uploaded Excel file to function.")
    print("Use within the Streamlit application for proper functionality.")