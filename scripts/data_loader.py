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
    
    def _clean_order_data(self, df):
        """Clean and standardize order data"""
        df = df.copy()
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Ensure numeric columns are numeric
        numeric_columns = ['Order No.', 'Shipment No.', 'Qty in Cases', 'Qty in Eaches']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Date', 'Sku Code'])
        
        # Fill missing quantities with 0
        df['Qty in Cases'] = df['Qty in Cases'].fillna(0)
        df['Qty in Eaches'] = df['Qty in Eaches'].fillna(0)
        
        return df
    
    def _clean_sku_master(self, df):
        """Clean and standardize SKU master data"""
        df = df.copy()
        
        # Ensure numeric columns are numeric
        numeric_columns = ['Sku Code', 'Case Config', 'Pallet Fit']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Sku Code'])
        
        # Fill missing values with defaults
        df['Case Config'] = df['Case Config'].fillna(1)
        df['Pallet Fit'] = df['Pallet Fit'].fillna(1)
        
        return df
    
    def _clean_receipt_data(self, df):
        """Clean and standardize receipt data"""
        df = df.copy()
        
        # Convert date column to datetime
        df['Receipt Date'] = pd.to_datetime(df['Receipt Date'], errors='coerce')
        
        # Ensure numeric columns are numeric
        numeric_columns = ['SKU ID', 'Quantity in Cases', 'Quantity in Eaches']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Receipt Date', 'SKU ID'])
        
        # Fill missing quantities with 0
        df['Quantity in Cases'] = df['Quantity in Cases'].fillna(0)
        df['Quantity in Eaches'] = df['Quantity in Eaches'].fillna(0)
        
        return df
    
    def _clean_inventory_data(self, df):
        """Clean and standardize inventory data"""
        df = df.copy()
        
        # Convert date column - handle DD.MM.YYYY format
        try:
            df['Calendar Day'] = pd.to_datetime(df['Calendar Day'], format='%d.%m.%Y', errors='coerce')
        except:
            df['Calendar Day'] = pd.to_datetime(df['Calendar Day'], errors='coerce')
        
        # Ensure numeric columns are numeric
        numeric_columns = ['SKU ID', 'Total Stock in Cases (In Base Unit of Measure)', 'Total Stock in Pieces (In Base Unit of Measue)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
            'unique_sites': df['Site'].nunique(),
            'total_stock_cases': df['Total Stock in Cases (In Base Unit of Measure)'].sum(),
            'sites': df['Site'].unique().tolist()
        }
        
        return validation

# Test function for standalone execution
if __name__ == "__main__":
    print("DataLoader module - ready for use")
    print("This module requires an uploaded Excel file to function.")
    print("Use within the Streamlit application for proper functionality.")