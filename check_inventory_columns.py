#!/usr/bin/env python3
"""
Check what columns are actually in the inventory data
"""

import pandas as pd

def check_inventory_columns():
    """Check what columns exist in the inventory data"""
    
    test_file_path = "Test Data/ITC_WMS_RAW_DATA.xlsx"
    
    print("🔍 Checking Inventory Data Columns")
    print("=" * 50)
    
    # Read just the inventory sheet without any processing
    try:
        raw_inventory_df = pd.read_excel(test_file_path, sheet_name='InventoryData')
        
        print(f"📊 Raw Inventory Data:")
        print(f"  Shape: {raw_inventory_df.shape}")
        print(f"  Columns: {list(raw_inventory_df.columns)}")
        print(f"  Data types:\n{raw_inventory_df.dtypes}")
        
        print(f"\n📋 Sample data:")
        print(raw_inventory_df.head())
        
    except Exception as e:
        print(f"❌ Error reading inventory data: {e}")

if __name__ == "__main__":
    check_inventory_columns()