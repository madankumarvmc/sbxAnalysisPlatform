#!/usr/bin/env python3
"""
Check the actual date format in the receipt data to identify DD/MM/YYYY vs MM/DD/YYYY issue
"""

import pandas as pd

def check_receipt_date_format():
    """Check how receipt dates are actually stored in the Excel file"""
    
    # Read the raw Excel file without any processing
    test_file_path = "Test Data/ITC_WMS_RAW_DATA.xlsx"
    
    print("🔍 Checking Raw Receipt Date Format")
    print("=" * 50)
    
    # Read just the receipt sheet without any processing
    raw_receipt_df = pd.read_excel(test_file_path, sheet_name='ReceiptData')
    
    print(f"📊 Raw Receipt Data Analysis:")
    print(f"  Total records: {len(raw_receipt_df)}")
    print(f"  Columns: {list(raw_receipt_df.columns)}")
    
    # Check the Receipt Date column specifically
    date_column = 'Receipt Date'
    if date_column in raw_receipt_df.columns:
        print(f"\n📅 Raw Receipt Date Analysis:")
        print(f"  Column type: {raw_receipt_df[date_column].dtype}")
        
        # Show raw values before any conversion
        print(f"\n📅 First 20 Raw Receipt Dates:")
        for i, date in enumerate(raw_receipt_df[date_column].head(20)):
            print(f"  {i+1:2d}: {date} (type: {type(date)})")
        
        # Check for patterns that suggest DD/MM/YYYY format
        print(f"\n🔍 Analyzing Date Patterns:")
        unique_dates = raw_receipt_df[date_column].unique()
        print(f"  Total unique dates: {len(unique_dates)}")
        
        # Show all unique dates to spot the pattern
        print(f"\n📅 All Unique Raw Dates:")
        for i, date in enumerate(sorted(unique_dates[:50])):  # Show first 50
            print(f"  {i+1:2d}: {date}")
            
        # Try parsing with different formats to see which makes more sense
        print(f"\n🧪 Testing Different Date Parsing:")
        
        # Test 1: Default pandas parsing (what's currently happening)
        try:
            default_parsed = pd.to_datetime(raw_receipt_df[date_column], errors='coerce')
            print(f"  Default parsing - Date range: {default_parsed.min()} to {default_parsed.max()}")
            print(f"  Default parsing - Unique dates: {default_parsed.nunique()}")
        except Exception as e:
            print(f"  Default parsing failed: {e}")
        
        # Test 2: DD/MM/YYYY format
        try:
            dd_mm_yyyy_parsed = pd.to_datetime(raw_receipt_df[date_column], format='%d/%m/%Y', errors='coerce')
            valid_dd_mm = dd_mm_yyyy_parsed.notna().sum()
            if valid_dd_mm > 0:
                print(f"  DD/MM/YYYY parsing - Date range: {dd_mm_yyyy_parsed.min()} to {dd_mm_yyyy_parsed.max()}")
                print(f"  DD/MM/YYYY parsing - Unique dates: {dd_mm_yyyy_parsed.nunique()}")
                print(f"  DD/MM/YYYY parsing - Valid conversions: {valid_dd_mm}/{len(raw_receipt_df)}")
        except Exception as e:
            print(f"  DD/MM/YYYY parsing failed: {e}")
        
        # Test 3: MM/DD/YYYY format
        try:
            mm_dd_yyyy_parsed = pd.to_datetime(raw_receipt_df[date_column], format='%m/%d/%Y', errors='coerce')
            valid_mm_dd = mm_dd_yyyy_parsed.notna().sum()
            if valid_mm_dd > 0:
                print(f"  MM/DD/YYYY parsing - Date range: {mm_dd_yyyy_parsed.min()} to {mm_dd_yyyy_parsed.max()}")
                print(f"  MM/DD/YYYY parsing - Unique dates: {mm_dd_yyyy_parsed.nunique()}")
                print(f"  MM/DD/YYYY parsing - Valid conversions: {valid_mm_dd}/{len(raw_receipt_df)}")
        except Exception as e:
            print(f"  MM/DD/YYYY parsing failed: {e}")
        
        # Test 4: DD-MM-YYYY format
        try:
            dd_mm_yyyy_dash_parsed = pd.to_datetime(raw_receipt_df[date_column], format='%d-%m-%Y', errors='coerce')
            valid_dd_mm_dash = dd_mm_yyyy_dash_parsed.notna().sum()
            if valid_dd_mm_dash > 0:
                print(f"  DD-MM-YYYY parsing - Date range: {dd_mm_yyyy_dash_parsed.min()} to {dd_mm_yyyy_dash_parsed.max()}")
                print(f"  DD-MM-YYYY parsing - Valid conversions: {valid_dd_mm_dash}/{len(raw_receipt_df)}")
        except Exception as e:
            print(f"  DD-MM-YYYY parsing failed: {e}")
            
    else:
        print(f"❌ Receipt Date column not found! Available columns: {list(raw_receipt_df.columns)}")

if __name__ == "__main__":
    check_receipt_date_format()