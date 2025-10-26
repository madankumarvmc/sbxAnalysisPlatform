# Order Analysis Module Documentation

## Overview
The `order_analysis.py` module provides comprehensive analysis of order patterns and trends for warehouse operations. It contains the `OrderAnalyzer` class which processes order data to identify patterns, calculate statistics, and generate insights for capacity planning and operational optimization.

## OrderAnalyzer Class

### Purpose
Analyzes order data to identify:
- Daily volume patterns and trends
- Statistical distributions for capacity planning
- Percentile analysis for operational thresholds
- Customer behavior patterns and seasonality
- Peak period identification
- Pick type classifications (Each Pick vs Case Pick)

---

## Class Initialization

### `__init__(order_data, sku_master=None, analysis_config=None)`

**Purpose**: Initialize the OrderAnalyzer with order data, SKU master data, and configuration parameters.

**⚡ MAJOR UPDATE**: Now requires `sku_master` parameter for accurate case equivalent volume calculations.

**Parameters**:
- `order_data` (pandas.DataFrame): Cleaned order data from DataLoader containing columns:
  - `Date`: Order date
  - `Order No.`: Unique order identifier
  - `Shipment No.`: Shipment number
  - `Sku Code`: Product SKU identifier
  - `Qty in Cases`: Quantity in cases
  - `Qty in Eaches`: Quantity in individual units
- `sku_master` (pandas.DataFrame, **NEW REQUIRED**): SKU master data containing:
  - `Sku Code`: Product SKU identifier
  - `Category`: Product category
  - `Case Config`: **CRITICAL** - Number of eaches per case for volume conversion
  - `Pallet Fit`: Cases per pallet
- `analysis_config` (dict, optional): Configuration parameters including:
  - `PERCENTILE_LEVELS`: List of percentiles for analysis (default: [95, 90, 85, 80, 75])
  - `DATE_RANGE`: Start and end dates for analysis
  - `FMS_THRESHOLDS`: Fast/Medium/Slow classification thresholds

**Returns**: OrderAnalyzer instance

**Internal Setup**:
- **NEW**: Stores SKU master data for case equivalent volume calculations
- Applies date range filtering if specified
- Initializes analysis result containers
- Sets up configuration parameters from config module

**🚨 BREAKING CHANGE**: This constructor signature has changed. Update all calling code to include `sku_master` parameter.

---

## 🔧 **CASE EQUIVALENT VOLUME STANDARDIZATION**

### **Critical Change - Volume Calculation Method**

**Previous Implementation (❌ INCORRECT)**:
- Mixed raw cases and eaches: `2 cases + 500 eaches = 502 "units"`
- Different calculation methods across functions
- Meaningless volume metrics for business planning

**New Implementation (✅ CORRECT)**:
- **All volume = Case Equivalent Volume**
- **Formula**: `Volume = Cases + (Eaches ÷ Case_Config)`
- **Example**: `2 cases + 500 eaches = 2 + (500÷250) = 4.0 case equivalent`
- **Consistent across ALL analysis functions**

### **Business Impact**

- **Accurate Capacity Planning**: Volume projections now mathematically correct
- **Consistent Metrics**: All volume calculations use same methodology  
- **Professional Reports**: Excel outputs show meaningful business metrics
- **Cross-SKU Comparisons**: Different products now comparable on same scale

### **Functions Updated with Case Equivalent Volume**

All functions now use `CaseEquivalentConverter` utility:

1. **`analyze_daily_patterns()`** - Added `Daily_Case_Equivalent_Volume` column
2. **`analyze_order_profiles()`** - Fixed `Units_Per_Order/Line` → `Case_Equivalent_Per_Order/Line`
3. **`analyze_outbound_summary_statistics()`** - All volume projections use case equivalent
4. **`analyze_monthly_volumes()`** - Monthly totals in case equivalent
5. **`analyze_enhanced_weekday_patterns()`** - Weekday averages in case equivalent

---

## Analysis Functions

### 1. `run_complete_analysis()`

**Purpose**: Main orchestrator function that runs all analysis modules and returns comprehensive results.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'success': True,
    'analysis_date': datetime.now(),
    'data_summary': {},  # Basic data statistics
    'daily_analysis': {},  # Daily patterns and volumes
    'volume_analysis': {},  # Volume statistics and distributions
    'percentile_analysis': {},  # Percentile analysis for capacity planning
    'sku_frequency': {},  # SKU frequency patterns
    'peak_analysis': {},  # Peak period identification
    'trends': {},  # Volume trends over time
    'outbound_summary': {},  # Comprehensive outbound statistics
    'order_profiles': {},  # Order profile ratios and metrics
    'monthly_volumes': {},  # Monthly volume patterns
    'enhanced_weekday_patterns': {}  # Enhanced weekday analysis
}
```

### 2. `analyze_daily_patterns()`

**Purpose**: Analyze daily order patterns, volumes, and day-of-week trends.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'daily_data': DataFrame,  # Daily aggregated data with columns:
        # - Date, Daily_Orders, Daily_Shipments, Daily_SKUs, Daily_Cases, Daily_Eaches
        # - Day_of_Week, Week_Number, Month, Cases_7Day_MA, Orders_7Day_MA
    'day_of_week_patterns': DataFrame,  # Day-of-week aggregated statistics
    'total_days': int,  # Total days in analysis
    'avg_daily_orders': float,  # Average daily order count
    'avg_daily_cases': float,  # Average daily case volume
    'busiest_day': str,  # Date of highest volume (YYYY-MM-DD)
    'busiest_day_volume': float  # Volume on busiest day
}
```

### 3. `analyze_volume_statistics()`

**Purpose**: Calculate comprehensive volume statistics and distributions for cases and orders.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'cases': {
        'total': float,  # Total cases across all days
        'mean': float,  # Average daily cases
        'median': float,  # Median daily cases
        'std': float,  # Standard deviation
        'min': float,  # Minimum daily cases
        'max': float,  # Maximum daily cases
        'cv': float  # Coefficient of variation
    },
    'orders': {
        'total': float,  # Total orders across all days
        'mean': float,  # Average daily orders
        'median': float,  # Median daily orders
        'std': float,  # Standard deviation
        'min': float,  # Minimum daily orders
        'max': float,  # Maximum daily orders
        'cv': float  # Coefficient of variation
    },
    'distribution': {
        'cases_skewness': float,  # Distribution skewness for cases
        'cases_kurtosis': float,  # Distribution kurtosis for cases
        'orders_skewness': float,  # Distribution skewness for orders
        'orders_kurtosis': float  # Distribution kurtosis for orders
    }
}
```

### 4. `analyze_percentiles()`

**Purpose**: Generate percentile analysis for capacity planning and operational thresholds.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'percentiles': {
        'p95': {
            'cases': float,  # 95th percentile for cases
            'orders': float,  # 95th percentile for orders
            'interpretation': str  # Business interpretation
        },
        'p90': {...},  # Similar structure for other percentiles
        'p85': {...},
        'p80': {...},
        'p75': {...}
    },
    'capacity_planning': {
        'normal_capacity': float,  # Recommended normal capacity (mean * 1.1)
        'peak_capacity': float,  # Recommended peak capacity (p95 * 1.05)
        'surge_capacity': float,  # Recommended surge capacity (max * 1.1)
        'utilization_at_normal': float,  # Utilization percentage at normal capacity
        'utilization_at_peak': float  # Utilization percentage at peak capacity
    }
}
```

### 5. `analyze_sku_frequency()`

**Purpose**: Analyze SKU ordering frequency patterns for Fast/Medium/Slow (FMS) classification.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'sku_details': DataFrame,  # SKU-level statistics with columns:
        # - Sku Code, Total_Order_Lines, Days_Ordered, Unique_Orders
        # - Total_Cases, Avg_Cases_Per_Line, Total_Eaches, Avg_Eaches_Per_Line
        # - Order_Frequency_Percent, Cumulative_Frequency_Percent, FMS_Category
    'fms_summary': DataFrame,  # Summary by FMS category with columns:
        # - FMS_Category, SKU_Count, Total_Cases, Avg_Days_Ordered, Avg_Frequency_Percent
    'total_skus': int,  # Total number of SKUs analyzed
    'analysis_period_days': int  # Number of days in analysis period
}
```

### 6. `analyze_peak_periods()`

**Purpose**: Identify peak periods and unusual volume patterns.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'peak_threshold': float,  # Threshold for peak days (mean + 1 std)
    'high_peak_threshold': float,  # Threshold for high peak days (mean + 2 std)
    'peak_days_count': int,  # Number of peak days
    'high_peak_days_count': int,  # Number of high peak days
    'peak_day_percentage': float,  # Percentage of days that are peak days
    'peak_day_patterns': dict,  # Peak days by day of week
    'avg_peak_volume': float,  # Average volume on peak days
    'max_peak_volume': float  # Maximum volume on peak days
}
```

### 7. `analyze_trends()`

**Purpose**: Analyze volume trends and growth patterns over time.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'monthly_summary': DataFrame,  # Monthly aggregated data with growth rates
    'growth_analysis': {
        'avg_monthly_cases_growth': float,  # Average monthly growth rate for cases
        'avg_monthly_orders_growth': float,  # Average monthly growth rate for orders
        'months_with_positive_growth': int,  # Count of months with positive growth
        'months_with_negative_growth': int  # Count of months with negative growth
    },
    'trend_direction': str,  # "Increasing", "Decreasing", or "Stable"
    'seasonality_detected': bool  # Whether seasonal patterns were detected
}
```

### 8. `analyze_outbound_summary_statistics()`

**Purpose**: Generate comprehensive outbound statistics with pick type breakdown for warehouse analytics.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'overall': {
        'annual_total': {'orders': int, 'lines': int, 'eaches': int, 'skus': int},
        'monthly_average': {'orders': int, 'lines': int, 'eaches': int, 'skus': int},
        'monthly_peak': {'orders': int, 'lines': int, 'eaches': int, 'skus': int},
        'daily_average': {'orders': int, 'lines': int, 'eaches': int, 'skus': int},
        'absolute_peak': {'orders': int, 'lines': int, 'eaches': int, 'skus': int},
        'design_peak': {'orders': int, 'lines': int, 'eaches': int, 'skus': int},
        'design_pa_ratios': {'orders_ratio': float, 'lines_ratio': float, 'eaches_ratio': float, 'skus_ratio': float}
    },
    'each_picks': {
        'annual_total': {'orders': int, 'lines': int, 'eaches': int},
        'daily_average': {'orders': int, 'lines': int, 'eaches': int}
    },
    'case_picks': {
        'annual_total': {'orders': int, 'lines': int, 'cases': int, 'eaches': int},
        'daily_average': {'orders': int, 'lines': int, 'cases': int, 'eaches': int}
    }
}
```

### 9. `analyze_order_profiles()`

**Purpose**: Calculate order profile statistics and key operational ratios.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'statistical_values': {
        'lines_per_order': float,  # Average lines per order
        'units_per_line': float,  # Average units per line
        'units_per_order': float,  # Average units per order
        'eaches_per_line': float,  # Average eaches per line
        'eaches_per_order': float,  # Average eaches per order
        'cases_per_line': float,  # Average cases per line
        'cases_per_order': float,  # Average cases per order
        'eaches_per_case': float,  # Ratio of eaches to cases
        'pick_lines_per_order': float,  # Average pick lines per order
        'each_lines_per_order': float,  # Average each pick lines per order
        'case_lines_per_order': float  # Average case pick lines per order
    },
    'order_level_stats': DataFrame,  # Statistical summary of order-level metrics
    'line_level_stats': DataFrame   # Statistical summary of line-level metrics
}
```

### 10. `analyze_monthly_volumes()`

**Purpose**: Analyze volume patterns by month with pick type breakdown.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'monthly_totals': {
        'overall': DataFrame,  # Monthly totals with columns: Month_Year, Orders, Lines, Eaches, SKUs
        'each_picks': DataFrame,  # Each picks monthly totals
        'case_picks': DataFrame   # Case picks monthly totals (includes Cases column)
    },
    'summary': {
        'total_months': int,  # Number of months analyzed
        'avg_monthly_orders': float,  # Average monthly orders
        'peak_month': str,  # Month with highest orders (e.g., "Jan-25")
        'peak_orders': int   # Number of orders in peak month
    }
}
```

### 11. `analyze_enhanced_weekday_patterns()`

**Purpose**: Enhanced weekday analysis with averages and pick type breakdown.

**Parameters**: None

**Returns**: Dictionary containing:
```python
{
    'existing_patterns': DataFrame,  # Original day-of-week patterns with statistics
    'weekday_averages': {
        'overall': DataFrame,  # Overall weekday averages: Day_of_Week, Orders, Lines, Eaches, SKUs
        'each_picks': DataFrame,  # Each picks weekday averages
        'case_picks': DataFrame   # Case picks weekday averages (includes Cases column)
    },
    'summary': {
        'busiest_day': str,  # Day with highest average orders
        'quietest_day': str,  # Day with lowest average orders
        'total_weeks_analyzed': float  # Number of weeks in analysis
    }
}
```

---

## Excel Generator Integration

### Call Chain in Application Flow

1. **In `app.py` (Line 470-471)**:
```python
order_analyzer = OrderAnalyzer(available_data['order_data'], analysis_config)
analysis_results['order_analysis'] = order_analyzer.run_complete_analysis()
```

2. **In `excel_generator.py` (Line 87)**:
```python
self._create_order_analysis_sheet(writer)
```

3. **In `_create_order_analysis_sheet()` method**:
The function processes the `order_analysis` results from the analysis_results dictionary and creates multiple sections in the Excel sheet.

### Data Flow Process

1. **Data Input**: Raw order data from Excel upload
2. **Analysis Execution**: `OrderAnalyzer.run_complete_analysis()` processes all data
3. **Results Storage**: Analysis results stored in `analysis_results['order_analysis']`
4. **Excel Generation**: `ExcelGenerator._create_order_analysis_sheet()` creates formatted Excel output

---

## Excel Output Format

The Order Analysis creates a comprehensive Excel sheet with sections in the following sequence:

### 1. Daily Operational Data
- **Format**: Raw daily data table
- **Columns**: Date, Daily_Orders, Daily_Shipments, Daily_SKUs, Daily_Cases, Daily_Eaches, Day_of_Week, Week_Number, Month, Cases_7Day_MA, Orders_7Day_MA
- **Data Source**: `analyze_daily_patterns()` daily_data

### 2. Enhanced Summary Statistics
- **Format**: Two-column table (Metric, Value)
- **Content**: Key metrics like average daily orders, cases, busiest day information
- **Data Source**: Multiple functions aggregated

### 3. Outbound Summary Statistics Table ✅ **UPDATED**

**Table Structure**: 12 columns with **FIXED** professional formatting
- **Row numbering**: #1-8 
- **Column Groups**: Overall (4 cols) | Eaches Only (3 cols) | Case Orders (3 cols)

| # | Description | Overall ||| Eaches Only ||| Case Orders |||
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | Overall Orders | Overall Lines | **Overall Volume** | Overall SKUs | Eaches Only Orders | Eaches Only Lines | **Each Only Volume** | Case Orders | Case Lines | **Case Volume** |
| 1 | Annual Total | X | X | **X.XX** | X | X | X | **X.XX** | X | X | **X.XX** |
| 2 | Monthly Average | X | X | **X.XX** | X | X | X | **X.XX** | X | X | **X.XX** |
| 3 | Monthly Peak Values | X | X | **X.XX** | X | X | X | **X.XX** | X | X | **X.XX** |
| 4 | Daily Average | X | X | **X.XX** | X | X | X | **X.XX** | X | X | **X.XX** |
| 6 | Absolute Peak | X | X | **X.XX** | X | X | X | **X.XX** | X | X | **X.XX** |
| 7 | Design Peak | X | X | **X.XX** | X | X | X | **X.XX** | X | X | **X.XX** |
| 8 | Design P/A Ratio | X | X | **X.XX** | X | X | X | **X.XX** | X | X | **X.XX** |

**🔥 KEY CHANGES**:
- **Volume columns**: Now show case equivalent volume (Cases + Eaches÷Case_Config)
- **Clear headers**: "Overall Volume", "Each Only Volume", "Case Volume"
- **No more confusion**: Eliminated "Each Pick Eaches", "Case Pick Eaches"
- **Professional format**: Consistent business terminology

**Data Source**: `analyze_outbound_summary_statistics()` results with case equivalent volume

### 4. Volume by Weekday - Averages Table

**Table Structure**: 12 columns showing weekly averages
- **Row numbering**: #1-7 (Sunday through Saturday)
- **Column Groups**: Overall (4 cols) | Each Picks (3 cols) | Case Picks (3 cols)

| # | Week Day | Overall ||| Each Picks ||| Case Picks |||
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | Orders | Lines | Eaches | SKUs | Orders | Lines | Each Pick Eaches | Orders | Lines | Case Pick Eaches |
| 1 | Sunday | X | X | X | X | X | X | X | X | X | X |
| 2 | Monday | X | X | X | X | X | X | X | X | X | X |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | Saturday | X | X | X | X | X | X | X | X | X | X |

**Data Source**: `analyze_enhanced_weekday_patterns()` results

### 5. Order Profiles Table

**Table Structure**: 13 columns in horizontal format
- **Single row**: Statistical Values with all key ratios

| Description | | Lines/Ord | Units/Line | Units/Ord | Ea Lines/Ord | Eaches/Line | Eaches/Ord | Cs Lns/Ord | Cases/Line | Cases/Ord | Eaches/Case | Pk Lns/Ord |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Statistical Values ==> | | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |

**Data Source**: `analyze_order_profiles()` statistical_values

### 6. Volume by Month - Totals Table

**Table Structure**: 12 columns showing monthly totals
- **Row numbering**: #7+ (continues from weekday section)
- **Column Groups**: Overall (4 cols) | Each Picks (3 cols) | Case Picks (3 cols)

| # | Month - Year | Overall ||| Each Picks ||| Case Picks |||
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | Orders | Lines | Eaches | SKUs | Orders | Lines | Each Pick Eaches | Orders | Lines | Case Pick Eaches |
| 7 | Jan-25 | X | X | X | X | X | X | X | X | X | X |
| 8 | Feb-25 | X | X | X | X | X | X | X | X | X | X |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Data Source**: `analyze_monthly_volumes()` results

### 7. Volume Analysis
- **Format**: Two-column table (Metric, Value)
- **Content**: Statistical measures for cases and orders (total, mean, median, std, min, max, CV)
- **Data Source**: `analyze_volume_statistics()` results

### 8. Percentile Analysis & Capacity Planning
- **Format**: Two-column table (Metric, Value)
- **Content**: Percentile values and capacity recommendations
- **Data Source**: `analyze_percentiles()` results

### 9. Peak Period Analysis
- **Format**: Two-column table (Metric, Value)
- **Content**: Peak period identification and patterns
- **Data Source**: `analyze_peak_periods()` results

### 10. Trend Analysis
- **Format**: Two-column table (Metric, Value)
- **Content**: Growth trends and trend quality metrics
- **Data Source**: `analyze_trends()` results

---

## Function Call Summary in Excel Generation

The `_create_order_analysis_sheet()` method in `excel_generator.py` processes the following function results:

1. **`daily_analysis`** → Daily Operational Data section + Enhanced Summary Statistics
2. **`outbound_summary`** → Outbound Summary Statistics Table (12 columns)
3. **`enhanced_weekday_patterns`** → Volume by Weekday Table (12 columns)  
4. **`order_profiles`** → Order Profiles Table (13 columns)
5. **`monthly_volumes`** → Volume by Month Table (12 columns)
6. **`volume_analysis`** → Volume Analysis section
7. **`percentile_analysis`** → Percentile Analysis & Capacity Planning section
8. **`peak_analysis`** → Peak Period Analysis section
9. **`trends`** → Trend Analysis section

Each section is professionally formatted with headers, sub-headers, and appropriate data types to create a comprehensive warehouse analysis report.