# Warehouse Analysis Tool V2 - Streamlit Application

A comprehensive warehouse data analysis tool with an intuitive web interface. Upload your Excel file, configure variables, run analysis, and download comprehensive Excel reports.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Access the Application
Open your browser and go to: `http://localhost:8501`

## 📁 Expected Data Format

The application expects an Excel file with these exact sheet names and columns:

### OrderData Sheet
- `Date` - Order date (YYYY-MM-DD format)
- `Order No.` - Order number (integer)
- `Shipment No.` - Shipment number (integer) 
- `Sku Code` - SKU identifier (text)
- `Qty in Cases` - Quantity in cases (integer)
- `Qty in Eaches` - Quantity in pieces (integer)

### SkuMaster Sheet
- `Category` - Product category (text)
- `Sku Code` - SKU identifier (integer)
- `Case Config` - Items per case (integer)
- `Pallet Fit` - Cases per pallet (integer)

### ReceiptData Sheet
- `Receipt Date` - Receipt date (YYYY-MM-DD format)
- `SKU ID` - SKU identifier (integer)
- `Shipment No` - Shipment number (text)
- `Truck No` - Truck identifier (text)
- `Batch` - Batch number (text, optional)
- `Quantity in Cases` - Received cases (integer)
- `Quantity in Eaches` - Received pieces (integer)

### InventoryData Sheet
- `Calendar Day` - Inventory date (DD.MM.YYYY format)
- `Site` - Warehouse site (text)
- `SKU ID` - SKU identifier (integer)
- `SKU Name` - SKU description (text)
- `Total Stock in Cases (In Base Unit of Measure)` - Stock in cases (integer)
- `Total Stock in Pieces (In Base Unit of Measue)` - Stock in pieces (float)

## 🎯 Features

### 📊 Analysis Types
- **Order Analysis**: Pattern analysis, trends, volume statistics
- **Inventory Analysis**: ABC classification, stock levels, reorder points
- **Manpower Analysis**: Productivity metrics, time studies, efficiency

### ⚙️ Configurable Variables
- **ABC Classification**: Customizable thresholds for A/B/C categories
- **FMS Classification**: Fast/Medium/Slow frequency analysis
- **Inventory Parameters**: Safety stock, reorder points, lead times
- **Manpower Metrics**: Efficiency targets, pick rates, shift configuration

### 📥 Excel Output
- Comprehensive multi-sheet Excel reports
- Executive summary with key metrics
- Detailed analysis for each category
- Variable configuration documentation
- Professional formatting

## 🔧 Usage Workflow

1. **📁 Upload Excel File**
   - Download template if needed
   - Upload your warehouse data Excel file
   - Verify sheet detection and structure

2. **⚙️ Configure Variables**
   - Set global parameters (date ranges, thresholds)
   - Configure analysis-specific variables
   - Save configuration

3. **🚀 Run Analysis**
   - Execute complete analysis across all detected sheets
   - Monitor progress with real-time updates
   - Wait for completion

4. **📥 Download Results**
   - Review analysis summary
   - Download comprehensive Excel report
   - Use insights for warehouse optimization

## 📄 Template Download

The application provides a template download feature that generates an Excel file with:
- Exact sheet names and column structure
- Realistic sample data (100+ records per sheet)
- Proper data types and formatting
- Ready-to-use format for your analysis

## 🛠️ Technical Requirements

- Python 3.7+
- Streamlit 1.28.0+
- pandas 2.0.0+
- openpyxl 3.1.0+
- Modern web browser

## 📈 Analysis Capabilities

- **Pattern Recognition**: Daily/weekly/monthly trends
- **Classification**: ABC-FMS cross-tabulation
- **Forecasting**: Volume and demand predictions
- **Optimization**: Inventory and manpower recommendations
- **Reporting**: Professional Excel outputs with charts

## 🔍 Troubleshooting

**File Upload Issues:**
- Ensure Excel file has exact sheet names
- Check column names match exactly (case-sensitive)
- Verify file is not corrupted

**Analysis Errors:**
- Validate data types (dates, numbers)
- Check for missing required columns
- Ensure adequate data volume

**Performance:**
- Large files may take several minutes
- Close other browser tabs
- Use date filters for faster processing

## 🚀 Future Enhancements

- Real-time dashboard visualizations
- Advanced forecasting models  
- Multi-site comparison analysis
- Automated report scheduling
- API integration capabilities

---

**Built with Streamlit for warehouse professionals** 📦