#!/usr/bin/env python3
"""
Warehouse Analysis Tool V2 - Streamlit Application (Simplified)

A streamlined tool for warehouse data analysis with Excel-first output approach.
Upload your Excel file, configure variables, run analysis, and download comprehensive reports.
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
import traceback

# Import configuration and analysis modules
import config
from scripts import (
    DataLoader, 
    OrderAnalyzer, 
    SKUAnalyzer, 
    ABCFMSAnalyzer, 
    ReceiptAnalyzer, 
    ExcelGenerator
)

# Configure page
st.set_page_config(
    page_title="Warehouse Analysis Tool V2",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 0.3rem;
        border-left: 4px solid #1f77b4;
        margin: 0.8rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 0.8rem;
        border-radius: 0.3rem;
        border-left: 4px solid #28a745;
        margin: 0.8rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 0.8rem;
        border-radius: 0.3rem;
        border-left: 4px solid #dc3545;
        margin: 0.8rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.3rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏢 Warehouse Analysis Tool V2</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Excel-First Analysis Tool</strong><br>
        Upload your warehouse data Excel file, configure analysis parameters, and generate comprehensive Excel reports.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🏢 Warehouse Analysis")
        
        # Navigation sections
        selected_step = st.radio(
            "Analysis Steps:",
            [
                "📁 Upload Excel File",
                "⚙️ Configure Variables", 
                "🚀 Run Analysis",
                "📥 Download Results"
            ],
            key="navigation_step"
        )
        
        # Show progress indicator
        show_progress_indicator()
    
    # Main content based on selected step
    if selected_step == "📁 Upload Excel File":
        show_upload_section()
    elif selected_step == "⚙️ Configure Variables":
        show_variables_section()
    elif selected_step == "🚀 Run Analysis":
        show_analysis_section()
    elif selected_step == "📥 Download Results":
        show_download_section()

def initialize_session_state():
    """Initialize all session state variables"""
    
    # File upload state
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'file_sheets' not in st.session_state:
        st.session_state.file_sheets = {}
    if 'file_validated' not in st.session_state:
        st.session_state.file_validated = False
    
    # Variable configuration state
    if 'variables_configured' not in st.session_state:
        st.session_state.variables_configured = False
    if 'analysis_variables' not in st.session_state:
        st.session_state.analysis_variables = config.get_variable_defaults()
    
    # Analysis state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'excel_report_ready' not in st.session_state:
        st.session_state.excel_report_ready = False

def show_progress_indicator():
    """Show progress indicator in sidebar"""
    st.markdown("### 📈 Progress")
    
    progress_steps = [
        ("📁 File Upload", st.session_state.uploaded_file is not None),
        ("⚙️ Variables", st.session_state.variables_configured),
        ("🚀 Analysis", st.session_state.analysis_complete),
        ("📥 Download", st.session_state.excel_report_ready)
    ]
    
    for step_name, completed in progress_steps:
        if completed:
            st.success(f"✅ {step_name}")
        else:
            st.info(f"⏳ {step_name}")

def show_upload_section():
    """Display the Excel file upload section"""
    st.markdown('<h2 class="section-header">📁 Upload Excel File</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload your warehouse data Excel file containing the required sheets:
    - **OrderData**: Order transactions and shipment details
    - **SkuMaster**: Product information and categories  
    - **InventoryData**: Stock levels and inventory details
    - **ReceiptData**: Receipt and inbound data
    """)
    
    # Template download and file upload in columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Template download button
        st.markdown("**📄 Need a template?**")
        if st.button("📥 Download Template", type="secondary", use_container_width=True):
            template_buffer = generate_template_excel()
            st.download_button(
                label="📄 Download Excel Template",
                data=template_buffer,
                file_name="Warehouse_Data_Template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
        Template includes:<br>
        • OrderData (100 sample orders)<br>
        • SkuMaster (50 SKUs)<br>
        • ReceiptData (75 receipts)<br>
        • InventoryData (80 records)<br>
        With realistic sample data matching your format.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your warehouse Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing your warehouse data sheets",
            key="main_file_uploader"
        )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        process_uploaded_file(uploaded_file)
    
    elif st.session_state.uploaded_file is not None:
        st.success(f"✅ File uploaded: {st.session_state.uploaded_file.name}")
        show_file_summary()

def process_uploaded_file(uploaded_file):
    """Process and validate the uploaded Excel file"""
    try:
        st.info("🔄 Processing uploaded file...")
        
        # Initialize data loader
        data_loader = DataLoader(uploaded_file, verbose=False)
        
        # Load all available data
        load_results = data_loader.load_all_data()
        
        if load_results['success']:
            # Store validation info in session state
            st.session_state.file_sheets = {}
            for data_type, data_df in load_results['data'].items():
                st.session_state.file_sheets[data_type] = {
                    'rows': len(data_df),
                    'columns': list(data_df.columns)
                }
            
            st.session_state.file_validated = True
            st.success(f"✅ File processed successfully: {uploaded_file.name}")
            show_file_summary()
        else:
            st.error("❌ File processing failed:")
            for error in load_results['errors']:
                st.error(f"• {error}")
            st.session_state.file_validated = False
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.session_state.file_validated = False

def show_file_summary():
    """Display summary of uploaded file"""
    st.markdown("### 📋 File Summary")
    
    if not st.session_state.file_sheets:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Detected Sheets:**")
        for sheet_name, sheet_info in st.session_state.file_sheets.items():
            st.write(f"✅ {sheet_name.replace('_', ' ').title()}: {sheet_info['rows']} rows")
    
    with col2:
        sheet_to_preview = st.selectbox(
            "Preview sheet columns:",
            list(st.session_state.file_sheets.keys()),
            key="sheet_preview_selector",
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if sheet_to_preview:
            st.write(f"**Columns in '{sheet_to_preview.replace('_', ' ').title()}':**")
            columns = st.session_state.file_sheets[sheet_to_preview]['columns']
            for i, col in enumerate(columns[:10], 1):  # Show first 10 columns
                st.write(f"{i}. {col}")
            if len(columns) > 10:
                st.write(f"... and {len(columns) - 10} more columns")

def show_variables_section():
    """Display the variable configuration section"""
    st.markdown('<h2 class="section-header">⚙️ Configure Variables</h2>', unsafe_allow_html=True)
    
    if not st.session_state.file_validated:
        st.warning("⚠️ Please upload and validate an Excel file first.")
        return
    
    st.markdown("""
    Configure the analysis parameters for your warehouse data analysis.
    These variables will be applied across all available analysis types based on the detected sheets.
    """)
    
    # Create variable configuration form using the existing variables from app.py
    with st.form("variables_form"):
        create_variables_form()
        
        submitted = st.form_submit_button("💾 Save Configuration", type="primary")
        
        if submitted:
            save_variables_configuration()
            st.success("✅ Variables configured successfully!")
            st.session_state.variables_configured = True

def create_variables_form():
    """Create the unified variables configuration form"""
    
    # Use the existing form creation logic from the original app.py
    # Global Variables Section
    st.markdown("#### 🌐 Global Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ABC Classification Thresholds**")
        abc_a_threshold = st.slider(
            "A Items Threshold (%)",
            min_value=50, max_value=90, 
            value=st.session_state.analysis_variables['global']['abc_a_threshold'], 
            step=5,
            key="abc_a_threshold"
        )
        abc_b_threshold = st.slider(
            "B Items Threshold (%)",
            min_value=abc_a_threshold, max_value=95, 
            value=st.session_state.analysis_variables['global']['abc_b_threshold'], 
            step=5,
            key="abc_b_threshold"
        )
    
    with col2:
        st.markdown("**FMS Classification Thresholds**")
        fms_fast_threshold = st.slider(
            "Fast Items Threshold (%)",
            min_value=50, max_value=90, 
            value=st.session_state.analysis_variables['order_analysis']['fms_fast_threshold'], 
            step=5,
            key="fms_fast_threshold"
        )
        fms_medium_threshold = st.slider(
            "Medium Items Threshold (%)",
            min_value=fms_fast_threshold, max_value=95, 
            value=st.session_state.analysis_variables['order_analysis']['fms_medium_threshold'], 
            step=5,
            key="fms_medium_threshold"
        )
        
        st.markdown("**Percentile Analysis**")
        percentile_levels = st.multiselect(
            "Percentiles for Analysis",
            options=[99, 95, 90, 85, 80, 75, 70, 65, 60],
            default=st.session_state.analysis_variables['order_analysis']['percentile_levels'],
            key="percentile_levels"
        )

def save_variables_configuration():
    """Save the variables configuration to session state"""
    
    # Update analysis variables with form values
    st.session_state.analysis_variables = {
        'global': {
            'abc_a_threshold': st.session_state.get('abc_a_threshold', 70),
            'abc_b_threshold': st.session_state.get('abc_b_threshold', 90),
            'currency_symbol': '$',
            'decimal_places': 2,
            'working_days_per_week': 5,
            'working_hours_per_day': 8,
        },
        'order_analysis': {
            'fms_fast_threshold': st.session_state.get('fms_fast_threshold', 70),
            'fms_medium_threshold': st.session_state.get('fms_medium_threshold', 90),
            'percentile_levels': st.session_state.get('percentile_levels', [95, 90, 85, 80, 75]),
            'vip_customer_threshold': 50,
            'seasonal_adjustment': False,
        },
        'inventory_analysis': {
            'safety_stock_days': 7,
            'reorder_point_days': 14,
            'default_lead_time': 7,
            'storage_cost_percent': 15.0,
        },
        'manpower_analysis': {
            'target_efficiency': 85,
            'standard_pick_rate': 60,
            'shifts_per_day': 1,
            'break_time_minutes': 30,
        }
    }

def show_analysis_section():
    """Display the analysis execution section"""
    st.markdown('<h2 class="section-header">🚀 Run Analysis</h2>', unsafe_allow_html=True)
    
    # Check prerequisites
    if not st.session_state.file_validated:
        st.warning("⚠️ Please upload and validate an Excel file first.")
        return
    
    if not st.session_state.variables_configured:
        st.warning("⚠️ Please configure analysis variables first.")
        return
    
    # Show analysis readiness
    st.markdown("### ✅ Ready for Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📁 **File**: {st.session_state.uploaded_file.name}")
        st.info(f"📊 **Sheets Detected**: {len(st.session_state.file_sheets)}")
    
    with col2:
        st.info(f"⚙️ **Variables**: Configured")
        
        # Show available analyses
        available_analyses = config.validate_sheet_availability(st.session_state.file_sheets)
        available_count = sum(available_analyses.values())
        st.info(f"📈 **Available Analyses**: {available_count}")
    
    # Analysis execution
    st.markdown("### 🎯 Execute Analysis")
    
    if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
        execute_warehouse_analysis()

def execute_warehouse_analysis():
    """Execute the complete warehouse analysis using the analysis modules"""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize data loader
        status_text.text("🔄 Initializing data loader...")
        progress_bar.progress(10)
        
        data_loader = DataLoader(st.session_state.uploaded_file, verbose=False)
        load_results = data_loader.load_all_data()
        
        if not load_results['success']:
            st.error("❌ Data loading failed:")
            for error in load_results['errors']:
                st.error(f"• {error}")
            return
        
        # Step 2: Create analysis configuration
        status_text.text("⚙️ Preparing analysis configuration...")
        progress_bar.progress(20)
        
        analysis_config = config.create_analysis_config(st.session_state.analysis_variables)
        
        # Step 3: Run available analyses
        analysis_results = {'data_loader': load_results}
        current_progress = 20
        
        available_data = load_results['data']
        
        # Get SKU master data for all analyses
        sku_master = available_data.get('sku_master')
        
        # Log SKU master availability for debugging
        if sku_master is not None:
            st.info(f"✅ SKU Master available: {len(sku_master)} SKUs loaded")
        else:
            st.warning("⚠️ SKU Master not available - using default case config (1 each = 1 case)")
        
        # Order Analysis
        if 'order_data' in available_data:
            status_text.text("📈 Running order pattern analysis...")
            progress_bar.progress(30)
            
            try:
                order_analyzer = OrderAnalyzer(available_data['order_data'], sku_master, analysis_config)  # ✅ FIXED: Added sku_master
                analysis_results['order_analysis'] = order_analyzer.run_complete_analysis()
                current_progress = 40
            except Exception as e:
                st.warning(f"⚠️ Order analysis failed: {str(e)}")
                analysis_results['order_analysis'] = {'error': str(e), 'success': False}
                current_progress = 40
        
        # SKU Analysis
        if 'order_data' in available_data:
            status_text.text("🏷️ Performing SKU analysis...")
            progress_bar.progress(current_progress)
            try:
                sku_analyzer = SKUAnalyzer(available_data['order_data'], sku_master, analysis_config)
                analysis_results['sku_analysis'] = sku_analyzer.run_complete_analysis()
                current_progress += 15
            except Exception as e:
                st.warning(f"⚠️ SKU analysis failed: {str(e)}")
                analysis_results['sku_analysis'] = {'error': str(e), 'success': False}
                current_progress += 15
        
        # ABC-FMS Analysis
        if 'order_data' in available_data:
            status_text.text("🎯 Executing ABC-FMS classification...")
            progress_bar.progress(current_progress)
            
            try:
                abc_fms_analyzer = ABCFMSAnalyzer(available_data['order_data'], sku_master, analysis_config)
                analysis_results['abc_fms_analysis'] = abc_fms_analyzer.run_complete_analysis()
                current_progress += 15
            except Exception as e:
                st.warning(f"⚠️ ABC-FMS analysis failed: {str(e)}")
                analysis_results['abc_fms_analysis'] = {'error': str(e), 'success': False}
                current_progress += 15
        
        # Receipt Analysis
        if 'receipt_data' in available_data:
            status_text.text("📦 Analyzing receipt patterns...")
            progress_bar.progress(current_progress)
            
            try:
                receipt_analyzer = ReceiptAnalyzer(available_data['receipt_data'], available_data.get('order_data'), sku_master, analysis_config)
                analysis_results['receipt_analysis'] = receipt_analyzer.run_complete_analysis()
                current_progress += 10
            except Exception as e:
                st.warning(f"⚠️ Receipt analysis failed: {str(e)}")
                analysis_results['receipt_analysis'] = {'error': str(e), 'success': False}
                current_progress += 10
        
        # Step 4: Generate Excel Report
        status_text.text("📋 Generating comprehensive Excel report...")
        progress_bar.progress(85)
        
        # Check if we have any successful analyses
        successful_analyses = [k for k, v in analysis_results.items() 
                             if k != 'data_loader' and (not isinstance(v, dict) or v.get('success', True) != False)]
        
        if not successful_analyses:
            st.error("❌ No analyses completed successfully. Cannot generate Excel report.")
            st.error("Please check your data format and try again.")
            return
        
        try:
            excel_generator = ExcelGenerator(analysis_results, st.session_state.analysis_variables, analysis_config)
            excel_buffer = excel_generator.generate_comprehensive_report()
        except Exception as e:
            st.error(f"❌ Excel generation failed: {str(e)}")
            st.error("Some analyses completed but report generation failed.")
            if st.checkbox("Show detailed error information"):
                st.code(traceback.format_exc())
            return
        
        # Step 5: Complete
        status_text.text("✅ Analysis completed successfully!")
        progress_bar.progress(100)
        
        # Update session state
        st.session_state.analysis_complete = True
        st.session_state.analysis_results = {
            'timestamp': datetime.now(),
            'file_name': st.session_state.uploaded_file.name,
            'excel_report': excel_buffer,
            'summary': {
                'analyses_completed': len([k for k in analysis_results.keys() if k != 'data_loader']),
                'total_orders': available_data.get('order_data', pd.DataFrame()).shape[0] if 'order_data' in available_data else 0,
                'unique_skus': available_data.get('order_data', pd.DataFrame())['Sku Code'].nunique() if 'order_data' in available_data else 0,
                'total_volume': available_data.get('order_data', pd.DataFrame())['Qty in Cases'].sum() if 'order_data' in available_data else 0
            }
        }
        st.session_state.excel_report_ready = True
        
        st.success("🎉 Analysis completed successfully! Go to Download Results to get your report.")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.error("Please check your data format and try again.")
        if st.checkbox("Show detailed error information"):
            st.code(traceback.format_exc())

def show_download_section():
    """Display the download results section"""
    st.markdown('<h2 class="section-header">📥 Download Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_complete:
        st.info("🔄 No analysis results available. Please run the analysis first.")
        return
    
    # Show analysis summary
    results = st.session_state.analysis_results
    
    st.markdown("### 📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Analysis Date", results['timestamp'].strftime("%Y-%m-%d %H:%M"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Analyses Completed", results['summary']['analyses_completed'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Orders", f"{results['summary']['total_orders']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Unique SKUs", results['summary']['unique_skus'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download section
    st.markdown("### 📥 Download Excel Report")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Comprehensive Excel Report includes:**
        - Executive Summary with key metrics
        - Order Analysis (patterns, trends, forecasts)
        - SKU Analysis (performance, ABC classification)
        - ABC-FMS Analysis (strategic segmentation)
        - Inventory Analysis (stock levels, recommendations)
        - Receipt Analysis (inbound patterns, efficiency)
        - Consolidated Recommendations
        - Configuration Documentation
        - Raw Data Summary
        """)
    
    with col2:
        # Provide download
        filename = f"Warehouse_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        st.download_button(
            label="📥 Download Excel Report",
            data=results['excel_report'],
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
        
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
        Report: {filename}<br>
        Size: {len(results['excel_report']) / 1024:.1f} KB<br>
        Generated: {results['timestamp'].strftime("%H:%M:%S")}
        </div>
        """, unsafe_allow_html=True)

def generate_template_excel():
    """Generate template Excel file (reuse from original app.py)"""
    
    import random
    from datetime import datetime, timedelta
    
    # Create Excel buffer
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        # OrderData Sheet - exact structure
        start_date = datetime(2025, 1, 1)
        order_data = []
        
        for i in range(100):  # Generate 100 sample orders
            date = start_date + timedelta(days=random.randint(0, 60))
            order_no = 144070000 + random.randint(1000, 9999)
            shipment_no = 1008160000 + random.randint(1000, 9999)
            sku_codes = ['FXT50005W', 'FXT11005I', 'FXT20010B', 'FXT30015G', 'FXT40020R']
            sku_code = random.choice(sku_codes)
            qty_cases = random.randint(1, 50)
            qty_eaches = random.randint(0, 20)
            
            order_data.append({
                'Date': date,
                'Order No.': order_no,
                'Shipment No.': shipment_no,
                'Sku Code': sku_code,
                'Qty in Cases': qty_cases,
                'Qty in Eaches': qty_eaches
            })
        
        order_df = pd.DataFrame(order_data)
        order_df.to_excel(writer, sheet_name='OrderData', index=False)
        
        # SkuMaster Sheet - exact structure
        sku_master_data = []
        categories = ['CG', 'Food', 'Beverages', 'Personal Care', 'Household']
        
        for i in range(50):  # Generate 50 SKUs
            sku_master_data.append({
                'Category': random.choice(categories),
                'Sku Code': 12 + i,
                'Case Config': random.choice([250, 500, 1000, 1500]),
                'Pallet Fit': random.choice([8, 10, 12, 15, 20])
            })
        
        sku_master_df = pd.DataFrame(sku_master_data)
        sku_master_df.to_excel(writer, sheet_name='SkuMaster', index=False)
        
        # ReceiptData Sheet - exact structure
        receipt_data = []
        
        for i in range(75):  # Generate 75 receipt records
            receipt_date = (start_date + timedelta(days=random.randint(0, 60))).strftime('%Y-%m-%d')
            sku_id = 12 + random.randint(0, 49)
            shipment_no = f'SH{1000 + random.randint(0, 999)}'
            truck_no = f'T{random.randint(100, 999):03d}'
            batch = f'B{2025000 + random.randint(1, 999)}' if random.random() > 0.3 else None  # 30% null values
            qty_cases = random.randint(1, 10)
            qty_eaches = random.randint(0, 50)
            
            receipt_data.append({
                'Receipt Date': receipt_date,
                'SKU ID': sku_id,
                'Shipment No': shipment_no,
                'Truck No': truck_no,
                'Batch': batch,
                'Quantity in Cases': qty_cases,
                'Quantity in Eaches': qty_eaches
            })
        
        receipt_df = pd.DataFrame(receipt_data)
        receipt_df.to_excel(writer, sheet_name='ReceiptData', index=False)
        
        # InventoryData Sheet - exact structure
        inventory_data = []
        sites = ['TWBH', 'MUMBAI', 'DELHI', 'BANGALORE', 'CHENNAI']
        sku_names = [
            'AIM YELLOW 37s', 'Arrow 24s', 'AIM MAXI 60s (WM)', 'Classmate 12s',
            'Sunfeast 24s', 'Bingo 36s', 'Aashirvaad 48s', 'Fiama 60s',
            'Vivel 24s', 'Savlon 12s', 'Boost 18s', 'Horlicks 24s'
        ]
        
        for i in range(80):  # Generate 80 inventory records
            calendar_day = (start_date + timedelta(days=random.randint(120, 150))).strftime('%d.%m.%Y')
            site = random.choice(sites)
            sku_id = 9000 + random.randint(100, 999)
            sku_name = random.choice(sku_names)
            stock_cases = random.randint(0, 200)
            stock_pieces = round(random.uniform(0, 100), 1)
            
            inventory_data.append({
                'Calendar Day': calendar_day,
                'Site': site,
                'SKU ID': sku_id,
                'SKU Name': sku_name,
                'Total Stock in Cases (In Base Unit of Measure)': stock_cases,
                'Total Stock in Pieces (In Base Unit of Measue)': stock_pieces  # Note: keeping the typo as in original
            })
        
        inventory_df = pd.DataFrame(inventory_data)
        inventory_df.to_excel(writer, sheet_name='InventoryData', index=False)
    
    output.seek(0)
    return output.getvalue()

if __name__ == "__main__":
    main()