# Screenshots Required for User Manual

Please capture the following 5 screenshots from the running Streamlit app at http://localhost:8501

## Required Screenshots:

### 0. sidebar_navigation.png (NEW)
- Capture the **left sidebar** showing all 4 navigation steps:
  - 📁 Upload Excel File
  - ⚙️ Configure Variables
  - 🚀 Run Analysis
  - 📥 Download Results
- This shows users the sequential workflow

### 1. step1_upload.png
- Navigate to **"📁 Upload Excel File"** in the sidebar
- Capture the file upload interface
- Include the drag & drop area and any validation messages

### 2. step2_configure.png  
- Navigate to **"⚙️ Configure Variables"** in the sidebar
- Capture the configuration panel
- Show the Case/Each conversion factor inputs

### 3. step3_run.png
- Navigate to **"🚀 Run Analysis"** in the sidebar
- Capture the Run Analysis button and surrounding interface
- If possible, show the progress indicator

### 4. step4_download.png
- Navigate to **"📥 Download Results"** in the sidebar
- Capture the download button
- Show the generated filename format if available

## How to Capture:
1. Open http://localhost:8501 in your browser
2. Use screenshot tool:
   - **Mac**: Cmd + Shift + 4 (select area)
   - **Windows**: Win + Shift + S (snipping tool)
3. Save each screenshot with the **exact filename** listed above
4. Place all screenshots in this `manual/screenshots/` folder

## Verification:
Once screenshots are added, open `manual/USER_MANUAL.html` in your browser to verify they load correctly.

The HTML has fallback placeholders that will show if screenshots are missing.