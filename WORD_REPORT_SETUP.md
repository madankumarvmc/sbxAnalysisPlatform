# MS Word Report Setup Guide

## Overview
The MS Word Report feature generates professional reports with AI-powered insights using Google Gemini. The system works with or without AI integration.

## Quick Start

### 1. Install Dependencies
```bash
pip install python-docx matplotlib google-generativeai python-dotenv
```

### 2. Configure Gemini API (Optional)
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```
3. Set environment variable:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```

### 3. Use the Feature
1. Run your analysis in the Streamlit app
2. Download Excel report first
3. Click "Generate Word Report" 
4. Download the generated .docx file

## Features

### With Gemini API
- ✅ AI-generated executive summary
- ✅ Intelligent analysis of receipt patterns
- ✅ Operational insights and recommendations
- ✅ Context-aware commentary on data

### Without Gemini API (Fallback Mode)
- ✅ Professional document structure
- ✅ All data tables included
- ✅ Charts and visualizations
- ✅ Predefined analysis text

## Current Report Sections

1. **Title Page** - Professional cover with metadata
2. **Executive Summary** - Key insights and recommendations
3. **Receipt Analysis** - Daily patterns, percentiles, and insights

## Architecture

```
scripts/word_report/
├── __init__.py              # Package exports
├── word_generator.py        # Main report generator
├── word_prompts.py         # AI prompt templates
├── gemini_client.py        # Google Gemini API client
└── chart_recreator.py      # Matplotlib chart generation
```

## Key Benefits

- **Flexible**: Works with or without AI
- **Professional**: Word default styling
- **Comprehensive**: Tables, charts, and insights
- **Secure**: Environment variable API key storage
- **Reliable**: Fallback text for robustness

## Troubleshooting

### Missing Dependencies
```bash
pip install python-docx google-generativeai matplotlib python-dotenv
```

### API Key Issues
- Check `.env` file exists and contains valid key
- Restart application after setting environment variables
- Reports work without API key using fallback text

### Import Errors
- Ensure all dependencies are installed
- Check Python path includes project directory

## Extending the System

### Adding New Sections
1. Add prompts to `word_prompts.py`
2. Create section method in `word_generator.py`
3. Call method in `generate_report()`

### Customizing Charts
- Modify `chart_recreator.py`
- Add new chart types as needed
- Charts automatically embedded in Word

### Prompt Engineering
- Edit prompts in `word_prompts.py`
- Test with different analysis data
- Adjust for specific business context

## File Outputs

- **Excel Report**: `.xlsx` with all analysis data
- **Word Report**: `.docx` with insights and visuals
- Both reports complement each other for complete analysis

## Next Steps

Ready to expand with:
- Order Analysis section
- SKU Performance analysis  
- Inventory recommendations
- Manpower planning insights
- Custom business metrics