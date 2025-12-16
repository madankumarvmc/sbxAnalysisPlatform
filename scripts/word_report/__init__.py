"""
Word Report Generation Module for Warehouse Analysis Tool

This module provides functionality to generate MS Word reports with:
- LLM-powered insights using Google Gemini
- Tables and charts from analysis results
- Professional document formatting
"""

from .word_generator import WordReportGenerator
from .gemini_client import GeminiClient

__all__ = ['WordReportGenerator', 'GeminiClient']