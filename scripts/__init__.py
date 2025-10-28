"""
Warehouse Analysis Tool V2 - Analysis Scripts Package

This package contains all the analysis modules for the warehouse analysis tool.
Each module performs a specific type of analysis and returns structured results.
"""

__version__ = "2.0.0"
__author__ = "Warehouse Analysis Team"

# Import all analysis modules for easy access
from .data_loader import DataLoader
from .order_analysis import OrderAnalyzer
from .sku_analysis import SKUAnalyzer
from .abc_fms_analysis import ABCFMSAnalyzer
from .receipt_analysis import ReceiptAnalyzer
from .inventory_analysis import InventoryAnalyzer
from .manpower_analysis import ManpowerAnalyzer
from .excel_generator import ExcelGenerator

__all__ = [
    'DataLoader',
    'OrderAnalyzer', 
    'SKUAnalyzer',
    'ABCFMSAnalyzer',
    'ReceiptAnalyzer',
    'InventoryAnalyzer',
    'ManpowerAnalyzer',
    'ExcelGenerator'
]