#!/usr/bin/env python3
"""
Excel Chart Generator - Central coordinator for chart placement in Excel workbooks

This module handles the placement of charts in Excel workbooks, managing
the creation of References and positioning of charts based on table locations.
"""

from openpyxl.chart import Reference
from openpyxl.utils import get_column_letter
from typing import Dict, Optional
import pandas as pd

# Import all chart modules
from .order_analysis_charts import OrderAnalysisCharts
from .receipt_analysis_charts import ReceiptAnalysisCharts
# from .sku_analysis_charts import SKUAnalysisCharts
# from .abc_fms_analysis_charts import ABCFMSAnalysisCharts
# from .inventory_analysis_charts import InventoryAnalysisCharts
# from .manpower_analysis_charts import ManpowerAnalysisCharts


class ExcelChartGenerator:
    """
    Coordinates chart placement in Excel workbooks.
    Handles Reference creation and positioning logic.
    """
    
    def __init__(self, worksheet):
        """
        Initialize with a worksheet object.
        
        Args:
            worksheet: openpyxl worksheet object
        """
        self.ws = worksheet
        
        # Initialize all chart modules
        self.order_charts = OrderAnalysisCharts()
        self.receipt_charts = ReceiptAnalysisCharts()
        
    def add_order_daily_trend_chart(self, table_position: Dict, columns_gap: int = 2) -> bool:
        """
        Add Order Profile daily trend chart to the right of the daily data table.
        
        Args:
            table_position: Dictionary with table location info:
                - 'row': Starting row of table (1-based Excel indexing)
                - 'col': Starting column of table (1-based)
                - 'num_rows': Number of data rows (excluding header)
                - 'num_cols': Number of columns in table
            columns_gap: Number of columns to leave between table and chart (default 2)
            
        Returns:
            bool: True if chart was added successfully, False otherwise
        """
        try:
            # Extract position info
            start_row = table_position['row']
            start_col = table_position['col']
            num_rows = table_position['num_rows']
            
            # Calculate rows
            header_row = start_row
            first_data_row = start_row + 1
            last_data_row = start_row + num_rows
            
            # Create References for the data
            # Map to actual columns: Date, Daily_Order_Lines, Daily_Orders, Daily_Shipments, Daily_SKUs
            
            # Dates (column A, no header)
            dates_ref = Reference(self.ws, 
                                 min_col=start_col, 
                                 min_row=first_data_row, 
                                 max_row=last_data_row)
            
            # Daily_Orders (column C, with header for series name)
            lines_ref = Reference(self.ws, 
                                min_col=start_col + 2,
                                min_row=header_row, 
                                max_row=last_data_row)
            
            # Daily_Shipments (column D, with header)
            customers_ref = Reference(self.ws, 
                                    min_col=start_col + 3,
                                    min_row=header_row, 
                                    max_row=last_data_row)
            
            # Daily_SKUs (column E, with header)
            shipments_ref = Reference(self.ws, 
                                    min_col=start_col + 4,
                                    min_row=header_row, 
                                    max_row=last_data_row)
            
            # Get chart from chart module
            chart = self.order_charts.create_daily_trend_chart(
                ws=self.ws,
                dates_ref=dates_ref,
                lines_ref=lines_ref,
                customers_ref=customers_ref,
                shipments_ref=shipments_ref
            )
            
            # Calculate chart position (to the right of table)
            chart_col = start_col + table_position['num_cols'] + columns_gap
            chart_position = f"{get_column_letter(chart_col)}{start_row}"
            
            # Add chart to worksheet
            self.ws.add_chart(chart, chart_position)
            
            print(f"✅ Order trend chart added at {chart_position}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not add order trend chart: {str(e)}")
            return False
    
    def add_order_volume_trend_chart(self, table_position: Dict, columns_gap: int = 2, 
                                   row_gap: int = 22) -> bool:
        """
        Add Daily Case Equivalent Volume trend chart below the main trend chart.
        
        Args:
            table_position: Dictionary with table location info
            columns_gap: Number of columns to leave between table and chart (default 2)
            row_gap: Number of rows below the first chart (default 12)
            
        Returns:
            bool: True if chart was added successfully, False otherwise
        """
        try:
            # Extract position info
            start_row = table_position['row']
            start_col = table_position['col']
            num_rows = table_position['num_rows']
            
            # Calculate rows
            header_row = start_row
            first_data_row = start_row + 1
            last_data_row = start_row + num_rows
            
            # Create References for the data
            # Dates (column A, no header)
            dates_ref = Reference(self.ws, 
                                 min_col=start_col, 
                                 min_row=first_data_row, 
                                 max_row=last_data_row)
            
            # Daily_Order_Lines (column B, with header)
            lines_ref = Reference(self.ws, 
                                min_col=start_col + 1,  # Column B
                                min_row=header_row, 
                                max_row=last_data_row)
            
            # Daily_Case_Equivalent_Volume (column H, with header)
            volume_ref = Reference(self.ws, 
                                  min_col=start_col + 7,  # Column H (8th column)
                                  min_row=header_row, 
                                  max_row=last_data_row)
            
            # Get chart from chart module
            chart = self.order_charts.create_volume_trend_chart(
                ws=self.ws,
                dates_ref=dates_ref,
                lines_ref=lines_ref,
                volume_ref=volume_ref
            )
            
            # Calculate chart position (below the first chart)
            chart_col = start_col + table_position['num_cols'] + columns_gap
            chart_row = start_row + row_gap  # Position below first chart
            chart_position = f"{get_column_letter(chart_col)}{chart_row}"
            
            # Add chart to worksheet
            self.ws.add_chart(chart, chart_position)
            
            print(f"✅ Order volume chart added at {chart_position}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not add order volume chart: {str(e)}")
            return False
    
    def add_order_percentile_chart(self, table_position: Dict, columns_gap: int = 2) -> bool:
        """
        Add percentile analysis column chart.
        
        Args:
            table_position: Dictionary with table location info
            columns_gap: Gap between table and chart
            
        Returns:
            bool: Success status
        """
        # To be implemented when needed
        pass
    
    def add_receipt_daily_trend_chart(self, table_position: Dict, columns_gap: int = 2) -> bool:
        """
        Add Receipt Profile daily trend chart.
        
        Args:
            table_position: Dictionary with table location info
            columns_gap: Gap between table and chart
            
        Returns:
            bool: Success status
        """
        try:
            # Extract position info
            start_row = table_position['row']
            start_col = table_position['col']
            num_rows = table_position['num_rows']
            
            # Calculate rows
            header_row = start_row
            first_data_row = start_row + 1
            last_data_row = start_row + num_rows
            
            # Create References for the data
            # Map to actual columns: Date, Daily_Receipt_Lines, Daily_SKUs, Daily_Shipments, Daily_Trucks
            
            # Dates (column A, no header)
            dates_ref = Reference(self.ws, 
                                 min_col=start_col, 
                                 min_row=first_data_row, 
                                 max_row=last_data_row)
            
            # Daily_Receipt_Lines (column B, with header for series name)
            lines_ref = Reference(self.ws, 
                                min_col=start_col + 1,
                                min_row=header_row, 
                                max_row=last_data_row)
            
            # Daily_Shipments (column D, with header)
            shipments_ref = Reference(self.ws, 
                                    min_col=start_col + 3,
                                    min_row=header_row, 
                                    max_row=last_data_row)
            
            # Daily_Trucks (column E, with header)
            trucks_ref = Reference(self.ws, 
                                 min_col=start_col + 4,
                                 min_row=header_row, 
                                 max_row=last_data_row)
            
            # Get chart from chart module
            chart = self.receipt_charts.create_receipt_trend_chart(
                ws=self.ws,
                dates_ref=dates_ref,
                lines_ref=lines_ref,
                shipments_ref=shipments_ref,
                trucks_ref=trucks_ref
            )
            
            # Calculate chart position (to the right of table)
            chart_col = start_col + table_position['num_cols'] + columns_gap
            chart_position = f"{get_column_letter(chart_col)}{start_row}"
            
            # Add chart to worksheet
            self.ws.add_chart(chart, chart_position)
            
            print(f"✅ Receipt trend chart added at {chart_position}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not add receipt trend chart: {str(e)}")
            return False
    
    def add_receipt_volume_trend_chart(self, table_position: Dict, columns_gap: int = 2,
                                     row_gap: int = 22) -> bool:
        """
        Add Receipt Volume trend chart below the main receipt chart.
        
        Args:
            table_position: Dictionary with table location info
            columns_gap: Number of columns to leave between table and chart (default 2)
            row_gap: Number of rows below the first chart (default 22)
            
        Returns:
            bool: True if chart was added successfully, False otherwise
        """
        try:
            # Extract position info
            start_row = table_position['row']
            start_col = table_position['col']
            num_rows = table_position['num_rows']
            
            # Calculate rows
            header_row = start_row
            first_data_row = start_row + 1
            last_data_row = start_row + num_rows
            
            # Create References for the data
            # Dates (column A, no header)
            dates_ref = Reference(self.ws, 
                                 min_col=start_col, 
                                 min_row=first_data_row, 
                                 max_row=last_data_row)
            
            # Daily_Case_Equivalent_Volume (column H, with header)
            volume_ref = Reference(self.ws, 
                                  min_col=start_col + 7,  # Column H (8th column)
                                  min_row=header_row, 
                                  max_row=last_data_row)
            
            # Get chart from chart module
            chart = self.receipt_charts.create_receipt_volume_chart(
                ws=self.ws,
                dates_ref=dates_ref,
                volume_ref=volume_ref
            )
            
            # Calculate chart position (below the first chart)
            chart_col = start_col + table_position['num_cols'] + columns_gap
            chart_row = start_row + row_gap  # Position below first chart
            chart_position = f"{get_column_letter(chart_col)}{chart_row}"
            
            # Add chart to worksheet
            self.ws.add_chart(chart, chart_position)
            
            print(f"✅ Receipt volume chart added at {chart_position}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not add receipt volume chart: {str(e)}")
            return False
    
    # Helper methods
    def _create_reference(self, col: int, start_row: int, end_row: int, 
                         include_header: bool = False) -> Reference:
        """
        Helper method to create a Reference object.
        
        Args:
            col: Column number (1-based)
            start_row: Starting row (1-based)
            end_row: Ending row (1-based)
            include_header: Whether to include header row
            
        Returns:
            Reference object
        """
        if include_header:
            start_row -= 1
            
        return Reference(self.ws, 
                        min_col=col, 
                        min_row=start_row, 
                        max_row=end_row)
    
    def _calculate_chart_position(self, table_position: Dict, 
                                 placement: str = 'right',
                                 gap: int = 2) -> str:
        """
        Calculate where to place a chart relative to a table.
        
        Args:
            table_position: Table location dictionary
            placement: 'right', 'bottom', or 'below_right'
            gap: Number of rows/columns gap
            
        Returns:
            Cell reference string (e.g., 'J2')
        """
        if placement == 'right':
            col = table_position['col'] + table_position['num_cols'] + gap
            row = table_position['row']
        elif placement == 'bottom':
            col = table_position['col']
            row = table_position['row'] + table_position['num_rows'] + gap + 1
        elif placement == 'below_right':
            col = table_position['col'] + table_position['num_cols'] + gap
            row = table_position['row'] + table_position['num_rows'] + gap + 1
        else:
            # Default to right
            col = table_position['col'] + table_position['num_cols'] + gap
            row = table_position['row']
            
        return f"{get_column_letter(col)}{row}"