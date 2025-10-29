#!/usr/bin/env python3
"""
Receipt Analysis Charts Module

This module handles chart creation for the Receipt Analysis sheet.
"""

from openpyxl.chart import LineChart, BarChart, PieChart, Reference
from typing import Optional


class ReceiptAnalysisCharts:
    """
    Creates chart objects for Receipt Analysis data.
    """
    
    def __init__(self):
        """Initialize with default styling configurations."""
        # Define color scheme
        self.colors = {
            'blue': '4472C4',
            'orange': 'ED7D31', 
            'gray': 'A5A5A5',
            'yellow': 'FFC000',
            'green': '70AD47',
            'red': 'C55A11'
        }
        
        # Default chart dimensions
        self.default_width = 20
        self.default_height = 10
        
    def create_receipt_trend_chart(self, ws, dates_ref: Reference,
                                  lines_ref: Reference,
                                  shipments_ref: Reference,
                                  trucks_ref: Reference) -> LineChart:
        """
        Create Receipt trend chart showing Lines, Shipments, and Trucks.
        
        Args:
            ws: Worksheet object (needed for Reference objects)
            dates_ref: Reference to date values (X-axis)
            lines_ref: Reference to Daily_Receipt_Lines data (with header)
            shipments_ref: Reference to Daily_Shipments data (with header)
            trucks_ref: Reference to Daily_Trucks data (with header)
            
        Returns:
            LineChart object configured and ready to be placed
        """
        # Create line chart
        chart = LineChart()
        chart.title = "Receipts Data - Daily Trend"
        chart.style = 2  # Professional style
        chart.width = self.default_width
        chart.height = self.default_height
        
        # Add data series to chart
        chart.add_data(lines_ref, titles_from_data=True)
        chart.add_data(shipments_ref, titles_from_data=True)
        chart.add_data(trucks_ref, titles_from_data=True)
        
        # Set categories (X-axis dates)
        chart.set_categories(dates_ref)
        
        # Style the series
        # Series 0: Daily_Receipt_Lines - Orange
        if len(chart.series) > 0:
            s1 = chart.series[0]
            s1.graphicalProperties.line.solidFill = self.colors['orange']
            s1.graphicalProperties.line.width = 25000  # Line width in EMUs
            s1.smooth = False  # No smoothing for accurate data representation
            
        # Series 1: Daily_Shipments - Blue
        if len(chart.series) > 1:
            s2 = chart.series[1]
            s2.graphicalProperties.line.solidFill = self.colors['blue']
            s2.graphicalProperties.line.width = 20000
            s2.smooth = False
            
        # Series 2: Daily_Trucks - Gray
        if len(chart.series) > 2:
            s3 = chart.series[2]
            s3.graphicalProperties.line.solidFill = self.colors['gray']
            s3.graphicalProperties.line.width = 20000
            s3.smooth = False
        
        # Configure axes
        chart.y_axis.title = "Count"
        chart.x_axis.title = "Date"
        chart.x_axis.tickLblPos = "low"
        
        # Remove gridlines for cleaner look
        chart.x_axis.majorGridlines = None
        
        # Legend position
        chart.legend.position = 'b'  # Bottom
        
        return chart
    
    def create_receipt_volume_chart(self, ws, dates_ref: Reference,
                                   volume_ref: Reference) -> LineChart:
        """
        Create Receipt volume chart for Case Equivalent Volume.
        
        Args:
            ws: Worksheet object
            dates_ref: Reference to date values (X-axis)
            volume_ref: Reference to Daily_Case_Equivalent_Volume data (with header)
            
        Returns:
            LineChart object configured and ready to be placed
        """
        # Create line chart
        chart = LineChart()
        chart.title = "Daily Receipt Case Equivalent Volume"
        chart.style = 2  # Professional style
        chart.width = self.default_width
        chart.height = self.default_height
        
        # Add data series
        chart.add_data(volume_ref, titles_from_data=True)
        
        # Set categories (X-axis dates)
        chart.set_categories(dates_ref)
        
        # Style the series
        # Volume line - Green
        if len(chart.series) > 0:
            s1 = chart.series[0]
            s1.graphicalProperties.line.solidFill = self.colors['green']
            s1.graphicalProperties.line.width = 25000  # Slightly thicker line
            s1.smooth = False
        
        # Configure axes
        chart.y_axis.title = "Volume"
        chart.x_axis.title = "Date"
        chart.x_axis.tickLblPos = "low"
        
        # Remove gridlines for cleaner look
        chart.x_axis.majorGridlines = None
        
        # Hide legend for single series chart
        chart.legend = None
        
        return chart