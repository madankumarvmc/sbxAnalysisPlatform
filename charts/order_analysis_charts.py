#!/usr/bin/env python3
"""
Order Analysis Charts Module

This module handles chart creation for the Order Analysis sheet.
Charts are created and returned, with positioning handled by the caller.
"""

from openpyxl.chart import LineChart, BarChart, PieChart, Reference
from typing import Optional


class OrderAnalysisCharts:
    """
    Creates chart objects for Order Analysis data.
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
        
    def create_daily_trend_chart(self, ws, dates_ref: Reference,
                                lines_ref: Reference, 
                                customers_ref: Reference,
                                shipments_ref: Reference) -> LineChart:
        """
        Create Order Profile trend chart showing Lines, Customers, and Shipments.
        
        Args:
            ws: Worksheet object (needed for Reference objects)
            dates_ref: Reference to date values (X-axis)
            lines_ref: Reference to #Lines data (with header)
            customers_ref: Reference to #Customer data (with header)
            shipments_ref: Reference to #Shipments data (with header)
            
        Returns:
            LineChart object configured and ready to be placed
        """
        # Create line chart
        chart = LineChart()
        chart.title = "Order Profile - Daily Trend"
        chart.style = 2  # Professional style
        chart.width = self.default_width
        chart.height = self.default_height
        
        # Add data series to chart
        chart.add_data(lines_ref, titles_from_data=True)
        chart.add_data(customers_ref, titles_from_data=True)
        chart.add_data(shipments_ref, titles_from_data=True)
        
        # Set categories (X-axis dates)
        chart.set_categories(dates_ref)
        
        # Style the series
        # Series 0: Daily_Orders - Blue
        if len(chart.series) > 0:
            s1 = chart.series[0]
            s1.graphicalProperties.line.solidFill = self.colors['blue']
            s1.graphicalProperties.line.width = 20000  # Line width in EMUs
            s1.smooth = False  # No smoothing for accurate data representation
            
        # Series 1: Daily_Shipments - Orange  
        if len(chart.series) > 1:
            s2 = chart.series[1]
            s2.graphicalProperties.line.solidFill = self.colors['orange']
            s2.graphicalProperties.line.width = 20000
            s2.smooth = False
            
        # Series 2: Daily_SKUs - Gray
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
    
    def create_volume_trend_chart(self, ws, dates_ref: Reference,
                                 lines_ref: Reference,
                                 volume_ref: Reference) -> LineChart:
        """
        Create Daily Order Lines & Case Equivalent Volume trend chart.
        
        Args:
            ws: Worksheet object (needed for Reference objects)
            dates_ref: Reference to date values (X-axis)
            lines_ref: Reference to Daily_Order_Lines data (with header)
            volume_ref: Reference to Daily_Case_Equivalent_Volume data (with header)
            
        Returns:
            LineChart object configured and ready to be placed
        """
        # Create line chart
        chart = LineChart()
        chart.title = "Daily Order Lines & Case Equivalent Volume"
        chart.style = 2  # Professional style
        chart.width = self.default_width
        chart.height = self.default_height
        
        # Add data series
        chart.add_data(lines_ref, titles_from_data=True)
        chart.add_data(volume_ref, titles_from_data=True)
        
        # Set categories (X-axis dates)
        chart.set_categories(dates_ref)
        
        # Style the series
        # Order Lines - Blue
        if len(chart.series) > 0:
            s1 = chart.series[0]
            s1.graphicalProperties.line.solidFill = self.colors['blue']
            s1.graphicalProperties.line.width = 25000  # Slightly thicker line
            s1.smooth = False
            
        # Volume line - Green
        if len(chart.series) > 1:
            s2 = chart.series[1]
            s2.graphicalProperties.line.solidFill = self.colors['green']
            s2.graphicalProperties.line.width = 25000  # Slightly thicker line
            s2.smooth = False
        
        # Configure axes
        chart.y_axis.title = "Count / Volume"
        chart.x_axis.title = "Date"
        chart.x_axis.tickLblPos = "low"
        
        # Remove gridlines for cleaner look
        chart.x_axis.majorGridlines = None
        
        # Show legend for two series chart
        chart.legend.position = 'b'  # Bottom
        
        return chart
    
    def create_percentile_chart(self, ws, categories_ref: Reference,
                              data_refs: list) -> BarChart:
        """
        Create column chart for percentile analysis.
        
        Args:
            ws: Worksheet object
            categories_ref: Reference to percentile levels
            data_refs: List of References to data series
            
        Returns:
            BarChart object configured as column chart
        """
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "Order Volume Percentiles"
        chart.width = 15
        chart.height = 10
        
        # Add data series
        for data_ref in data_refs:
            chart.add_data(data_ref, titles_from_data=True)
        
        chart.set_categories(categories_ref)
        
        # Configure axes
        chart.y_axis.title = "Volume"
        chart.x_axis.title = "Percentile"
        
        return chart
    
    def create_top_skus_chart(self, ws, categories_ref: Reference,
                             data_ref: Reference, num_skus: int = 10) -> BarChart:
        """
        Create horizontal bar chart for top SKUs.
        
        Args:
            ws: Worksheet object
            categories_ref: Reference to SKU names
            data_ref: Reference to volume data (with header)
            num_skus: Number of top SKUs being shown
            
        Returns:
            BarChart object configured as horizontal bar chart
        """
        chart = BarChart()
        chart.type = "bar"  # Horizontal bar
        chart.style = 10
        chart.title = f"Top {num_skus} SKUs by Volume"
        chart.width = 15
        chart.height = 10
        
        # Add data
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(categories_ref)
        
        # Configure axes
        chart.x_axis.title = "Volume"
        chart.y_axis.title = "SKU"
        
        # Color bars
        if len(chart.series) > 0:
            chart.series[0].graphicalProperties.solidFill = self.colors['blue']
        
        return chart