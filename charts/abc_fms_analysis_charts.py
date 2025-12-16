#!/usr/bin/env python3
"""
ABC FMS Analysis Charts Module

Creates charts for ABC-FMS cross-classification analysis including
the stacked bar chart showing distribution of SKUs, Volume, and Lines.
"""

from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import DataPoint
from typing import Optional


class ABCFMSAnalysisCharts:
    """
    Chart creation methods for ABC-FMS Analysis.
    """
    
    def create_abc_fms_distribution_chart(self, ws, data_ref: Reference, 
                                         categories_ref: Reference) -> BarChart:
        """
        Create stacked bar chart showing ABC-FMS distribution.
        
        The chart shows three bars (SKU, Volume, Lines) with three segments each:
        - AF (A class + Fast): High-value fast movers
        - Rest: All other combinations  
        - CS (C class + Slow): Low-value slow movers
        
        Args:
            ws: Worksheet object
            data_ref: Reference to data values (3x3 grid: AF/Rest/CS × SKU/Volume/Lines)
            categories_ref: Reference to category labels (SKU, Volume, Lines)
            
        Returns:
            BarChart: Configured stacked bar chart
        """
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "stacked"
        chart.overlap = 100
        
        # Chart properties
        chart.title = "ABC-FMS Distribution Analysis"
        chart.x_axis.title = "Metrics"
        chart.y_axis.title = "Percentage (%)"
        
        # Size
        chart.width = 12
        chart.height = 8
        
        # Add data
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(categories_ref)
        
        # Style the series with appropriate colors
        # Series 0: AF (green - high performers)
        if len(chart.series) > 0:
            chart.series[0].graphicalProperties.solidFill = "90EE90"  # Light green
            
        # Series 1: Rest (gray - middle performers)
        if len(chart.series) > 1:
            chart.series[1].graphicalProperties.solidFill = "D3D3D3"  # Light gray
            
        # Series 2: CS (light blue - low performers)  
        if len(chart.series) > 2:
            chart.series[2].graphicalProperties.solidFill = "87CEEB"  # Sky blue
        
        # Format y-axis as percentage
        chart.y_axis.numFmt = '0"%"'
        chart.y_axis.majorGridlines = None
        
        # Legend position
        chart.legend.position = 'r'
        
        return chart