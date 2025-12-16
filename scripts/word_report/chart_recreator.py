"""
Chart Recreation Module for Word Report Generation

This module creates charts from analysis data using matplotlib,
optimized for embedding in Word documents.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import pandas as pd
import numpy as np
import io
from datetime import datetime
from typing import Optional, Tuple, Any

# Set default style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')

class ChartCreator:
    """Create charts for Word report embedding."""
    
    def __init__(self, figsize: Tuple[float, float] = (10, 6), dpi: int = 150):
        """
        Initialize chart creator with default settings.
        
        Args:
            figsize: Default figure size (width, height) in inches
            dpi: Dots per inch for chart resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def create_receipt_volume_chart(self, daily_data: pd.DataFrame) -> io.BytesIO:
        """
        Create daily receipt volume trend chart.
        
        Args:
            daily_data: DataFrame with receipt data (should have date and volume columns)
        
        Returns:
            BytesIO buffer containing the chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Identify date and volume columns
        date_col = None
        volume_col = None
        
        for col in daily_data.columns:
            if 'date' in col.lower():
                date_col = col
            elif 'case' in col.lower() or 'qty' in col.lower() or 'volume' in col.lower():
                volume_col = col
        
        if not date_col or not volume_col:
            # Fallback to first two columns
            date_col = daily_data.columns[0]
            volume_col = daily_data.columns[1]
        
        # Convert dates if needed
        try:
            dates = pd.to_datetime(daily_data[date_col])
        except:
            dates = range(len(daily_data))
        
        volumes = daily_data[volume_col]
        
        # Create the plot
        ax.plot(dates, volumes, marker='o', linewidth=2, markersize=6, 
                color=self.color_palette[0], label='Daily Volume')
        
        # Add average line
        avg_volume = volumes.mean()
        ax.axhline(y=avg_volume, color=self.color_palette[1], linestyle='--', 
                   alpha=0.7, label=f'Average: {avg_volume:.0f}')
        
        # Formatting
        ax.set_title('Daily Receipt Volume Trend', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volume (Cases)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Format x-axis dates if applicable
        if isinstance(dates.iloc[0] if hasattr(dates, 'iloc') else dates[0], pd.Timestamp):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer
    
    def create_order_volume_chart(self, daily_data: pd.DataFrame) -> io.BytesIO:
        """
        Create daily order volume trend chart.
        
        Args:
            daily_data: DataFrame with order data
        
        Returns:
            BytesIO buffer containing the chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Similar logic to receipt chart
        date_col = None
        volume_col = None
        
        for col in daily_data.columns:
            if 'date' in col.lower():
                date_col = col
            elif 'order' in col.lower() or 'qty' in col.lower() or 'volume' in col.lower():
                volume_col = col
        
        if not date_col or not volume_col:
            date_col = daily_data.columns[0]
            volume_col = daily_data.columns[1]
        
        try:
            dates = pd.to_datetime(daily_data[date_col])
        except:
            dates = range(len(daily_data))
        
        volumes = daily_data[volume_col]
        
        # Create bar chart for orders
        ax.bar(dates, volumes, color=self.color_palette[2], alpha=0.7, label='Daily Orders')
        
        # Add trend line
        z = np.polyfit(range(len(volumes)), volumes, 1)
        p = np.poly1d(z)
        ax.plot(dates, p(range(len(volumes))), color=self.color_palette[3], 
                linestyle='--', linewidth=2, label='Trend')
        
        # Formatting
        ax.set_title('Daily Order Volume Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Order Volume', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right')
        
        if isinstance(dates.iloc[0] if hasattr(dates, 'iloc') else dates[0], pd.Timestamp):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer
    
    def create_abc_distribution_chart(self, abc_data: dict) -> io.BytesIO:
        """
        Create ABC classification distribution pie chart.
        
        Args:
            abc_data: Dictionary with A, B, C percentages or counts
        
        Returns:
            BytesIO buffer containing the chart image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Prepare data
        categories = ['A Items', 'B Items', 'C Items']
        
        # Extract values (handle different data structures)
        if isinstance(abc_data, dict):
            values = []
            counts = []
            for cat in ['A', 'B', 'C']:
                if cat in abc_data:
                    if isinstance(abc_data[cat], dict):
                        values.append(abc_data[cat].get('percentage', 0))
                        counts.append(abc_data[cat].get('count', 0))
                    else:
                        values.append(float(abc_data[cat]))
                        counts.append(0)
                else:
                    values.append(0)
                    counts.append(0)
        else:
            # Fallback sample data
            values = [20, 30, 50]
            counts = [100, 150, 250]
        
        # Pie chart for percentages
        colors = [self.color_palette[0], self.color_palette[1], self.color_palette[2]]
        wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
        ax1.set_title('ABC Classification by Value', fontsize=12, fontweight='bold')
        
        # Bar chart for counts
        if sum(counts) > 0:
            ax2.bar(categories, counts, color=colors)
            ax2.set_title('ABC Classification by Count', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of SKUs')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.bar(categories, values, color=colors)
            ax2.set_title('ABC Classification Distribution', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Percentage')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer
    
    def create_percentile_chart(self, percentile_data: pd.DataFrame, title: str = "Percentile Analysis") -> io.BytesIO:
        """
        Create percentile analysis chart.
        
        Args:
            percentile_data: DataFrame with percentile levels and values
            title: Chart title
        
        Returns:
            BytesIO buffer containing the chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract percentile levels and values
        if 'Percentile' in percentile_data.columns:
            percentiles = percentile_data['Percentile']
            values = percentile_data.iloc[:, 1]  # Second column as values
        else:
            percentiles = percentile_data.index
            values = percentile_data.iloc[:, 0]
        
        # Create step chart
        ax.step(percentiles, values, where='mid', linewidth=2, 
                color=self.color_palette[4], label='Volume at Percentile')
        ax.fill_between(percentiles, values, step='mid', alpha=0.3, 
                        color=self.color_palette[4])
        
        # Add markers for key percentiles
        key_percentiles = [50, 75, 90, 95]
        for kp in key_percentiles:
            if kp in percentiles.values:
                idx = list(percentiles.values).index(kp)
                ax.plot(kp, values.iloc[idx], 'ro', markersize=8)
                ax.annotate(f'P{kp}: {values.iloc[idx]:.0f}', 
                           xy=(kp, values.iloc[idx]), 
                           xytext=(5, 5), textcoords='offset points')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Percentile', fontsize=12)
        ax.set_ylabel('Volume', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer
    
    def create_inventory_turnover_chart(self, inventory_data: pd.DataFrame) -> io.BytesIO:
        """
        Create inventory turnover analysis chart.
        
        Args:
            inventory_data: DataFrame with inventory metrics
        
        Returns:
            BytesIO buffer containing the chart image
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Sample data structure handling
        if 'SKU' in inventory_data.columns and 'Turnover' in inventory_data.columns:
            # Top 10 SKUs by turnover
            top_skus = inventory_data.nlargest(10, 'Turnover')
            
            ax1.barh(range(len(top_skus)), top_skus['Turnover'].values, 
                    color=self.color_palette[5])
            ax1.set_yticks(range(len(top_skus)))
            ax1.set_yticklabels(top_skus['SKU'].values)
            ax1.set_title('Top 10 SKUs by Turnover Rate', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Turnover Rate')
            ax1.grid(True, alpha=0.3, axis='x')
        
        # Inventory value distribution
        if 'Value' in inventory_data.columns:
            ax2.hist(inventory_data['Value'], bins=20, color=self.color_palette[6], 
                    alpha=0.7, edgecolor='black')
            ax2.set_title('Inventory Value Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Inventory Value ($)')
            ax2.set_ylabel('Number of SKUs')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer
    
    def create_manpower_chart(self, manpower_data: dict) -> io.BytesIO:
        """
        Create manpower requirement visualization.
        
        Args:
            manpower_data: Dictionary with manpower metrics
        
        Returns:
            BytesIO buffer containing the chart image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Staffing requirements by function
        if 'by_function' in manpower_data:
            functions = list(manpower_data['by_function'].keys())
            requirements = list(manpower_data['by_function'].values())
            
            ax1.bar(functions, requirements, color=self.color_palette[:len(functions)])
            ax1.set_title('Staffing Requirements by Function', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Staff')
            ax1.set_xlabel('Function')
            ax1.grid(True, alpha=0.3, axis='y')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Efficiency comparison
        if 'efficiency' in manpower_data:
            categories = ['Current', 'Target', 'Best Practice']
            values = [
                manpower_data['efficiency'].get('current', 70),
                manpower_data['efficiency'].get('target', 85),
                manpower_data['efficiency'].get('best_practice', 95)
            ]
            
            colors = [self.color_palette[7], self.color_palette[8], self.color_palette[9]]
            bars = ax2.bar(categories, values, color=colors)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val}%', ha='center', va='bottom')
            
            ax2.set_title('Efficiency Comparison', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Efficiency (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        return buffer


# Standalone functions for direct use
def create_receipt_volume_chart(daily_data: pd.DataFrame) -> io.BytesIO:
    """Create receipt volume chart using default settings."""
    creator = ChartCreator()
    return creator.create_receipt_volume_chart(daily_data)

def create_order_volume_chart(daily_data: pd.DataFrame) -> io.BytesIO:
    """Create order volume chart using default settings."""
    creator = ChartCreator()
    return creator.create_order_volume_chart(daily_data)

def create_abc_distribution_chart(abc_data: dict) -> io.BytesIO:
    """Create ABC distribution chart using default settings."""
    creator = ChartCreator()
    return creator.create_abc_distribution_chart(abc_data)

def create_percentile_chart(percentile_data: pd.DataFrame, title: str = "Percentile Analysis") -> io.BytesIO:
    """Create percentile chart using default settings."""
    creator = ChartCreator()
    return creator.create_percentile_chart(percentile_data, title)