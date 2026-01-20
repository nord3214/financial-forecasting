import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

class FinancialEDA:
    def __init__(self, df, output_path="plots/eda/"):
        self.df = df
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Define class-wide constants for styling
        self.primary_color = '#1a4a73'  # Professional Navy
        self.secondary_color = '#808080' # Professional Grey
        self.font_family = 'serif'       # Matches LaTeX/Thesis body text
        
        # Apply global Matplotlib settings
        self._apply_global_style()

    def _apply_global_style(self):
        """Sets the visual standard for all plots in the thesis."""
        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.spines.top': False,    
            'axes.spines.right': False,  
            'figure.titlesize': 14,
            'savefig.dpi': 300,        
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight'   
        })

    def plot_time_series(self, column, title, unit="EUR"):
        """Creates a standardized time series plot."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(self.df.index, self.df[column], 
                color=self.primary_color, 
                linewidth=1.5, 
                label=column)
        
        ax.set_title(title, pad=20)
        ax.set_ylabel(f"{column} ({unit})")
        ax.set_xlabel("Date")
        
        # Save using the standardized path and format
        save_path = os.path.join(self.output_path, f"{column}_timeseries.pdf")
        plt.savefig(save_path)
        plt.close()

    def check_adfuller(self, column):
        result = adfuller(self.df[column].dropna())
        print(f"ADF Statistic for {column}: {result[0]}")
        print(f"p-value: {result[1]}")
        return result[1]

    def plot_correlation(self, feature_cols):
        plt.figure(figsize=(10, 8))
        corr = self.df[feature_cols].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title("Feature Correlation Heatmap")
        plt.savefig(f"{self.output_path}correlation_matrix.pdf")
        plt.close()

    def plot_distribution(self, column, range=None):
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[column].dropna(), kde=True, bins=100, binrange=range)
        plt.xlim(range)
        plt.title(f"Distribution of {column}")
        plt.savefig(f"{self.output_path}{column}_dist.pdf")
        plt.close()