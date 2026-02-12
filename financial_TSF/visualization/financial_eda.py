from time import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates

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

    def plot_time_series(self, column, title, unit: str | None ="USD"):
        """Creates a standardized time series plot."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(self.df.index, self.df[column], 
                color=self.primary_color, 
                linewidth=1.5, 
                label=column)
        
        ax.set_title(title, pad=20)
        if unit:
            unit = f" ({unit})"
        else:
            unit = ""
        ax.set_ylabel(f"{column}{unit}")
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
        data = self.df[column].dropna()
        
        sns.histplot(data, kde=True, bins=100, binrange=range, stat="density", alpha=0.4)
        plt.xlim(range) if range else None
        
        #calc dist params
        mu, std = norm.fit(data) 
        
        if range:
            xmin, xmax = range
        else:
            xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        
        p = norm.pdf(x, mu, std)
        
        plt.plot(x, p, 'r', linewidth=2, label=f'Normal ($\mu={mu:.2e}, \sigma={std:.2e}$)')
        
        plt.legend()
        plt.title(f"Distribution of {column}")
        plt.savefig(f"{self.output_path}{column}_dist.pdf")
        plt.close()


    def plot_split_points(self, df_data: list[pd.DataFrame], val_start_timestamp: pd.Timestamp, test_start_timestamp: pd.Timestamp):
        # put in df
        tickers = [df['ticker'].iloc[0] for df in df_data]
        start_dates = [df.index.min() for df in df_data]
        end_dates = [df.index.max() for df in df_data]
        
        df = pd.DataFrame({
            'ticker': tickers, 
            'start_date': start_dates, 
            'end_date': end_dates
        })
        

        df_plot = df.sort_values(by='start_date', ascending=False).head(30).copy()
        
        df_plot.reset_index(drop=True, inplace=True)

        global_end = max(end_dates)
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, row in df_plot.iterrows():
            s = mdates.date2num(row['start_date'])
            v = mdates.date2num(val_start_timestamp)
            t = mdates.date2num(test_start_timestamp)
            e = mdates.date2num(row['end_date'])
            
            # Training
            if row['start_date'] < val_start_timestamp:
                ax.hlines(i, s, v, colors='darkgrey', lw=10)
            
            # Validation
            v_draw_start = max(s, v)
            if v_draw_start < t:
                ax.hlines(i, v_draw_start, t, colors='grey', lw=10)
                
            # Testing
            t_draw_start = max(s, t)
            if t_draw_start < e:
                ax.hlines(i, t_draw_start, e, colors='lightgrey', lw=10)

        line_coords = [mdates.date2num(val_start_timestamp), 
                    mdates.date2num(test_start_timestamp), 
                    mdates.date2num(global_end)]
        
        ax.vlines(line_coords, ymin=-1, ymax=len(df_plot), colors='black', linestyles='--', alpha=0.5)

        y_text = len(df_plot) - 0.3
        
        def get_mid(d1, d2):
            return float(mdates.date2num(d1 + (d2 - d1) / 2))

        absolute_start = df_plot['start_date'].min()
        
        ax.text(get_mid(absolute_start, val_start_timestamp), y_text, 'Training', ha='center', weight='bold')
        ax.text(get_mid(val_start_timestamp, test_start_timestamp), y_text, 'Validation', ha='center', weight='bold')
        ax.text(get_mid(test_start_timestamp, global_end), y_text, 'Testing', ha='center', weight='bold')

        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['ticker'])
        
        ax.set_ylim(-1, len(df_plot) + 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}data_split_points.pdf")
        plt.show()
                


    def plot_daily_volume(self, date_str):
        """
        Plots volume for a specific day to show intraday patterns.
        Example date_str: '2023-10-27'
        """
        volume_col_str = 'volume'
        #Filter for the specific day
        daily_data = self.df.loc[date_str]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Using fill_between creates
        ax.fill_between(daily_data.index, daily_data[volume_col_str], 
                        color=self.primary_color, alpha=0.3)
        ax.plot(daily_data.index, daily_data[volume_col_str], 
                color=self.primary_color, linewidth=1.5)
        
        # Format X-axis to show hours and minutes (HH:MM)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax.set_title(f"Intraday Trading Volume: {date_str}", pad=20)
        ax.set_ylabel("Volume (Shares)")
        ax.set_xlabel("Time of Day")
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_path, f"volume_{date_str}.pdf")
        plt.savefig(save_path)
        plt.close()
                