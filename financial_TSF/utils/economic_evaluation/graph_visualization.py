from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import StepResult
from financial_TSF.utils.utils import get_result_path


class GraphVisualization:
    def __init__(self, config: AppConfig, step_result: StepResult):
        self.config = config
        self.step_result = step_result

        self.primary_color   = '#1a4a73'   # Navy  – actual values
        self.secondary_color = '#e07b39'   # Orange – predicted values
        self.output_path = get_result_path(config, step_result) / "plots" / "predictions"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._apply_global_style()

    def _apply_global_style(self):
        """Sets the visual standard for all plots in the thesis."""
        plt.rcParams.update({
            'font.family': 'serif',
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
            'savefig.bbox': 'tight',
        })

    def visualize_weekly_predictions(self):
        """
        Plots the first x days of actual vs. predicted return for the
        first horizon and the first ticker in split_data.
        """
        split     = self.step_result.split_data[0]
        y_test    = split["y_test"]
        y_pred    = split["y_test_pred"]
        col       = y_test.columns[0]   # first horizon column

        # MultiIndex: level 0 is the timestamp – keep first 7 calendar days
        timestamps = y_test.index.get_level_values(0)
        display_dates = pd.Timedelta(days=7)
        display_dates = None
        if display_dates: 
            cutoff = timestamps.min() + display_dates
        else:
            cutoff = timestamps.max()
        
        mask       = timestamps < cutoff

        actual    = y_test.loc[mask, col]
        predicted = y_pred.loc[mask, col]
        dates     = actual.index.get_level_values(0)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(actual.values,    color=self.primary_color,   linewidth=1.5, label="Ground Truth")
        ax.plot(predicted.values, color=self.secondary_color, linewidth=1.5, label="Forecast")
        ax.set_title(f"Return Prediction — {col} (first 7 days)")
        ax.set_xlabel("Number of Predictions")
        ax.set_ylabel("Log Return")
        ax.legend()
        plt.tight_layout()
        save_path = self.output_path / f"weekly_predictions_{col}.pdf"
        plt.savefig(save_path)
        plt.close()
        print(f"[GraphVisualization] Saved weekly predictions plot -> {save_path}")



    def run(self):
        """Generate all prediction visualizations."""
        self.visualize_weekly_predictions()

