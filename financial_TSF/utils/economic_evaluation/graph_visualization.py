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
        self.idx_of_ticker_to_plot = 0

        self.primary_color   = '#1a4a73'   # Navy  – actual values
        self.secondary_color = "#dd6f25"   # Orange – predicted values
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
        split     = self.step_result.split_data[self.idx_of_ticker_to_plot]
        y_test    = split["y_test"]
        y_pred    = split["y_test_pred"]
        cols      = y_test.columns   # last horizon column when using all from thesis lockback window = 48

        # MultiIndex: level 0 is the timestamp – keep first 7 calendar days
        timestamps = y_test.index.get_level_values(0)
        display_dates = pd.Timedelta(days=7)
        display_dates = None
        
        for col in cols:
            horizon = col.split("_")[-1]  # get number of horizons form column name
            graph_title = f"Log-return of the next {horizon}-Intervals" 
            if display_dates: 
                cutoff = timestamps.min() + display_dates
                graph_title += f" (first {display_dates.days} days)"
            else:
                cutoff = timestamps.max()
            
            mask       = timestamps < cutoff

            actual    = y_test.loc[mask, col]
            predicted = y_pred.loc[mask, col]
            dates     = actual.index.get_level_values(0)
            
            plot_df = pd.DataFrame({
                'Date': dates,
                'Actual': actual.values,
                'Predicted': predicted.values
            }).sort_values('Date')
            
            expected_step = pd.Timedelta(minutes=5)
            
            gap_mask = plot_df["Date"].diff() > expected_step * 1.5

        
            plot_df.loc[gap_mask, ["Actual", "Predicted"]] = np.nan

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(plot_df["Date"], plot_df["Actual"],    color=self.primary_color,   linewidth=1.5, label="Ground Truth")
            ax.plot(plot_df["Date"], plot_df["Predicted"], color=self.secondary_color, linewidth=1.5, label="Forecast")
            ax.set_title(graph_title)
            ax.set_xlabel("Date of Predictions")
            ax.set_ylabel("Log-return")
            ax.legend(loc="upper right")
            plt.tight_layout()
            save_path = self.output_path / f"weekly_predictions_{col}.pdf"
            plt.savefig(save_path)
            plt.close()
            print(f"[GraphVisualization] Saved weekly predictions plot -> {save_path}")

    def visualize_prediction_scatter(self):
        """Plots predicted returns against ground truth returns for each horizon."""
        split  = self.step_result.split_data[self.idx_of_ticker_to_plot]
        y_test = split["y_test"]
        y_pred = split["y_test_pred"]

        for col in y_test.columns:
            actual = y_test[col].to_numpy()
            predicted = y_pred[col].to_numpy()
            
            lower = min(actual.min(), predicted.min())
            upper = max(actual.max(), predicted.max())
            pad = (upper - lower) * 0.05 if upper > lower else 1e-6

            horizon = col.split("_")[-1]
            graph_title = f"Predicted vs Ground Truth ({horizon}-Intervals)"

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(actual, predicted, label='Predictions', color=self.secondary_color)
            ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color=self.primary_color
                    , linestyle='--', label='Perfect Prediction')

            ax.set_xlim(lower - pad, upper + pad)
            ax.set_ylim(lower - pad, upper + pad)
            # ax.set_aspect('equal', adjustable='box')
            ax.set_title(graph_title)
            ax.set_xlabel("Ground Truth Log-return")
            ax.set_ylabel("Predicted Log-return")
            ax.legend(loc="upper left")

            plt.tight_layout()
            save_path = self.output_path / f"scatter_predictions_{col}.pdf"
            plt.savefig(save_path)
            plt.close()
            print(f"[GraphVisualization] Saved prediction scatter plot -> {save_path}")



    def run(self):
        """Generate all prediction visualizations."""
        self.visualize_weekly_predictions()
        self.visualize_prediction_scatter()

