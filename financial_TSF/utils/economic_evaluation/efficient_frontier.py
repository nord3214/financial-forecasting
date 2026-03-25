import array
from typing import Literal
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import SplitData, StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd

from financial_TSF.utils.economic_evaluation.graph_visualization import GraphVisualization
from financial_TSF.utils.economic_evaluation.top_k_simulator import TopKSimulator
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models


class PortfolioOptimizer:
    
    def __init__(self, split_data: list[SplitData], horizons: list[int]):
        """split data needs to be already unscaled for this"""
        self.split_data = split_data
        self.horizons = horizons


    def transform_predictions_into_py_opt_format(self, horizon: int, dataset: Literal["y_test_pred", "y_test", "y_train"]) -> pd.DataFrame:
        """
            Transform predictions into format for portfolio optimization library PyPortfolioOpt to compute optimal portfolio weights based on predicted returns.
            dataframe with columns as tickers and rows as time steps
            only the horizon ith value is used so ~ number of rows is reduced by factor of horizon 
        """
        series_list = []
        for split in self.split_data:
            ticker: str = split['ticker_name']
            target = split[dataset]
            target_horizon = target[[f"target_{horizon}"]].rename(columns={f"target_{horizon}": ticker})
            target_horizon.index = target_horizon.index.droplevel(1)
            series_list.append(target_horizon)
            
            print(f"ticker: {ticker}, target_horizon shape: {target_horizon.shape}")

        # :TODO does inner join makes sense here?
        targets = pd.concat(series_list, axis=1, join='inner')
        targets = targets.iloc[::horizon]
            
        print(f"final prediction length: {targets.shape[0]}")
            
        return targets
    
    def run_portfolio_optimization_for_horizon(self, horizon: int):
        
        use_naive_forecast = False
        
        
            
        
        y_pred = self.transform_predictions_into_py_opt_format(horizon=horizon, dataset="y_test_pred")
        y_ground_truth = self.transform_predictions_into_py_opt_format(horizon=horizon, dataset="y_test")
        y_train_hist = self.transform_predictions_into_py_opt_format(horizon=horizon, dataset="y_train")
        
        if use_naive_forecast:
            # naive baseline: predict the in-sample mean log-return for every test period
            train_mean = y_train_hist.mean()          # Series: ticker -> mean log-return
            y_pred = pd.DataFrame(
                np.tile(train_mean.values, (len(y_pred), 1)),
                index=y_pred.index,
                columns=y_pred.columns,
            )

        # Intraday frequency: 252 trading days * 78 five-minute bars/day, stepped by horizon
        # e.g. horizon=48 -> each period = 48 * 5min = 4h -> ~409 periods/year
        BARS_PER_TRADING_DAY = 78  # 6.5h * 12 bars/h
        PERIODS_PER_YEAR = int(252 * BARS_PER_TRADING_DAY / horizon)

        # Covariance estimated on in-sample (training) log returns only — no look-ahead bias.
        # At test time, these returns are entirely in the past and would be available to any
        # real-world investor. Ledoit-Wolf shrinkage stabilises the estimate when T is modest.
        S = risk_models.CovarianceShrinkage(
            y_train_hist, returns_data=True, frequency=PERIODS_PER_YEAR
        ).ledoit_wolf()
        
        # # Exponentially weighted — more weight on recent data, better for non-stationary
        # S = risk_models.exp_cov(y_train_hist, returns_data=True, frequency=PERIODS_PER_YEAR, span=180)

        # # Sample covariance — no shrinkage, noisiest
        # S = risk_models.sample_cov(y_train_hist, returns_data=True, frequency=PERIODS_PER_YEAR)


        portfolio_log_returns = []

        for idx, pred_row in y_pred.iterrows():
            # annualize the predicted log return for this period
            mu = np.exp(pred_row * PERIODS_PER_YEAR) - 1  # geometric annualized simple return
            # optimize weights based on predicted returns
            ef = EfficientFrontier(mu, S)
            try:
                ef.max_sharpe()
            except Exception:
                # fallback to min volatility if max_sharpe is infeasible (e.g. all mu <= 0)
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()

            weights = pd.Series(ef.clean_weights())  # ticker -> weight

            # realize portfolio return: dot product of weights with actual log returns
            actual_row = y_ground_truth.loc[idx]
            realized_return = float(weights.dot(actual_row))
            portfolio_log_returns.append({"timestamp": idx, "portfolio_log_return": realized_return})

        portfolio_returns = pd.DataFrame(portfolio_log_returns).set_index("timestamp")
        cumulative_return = portfolio_returns["portfolio_log_return"].sum()
        annualized_return = cumulative_return * PERIODS_PER_YEAR / len(portfolio_returns)
        sharpe = (
            portfolio_returns["portfolio_log_return"].mean()
            / portfolio_returns["portfolio_log_return"].std()
            * np.sqrt(PERIODS_PER_YEAR)
                )
        print(f"[PyPortfolioOpt Backtest] Periods: {len(portfolio_returns)}, "
                f"Cumulative log-return: {cumulative_return:.4f}, "
                f"Annualized return: {annualized_return:.4f}, "
                f"Sharpe: {sharpe:.4f}")
        
        result: dict = {
            "horizon": horizon,
            "periods_rebalancing": len(portfolio_returns),
            "cumulative_log_return": cumulative_return,
            "annualized_return": annualized_return,
            "sharpe": sharpe
        }
        
        return result
        
    def run(self): 
        
        results = []
        
        for horizon in self.horizons:
            print(f"Running portfolio optimization for horizon {horizon}...")
            
            results.append(self.run_portfolio_optimization_for_horizon(horizon))
            
        print("\n--- Portfolio Optimization Summary ---")
        for res in results:
            print(f"Horizon {res['horizon']}: "
                    f"Periods: {res['periods_rebalancing']}, "
                    f"Cumulative log-return: {res['cumulative_log_return']:.4f}, "
                    f"Annualized return: {res['annualized_return']:.4f}, "
                    f"Sharpe: {res['sharpe']:.4f}")
            
        return results
            
