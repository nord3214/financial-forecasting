from typing import List
from matplotlib import pyplot as plt
from financial_TSF.data_models.step_datamodel import SplitData
import pandas as pd
import numpy as np


class TopKSimulator:
    # :TODO maybe make more performant by using vectorized operations instead of loops
    def __init__(self, split_data_list: List[SplitData], transaction_cost_bps: float = 2.0):
        """        
        :param split_data_list: corresponding dict of tickers 
        :type split_data_list: List[SplitData]
        :param transaction_cost_bps: cost per transaction(buy or sell) in basis points (1 bps = 0.01%)
        :type transaction_cost_bps: float
        """
        self.data: List[SplitData] = split_data_list
        self.cost_pct = transaction_cost_bps / 10000  # convert bps to decimal percentage
        self.return_matrix: pd.DataFrame | None = None # DataFrame with actual returns for each ticker and time step for 1 interval
        self.no_of_entries = None
        self.annualized_sharpe = None
        self.hit_rate = None
        
        
        
    def prepare_data(self, horizon_col: str, actual_return_col: str = 'return_next_interval'):
        """
        Transforms tickerdata into global DataFrame for Top-K evaluation
        :param horizon_col: column name for predicted returns at specific horizon
        :param actual_return_col: optional column name for actual returns at specific horizon 
        if None - hold to maturity used
        if str column name need to specify log returns of current interval 
        """
                    
        pred_frames = []
        actual_frames = []
        
        print(f"Aggregating data for signal: {horizon_col} ...")
        
        for item in self.data:
            ticker_name = item['ticker_name']
            
            # get scaler and data
            y_scaler = item['y_scaler']
            y_test = item['y_test']
            y_test_pred = item['y_test_pred']
            
            actual_unscaled = item['log_return_horizon1']
            
            # inverse transform predictions and actuals
            pred_unscaled = y_scaler.inverse_transform(y_test_pred)
            
            # convert back to DataFrames with original timestamp Index and Column names
            df_pred = pd.DataFrame(pred_unscaled, columns=y_test_pred.columns, index=y_test_pred.index)
            # :TODO have to set index to y_test index??? 
            df_actual = actual_unscaled.loc[y_test.index]
            if len(df_actual) != len(df_pred):
                raise ValueError(f"Length mismatch between actual returns and predicted returns for ticker {ticker_name}")
            
            # only consider the specific horizon column
            series_pred = df_pred[horizon_col].rename(ticker_name)
            # use actual log return over next interval for economic
            series_actual = df_actual[actual_return_col].rename(ticker_name)
            
            # remove ticker multiindex level for concatenation so can be concatenated on index timestamp
            pred_frames.append(series_pred.droplevel('ticker'))
            actual_frames.append(series_actual.droplevel('ticker'))
            
        # concatenate all ticker series into a single DataFrame on axis 1 one row per timestamp with all tickers
        self.signal_matrix: pd.DataFrame = pd.concat(pred_frames, axis=1)
        self.return_matrix = pd.concat(actual_frames, axis=1)
        
        # :TODO think about missing data handling here!
        # self.signal_matrix.dropna(axis=0, how='any', inplace=True)
        # self.returns_matrix = self.returns_matrix.reindex(self.signal_matrix.index)
        
        print(f"Data prepared. Time steps: {self.signal_matrix.shape[0]}, Tickers: {self.signal_matrix.shape[1]}")
        
    def run_simulation(self, k: int = 5):
        """
        Run Top-K trading simulation (Vectorized Version)
        """
        if not hasattr(self, 'signal_matrix') or not hasattr(self, 'return_matrix') or self.return_matrix is None:
            raise ValueError("Data not prepared. Call prepare_data() before running simulation.")

        print(f"Simulation Top-{k} Long/Short trading strategy (Vectorized)...")

        # align returns with signals
        signals = self.signal_matrix
        returns = self.return_matrix.loc[signals.index]
        
        # convert log returns to simple returns for PnL calculation (simple returns are additive for portfolio return calculation
        simple_returns = np.exp(returns) - 1

        # check if we have enough valid tickers to trade at each time step
        valid_counts = signals.count(axis=1)
        valid_mask = valid_counts >= (k * 2)

        # rank tickers by predicted signal 
        ranks_desc = signals.rank(axis=1, method='first', ascending=False)
        # Ascending=True: Niedrigste Werte haben Rang 1 -> Short
        ranks_asc = signals.rank(axis=1, method='first', ascending=True)

        # create masks for long and short positions of top k tickers
        long_mask = ranks_desc <= k
        short_mask = ranks_asc <= k

        # calculate average returns for long and short positions with the actual simple returns
        avg_long_ret = (simple_returns * long_mask).sum(axis=1) / k
        avg_short_ret = (simple_returns * short_mask).sum(axis=1) / k

        # gross Return: 50% Long - 50% Short
        gross_ret = 0.5 * avg_long_ret - 0.5 * avg_short_ret

        # calculate turnover: how many positions are changed compared to previous time step with logic and operation
        long_changes = (long_mask & ~long_mask.shift(1).fillna(False)).sum(axis=1)
        short_changes = (short_mask & ~short_mask.shift(1).fillna(False)).sum(axis=1)
        
        total_changes = long_changes + short_changes

        # calculate turnover cost as percentage of portfolio value, assuming equal weight per position and cost per trade
        turnover_cost = (total_changes / k) * self.cost_pct

        # Net Return after costs
        net_ret = gross_ret - turnover_cost

        # if not enough valid tickers to trade, set return to 0 for that time step
        net_ret = net_ret.where(valid_mask, 0.0)

        # 
        self.results = pd.DataFrame({
            'net_return': net_ret
        }, index=signals.index)

        self.results['cumulative_return'] = (1 + self.results['net_return']).cumprod()
        
        self.hit_rate = self.get_hit_rate(long_mask, short_mask, simple_returns, valid_mask)
        self.annualized_sharpe = self.calculate_sharpe_ratio(net_ret)
        self.no_of_entries = total_changes.sum()
        self.percentage_return = (self.results['cumulative_return'].iloc[-1] - 1) * 100
        
        
    def plot_equity_curve(self):
        """
        Plot equity curve of the Top-K trading strategy
        """
        if not hasattr(self, 'results'):
            raise ValueError("Simulation not run yet. Call run_simulation() before plotting.")

        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['cumulative_return'], label='Top-K L/S Strategy')
        plt.title('Equity Curve (Net of Costs)')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    
    def get_hit_rate(self, long_mask: pd.DataFrame, short_mask: pd.DataFrame, simple_returns: np.ndarray, valid_mask: pd.Series) -> int:
        """
        Calculate total number of trades executed in the simulation for each time step
        a ticker held for 10time intervals counts as 10 trades - at each time step it is check if ticker was profitable (hit)
        """
    
        # For long, success is return > 0; for shorts, success is return < 0
        long_hits = (long_mask & (simple_returns > 0)).sum(axis=1)
        short_hits = (short_mask & (simple_returns < 0)).sum(axis=1)

        # Total successful positions per timestamp
        total_hits = long_hits + short_hits

        # count active positions of current time step
        # only consider returns where data is available
        non_nan_returns = ~np.isnan(simple_returns)
        # active positions only count if we had enough valid tickers to trade at that time step
        active_positions = ((long_mask.values | short_mask.values) & non_nan_returns).sum(axis=1)
        # mask to consider only time steps where we had enough valid tickers to trade
        total_trades_made = active_positions[valid_mask].sum()
        total_successes = total_hits[valid_mask].sum()

        # gain percentage of successful trades 
        hit_rate = total_successes / total_trades_made if total_trades_made > 0 else 0
        print(f"Hit Rate: {hit_rate:.2%}")
        
        return hit_rate

    def calculate_sharpe_ratio(self, net_pnl_series: pd.Series) -> float:
        """
        Calculate annualized Sharpe Ratio for simple returns.
        :param net_pnl_series: Series of net PnL returns from the trading simulation (simple returns, not log returns altough relatively negligible for very small returns)
        :return: Annualized Sharpe Ratio
        """
        # Ensure we drop any NaNs from the simulation (e.g., initial periods)
        clean_returns = net_pnl_series.dropna()
        
        if clean_returns.empty:
            return 0.0

        # Mean and Std of simple returns
        mean_return = clean_returns.mean() 
        std_return = clean_returns.std()
        
        # Avoid division by zero if there is no volatility
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        # Risk-free rate (Rf) is assumed 0 for intraday strategies
        sharpe_per_period = mean_return / std_return
        
        # Annualization: sqrt(Total periods in a year)
        # 6.5 hours * 12 intervals/hour = 78 bars/day
        bars_per_day = (6.5 * 60) / 5 
        annualization_factor = np.sqrt(252 * bars_per_day)
        
        return float(sharpe_per_period * annualization_factor)
        
        
    
            
            
            
            
                
            
            
            

        
        
        
            
            
            