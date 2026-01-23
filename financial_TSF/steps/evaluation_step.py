from typing import List
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from financial_TSF.data_models.config_data_model import AppConfig
from financial_TSF.data_models.step_datamodel import SplitData, StepResult
from financial_TSF.steps.base_step import PipelineStep
import pandas as pd
import numpy as np


class EvaluationStep(PipelineStep):
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self, step_result: StepResult) -> StepResult:
        
        step_result = self.map_predictions_to_tickers(step_result)
                
        # machine learning model evaluation
        step_result.metrics = self.compute_multistep_metrics(step_result.y_test, step_result.y_test_pred, step_result)
        
        
        # ------------------------- economic evaluation -------------------------
        
        # sim = TopKSimulator(step_result.split_data, transaction_cost_bps=1.0)
        # sim.prepare_data(horizon_col=f"target_{self.config.settings.forecast_horizon[0]}")
        
        # result_df = sim.run_simulation(k=5)
        # metrics = sim.get_metrics()
        # print("Top-K Trading Strategy Metrics: ", metrics)
        
        # sim.plot_equity_curve()
        
        return step_result
    
    def map_predictions_to_tickers(self, step_result: StepResult) -> StepResult:
        """
            Map global y_test_pred to each ticker's in split_data y_test_pred
        """
    
        start_idx = 0
        for split in step_result.split_data:
            n_rows = split['y_test'].shape[0]
            current_slice = step_result.y_test_pred[start_idx:start_idx + n_rows]
            
            # assign the y_pred slice to the corresponding ticker in split_data
            split['y_test_pred'] = pd.DataFrame(
                current_slice,
                columns=split['y_test'].columns,
                index=split['y_test'].index
            )
            
            # validate for using correct prediction
            if not split['y_test_pred'].shape == split['y_test'].shape:
                raise ValueError("Shape mismatch between y_test and y_test_pred in split_data mapping")
            if not (split['y_test'].index == split['y_test_pred'].index).all():
                raise ValueError("Index mismatch between y_test and y_test_pred in split_data mapping")
            # compare if y_test values are the same - to induce operating on the correct ticker slices
            if not (split['y_test'].values == step_result.y_test[start_idx:start_idx + n_rows].values).all():
                raise ValueError("Probably ticker mismatch between y_test in split_data and global y_test in StepResult")

            start_idx += n_rows
            
        return step_result
    
    def compute_hitrate(self, y_true, y_pred) -> float:
        """
            Compute hit rate for direction of returns
        """

        return np.mean(np.sign(y_true.values) == np.sign(y_pred.values))
    
    def calculate_sharpe_ratio(self, net_pnl_series: pd.Series) -> float:
        """
            Calculate annualized Sharpe Ratio
        """
        # calculta mean return easy with log returns because they are additive
        mean_return = net_pnl_series.mean() 
        
        std_return = net_pnl_series.std()
        
        if std_return == 0:
            return 0.0
        
        # risk free rate is assumed to be 0 for simplicity and short time frames
        sharpe_per_period = mean_return / std_return
        
        # number of bars per day
        bars_per_day = (6.5 * 60) / 5  # assuming 5-minute intervals in a 6.5-hour trading day

        annualization_factor = np.sqrt(252 * bars_per_day)
        
        annualized_sharpe = sharpe_per_period * annualization_factor
        
        return annualized_sharpe
    
    def calculate_percentage_return(self, cumulative_pnl: pd.Series) -> float:
        """
            Calculate percentage return from cumulative PnL series
        """
        percentage_return = (np.exp(cumulative_pnl.iloc[-1]) - 1) * 100 
        return percentage_return 
    
    def calculate_number_of_trades(self, positions: pd.Series) -> float:
        """
            Calculate turnover factor from position series
        """
        total_trades = positions.diff().abs().sum() *2 # each trade involves both entry and exit
        
        return int(total_trades)
    
    
    def compute_multistep_metrics(self, y_true, y_pred, step_result: StepResult = StepResult()) -> dict[str, float]:
        """
            H = number of forecast horizons
            y_true: shape (N, H)
            y_pred: shape (N, H)
        """     
        
        metrics = {}
        
        for i, horizon in enumerate(self.config.settings.forecast_horizon):
            # :TODO convert to dataframe or different indexing!
            y_true_horizon = pd.DataFrame(y_true).iloc[:, i]
            y_pred_horizon = pd.DataFrame(y_pred).iloc[:, i]
                     
            
            # use_naive_approach = False
        
            # if use_naive_approach:
            #     y_pred_horizon = y_true_horizon.shift(horizon, fill_value=0)
            #     # y_pred_horizon = np.roll(y_true, shift=-2, axis=1)
                
            #     print("Prediction: ", y_pred_horizon.head(20))
            #     print("True: ", y_true_horizon.head(20))
                
            #     df = pd.concat([y_pred_horizon, y_true_horizon], axis=1)
            #     print("Comparison: ", df.head(20))
            
            #trading simulation evaluation for each horizon
            trading_simulator = TradingSimulation(self.config, horizon)  
            backtest_results = trading_simulator.evaluate_horizon(y_true_horizon, y_pred_horizon, threshold=0.001)
            
            if not (backtest_results['y_true'].to_numpy() == y_true_horizon.to_numpy()).all():
                raise ValueError("Mismatch between backtest y_true and input y_true_horizon cant reset index properly")
            y_pred_traded = backtest_results['position'].abs() * y_pred_horizon.reset_index(drop=True)
            y_true_traded = backtest_results['position'].abs() * y_true_horizon.reset_index(drop=True)
            
            #only the acutal traded values
            y_true_traded = y_true_traded[y_true_traded != 0]
            y_pred_traded = y_pred_traded[y_pred_traded != 0]
            
        

            

            metrics[f"Horizon_{horizon}"] = {
                "MAE": mean_absolute_error(y_true_horizon, y_pred_horizon),
                "MSE": mean_squared_error(y_true_horizon, y_pred_horizon),
                "RMSE": root_mean_squared_error(y_true_horizon, y_pred_horizon),
                "R2": r2_score(y_true_horizon, y_pred_horizon),
                "Hit_Rate": self.compute_hitrate(y_true_traded, y_pred_traded),
                "Sharpe_Ratio": self.calculate_sharpe_ratio(backtest_results['net_pnl']),
                "Percentage_Return": self.calculate_percentage_return(backtest_results['cumulative_pnl']), 
                "Number_of_Trades": self.calculate_number_of_trades(backtest_results['position'])
            }
            
        metrics["Global"] = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }
        
        print("Multi-step Forecasting Metrics: ", metrics)    
        return metrics
    
class TradingSimulation:
    def __init__(self, config: AppConfig, horizon: int):
        self.config = config
        self.trading_fee = 0.0002  # Example trading fee per trade 
        self.horizon = horizon
        
    def evaluate_horizon(self, y_true, y_pred, threshold: float) -> pd.DataFrame:
        """
            Evaluate trading strategy based on predictions for a specific horizon
        """
        df = pd.DataFrame(index=range(len(y_pred)))
        df['y_true'] = y_true.reset_index(drop=True)
        df['y_pred'] = y_pred.reset_index(drop=True)
                
        df['position'] = 0
        df['trade_costs'] = 0.0
        
        # loop to only enter one position at a time 
        i = 0
        while i < len(df):
            current_pred = df['y_pred'].iloc[i]
            
            # Signal-Check
            if current_pred > threshold:
                df.loc[i, 'position'] = 1
                df.loc[i, 'trade_costs'] = self.trading_fee * 2  
                i += self.horizon  # skip next 'horizon' steps to simulate cooldown
            elif current_pred < -threshold:
                df.loc[i, 'position'] = -1
                df.loc[i, 'trade_costs'] = self.trading_fee * 2
                i += self.horizon  # skip next 'horizon' steps to simulate cooldown
            else:
                i += 1  # No signal, check next step
                
        df['net_pnl'] = (df['position'] * df['y_true']) - df['trade_costs']
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
    
        return df

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
        self.return_matrix: pd.DataFrame = pd.concat(actual_frames, axis=1)
        
        # :TODO think about missing data handling here!
        # self.signal_matrix.dropna(axis=0, how='any', inplace=True)
        # self.returns_matrix = self.returns_matrix.reindex(self.signal_matrix.index)
        
        print(f"Data prepared. Time steps: {self.signal_matrix.shape[0]}, Tickers: {self.signal_matrix.shape[1]}")
        
    def run_simulation(self, k:int = 5):
        """
        Run Top-K trading simulation based on prepared signal and actual return matrice 
        """
        
        if not hasattr(self, 'signal_matrix') or not hasattr(self, 'return_matrix'):
            raise ValueError("Data not prepared. Call prepare_data() before running simulation.")
        
        portfolio_returns = []
        timestamps = []
        turnover_tracker = []
        
        prev_longs = set()
        prev_shorts = set()
        
        print(f"Simulation Top-{k} Long/Short trading strategy...")
        
        # calculate returns for each time step
        for timestamp, row_preds in self.signal_matrix.iterrows():
            
            # NaN values dropped for ranking
            valid_preds = row_preds.dropna()
            
            # check if enough valid tickers for trading
            if len(valid_preds) < k*2: 
                portfolio_returns.append(0.0)
                timestamps.append(timestamp)
                continue
            
            # select str top-k and bottom-k tickers to trade 
            # :TODO handle if best is negative or worst is positive - no trade should be made in that case
            long_tickers: list[str] = valid_preds.nlargest(k).index.to_list()
            short_tickers: list[str] = valid_preds.nsmallest(k).index.to_list()
            
            # get actual returns for selected tickers
            actual_log_returns = self.return_matrix.loc[timestamp] # type: ignore
            actual_simple_returns = np.exp(actual_log_returns) - 1  # convert log returns to simple returns
            
            
            avg_long_return = actual_simple_returns[long_tickers].mean()
            avg_short_return = actual_simple_returns[short_tickers].mean()
            
            # assuming 50% allocation to longs and shorts respectively
            gross_return = 0.5 * avg_long_return - 0.5 * avg_short_return
            
            current_longs = set(long_tickers)
            current_shorts = set(short_tickers)
            
            # how many new trades are made compared to previous positions
            changes: int = len(current_longs-prev_longs) + len(current_shorts - prev_shorts)
            
            # weighting of turnover cost assuming equal weight(invest) per position
            turnover_cost = ((changes*2) / (2*k)) * self.cost_pct
            
            net_return = gross_return - turnover_cost
            
            portfolio_returns.append(net_return)
            timestamps.append(timestamp)
            
            prev_longs = current_longs
            prev_shorts = current_shorts
            
        # result DataFrame
        self.results = pd.DataFrame({
            'net_return': portfolio_returns
        }, index=timestamps)
        
        # cumulative return ???
        self.results['cumulative_return'] = (1 + self.results['net_return']).cumprod()
        
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
        
    def get_metrics(self):
        rets = self.results['net_return']
        
       # Annualization factor assuming 252 trading days and 78 5-minute intervals per day
        intervals_per_year = 252 * 78 
        
        sharpe = (rets.mean() / rets.std()) * np.sqrt(intervals_per_year)
        total_return = self.results['cumulative_return'].iloc[-1] - 1
        
        # Max Drawdown
        cum = self.results['cumulative_return']
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()
        
        return {
            "Sharpe Ratio": round(sharpe, 2),
            "Total Return %": round(total_return * 100, 2),
            "Max Drawdown %": round(max_dd * 100, 2)
        }
        
    
            
            
            
            
                
            
            
            

        
        
        
            
            
            