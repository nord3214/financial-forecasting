from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureVisualization:
    dfs: list[pd.DataFrame]
    columns: list[str]
    
    def __init__(self, dfs: list[pd.DataFrame], folder: str = ''):
        """Initialize the FeatureVisualization class with a list of DataFrames but limits it to 10DataFrames exclusively and a folder name for saving plots."""
        dfs = dfs[0:10]
        self.dfs = dfs
        self.columns = dfs[0].columns.tolist()
        self.file_path = Path(f'plots/{folder}')
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.primary_color = '#1a4a73'
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
            'savefig.bbox': 'tight'   
        })
        
    
    def visualize_feature_distributions(self):
        import matplotlib.pyplot as plt
        
        for col in self.columns:
            plt.figure()
            for i, df in enumerate(self.dfs):
                plt.hist(df[col].dropna(), bins=100, alpha=0.5, label=f'DataFrame {i+1}')
            plt.title(f'Feature Distribution for {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            
    def visualize_feature_boxplots(self):
        """Visualize boxplots for each feature across all DataFrames using seaborn."""


        for col in self.columns:            
            plot_data = []
            
            for i, df in enumerate(self.dfs):
                ticker = f"DataFrame {i+1}"
                try: 
                    if "ticker" in df.columns:
                        ticker = df["ticker"].iloc[0]
                    else: 
                        ticker = df.index.get_level_values('ticker')[0]
                except Exception as e:
                    ticker = f"DataFrame {i+1}"
                    
                if col in df.columns:
                    # Drop NaNs for plotting
                    valid_values = df[col].dropna()
                    if not valid_values.empty:
                        temp_df = pd.DataFrame({
                            'Value': valid_values,
                            'Ticker': ticker
                        })
                        plot_data.append(temp_df)
            
            if plot_data:
                combined_df = pd.concat(plot_data, ignore_index=True)
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Ticker', y='Value', data=combined_df, color=self.primary_color)
                plt.title(f'Feature Boxplot: {col}')
                plt.rotation = 45 # Rotation on x-axis labels if needed, usually done via plt.xticks()
                plt.xticks(rotation=45)
                
                plt.savefig(self.file_path / f'{col}_boxplot.png')
                plt.close()

                # plt.show()

    def run(self):
        
        self.visualize_feature_boxplots()
        ticker_feature_stats = []
        # for df in self.dfs:
        #     ticker_feature_stats.append(df.describe(percentiles=[0.01, 0.05, 0.95, 0.99]))
            

            
            
            
            
            

        
    