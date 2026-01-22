from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureVisualization:
    dfs: list[pd.DataFrame]
    columns: list[str]
    
    def __init__(self, dfs: list[pd.DataFrame], folder: str = ''):
        self.dfs = dfs
        self.columns = dfs[0].columns.tolist()
        self.file_path = Path(f'plots/{folder}')
        self.file_path.mkdir(parents=True, exist_ok=True)
        
    
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
                ticker = df["ticker"].iloc[0] if "ticker" in df.columns else f"DataFrame {i+1}"
                if col in df.columns:
                    # Drop NaNs for plotting
                    valid_values = df[col].dropna()
                    if not valid_values.empty:
                        temp_df = pd.DataFrame({
                            'Value': valid_values,
                            'DataFrame': ticker
                        })
                        plot_data.append(temp_df)
            
            if plot_data:
                combined_df = pd.concat(plot_data, ignore_index=True)
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='DataFrame', y='Value', data=combined_df)
                plt.title(f'Feature Boxplots: {col}')
                plt.rotation = 45 # Rotation on x-axis labels if needed, usually done via plt.xticks()
                plt.xticks(rotation=45)
                
                plt.savefig(self.file_path / f'{col}_boxplot.png')
                plt.close()

                # plt.show()

    def run(self):
        
        self.visualize_feature_boxplots()
        ticker_feature_stats = []
        for df in self.dfs:
            ticker_feature_stats.append(df.describe(percentiles=[0.01, 0.05, 0.95, 0.99]))
            

            
            
            
            
            

        
    