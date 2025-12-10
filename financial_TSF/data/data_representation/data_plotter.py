import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("dark_background")

df = pd.read_csv("data/concatenated/AAPL/AAPL_intraday_5min_full.csv")

overall_change = df['close'].iloc[-1] - df['close'].iloc[0]
percentage_change = (overall_change / df['close'].iloc[0]) * 100
print(f"Overall Change: {overall_change}")
print(f"Percentage Change: {percentage_change:.2f}%")

quantiles_to_calculate = [0.001, 0.999]
horizon = 2
# calculate log return
df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
print(df.describe())

df[f"target{1}"] = df['Log_Return'].rolling(window=horizon).sum().shift(-horizon)

print(df.tail(14))


# calculate quantiles
df_quantiles = df['Log_Return'].quantile(quantiles_to_calculate)
print("1st Percentile (Log Return): ", df_quantiles[quantiles_to_calculate[0]])
print("99th Percentile (Log Return): ", df_quantiles[quantiles_to_calculate[1]])


df_filtered = df[(df['Log_Return'] >= df_quantiles[quantiles_to_calculate[0]]) & (df['Log_Return'] <= df_quantiles[quantiles_to_calculate[1]])]


return_first_percentile = df[df['Log_Return'] <= df_quantiles[quantiles_to_calculate[0]]]
returns_first_percentile = return_first_percentile.sum()
print("Sum of returns below 1st Percentile (Log Return): ", returns_first_percentile['Log_Return'])


return_ninety_ninth_percentile = df[df['Log_Return'] >= df_quantiles[quantiles_to_calculate[1]]]
returns_ninety_ninth_percentile = return_ninety_ninth_percentile.sum()
print("Sum of returns above 99th Percentile (Log Return): ", returns_ninety_ninth_percentile['Log_Return'])

# difference 
difference = returns_ninety_ninth_percentile['Log_Return'] + returns_first_percentile['Log_Return']
print("Difference between sums of extreme quantiles (Log Return): ", difference)
    
overall_change_log = df_filtered['Log_Return'].sum()
percentage_change_log = (np.exp(overall_change_log) - 1) * 100
print(f"Overall quantile Change (Log): {overall_change_log}")
print(f"Percentage Change (Log) w quantiles: {percentage_change_log:.2f}%")

# calculate historgram data
hist_data = np.histogram(df_filtered['Log_Return'].dropna(), bins=1000)

# plot figues
fig, ax = plt.subplots(3, 1, figsize=(12, 6))

# plot
ax[0].plot(hist_data[1][:-1], hist_data[0], label='Log Return')
plt.show()