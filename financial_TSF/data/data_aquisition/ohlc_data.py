
from pathlib import Path
from settings import years, months, symbols, parameters
from api_call import make_api_call
from pandas import read_csv, concat
import os



def fetch_ohlc_data():
    """Fetch OHLC data from Alpha Vantage API and save specificed Tickers to local CSV files."""
    absolute_path = Path(f"G:\\TSFdata/{parameters['interval']}")
    for symbol in symbols:
        
        request_apis = []
        
        for year in years:
            for month in months:
            
                save_path = absolute_path / f"raw/{symbol}/{year}-{month}_intraday_{parameters['interval']}.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # save_path_old = Path(f"data/raw/{symbol}/{year}-{month}_intraday_{parameters['interval']}.csv")
                if save_path.exists():
                    print(f"File for {symbol} in {month}/{year} already exists. Skipping download.")
                    continue
                print(f"Fetching data for {symbol} in {month}/{year}")
                api_request = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={parameters["interval"]}&month={year}-{month}&outputsize=full&extended_hours=false&apikey={parameters["apikey"]}&datatype=csv'
                print("With API request: ", api_request) 
                print(make_api_call(api_request, save_path=save_path))
                request_apis.append(api_request)
                
        request_apis_path = absolute_path / f"api_calls/{symbol}_api_requests_{parameters['interval']}.txt"
        request_apis_path.parent.mkdir(parents=True, exist_ok=True)            
        with open(request_apis_path, "w") as f:
            for api in request_apis:
                f.write(api + "\n")


def concat_data_from_ticker():
    """Concatenate OHLCV data from all CSV files of a specific ticker into a single CSV"""
    for symbol in symbols:
        all_files = sorted(Path(f"data/raw/{symbol}/").glob("*.csv"))
        combined_data = []
        for file in all_files:
            if os.path.getsize(file) < 10000:
                print(f"File {file} is too small, likely incomplete. Skipping.")
                continue
            else:
                df = read_csv(file)
                df = df.sort_values(by='timestamp')
                combined_data.append(df)
        if combined_data:

            save_dir = f"data/concatenated/{symbol}/"
            os.makedirs(save_dir, exist_ok=True)
            final_df = concat(combined_data, ignore_index=True)
            final_df.to_csv(f"{save_dir}{symbol}_intraday_{parameters['interval']}_full.csv", index=False)


concat_data_from_ticker()

# fetch_ohlc_data()