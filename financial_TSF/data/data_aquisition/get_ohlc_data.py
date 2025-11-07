
from pathlib import Path
from settings import years, months, symbols, parameters
from api_call import make_api_call





for year in years:
    if year == 2024:
        for month in months:
            if month == "11":
                for symbol in symbols:
                    save_path = Path(f"data/raw/{symbol}/{year}-{month}_intraday_{parameters['interval']}.csv")
                    if save_path.exists():
                        print(f"File for {symbol} in {month}/{year} already exists. Skipping download.")
                        continue
                    print(f"Fetching data for {symbol} in {month}/{year}")
                    api_request = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={parameters["interval"]}&month={year}-{month}&outputsize=full&extended_hours=false&apikey={parameters["apikey"]}&datatype=csv'
                    print(make_api_call(api_request, save_path=save_path))