import json
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[3] / ".env")
api_key = os.getenv("ALPHAVANTAGE_API_KEY")
print("Api key: ", api_key)

parameters = {
    'interval' : '5min',
    'month' : 'yyyy-mm',
    'apikey' : api_key
}


years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2025]


months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

symbols = json.load(open("financial_TSF\data\data_aquisition\companies31.12.25.json"))