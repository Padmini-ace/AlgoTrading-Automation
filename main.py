# Required Libraries:
# pip install yfinance pandas numpy ta scikit-learn gspread oauth2client

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import socket

# Force Python to use IPv4 instead of IPv6
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_wrapper(host, *args, **kwargs):
    return [ai for ai in orig_getaddrinfo(host, *args, **kwargs) if ai[0] == socket.AF_INET]
socket.getaddrinfo = getaddrinfo_wrapper
# Google Sheets Libraries
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------- GOOGLE SHEETS CONNECTION ----------
def connect_google_sheets(sheet_name="AlgoTrading"):
    try:
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open(sheet_name)
        logging.info("✅ Connected to Google Sheets successfully.")
        return sheet
    except Exception as e:
        logging.error(f"⚠️ Google Sheets connection failed: {e}")
        return None

# --------- FETCH DATA ----------
def fetch_data(ticker, period="6mo"):
    logging.info(f"Fetching data for {ticker}...")
    df = yf.download(ticker + ".NS", period=period, interval="1d")
    df.dropna(inplace=True)
    return df

# --------- ADD INDICATORS ----------
def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"].squeeze(), window=14).rsi()
    df["20MA"] = df["Close"].rolling(window=20).mean()
    df["50MA"] = df["Close"].rolling(window=50).mean()
    return df


# --------- GENERATE SIGNALS ----------
def generate_signals(df):
    df["Signal"] = 0
    df["Signal"] = np.where((df["RSI"] < 30) & (df["20MA"] > df["50MA"]), 1,
                    np.where((df["RSI"] > 70) & (df["20MA"] < df["50MA"]), -1, 0))
    return df

# --------- BACKTEST STRATEGY ----------
def backtest(df):
    df["Return"] = df["Close"].pct_change()
    df["StrategyReturn"] = df["Signal"].shift(1) * df["Return"]
    total_return = df["StrategyReturn"].cumsum().iloc[-1] * 100
    win_ratio = (df[df["Signal"] != 0]["StrategyReturn"] > 0).mean() * 100
    return total_return, win_ratio

# --------- ML PREDICTION ----------
def ml_predict(df):
    df = df.dropna()
    X = df[["RSI", "20MA", "50MA"]]
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)  # Next day up/down
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train.ravel())  # Flatten y_train
    preds = model.predict(X_test)
    acc = accuracy_score(y_test.ravel(), preds) * 100  # Flatten y_test too
    return acc


# --------- LOG TO GOOGLE SHEETS ----------
def log_to_sheets(sheet, stock, total_return, win_ratio, acc):
    try:
        worksheet = sheet.worksheet("TradeLog")
    except:
        worksheet = sheet.add_worksheet(title="TradeLog", rows="100", cols="10")
        worksheet.append_row(["Stock", "TotalReturn%", "WinRatio%", "MLAccuracy%", "Timestamp"])
    
    # Handle NaN values before sending to Google Sheets
    safe_total_return = 0 if pd.isna(total_return) else round(total_return, 2)
    safe_win_ratio = 0 if pd.isna(win_ratio) else round(win_ratio, 2)
    safe_acc = 0 if pd.isna(acc) else round(acc, 2)

    worksheet.append_row([stock, safe_total_return, safe_win_ratio, safe_acc, str(datetime.now())])
    logging.info(f"Logged {stock} results to Google Sheets.")


# --------- MAIN FUNCTION ----------
def run_algo():
    stocks = ["RELIANCE", "TCS", "HDFCBANK"]
    sheet = connect_google_sheets()

    for stock in stocks:
        df = fetch_data(stock)
        df = add_indicators(df)
        df = generate_signals(df)
        total_return, win_ratio = backtest(df)
        acc = ml_predict(df)

        print(f"\n📊 {stock} Results:")
        print(f"Total Return: {total_return:.2f}% | Win Ratio: {win_ratio:.2f}% | ML Accuracy: {acc:.2f}%")

        if sheet:
            log_to_sheets(sheet, stock, total_return, win_ratio, acc)

if __name__ == "__main__":
    run_algo()
