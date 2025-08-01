import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Connect to Google Sheets
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("AlgoTrading").worksheet("TradeLog")
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Streamlit Dashboard
st.set_page_config(page_title="AlgoTrading Dashboard", layout="wide")
st.title("📊 AlgoTrading Performance Dashboard")

if not df.empty:
    st.subheader("Latest Results")
    last = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stock", last["Stock"])
    col2.metric("Return", f"{last['TotalReturn%']}%")
    col3.metric("Win Ratio", f"{last['WinRatio%']}%")
    col4.metric("ML Accuracy", f"{last['MLAccuracy%']}%")

    st.subheader("All Trades Log")
    st.dataframe(df)

    st.subheader("Performance Chart")
    st.line_chart(df[["TotalReturn%", "MLAccuracy%"]])
else:
    st.warning("No data found in TradeLog yet. Run main.py first!")
