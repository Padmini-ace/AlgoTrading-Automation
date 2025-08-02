import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import pandas as pd
from datetime import datetime

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Open Google Sheet
SHEET_NAME = "YourSheetName"   # 🔹 Replace with your Google Sheet's name
worksheet = client.open(SHEET_NAME).worksheet("TradeLog")

# Fetch data
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# Streamlit dashboard
st.set_page_config(page_title="Algo Trading Dashboard", layout="wide")

st.title("📊 Algo Trading Dashboard")
st.subheader("Live Trade Log Results")

if df.empty:
    st.warning("No data found in Google Sheet.")
else:
    # Show trade log
    st.dataframe(df)

    # Summary stats
    st.subheader("📈 Performance Summary")
    avg_return = df["TotalReturn%"].mean()
    avg_accuracy = df["MLAccuracy%"].mean()
    st.write(f"✅ Average Return: {avg_return:.2f}%")
    st.write(f"🎯 Average ML Accuracy: {avg_accuracy:.2f}%")

    # Chart visualization
    st.subheader("📊 Returns Over Time")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    st.line_chart(df.set_index("Timestamp")[["TotalReturn%"]])

