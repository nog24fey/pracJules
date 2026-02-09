# Nikkei 225 Buzz & Price Tracker

This application traces the stock price and "topic buzz" (news volume) for major Nikkei 225 companies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Collect Data**:
   Run the data collector script to fetch the latest stock prices and news data. You can run this periodically (e.g., via cron) or click the "Refresh" button in the app.
   ```bash
   python3 data_collector.py
   ```

2. **Run Dashboard**:
   Start the Streamlit app to visualize the data.
   ```bash
   streamlit run app.py
   ```

## Features

- **Stock Price**: Fetches real-time/delayed prices via Yahoo Finance.
- **Buzz Score**: Counts the volume of recent news articles via Google News RSS.
- **Sentiment Score**: A simple keyword-based sentiment analysis of news titles.
- **Visualizations**: Interactive charts for price and buzz trends.

## Configuration

You can add more tickers to the `TICKERS` dictionary in `data_collector.py`.
