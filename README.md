# Nikkei 225 Buzz & Price Tracker

This project provides a Jupyter Notebook to trace the stock price and "topic buzz" (news volume) for major Nikkei 225 companies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: The notebook itself will also attempt to install necessary packages in the first cell.)

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook nikkei_buzz_tracker.ipynb
   ```

## Usage

1. Open `nikkei_buzz_tracker.ipynb`.
2. Run all cells.
   - The notebook will fetch the latest stock prices via Yahoo Finance.
   - It will search Google News RSS for article counts (buzz).
   - It will update the local data file `market_data.csv`.
   - It will display interactive charts (Price, Buzz, Sentiment) and a table of the latest headlines directly in the notebook.

## Features

- **Stock Price**: Fetches real-time/delayed prices via Yahoo Finance.
- **Buzz Score**: Counts the volume of recent news articles via Google News RSS.
- **Sentiment Score**: A simple keyword-based sentiment analysis of news titles.
- **Visualizations**: Interactive Plotly charts.

## Configuration

You can modify the `TICKERS` dictionary in the notebook to add or remove companies.
