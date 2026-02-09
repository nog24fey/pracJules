import yfinance as yf
import feedparser
import pandas as pd
import datetime
import os
import urllib.parse
import time

# Nikkei 225 subset for MVP
TICKERS = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "9983.T": "ファーストリテイリング",
    "8035.T": "東京エレクトロン",
    "6861.T": "キーエンス",
    "7974.T": "任天堂",
    "9432.T": "日本電信電話",
    "6501.T": "日立製作所",
    "8306.T": "三菱UFJフィナンシャル・グループ"
}

DATA_FILE = "market_data.csv"

def fetch_stock_price(ticker):
    """
    Fetches the current stock price.
    """
    try:
        stock = yf.Ticker(ticker)
        # Try fast_info first
        try:
            price = stock.fast_info.last_price
        except:
            # Fallback to history
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
            else:
                return None
        return price
    except Exception as e:
        print(f"Error fetching stock for {ticker}: {e}")
        return None

def fetch_buzz_and_sentiment(company_name):
    """
    Fetches news from Google News RSS for the company name.
    Returns:
        buzz_score (int): Number of articles found (proxy for attention).
        sentiment_score (float): Simple keyword-based sentiment.
        top_headline (str): The most recent headline.
    """
    encoded_query = urllib.parse.quote(company_name)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"

    try:
        feed = feedparser.parse(rss_url)
        entries = feed.entries

        buzz_score = len(entries)

        # Simple sentiment heuristic for Japanese titles
        positive_keywords = ["急騰", "上昇", "好調", "最高益", "増益", "買収", "提携", "ストップ高", "成長", "期待"]
        negative_keywords = ["急落", "下落", "不調", "赤字", "減益", "中止", "撤退", "ストップ安", "懸念", "失望"]

        total_score = 0
        for entry in entries:
            title = entry.title
            score = 0
            for pk in positive_keywords:
                if pk in title:
                    score += 1
            for nk in negative_keywords:
                if nk in title:
                    score -= 1
            total_score += score

        avg_sentiment = total_score / buzz_score if buzz_score > 0 else 0

        if entries:
            top_headline = entries[0].title
        else:
            top_headline = "No recent news"

        return buzz_score, avg_sentiment, top_headline

    except Exception as e:
        print(f"Error fetching news for {company_name}: {e}")
        return 0, 0, "Error"

def update_data():
    """
    Main function to fetch all data and append to CSV.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []

    print(f"Starting data update at {timestamp}...")

    # Check if file exists to load existing data
    existing_data = []
    if os.path.exists(DATA_FILE):
        try:
            df_existing = pd.read_csv(DATA_FILE)
            existing_data = df_existing.to_dict('records')
        except Exception as e:
            print(f"Error reading existing data: {e}")

    for ticker, name in TICKERS.items():
        print(f"Processing {name} ({ticker})...")

        price = fetch_stock_price(ticker)
        if price is None:
             print(f"Skipping {name} due to stock fetch error.")
             continue

        buzz, sentiment, headline = fetch_buzz_and_sentiment(name)

        new_rows.append({
            "timestamp": timestamp,
            "ticker": ticker,
            "name": name,
            "price": price,
            "buzz_score": buzz,
            "sentiment_score": sentiment,
            "top_headline": headline
        })
        time.sleep(1) # Be nice to APIs

    if not new_rows:
        print("No new data collected.")
        return pd.DataFrame(existing_data)

    df_new = pd.DataFrame(new_rows)

    if existing_data:
        df_combined = pd.concat([pd.DataFrame(existing_data), df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Ensure numeric columns are correct type
    df_combined['price'] = pd.to_numeric(df_combined['price'], errors='coerce')
    df_combined['buzz_score'] = pd.to_numeric(df_combined['buzz_score'], errors='coerce')
    df_combined['sentiment_score'] = pd.to_numeric(df_combined['sentiment_score'], errors='coerce')

    df_combined.to_csv(DATA_FILE, index=False)
    print("Data update complete.")
    return df_combined

if __name__ == "__main__":
    update_data()
