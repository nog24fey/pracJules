import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_collector import update_data, TICKERS

st.set_page_config(page_title="Nikkei 225 Buzz Tracker", layout="wide")

DATA_FILE = "market_data.csv"

def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the data collector first.")
        return pd.DataFrame()

st.title("📈 Nikkei 225 Buzz & Price Tracker")
st.markdown("This app traces the stock price and news 'buzz' (volume of news) for major Nikkei 225 companies.")

# Sidebar
st.sidebar.header("Controls")

if st.sidebar.button("🔄 Refresh Data Now"):
    with st.spinner("Fetching latest data..."):
        update_data()
    st.success("Data updated!")
    st.rerun()

df = load_data()

if not df.empty:
    # Filter by Ticker
    all_tickers = df['ticker'].unique()
    # Create a mapping for the multiselect
    ticker_options = [f"{t} ({TICKERS.get(t, 'Unknown')})" for t in all_tickers]

    selected_options = st.sidebar.multiselect(
        "Select Companies",
        ticker_options,
        default=ticker_options[:3] # Select first 3 by default
    )

    # Extract selected tickers
    selected_tickers = [opt.split(" ")[0] for opt in selected_options]

    if selected_tickers:
        df_filtered = df[df['ticker'].isin(selected_tickers)]

        # --- Metrics Row (Latest) ---
        st.subheader("Latest Snapshot")
        cols = st.columns(len(selected_tickers))

        for idx, ticker in enumerate(selected_tickers):
            company_name = TICKERS.get(ticker, ticker)
            latest_row = df_filtered[df_filtered['ticker'] == ticker].iloc[-1]

            # Calculate change if possible
            prev_price = 0
            price_delta = 0
            if len(df_filtered[df_filtered['ticker'] == ticker]) > 1:
                prev_row = df_filtered[df_filtered['ticker'] == ticker].iloc[-2]
                prev_price = prev_row['price']
                price_delta = latest_row['price'] - prev_price

            with cols[idx % len(cols)]:
                st.metric(
                    label=company_name,
                    value=f"¥{latest_row['price']:,.0f}",
                    delta=f"{price_delta:,.0f} JPY" if price_delta != 0 else None
                )
                st.caption(f"Buzz: {latest_row['buzz_score']} articles | Sentiment: {latest_row['sentiment_score']:.2f}")

        # --- Charts ---
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stock Price History")
            fig_price = px.line(
                df_filtered,
                x="timestamp",
                y="price",
                color="name",
                markers=True,
                title="Stock Price (JPY)"
            )
            st.plotly_chart(fig_price, use_container_width=True)

        with col2:
            st.subheader("Buzz Volume History")
            fig_buzz = px.bar(
                df_filtered,
                x="timestamp",
                y="buzz_score",
                color="name",
                barmode="group",
                title="News Volume (Buzz)"
            )
            st.plotly_chart(fig_buzz, use_container_width=True)

        st.subheader("Sentiment Analysis")
        st.markdown("Sentiment score based on keyword analysis of news titles (Positive > 0, Negative < 0).")
        fig_sent = px.line(
            df_filtered,
            x="timestamp",
            y="sentiment_score",
            color="name",
            markers=True
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        # --- Latest News ---
        st.divider()
        st.subheader("Latest Headlines")
        latest_news = df_filtered.sort_values("timestamp", ascending=False).drop_duplicates("ticker")

        for _, row in latest_news.iterrows():
            st.markdown(f"**{row['name']}**: {row['top_headline']} ({row['timestamp']})")

        # --- Raw Data ---
        with st.expander("View Raw Data"):
            st.dataframe(df_filtered)

    else:
        st.info("Please select at least one company from the sidebar.")
else:
    st.warning("No data available.")
