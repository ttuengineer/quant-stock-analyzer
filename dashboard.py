import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from stock_analyzer import StockAnalyzer
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Opportunity Scanner", layout="wide")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data():
    analyzer = StockAnalyzer()
    # Get ALL stocks, not just top 50
    return analyzer.analyze_all_stocks(top_n=500)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_chart_data(ticker, period="3mo"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def plot_stock_chart(ticker, df):
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))

    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='lightblue',
        opacity=0.3
    ))

    # Add moving averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA20'],
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA50'],
        name='SMA 50',
        line=dict(color='blue', width=1)
    ))

    fig.update_layout(
        title=f"{ticker} Price Chart",
        yaxis_title="Price",
        xaxis_title="Date",
        height=500,
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False
    )

    return fig

def main():
    st.title("🚀 Stock Opportunity Scanner")
    st.markdown("### AI-Powered Investment Analysis Tool")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Score filter
    min_score = st.sidebar.slider("Minimum Opportunity Score", 0, 100, 50)

    # RSI filter
    rsi_range = st.sidebar.slider("RSI Range", 0, 100, (20, 80))

    # P/E filter
    max_pe = st.sidebar.number_input("Max P/E Ratio", value=50, min_value=1)

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    with st.spinner("Analyzing stocks... This may take a minute..."):
        df = load_stock_data()

    # Apply filters
    filtered_df = df[
        (df['Score'] >= min_score) &
        (df['RSI'] >= rsi_range[0]) &
        (df['RSI'] <= rsi_range[1])
    ]

    # Filter P/E if available
    if 'P/E' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['P/E'].isna()) | (filtered_df['P/E'] <= max_pe)]

    # Main content
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Stocks Analyzed", len(df))

    with col2:
        st.metric("Opportunities Found", len(filtered_df))

    with col3:
        avg_score = filtered_df['Score'].mean() if not filtered_df.empty else 0
        st.metric("Avg Opportunity Score", f"{avg_score:.1f}")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Top Opportunities", "📈 Charts", "🎯 Buy Signals", "📉 Value Plays"])

    with tab1:
        st.subheader("Top Investment Opportunities")

        if not filtered_df.empty:
            # Format the dataframe for display
            display_df = filtered_df.copy()

            # Format numeric columns
            format_cols = {
                'Score': '{:.1f}',
                'Current Price': '${:.2f}',
                '1M Change %': '{:.1f}%',
                '3M Change %': '{:.1f}%',
                'RSI': '{:.1f}',
                'Volume Ratio': '{:.2f}',
                'P/E': '{:.1f}',
                'PEG': '{:.2f}',
                'ROE': '{:.2%}',
                'Profit Margin': '{:.2%}',
                'Analyst Rating': '{:.1f}',
                '52W Low %': '{:.1f}%',
                '52W High %': '{:.1f}%'
            }

            # Color code the Score column
            def color_score(val):
                if val >= 70:
                    return 'background-color: #90EE90'  # Light green
                elif val >= 50:
                    return 'background-color: #FFFFE0'  # Light yellow
                else:
                    return 'background-color: #FFB6C1'  # Light red

            styled_df = display_df.style.applymap(color_score, subset=['Score'])

            st.dataframe(styled_df, height=500)

            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "stock_opportunities.csv",
                "text/csv"
            )
        else:
            st.warning("No stocks match the current filters. Try adjusting your criteria.")

    with tab2:
        st.subheader("Stock Charts")

        if not filtered_df.empty:
            selected_ticker = st.selectbox(
                "Select a stock to chart:",
                filtered_df['Ticker'].tolist()
            )

            period = st.select_slider(
                "Time Period",
                options=["1mo", "3mo", "6mo", "1y"],
                value="3mo"
            )

            if selected_ticker:
                chart_data = get_stock_chart_data(selected_ticker, period)
                fig = plot_stock_chart(selected_ticker, chart_data)
                st.plotly_chart(fig, width='stretch')

                # Stock info
                col1, col2, col3, col4 = st.columns(4)
                stock_row = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]

                col1.metric("Opportunity Score", f"{stock_row['Score']:.1f}")
                col2.metric("Current Price", f"${stock_row['Current Price']:.2f}")
                col3.metric("1M Change", f"{stock_row['1M Change %']:.1f}%")
                col4.metric("RSI", f"{stock_row['RSI']:.1f}")

    with tab3:
        st.subheader("Strong Buy Signals")

        # Filter for strong buy signals
        buy_signals = filtered_df[
            (filtered_df['Score'] > 65) &
            (filtered_df['RSI'] < 70) &
            (filtered_df['1M Change %'] > -5)
        ].sort_values('Score', ascending=False)

        if not buy_signals.empty:
            st.success(f"Found {len(buy_signals)} strong buy signals!")

            for _, row in buy_signals.head(5).iterrows():
                with st.expander(f"🎯 {row['Ticker']} - Score: {row['Score']:.1f}"):
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Price", f"${row['Current Price']:.2f}")
                    col2.metric("P/E Ratio", f"{row['P/E']:.1f}" if pd.notna(row['P/E']) else "N/A")
                    col3.metric("RSI", f"{row['RSI']:.1f}")

                    st.write("**Key Indicators:**")
                    st.write(f"- 1-Month Performance: {row['1M Change %']:.1f}%")
                    st.write(f"- 3-Month Performance: {row['3M Change %']:.1f}%")
                    st.write(f"- Distance from 52W Low: {row['52W Low %']:.1f}%")
                    st.write(f"- Analyst Rating: {row['Analyst Rating']:.1f}" if pd.notna(row['Analyst Rating']) else "N/A")
        else:
            st.info("No strong buy signals at current filter settings. Try adjusting your criteria.")

    with tab4:
        st.subheader("Value Investment Opportunities")

        # Filter for value plays
        value_plays = filtered_df[
            (filtered_df['P/E'].notna()) &
            (filtered_df['P/E'] < 20) &
            (filtered_df['P/E'] > 0)
        ].sort_values('P/E')

        if not value_plays.empty:
            # Create scatter plot
            fig = px.scatter(
                value_plays,
                x='P/E',
                y='Score',
                size='Current Price',
                color='RSI',
                hover_data=['Ticker', 'Current Price', '1M Change %'],
                title="Value vs Opportunity Score",
                color_continuous_scale='RdYlGn_r'
            )

            st.plotly_chart(fig, width='stretch')

            st.write("**Top Value Picks (Low P/E, High Score):**")
            value_picks = value_plays.head(10)[['Ticker', 'Score', 'P/E', 'Current Price', 'PEG', 'ROE']]
            st.dataframe(value_picks)
        else:
            st.info("No value plays found with current filters.")

    # Footer
    st.markdown("---")
    st.markdown("### How the Scoring Works:")
    with st.expander("📚 Methodology"):
        st.write("""
        **The Opportunity Score (0-100) combines:**

        **Technical Indicators (40%)**
        - Price momentum (1-month and 3-month changes)
        - Moving average positions (20, 50-day)
        - RSI levels (oversold/overbought)
        - MACD signals
        - Volume surges
        - Distance from 52-week high/low

        **Fundamental Analysis (60%)**
        - P/E ratio (value assessment)
        - PEG ratio (growth vs price)
        - Profit margins
        - Return on equity (ROE)
        - Revenue growth
        - Analyst recommendations

        **Higher scores indicate:**
        - Strong momentum with good value
        - Positive technical signals
        - Solid fundamentals
        - Potential for price appreciation
        """)

    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. Always do your own research and consider consulting a financial advisor before making investment decisions.")

if __name__ == "__main__":
    main()