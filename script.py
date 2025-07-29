# AI Stock Market Chatbot with LLaMA, yfinance, and Streamlit
#
# This script creates a web-based chatbot that leverages a local LLaMA model (via Ollama)
# to provide financial information, including real-time stock data, historical performance,
# financial statements, and explanations of financial concepts.
#
# Author: Gemini
# Date: July 29, 2025
#
# To Run This App:
# 1. Install prerequisites:
#    pip install streamlit pandas yfinance matplotlib plotly ollama
#
# 2. Install and run Ollama:
#    - Download from https://ollama.com
#    - Run the Ollama application.
#    - Pull a model: `ollama pull llama3` (or another model of your choice)
#
# 3. Run the Streamlit app:
#    streamlit run your_script_name.py

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ollama
import json
import re
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple, Dict, List, Union

# --- Page Configuration ---
st.set_page_config(
    page_title="Local AI Stock Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Matplotlib Style ---
# Use a style that looks good in a dark-themed Streamlit app
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#1E1E1E",
    "axes.facecolor": "#1E1E1E",
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": "#CCCCCC",
    "xtick.color": "#CCCCCC",
    "ytick.color": "#CCCCCC",
    "grid.color": "#444444",
    "text.color": "#FFFFFF",
    "figure.figsize": (12, 6)
})

# --- Ollama Model Configuration ---
OLLAMA_MODEL = 'llama3:8b-instruct-q3_K_M' # Make sure you have pulled this model with `ollama pull llama3`

# --- Debug Flag ---
DEBUG = False  # Set to True to show LLaMA tool commands

# --- Period Labels Mapping ---
PERIOD_LABELS = {
    "7d": "7 Days",
    "30d": "30 Days",
    "90d": "90 Days",
    "1y": "1 Year",
    "5y": "5 Years"
}

# --- Helper Functions ---

def get_stock_ticker(name: str) -> str:
    """
    Tries to find a stock ticker using yfinance's search.
    This is a simple implementation and might not always be accurate.
    For Indian stocks, it's better to use ".NS" suffix for NSE, e.g., "RELIANCE.NS".
    """
    try:
        # yfinance search is not officially supported and can be unreliable.
        # A more robust solution would use a dedicated financial data API.
        # For this example, we'll rely on common conventions.
        if "tcs" in name.lower(): return "TCS.NS"
        if "reliance" in name.lower(): return "RELIANCE.NS"
        if "hdfc" in name.lower(): return "HDFCBANK.NS"
        if "infosys" in name.lower(): return "INFY.NS"
        
        # A simple fallback for US stocks
        results = yf.Tickers(name).tickers
        if results:
            return list(results.keys())[0]
        return name.upper() # Default to uppercase name as ticker
    except Exception:
        return name.upper()

def get_stock_data(ticker_symbol: str) -> Tuple[Dict, Optional[str]]:
    """Fetches real-time data for a given stock ticker. Returns (data, error)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Check if data was found
        if not info or 'currentPrice' not in info:
            return None, f"Could not find data for ticker '{ticker_symbol}'. Please check the symbol and try again."

        data = {
            "name": info.get('longName', ticker_symbol),
            "symbol": info.get('symbol', ticker_symbol),
            "price": info.get('currentPrice', 'N/A'),
            "day_high": info.get('dayHigh', 'N/A'),
            "day_low": info.get('dayLow', 'N/A'),
            "volume": info.get('volume', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "market_cap": info.get('marketCap', 'N/A'),
            "previous_close": info.get('previousClose', 'N/A'),
        }
        return data, None
    except Exception as e:
        st.error(f"An error occurred while fetching data for {ticker_symbol} from yfinance. Please try again later.")
        return None, f"An error occurred while fetching data for {ticker_symbol}."

@st.cache_data(show_spinner=False)
def get_historical_data(ticker_symbol: str, period: str = "1mo") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetches historical stock data. Returns (DataFrame, error)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period)
        if hist.empty:
            st.warning(f"No historical data found for '{ticker_symbol}' for the period '{period}'. Please check the ticker and period.")
            return None, f"Could not retrieve historical data for '{ticker_symbol}' for the period '{period}'."
        return hist, None
    except Exception as e:
        st.error(f"An error occurred while fetching historical data for {ticker_symbol}. Please try again later.")
        return None, f"An error occurred while fetching historical data for {ticker_symbol}."

@st.cache_data(show_spinner=False)
def get_balance_sheet(ticker_symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetches the balance sheet for a given stock ticker. Returns (DataFrame, error)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        balance_sheet = stock.balance_sheet
        if balance_sheet.empty:
            st.warning(f"No balance sheet data found for '{ticker_symbol}'.")
            return None, f"Could not retrieve balance sheet for '{ticker_symbol}'."
        return balance_sheet, None
    except Exception as e:
        st.error(f"An error occurred while fetching the balance sheet for {ticker_symbol}. Please try again later.")
        return None, f"An error occurred while fetching the balance sheet for {ticker_symbol}: {e}"

@st.cache_data(show_spinner=False)
def get_income_statement(ticker_symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetches the income statement for a given stock ticker. Returns (DataFrame, error)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        income_stmt = stock.income_stmt
        if income_stmt.empty:
            st.warning(f"No income statement data found for '{ticker_symbol}'.")
            return None, f"Could not retrieve income statement for '{ticker_symbol}'."
        return income_stmt, None
    except Exception as e:
        st.error(f"An error occurred while fetching the income statement for {ticker_symbol}. Please try again later.")
        return None, f"An error occurred while fetching the income statement for {ticker_symbol}: {e}"


def plot_price_history(df: pd.DataFrame, ticker_symbol: str, period_label: str) -> None:
    """Generates a line chart for historical price data using Matplotlib."""
    fig, ax = plt.subplots()
    df['Close'].plot(ax=ax, color='cyan', grid=True)
    ax.set_title(f"{ticker_symbol} - Closing Price ({period_label})", color='white')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend([f'{ticker_symbol} Close'])
    st.pyplot(fig)

def plot_volume(df: pd.DataFrame, ticker_symbol: str, period_label: str) -> None:
    """Generates a bar chart for trading volume using Matplotlib."""
    fig, ax = plt.subplots()
    df['Volume'].plot(kind='bar', ax=ax, color='magenta')
    ax.set_title(f"{ticker_symbol} - Trading Volume ({period_label})", color='white')
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    # Improve x-axis labels for readability
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in df.index], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

def plot_candlestick(df: pd.DataFrame, ticker_symbol: str, period_label: str) -> None:
    """Generates an interactive candlestick chart using Plotly."""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                           open=df['Open'],
                                           high=df['High'],
                                           low=df['Low'],
                                           close=df['Close'])])
    fig.update_layout(
        title=f'{ticker_symbol} Candlestick Chart ({period_label})',
        yaxis_title='Stock Price (USD)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

def call_llama(prompt: str) -> Optional[str]:
    """Sends a prompt to the local LLaMA model and gets a response as a string, or None on error."""
    try:
        # The ollama library returns a dictionary, the response is in the 'response' key
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        st.error("Please ensure Ollama is running and you have pulled the model with `ollama pull llama3`.")
        return None

def format_number(n: Union[float, int, str, None]) -> str:
    """Format large numbers with commas and units (K, M, B)."""
    if n is None or n == 'N/A':
        return 'N/A'
    try:
        n = float(n)
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        else:
            return f"{n:,.2f}"
    except Exception:
        return str(n)

# --- System Prompt for LLaMA ---
# This prompt guides the model to act as a financial assistant and use specific tool commands.
SYSTEM_PROMPT = """
You are a sophisticated AI financial assistant. Your role is to interpret user queries and respond with either a direct textual answer or by using a specific tool.

You have access to the following tools:
1. `[GET_PRICE:TICKER]` - To get the current price and key stats of a stock.
   - Example: User asks "What is the price of Apple?". You respond with `[GET_PRICE:AAPL]`.

2. `[PLOT_HISTORY:TICKER:PERIOD]` - To plot the historical price and volume of a stock.
   - `PERIOD` can be `7d`, `30d`, `90d`, `1y`, `5y`.
   - Example: User asks "Show me the chart for Google over the last month". You respond with `[PLOT_HISTORY:GOOGL:30d]`.

3. `[PLOT_CANDLESTICK:TICKER:PERIOD]` - To show a detailed candlestick chart.
   - Example: User asks "Give me a technical view of Microsoft for 90 days". You respond with `[PLOT_CANDLESTICK:MSFT:90d]`.

4. `[COMPARE:TICKER1:TICKER2:PERIOD]` - To compare two stocks over a period.
   - Example: User asks "Compare Reliance and HDFC over the past month". You respond with `[COMPARE:RELIANCE.NS:HDFCBANK.NS:30d]`.

5. `[GET_BALANCE_SHEET:TICKER]` - To get the latest balance sheet of a company.
    - Example: User asks "Show me Apple's balance sheet". You respond with `[GET_BALANCE_SHEET:AAPL]`.

6. `[GET_INCOME_STATEMENT:TICKER]` - To get the latest income statement of a company.
    - Example: User asks "What is the income statement for MSFT?". You respond with `[GET_INCOME_STATEMENT:MSFT]`.

7. `[EXPLAIN:CONCEPT]` - For financial term explanations.
   - Example: User asks "What is a P/E ratio?". You respond with `[EXPLAIN:P/E ratio]`. The user's query will be sent back to you for a full explanation.

**Rules:**
- **ALWAYS** use the tool format when a tool is applicable.
- For Indian stocks (like TCS, Reliance), use the `.NS` suffix (e.g., `TCS.NS`, `RELIANCE.NS`). For major US stocks (Apple, Google), use their standard tickers (e.g., `AAPL`, `GOOGL`).
- If you are unsure about a ticker, make a best guess.
- If the user asks a general question that doesn't fit a tool (e.g., "What are some good tech stocks?"), answer it directly without using a tool command.
- Only respond with the tool command, nothing else.
"""

def get_llama_tool_command(user_query: str) -> str:
    """Gets the specific tool command from LLaMA for a user query."""
    prompt = f"{SYSTEM_PROMPT}\n\nUser Query: \"{user_query}\"\n\nYour Response:"
    response = call_llama(prompt)
    return response.strip() if response else ""


# --- Streamlit UI ---
st.title("📈 Local AI Stock Analyst")
st.caption(f"Powered by a local LLaMA model (`{OLLAMA_MODEL}`) and `yfinance`")

# Sidebar for instructions and controls
with st.sidebar:
    st.header("How it Works")
    st.markdown("""
    This chatbot uses a local Large Language Model (LLaMA 3) to understand your questions about the stock market.

    1.  **Type a query** in the chat box.
    2.  The **LLaMA model** interprets your request and decides which financial tool to use.
    3.  The app fetches **live data** using the `yfinance` library.
    4.  **Charts and data** are displayed in the chat.

    **Example Queries:**
    - *What is the current price of Tesla?*
    - *Explain what P/E ratio means.*
    - *Show me the chart for INFY.NS over the last 90 days.*
    - *What is Apple's balance sheet?*
    - *Show me the income statement for Microsoft.*
    - *Compare Reliance and HDFC over the past month.*
    """)
    st.header("Setup Check")
    st.info("Make sure **Ollama is running** on your computer and you have pulled a model (e.g., `ollama pull llama3`).")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with the stock market today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if content is a dict (custom data) or string
        if isinstance(message["content"], dict):
            if message["content"]["type"] == "stock_data":
                st.markdown(message["content"]["text"])
            elif message["content"]["type"] == "financial_statement":
                st.markdown(f"#### {message['content']['title']}")
                st.info("Financial data is available above, but not stored in chat history for performance reasons.")
            elif message["content"]["type"] == "plot":
                 st.markdown(f"*(Displayed a {message['content']['plot_type']} for {message['content']['ticker']})*")
        else:
            st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("Ask about stocks..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display thinking indicator and get LLaMA's response
    with st.chat_message("assistant"):
        with st.spinner("🧠 LLaMA is thinking..."):
            tool_command = get_llama_tool_command(prompt)

        # --- Tool Execution Logic ---
        if not tool_command:
             st.warning("The model could not determine an action. Please try rephrasing your query.")
        
        if DEBUG:
            st.write(f"**LLaMA suggested action:** `{tool_command}`")
        
        elif tool_command.startswith("[GET_PRICE:"):
            ticker = tool_command.split(":")[1].strip("]")
            with st.spinner(f"Fetching data for {ticker}..."):
                data, error = get_stock_data(ticker)
                if error:
                    st.error(error)
                else:
                    response_text = f"""
                    ### Stock Information for {data['name']} ({data['symbol']})
                    | Metric          | Value                  |
                    |-----------------|------------------------|
                    | **Current Price** | **{format_number(data['price'])}** |
                    | Day High        | {format_number(data['day_high'])}     |
                    | Day Low         | {format_number(data['day_low'])}      |
                    | Previous Close  | {format_number(data['previous_close'])}|
                    | Volume          | {format_number(data['volume'])}     |
                    | Market Cap      | {format_number(data['market_cap'])}  |
                    | P/E Ratio       | {format_number(data['pe_ratio'])}  |
                    """
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "stock_data", "text": response_text}})

        elif tool_command.startswith("[PLOT_HISTORY:"):
            parts = tool_command.strip("[]").split(":")
            ticker, period = parts[1], parts[2]
            period_label = PERIOD_LABELS.get(period, "Custom Period")
            
            with st.spinner(f"Generating charts for {ticker} over {period_label}..."):
                hist_df, error = get_historical_data(ticker, period)
                if error:
                    st.warning(error)
                else:
                    st.markdown(f"#### Historical Data for {ticker} ({period_label})")
                    plot_price_history(hist_df, ticker, period_label)
                    plot_volume(hist_df, ticker, period_label)
                    if hist_df is not None and not hist_df.empty:
                        csv = hist_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Historical Data as CSV",
                            data=csv,
                            file_name=f"{ticker}_history_{period}.csv",
                            mime="text/csv"
                        )
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "plot", "plot_type": "history", "ticker": ticker, "period": period}})

        elif tool_command.startswith("[PLOT_CANDLESTICK:"):
            parts = tool_command.strip("[]").split(":")
            ticker, period = parts[1], parts[2]
            period_label = PERIOD_LABELS.get(period, "30 Days")

            with st.spinner(f"Generating candlestick chart for {ticker}..."):
                hist_df, error = get_historical_data(ticker, period)
                if error:
                    st.warning(error)
                else:
                    plot_candlestick(hist_df, ticker, period_label)
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "plot", "plot_type": "candlestick", "ticker": ticker, "period": period}})
        
        elif tool_command.startswith("[COMPARE:"):
            parts = tool_command.strip("[]").split(":")
            ticker1, ticker2, period = parts[1], parts[2], parts[3]
            period_label = PERIOD_LABELS.get(period, "30 Days")

            with st.spinner(f"Comparing {ticker1} and {ticker2}..."):
                hist1, err1 = get_historical_data(ticker1, period)
                hist2, err2 = get_historical_data(ticker2, period)

                if err1 or err2:
                    st.warning(err1 or err2)
                else:
                    # Normalize data to compare performance
                    normalized1 = (hist1['Close'] / hist1['Close'].iloc[0]) * 100
                    normalized2 = (hist2['Close'] / hist2['Close'].iloc[0]) * 100
                    
                    fig, ax = plt.subplots()
                    normalized1.plot(ax=ax, label=f'{ticker1} Performance', color='cyan')
                    normalized2.plot(ax=ax, label=f'{ticker2} Performance', color='magenta')
                    ax.set_title(f"Stock Performance Comparison ({period_label})", color='white')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Normalized Price (Starts at 100)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "plot", "plot_type": "comparison", "ticker": f"{ticker1} vs {ticker2}", "period": period}})

        elif tool_command.startswith("[GET_BALANCE_SHEET:"):
            ticker = tool_command.split(":")[1].strip("]")
            with st.spinner(f"Fetching Balance Sheet for {ticker}..."):
                balance_sheet, error = get_balance_sheet(ticker)
                if error:
                    st.warning(error)
                else:
                    title = f"Balance Sheet for {ticker}"
                    st.markdown(f"#### {title}")
                    st.dataframe(balance_sheet)
                    if balance_sheet is not None and not balance_sheet.empty:
                        csv = balance_sheet.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Balance Sheet as CSV",
                            data=csv,
                            file_name=f"{ticker}_balance_sheet.csv",
                            mime="text/csv"
                        )
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "financial_statement", "title": title, "ticker": ticker}})


        elif tool_command.startswith("[GET_INCOME_STATEMENT:"):
            ticker = tool_command.split(":")[1].strip("]")
            with st.spinner(f"Fetching Income Statement for {ticker}..."):
                income_statement, error = get_income_statement(ticker)
                if error:
                    st.warning(error)
                else:
                    title = f"Income Statement for {ticker}"
                    st.markdown(f"#### {title}")
                    st.dataframe(income_statement)
                    if income_statement is not None and not income_statement.empty:
                        csv = income_statement.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Income Statement as CSV",
                            data=csv,
                            file_name=f"{ticker}_income_statement.csv",
                            mime="text/csv"
                        )
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "financial_statement", "title": title, "ticker": ticker}})


        elif tool_command.startswith("[EXPLAIN:"):
            concept = tool_command.split(":")[1].strip("]")
            with st.spinner(f"Asking LLaMA to explain '{concept}'..."):
                explanation_prompt = f"Explain the financial term '{concept}' in a clear and concise way, as if you were talking to a beginner."
                explanation = call_llama(explanation_prompt)
                st.markdown(explanation)
                st.session_state.messages.append({"role": "assistant", "content": explanation})

        else: # Default case: LLaMA provides a direct answer
            st.markdown(tool_command)
            st.session_state.messages.append({"role": "assistant", "content": tool_command})
