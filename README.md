# Local AI Stock Market Chatbot

A privacy-first, interactive stock market chatbot powered by a local LLaMA model (via Ollama), Streamlit, and yfinance.

---

## 🚀 Overview
This app lets you chat in natural language to get real-time stock prices, historical charts, financial statements, news headlines, and AI-powered explanations—all running locally for privacy and speed.

---

## ✨ Features
- **Natural Language Chat**: Ask about stocks, financial terms, or request charts and comparisons.
- **Real-Time Data**: Get up-to-date prices, volume, and key stats for US and Indian stocks.
- **Interactive Charts**: View historical price, volume, and candlestick charts (Matplotlib & Plotly).
- **Financial Statements**: Instantly fetch balance sheets and income statements.
- **News Headlines**: See the latest news for any stock (if enabled).
- **AI Explanations**: LLaMA explains financial concepts in simple terms.
- **CSV Export**: Download historical data and statements as CSV.
- **Runs Locally**: All AI and data processing is done on your machine—no cloud required.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd finbot
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and Run Ollama
- Download Ollama from [https://ollama.com](https://ollama.com)
- Start the Ollama application.
- Pull a model (e.g., LLaMA 3):
  ```bash
  ollama pull llama3
  ```

### 4. Run the Streamlit App
```bash
streamlit run script.py
```

---

## 💡 Usage
- Type your query in the chat box (e.g., "What is the price of Apple?", "Show me the chart for TCS.NS for 1 year.")
- The AI will interpret your request and fetch data, show charts, or explain concepts.
- Download data as CSV using the provided buttons.

---

## 📦 Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies
- Ollama (for local LLaMA model inference)

---

## 🙏 Credits
- [Streamlit](https://streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Ollama](https://ollama.com)
- [Matplotlib](https://matplotlib.org/), [Plotly](https://plotly.com/)

---

## 📄 License
MIT License (see LICENSE file)
