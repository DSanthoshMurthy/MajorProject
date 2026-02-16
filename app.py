import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from thefuzz import process
import requests
import io
import os
import logging

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Stock Model Predicter", page_icon="📈", layout="centered")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. LOAD AI MODELS (LOCAL -> FALLBACK) ---
@st.cache_resource
def load_models():
    # A. Load NER (Company Detection)
    # We keep this online because the local model is just for sentiment
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

    # B. Load FinBERT (Sentiment)
    local_path = "model/" # Looks for model.safetensors and config.json here
    
    st.write("🔄 Loading Sentiment Model...")
    
    try:
        # Try loading local files first
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Directory {local_path} not found.")
            
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path)
        
        finbert = pipeline("text-classification", model=model, tokenizer=tokenizer)
        st.toast("✅ Using Custom Local Model!", icon="📂")
        print("Loaded local model from Model/")
        
    except Exception as e:
        logger.warning(f"⚠️ Local model failed: {e}. Falling back to standard FinBERT.")
        print(f"Fallback triggered: {e}")
        finbert = pipeline("text-classification", model="ProsusAI/finbert")
        st.toast("⚠️ Using Standard Online Model", icon="☁️")

    return ner, finbert

ner_pipeline, finbert_pipeline = load_models()

# --- 2. HELPER FUNCTIONS (SEEDING) ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 3. DYNAMIC TICKER MAPPING ---
@st.cache_data
def load_ticker_map():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(io.BytesIO(response.content))
            ticker_map = {}
            for _, row in df.iterrows():
                company_name = str(row['NAME OF COMPANY']).strip()
                symbol = f"{row['SYMBOL']}.NS"
                ticker_map[company_name] = symbol
                ticker_map[str(row['SYMBOL'])] = symbol
            return ticker_map, list(ticker_map.keys())
        else:
            raise Exception("NSE Download failed")
    except Exception as e:
        fallback_map = {
            "RELIANCE INDUSTRIES": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS",
            "JINDAL STEEL & POWER": "JINDALSTEL.NS", "JSW STEEL": "JSWSTEEL.NS",
            "INFOSYS": "INFY.NS", "HDFC BANK": "HDFCBANK.NS", "ZOMATO": "ZOMATO.NS"
        }
        return fallback_map, list(fallback_map.keys())

TICKER_MAP, COMPANY_NAMES = load_ticker_map()

# --- 4. MODEL CLASSES ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32).requires_grad_()
        c0 = torch.zeros(2, x.size(0), 32).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def get_ticker_from_text(text):
    entities = ner_pipeline(text)
    orgs = [e['word'] for e in entities if e['entity_group'] == 'ORG']
    
    if not orgs:
        words = text.split()
        for word in words:
            match, score = process.extractOne(word, COMPANY_NAMES)
            if score > 90:
                orgs = [match]
                break

    if not orgs:
        return None, "No Company Found"
    
    best_match, score = process.extractOne(orgs[0], COMPANY_NAMES)
    if score > 60:
        return TICKER_MAP[best_match], best_match
    return None, f"Unsure ({orgs[0]})"

def train_and_predict(ticker):
    set_seed(42)
    
    df = yf.download(ticker, period="2y", progress=False)
    if df.empty: return None, None
    
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(ticker, axis=1, level=1)
        except: df.columns = df.columns.droplevel(1)
        
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    lookback = 60
    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback])
        
    X_train = torch.from_numpy(np.array(X)).float()
    y_train = torch.from_numpy(np.array(y)).float()
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 20
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
    last_sequence = scaled_data[-lookback:]
    last_tensor = torch.from_numpy(np.array([last_sequence])).float()
    
    with torch.no_grad():
        pred_scaled = model(last_tensor).item()
        
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    curr_price = data[-1][0]
    
    return curr_price, pred_price

# --- 5. UI LAYOUT ---
st.title("🤖 AI Stock Model Predicter (Smart Mode)")
st.markdown("Paste a news headline. Uses **Local Custom Model** if available.")

with st.form("input_form"):
    news_text = st.text_area("News Headline:", height=100)
    submitted = st.form_submit_button("🔮 Analyze Market")

if submitted and news_text:
    
    with st.spinner("🕵️ Scanning NSE Database..."):
        ticker, name = get_ticker_from_text(news_text)
        
    if not ticker:
        st.error(f"Could not identify a stock. (Debug: {name})")
    else:
        st.success(f"**Identified:** {name} ({ticker})")
            
        with st.spinner("🧠 Analyzing Sentiment..."):
            # Using the loaded pipeline (either local or fallback)
            sent_result = finbert_pipeline(news_text)[0]
            label = sent_result['label']
            score = sent_result['score']
            
            if label == "positive": sent_val = score
            elif label == "negative": sent_val = -score
            else: sent_val = 0
            
        with st.spinner(f"📉 Training AI on {ticker}..."):
            curr_price, pred_price = train_and_predict(ticker)
            
        if curr_price:
            tech_change = ((pred_price - curr_price) / curr_price) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                if label == "positive":
                    st.metric("News Sentiment", f"{label.upper()}", f"{score:.2f}", delta_color="normal")
                elif label == "negative":
                    st.metric("News Sentiment", f"{label.upper()}", f"-{score:.2f}", delta_color="inverse")
                else:
                    st.metric("News Sentiment", f"{label.upper()}", f"{score:.2f}", delta_color="off")
            
            with col2:
                st.metric("Tech Prediction", f"₹{pred_price:.2f}", f"{tech_change:.2f}%")
                
            st.divider()
            
            final_score = tech_change + (sent_val * 2.0)
            
            if tech_change < 0 and sent_val > 0.5:
                 verdict = "🔄 POTENTIAL REVERSAL (Buy)"
                 color = "blue"
            elif tech_change > 0 and sent_val < -0.5:
                 verdict = "⚠️ POTENTIAL CRASH (Sell)"
                 color = "orange"
            elif final_score > 0.5:
                verdict = "🚀 STRONG BUY"
                color = "green"
            elif final_score < -0.5:
                verdict = "🔻 STRONG SELL"
                color = "red"
            else:
                verdict = "⚖️ HOLD / UNCERTAIN"
                color = "#FFC300" 
                
            st.markdown(f"<h3 style='text-align: center; color: {color};'>{verdict}</h3>", unsafe_allow_html=True)
        else:
            st.error("Could not fetch market data.")