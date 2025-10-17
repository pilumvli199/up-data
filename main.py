import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import io
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import json
import time

# ======================== CONFIGURATION ========================
UPSTOX_API_KEY = "your_upstox_api_key"
UPSTOX_ACCESS_TOKEN = "your_upstox_access_token"
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_telegram_chat_id"

# Upstox API endpoints
BASE_URL = "https://api.upstox.com/v2"
HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
}

# ======================== NIFTY 50 STOCKS ========================
NIFTY_50_STOCKS = [
    "NSE_EQ|INE040A01034",  # HDFCBANK
    "NSE_EQ|INE002A01018",  # RELIANCE
    "NSE_EQ|INE467B01029",  # TATAMOTORS
    "NSE_EQ|INE123W01016",  # INFY
    "NSE_EQ|INE009A01021",  # ICICIBANK
    "NSE_EQ|INE155A01022",  # TATASTEEL
    "NSE_EQ|INE019A01038",  # AXISBANK
    "NSE_EQ|INE081A01020",  # HINDUNILVR
    "NSE_EQ|INE028A01039",  # BHARTIARTL
    "NSE_EQ|INE066A01021",  # ITC
    "NSE_EQ|INE018A01030",  # WIPRO
    "NSE_EQ|INE062A01020",  # SBIN
    "NSE_EQ|INE120A01034",  # SUNPHARMA
    "NSE_EQ|INE030A01027",  # KOTAKBANK
    "NSE_EQ|INE528G01035",  # ULTRACEMCO
    "NSE_EQ|INE854D01024",  # TITAN
    "NSE_EQ|INE205A01025",  # MARUTI
    "NSE_EQ|INE106A01023",  # LT
    "NSE_EQ|INE758T01015",  # TECHM
    "NSE_EQ|INE239A01016",  # HCLTECH
    "NSE_EQ|INE860A01027",  # HUL
    "NSE_EQ|INE256A01028",  # BAJFINANCE
    "NSE_EQ|INE101D01020",  # POWERGRID
    "NSE_EQ|INE121A01024",  # ASIANPAINT
    "NSE_EQ|INE021A01026",  # M&M
    "NSE_EQ|INE192A01025",  # NESTLEIND
    "NSE_EQ|INE180A01020",  # ADANIPORTS
    "NSE_EQ|INE169A01031",  # DRREDDY
    "NSE_EQ|INE238A01034",  # BAJAJFINSV
    "NSE_EQ|INE523A01024",  # ONGC
    "NSE_EQ|INE742F01042",  # ADANIENT
    "NSE_EQ|INE089A01023",  # JSWSTEEL
    "NSE_EQ|INE216A01030",  # COALINDIA
    "NSE_EQ|INE288A01027",  # HINDALCO
    "NSE_EQ|INE070A01015",  # NTPC
    "NSE_EQ|INE145A01014",  # SBILIFE
    "NSE_EQ|INE001A01036",  # EICHERMOT
    "NSE_EQ|INE752E01010",  # HDFCLIFE
    "NSE_EQ|INE040A01026",  # GRASIM
    "NSE_EQ|INE114A01011",  # DIVISLAB
    "NSE_EQ|INE195A01028",  # CIPLA
    "NSE_EQ|INE129A01019",  # BPCL
    "NSE_EQ|INE079A01024",  # APOLLOHOSP
    "NSE_EQ|INE047A01021",  # TATACONSUM
    "NSE_EQ|INE755A01021",  # BRITANNIA
    "NSE_EQ|INE220A01025",  # INDUSINDBK
    "NSE_EQ|INE110A01025",  # HEROMOTOCO
    "NSE_EQ|INE160A01022",  # BAJAJ-AUTO
    "NSE_EQ|INE213A01029",  # TATAPOWER
    "NSE_EQ|INE797F01020",  # ADANIGREEN
]

NIFTY_50_INDEX = "NSE_INDEX|Nifty 50"

# ======================== HELPER FUNCTIONS ========================

def get_historical_candles(instrument_key, interval="5minute", days=5):
    """
    Fetch historical candle data from Upstox
    interval: 1minute, 5minute, 10minute, 30minute, 60minute, 1day, 1week, 1month
    """
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        url = f"{BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
        
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'success' and 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            # Limit to 500 candles
            return candles[:500] if len(candles) > 500 else candles
        return []
    except Exception as e:
        print(f"Error fetching candles for {instrument_key}: {str(e)}")
        return []

def get_option_chain(symbol="NIFTY"):
    """
    Fetch option chain data from Upstox
    """
    try:
        expiry_date = get_next_expiry()
        url = f"{BASE_URL}/option/chain"
        
        params = {
            "instrument_key": f"NSE_FO|{symbol}",
            "expiry_date": expiry_date
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'success':
            return data['data']
        return None
    except Exception as e:
        print(f"Error fetching option chain: {str(e)}")
        return None

def get_next_expiry():
    """Get next Thursday expiry date"""
    today = datetime.now()
    days_ahead = 3 - today.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

def create_candlestick_chart(candles, title):
    """
    Create TradingView style candlestick chart
    candles format: [[timestamp, open, high, low, close, volume], ...]
    """
    if not candles or len(candles) == 0:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_facecolor('white')
    
    # Plot candlesticks
    for idx, row in df.iterrows():
        color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'  # Green/Red
        
        # Draw candlestick body
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        
        # Body
        rect = mpatches.Rectangle((idx, bottom), 0.6, height, 
                                   facecolor=color, edgecolor=color, linewidth=1)
        ax.add_patch(rect)
        
        # Wick (high-low line)
        ax.plot([idx + 0.3, idx + 0.3], [row['low'], row['high']], 
                color=color, linewidth=1)
    
    # Formatting
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(df['low'].min() * 0.999, df['high'].max() * 1.001)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # X-axis labels (show every 50th candle)
    step = max(len(df) // 10, 1)
    xticks = range(0, len(df), step)
    xticklabels = [df.iloc[i]['timestamp'].strftime('%d-%m %H:%M') for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

def format_option_chain_data(option_data):
    """Format option chain data into readable text"""
    if not option_data:
        return "Option chain data not available"
    
    text = "📊 *NIFTY 50 OPTION CHAIN DATA* 📊\n\n"
    text += f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text += f"📅 Expiry: {option_data.get('expiry_date', 'N/A')}\n\n"
    
    # Process call and put options
    for option in option_data.get('data', [])[:20]:  # Limit to 20 strikes
        strike = option.get('strike_price', 'N/A')
        
        text += f"━━━━━ Strike: {strike} ━━━━━\n"
        
        # Call Option
        call_data = option.get('call_options', {})
        if call_data:
            text += f"📞 *CALL*\n"
            text += f"  LTP: ₹{call_data.get('last_price', 0):.2f}\n"
            text += f"  OI: {call_data.get('oi', 0):,}\n"
            text += f"  OI Chg: {call_data.get('oi_change', 0):,}\n"
            text += f"  Volume: {call_data.get('volume', 0):,}\n"
            text += f"  Delta: {call_data.get('delta', 0):.4f}\n"
            text += f"  Theta: {call_data.get('theta', 0):.4f}\n\n"
        
        # Put Option
        put_data = option.get('put_options', {})
        if put_data:
            text += f"📉 *PUT*\n"
            text += f"  LTP: ₹{put_data.get('last_price', 0):.2f}\n"
            text += f"  OI: {put_data.get('oi', 0):,}\n"
            text += f"  OI Chg: {put_data.get('oi_change', 0):,}\n"
            text += f"  Volume: {put_data.get('volume', 0):,}\n"
            text += f"  Delta: {put_data.get('delta', 0):.4f}\n"
            text += f"  Theta: {put_data.get('theta', 0):.4f}\n\n"
    
    return text

async def send_to_telegram(bot_token, chat_id, message=None, photo=None, caption=None):
    """Send message or photo to Telegram"""
    try:
        bot = Bot(token=bot_token)
        
        if photo:
            await bot.send_photo(chat_id=chat_id, photo=photo, caption=caption, parse_mode='Markdown')
        elif message:
            # Split long messages
            if len(message) > 4096:
                parts = [message[i:i+4096] for i in range(0, len(message), 4096)]
                for part in parts:
                    await bot.send_message(chat_id=chat_id, text=part, parse_mode='Markdown')
            else:
                await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        
        return True
    except TelegramError as e:
        print(f"Telegram error: {str(e)}")
        return False
    except Exception as e:
        print(f"Error sending to Telegram: {str(e)}")
        return False

# ======================== MAIN FUNCTION ========================

async def main():
    print("🚀 Starting Upstox Market Data Bot...")
    print("=" * 60)
    
    # 1. Fetch and send Nifty 50 Index data
    print("\n📈 Fetching NIFTY 50 Index data...")
    nifty_candles = get_historical_candles(NIFTY_50_INDEX, interval="5minute", days=5)
    
    if nifty_candles:
        print(f"✅ Fetched {len(nifty_candles)} candles for NIFTY 50")
        chart = create_candlestick_chart(nifty_candles, "NIFTY 50 - 5 Min Chart (Last 500 Candles)")
        if chart:
            await send_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                 photo=chart, caption="📊 *NIFTY 50 Index Chart*")
        time.sleep(2)
    
    # 2. Fetch and send Option Chain data
    print("\n📊 Fetching Option Chain data...")
    option_data = get_option_chain("NIFTY")
    
    if option_data:
        print("✅ Fetched Option Chain data")
        option_text = format_option_chain_data(option_data)
        await send_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message=option_text)
        time.sleep(2)
    
    # 3. Fetch and send Nifty 50 stocks data
    print("\n📈 Fetching NIFTY 50 Stocks data...")
    
    for idx, stock in enumerate(NIFTY_50_STOCKS[:10], 1):  # Process first 10 stocks as example
        print(f"Processing {idx}/10: {stock}")
        
        candles = get_historical_candles(stock, interval="5minute", days=5)
        
        if candles and len(candles) > 0:
            stock_name = stock.split('|')[1] if '|' in stock else stock
            chart = create_candlestick_chart(candles, f"{stock_name} - 5 Min Chart")
            
            if chart:
                await send_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
                                     photo=chart, caption=f"📊 *{stock_name}*")
            time.sleep(3)  # Rate limiting
    
    print("\n" + "=" * 60)
    print("✅ All data sent successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
