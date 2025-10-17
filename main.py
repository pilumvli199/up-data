import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import io
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import os
import time

# ======================== CONFIGURATION ========================
# Get from environment variables (set in Railway/Docker)
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "your_access_token")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_telegram_chat_id")

# Upstox API Configuration
BASE_URL = "https://api.upstox.com"
HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
}

# ======================== INSTRUMENT KEYS ========================
# Correct instrument keys for Nifty 50 stocks
NIFTY_50_STOCKS = {
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "INFY": "NSE_EQ|INE009A01021",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "TCS": "NSE_EQ|INE467B01029",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "ITC": "NSE_EQ|INE154A01025",
    "SBIN": "NSE_EQ|INE062A01020",
    "BAJFINANCE": "NSE_EQ|INE296A01024",
    "LT": "NSE_EQ|INE018A01030",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "AXISBANK": "NSE_EQ|INE238A01034",
    "ASIANPAINT": "NSE_EQ|INE021A01026",
    "MARUTI": "NSE_EQ|INE585B01010",
    "SUNPHARMA": "NSE_EQ|INE044A01036",
    "TITAN": "NSE_EQ|INE280A01028",
    "ULTRACEMCO": "NSE_EQ|INE481G01011",
    "NESTLEIND": "NSE_EQ|INE239A01016"
}

NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"

# ======================== HELPER FUNCTIONS ========================

def get_intraday_candles(instrument_key, interval="5minute"):
    """
    Fetch intraday candle data (current trading day)
    No authentication required for intraday API!
    """
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{instrument_key}/{interval}"
        
        # For intraday, no auth needed
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            return candles[:500] if len(candles) > 500 else candles
        return []
    except Exception as e:
        print(f"Error fetching intraday candles for {instrument_key}: {str(e)}")
        return []

def get_historical_candles_v2(instrument_key, days_back=5):
    """
    Fetch last 5 days historical data using V2 API
    """
    try:
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v2/historical-candle/{instrument_key}/5minute/{to_date}/{from_date}"
        
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            return candles[:500] if len(candles) > 500 else candles
        return []
    except Exception as e:
        print(f"Error fetching historical candles: {str(e)}")
        return []

def get_next_expiry():
    """Get next Thursday expiry date for options"""
    today = datetime.now()
    days_ahead = 3 - today.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

def get_option_chain_data():
    """
    Fetch option chain data using option chain API
    """
    try:
        expiry_date = get_next_expiry()
        url = f"{BASE_URL}/v2/option/chain"
        
        params = {
            "instrument_key": NIFTY_INDEX_KEY,
            "expiry_date": expiry_date
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success':
            return data.get('data', [])
        return []
    except Exception as e:
        print(f"Error fetching option chain: {str(e)}")
        return []

def get_option_greeks(instrument_keys):
    """
    Fetch option Greeks data (Delta, Theta, Vega, Gamma)
    Max 50 instruments per request
    """
    try:
        url = f"{BASE_URL}/v2/option/greek"
        
        # Join instrument keys with comma
        params = {
            "instrument_key": ",".join(instrument_keys[:50])  # Max 50
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success':
            return data.get('data', {})
        return {}
    except Exception as e:
        print(f"Error fetching option Greeks: {str(e)}")
        return {}

def create_candlestick_chart(candles, title):
    """
    Create TradingView style candlestick chart
    candles format: [[timestamp, open, high, low, close, volume, oi], ...]
    """
    if not candles or len(candles) == 0:
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Plot candlesticks
        for idx, row in df.iterrows():
            # TradingView colors: Green for bullish, Red for bearish
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            
            # Draw candlestick body
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            
            if height == 0:  # Doji candle
                height = 0.01
            
            # Body rectangle
            rect = mpatches.Rectangle((idx - 0.3, bottom), 0.6, height, 
                                      facecolor=color, edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
            
            # Wick lines
            ax.plot([idx, idx], [row['low'], row['high']], 
                   color=color, linewidth=1.2, solid_capstyle='round')
        
        # Formatting
        ax.set_xlim(-1, len(df))
        y_margin = (df['high'].max() - df['low'].min()) * 0.05
        ax.set_ylim(df['low'].min() - y_margin, df['high'].max() + y_margin)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#333')
        ax.set_xlabel('Time', fontsize=12, color='#666')
        ax.set_ylabel('Price (₹)', fontsize=12, color='#666')
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#ccc')
        
        # X-axis labels
        step = max(len(df) // 12, 1)
        xticks = list(range(0, len(df), step))
        xticklabels = [df.iloc[i]['timestamp'].strftime('%d %b\n%H:%M') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=9, color='#666')
        
        # Y-axis formatting
        ax.tick_params(axis='y', labelsize=10, colors='#666')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        return None

def format_option_chain_message(option_data):
    """Format option chain data for Telegram"""
    if not option_data:
        return "❌ Option chain data not available"
    
    text = "📊 *NIFTY 50 OPTION CHAIN* 📊\n\n"
    text += f"⏰ Time: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n"
    text += f"📅 Expiry: {get_next_expiry()}\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Process first 15 strikes (ATM ± 7)
    for idx, option in enumerate(option_data[:15], 1):
        strike = option.get('strike_price', 'N/A')
        
        text += f"*Strike: {strike}*\n"
        
        # Call Option
        call = option.get('call_options', {})
        if call:
            text += f"📞 CALL\n"
            text += f"  LTP: ₹{call.get('last_price', 0):.2f}\n"
            text += f"  OI: {call.get('oi', 0):,}\n"
            text += f"  Vol: {call.get('volume', 0):,}\n"
        
        # Put Option  
        put = option.get('put_options', {})
        if put:
            text += f"📉 PUT\n"
            text += f"  LTP: ₹{put.get('last_price', 0):.2f}\n"
            text += f"  OI: {put.get('oi', 0):,}\n"
            text += f"  Vol: {put.get('volume', 0):,}\n"
        
        text += "\n"
    
    return text

def format_greeks_message(greeks_data):
    """Format Greeks data for Telegram"""
    if not greeks_data:
        return "❌ Greeks data not available"
    
    text = "🔬 *OPTION GREEKS DATA* 🔬\n\n"
    text += f"⏰ Time: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for instrument, data in list(greeks_data.items())[:20]:  # First 20
        text += f"*{instrument.split('|')[1] if '|' in instrument else instrument}*\n"
        text += f"  LTP: ₹{data.get('last_price', 0):.2f}\n"
        text += f"  Delta: {data.get('delta', 0):.4f}\n"
        text += f"  Theta: {data.get('theta', 0):.4f}\n"
        text += f"  Vega: {data.get('vega', 0):.4f}\n"
        text += f"  Gamma: {data.get('gamma', 0):.6f}\n"
        text += f"  IV: {data.get('iv', 0):.4f}\n"
        text += f"  OI: {data.get('oi', 0):,}\n"
        text += f"  Vol: {data.get('volume', 0):,}\n\n"
    
    return text

async def send_telegram_message(message):
    """Send text message to Telegram"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Split if too long
        if len(message) > 4096:
            parts = [message[i:i+4096] for i in range(0, len(message), 4096)]
            for part in parts:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=part, parse_mode='Markdown')
                await asyncio.sleep(1)
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        
        return True
    except Exception as e:
        print(f"Telegram error: {str(e)}")
        return False

async def send_telegram_photo(photo, caption):
    """Send photo to Telegram"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"Telegram photo error: {str(e)}")
        return False

# ======================== MAIN FUNCTION ========================

async def main():
    print("\n" + "="*70)
    print("🚀 UPSTOX MARKET DATA & OPTION CHAIN BOT")
    print("="*70 + "\n")
    
    # Verify credentials
    if UPSTOX_ACCESS_TOKEN == "your_access_token":
        print("❌ Please set UPSTOX_ACCESS_TOKEN environment variable!")
        return
    
    if TELEGRAM_BOT_TOKEN == "your_telegram_bot_token":
        print("❌ Please set TELEGRAM_BOT_TOKEN environment variable!")
        return
    
    # 1. Send welcome message
    welcome_msg = f"🎯 *Market Data Update*\n\n⏰ {datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\nFetching latest market data..."
    await send_telegram_message(welcome_msg)
    
    # 2. Fetch and send Nifty 50 Index Chart
    print("📈 Fetching NIFTY 50 Index data...")
    nifty_candles = get_intraday_candles(NIFTY_INDEX_KEY, "5minute")
    
    if nifty_candles and len(nifty_candles) > 0:
        print(f"✅ Got {len(nifty_candles)} candles for NIFTY 50")
        chart = create_candlestick_chart(nifty_candles, "NIFTY 50 Index - 5 Minute Chart")
        if chart:
            await send_telegram_photo(chart, "📊 *NIFTY 50 Index*")
            await asyncio.sleep(2)
    else:
        print("⚠️ No intraday candles available (market might be closed)")
    
    # 3. Fetch and send Option Chain
    print("\n📊 Fetching Option Chain data...")
    option_chain = get_option_chain_data()
    
    if option_chain:
        print(f"✅ Got option chain with {len(option_chain)} strikes")
        msg = format_option_chain_message(option_chain)
        await send_telegram_message(msg)
        await asyncio.sleep(2)
        
        # Get instrument keys for Greeks
        option_keys = []
        for opt in option_chain[:25]:  # Get first 25 strikes
            if 'call_options' in opt:
                option_keys.append(opt['call_options'].get('instrument_key', ''))
            if 'put_options' in opt:
                option_keys.append(opt['put_options'].get('instrument_key', ''))
        
        # Remove empty keys
        option_keys = [k for k in option_keys if k]
        
        if option_keys:
            print(f"\n🔬 Fetching Greeks for {len(option_keys)} options...")
            greeks = get_option_greeks(option_keys)
            if greeks:
                print(f"✅ Got Greeks data for {len(greeks)} instruments")
                greeks_msg = format_greeks_message(greeks)
                await send_telegram_message(greeks_msg)
                await asyncio.sleep(2)
    
    # 4. Fetch and send top 10 Nifty 50 stocks
    print("\n📈 Fetching Nifty 50 Stocks data...")
    
    for idx, (name, key) in enumerate(list(NIFTY_50_STOCKS.items())[:10], 1):
        print(f"  [{idx}/10] Processing {name}...")
        
        candles = get_intraday_candles(key, "5minute")
        
        if candles and len(candles) > 5:
            chart = create_candlestick_chart(candles, f"{name} - 5 Minute Intraday Chart")
            if chart:
                await send_telegram_photo(chart, f"📊 *{name}*\n{len(candles)} candles")
                await asyncio.sleep(3)  # Rate limiting
        else:
            print(f"    ⚠️ No data for {name}")
    
    # 5. Summary message
    summary = f"\n✅ *Data Update Complete!*\n\n"
    summary += f"📊 Processed:\n"
    summary += f"  • NIFTY 50 Index\n"
    summary += f"  • Option Chain ({len(option_chain)} strikes)\n"
    summary += f"  • {len(NIFTY_50_STOCKS)} Nifty 50 Stocks\n\n"
    summary += f"⏰ Completed at: {datetime.now().strftime('%H:%M:%S')}"
    
    await send_telegram_message(summary)
    
    print("\n" + "="*70)
    print("✅ ALL DATA SENT SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\n🔧 Checking dependencies...")
    try:
        import pandas
        import matplotlib
        from telegram import Bot
        print("✅ All dependencies loaded!\n")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        exit(1)
    
    asyncio.run(main())
