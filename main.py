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
import pytz

# ======================== CONFIGURATION ========================
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
NIFTY_50_STOCKS = {
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "INFY": "NSE_EQ|INE009A01021",
    "ICICIBANK": "NSE_EQ|INE090A01021",
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
    "NESTLEIND": "NSE_EQ|INE239A01016",
    "TATAMOTORS": "NSE_EQ|INE155A01022"
}

NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"

# ======================== HELPER FUNCTIONS ========================

def get_historical_candles_v3(instrument_key, interval="5", unit="minute", days=5):
    """
    Fetch historical candles using V3 API (supports 5 minute interval!)
    interval: 1, 5, 10, 15, 30 (for minutes)
    unit: minute, day, week, month
    """
    try:
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # V3 API endpoint
        url = f"{BASE_URL}/v3/historical-candle/{instrument_key}/{interval}/{unit}/{to_date}/{from_date}"
        
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            # Limit to 500 candles
            return candles[:500] if len(candles) > 500 else candles
        return []
    except Exception as e:
        print(f"Error fetching V3 candles for {instrument_key}: {str(e)}")
        return []

def get_intraday_candles_v3(instrument_key, interval="5", unit="minute"):
    """
    Fetch intraday candles using V3 Intraday API (current day only)
    interval: 1, 5, 10, 15, 30
    unit: minute
    """
    try:
        # V3 Intraday API endpoint
        url = f"{BASE_URL}/v3/historical-candle/intraday/{instrument_key}/{interval}/{unit}"
        
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            return candles[:500] if len(candles) > 500 else candles
        return []
    except Exception as e:
        print(f"Error fetching intraday V3 candles: {str(e)}")
        return []

def get_market_quote(instrument_keys):
    """
    Get real-time market quotes using V3 API
    Max 500 instruments per request
    """
    try:
        url = f"{BASE_URL}/v3/market-quote/quotes"
        
        params = {
            "instrument_key": ",".join(instrument_keys[:500])
        }
        
        response = requests.get(url, headers={"Accept": "application/json"}, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success':
            return data.get('data', {})
        return {}
    except Exception as e:
        print(f"Error fetching market quotes: {str(e)}")
        return {}

def get_option_chain_data():
    """
    Fetch option chain data
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

def get_next_expiry():
    """Get next Thursday expiry date"""
    today = datetime.now()
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

def create_candlestick_chart(candles, title, show_volume=False):
    """
    Create TradingView style candlestick chart
    candles format: [[timestamp, open, high, low, close, volume, oi], ...]
    """
    if not candles or len(candles) == 0:
        return None
    
    try:
        # Handle different candle formats
        if len(candles[0]) >= 6:
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
        else:
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        df = pd.DataFrame(candles, columns=cols[:len(candles[0])])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create figure
        if show_volume and 'volume' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                          facecolor='white', 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(16, 8), facecolor='white')
            ax2 = None
        
        ax1.set_facecolor('white')
        
        # Plot candlesticks
        for idx, row in df.iterrows():
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            
            if height == 0:
                height = 0.01
            
            # Body
            rect = mpatches.Rectangle((idx - 0.3, bottom), 0.6, height, 
                                      facecolor=color, edgecolor=color, linewidth=1.5)
            ax1.add_patch(rect)
            
            # Wick
            ax1.plot([idx, idx], [row['low'], row['high']], 
                    color=color, linewidth=1.2, solid_capstyle='round')
        
        # Formatting
        ax1.set_xlim(-1, len(df))
        y_margin = (df['high'].max() - df['low'].min()) * 0.05
        ax1.set_ylim(df['low'].min() - y_margin, df['high'].max() + y_margin)
        
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#333')
        ax1.set_ylabel('Price (₹)', fontsize=12, color='#666')
        ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#ccc')
        
        # X-axis
        step = max(len(df) // 12, 1)
        xticks = list(range(0, len(df), step))
        xticklabels = [df.iloc[i]['timestamp'].strftime('%d %b\n%H:%M') for i in xticks]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=9, color='#666')
        ax1.tick_params(axis='y', labelsize=10, colors='#666')
        
        # Volume subplot
        if ax2 and 'volume' in df.columns:
            ax2.set_facecolor('white')
            for idx, row in df.iterrows():
                color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                ax2.bar(idx, row['volume'], color=color, alpha=0.5, width=0.8)
            
            ax2.set_xlim(-1, len(df))
            ax2.set_ylabel('Volume', fontsize=11, color='#666')
            ax2.set_xlabel('Time', fontsize=11, color='#666')
            ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#ccc')
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticklabels, fontsize=9, color='#666')
            ax2.tick_params(axis='y', labelsize=9, colors='#666')
        
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
    """Format option chain data"""
    if not option_data:
        return "❌ Option chain data not available (check market hours)"
    
    text = "📊 *NIFTY 50 OPTION CHAIN* 📊\n\n"
    text += f"⏰ Time: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n"
    text += f"📅 Expiry: {get_next_expiry()}\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for idx, option in enumerate(option_data[:15], 1):
        strike = option.get('strike_price', 'N/A')
        
        text += f"*Strike: {strike}*\n"
        
        # Call Option
        call = option.get('call_options', {})
        if call:
            text += f"📞 CALL: LTP ₹{call.get('last_price', 0):.2f} | "
            text += f"OI {call.get('oi', 0):,} | Vol {call.get('volume', 0):,}\n"
        
        # Put Option
        put = option.get('put_options', {})
        if put:
            text += f"📉 PUT: LTP ₹{put.get('last_price', 0):.2f} | "
            text += f"OI {put.get('oi', 0):,} | Vol {put.get('volume', 0):,}\n"
        
        text += "\n"
    
    return text

def format_market_status():
    """Check if market is open"""
    now = datetime.now()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    time_now = now.time()
    
    # Market hours: Mon-Fri, 9:15 AM - 3:30 PM
    market_open = time_now >= datetime.strptime("09:15", "%H:%M").time()
    market_close = time_now <= datetime.strptime("15:30", "%H:%M").time()
    is_weekday = weekday < 5
    
    is_open = is_weekday and market_open and market_close
    
    status = "🟢 MARKET OPEN" if is_open else "🔴 MARKET CLOSED"
    
    if not is_weekday:
        return f"{status}\n⚠️ Weekend - Market opens Monday 9:15 AM"
    elif not market_open:
        return f"{status}\n⚠️ Market opens at 9:15 AM"
    elif not market_close:
        return f"{status}\n⚠️ Market closed at 3:30 PM"
    
    return status

async def send_telegram_message(message):
    """Send text message to Telegram"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        
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
    print("🚀 UPSTOX MARKET DATA BOT - V3 API")
    print("="*70 + "\n")
    
    # Check credentials
    if UPSTOX_ACCESS_TOKEN == "your_access_token":
        print("❌ Set UPSTOX_ACCESS_TOKEN environment variable!")
        return
    
    if TELEGRAM_BOT_TOKEN == "your_telegram_bot_token":
        print("❌ Set TELEGRAM_BOT_TOKEN environment variable!")
        return
    
    # Market status
    market_status = format_market_status()
    print(market_status)
    print()
    
    # Welcome message
    welcome = f"🎯 *Market Data Update*\n\n{market_status}\n⏰ {datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\nFetching data using Upstox V3 API..."
    await send_telegram_message(welcome)
    
    # 1. NIFTY 50 Index - Try intraday first, then historical
    print("📈 Fetching NIFTY 50 Index data...")
    
    nifty_candles = get_intraday_candles_v3(NIFTY_INDEX_KEY, "5", "minute")
    
    if not nifty_candles or len(nifty_candles) < 5:
        print("  ⚠️ No intraday data, trying historical (last 5 days)...")
        nifty_candles = get_historical_candles_v3(NIFTY_INDEX_KEY, "5", "minute", 5)
    
    if nifty_candles and len(nifty_candles) > 0:
        print(f"  ✅ Got {len(nifty_candles)} candles")
        chart = create_candlestick_chart(nifty_candles, "NIFTY 50 - 5 Minute Chart", show_volume=True)
        if chart:
            await send_telegram_photo(chart, "📊 *NIFTY 50 Index*\n5-minute candles")
            await asyncio.sleep(2)
    else:
        await send_telegram_message("⚠️ NIFTY 50 data unavailable (market closed or API issue)")
    
    # 2. Option Chain
    print("\n📊 Fetching Option Chain...")
    option_chain = get_option_chain_data()
    
    if option_chain:
        print(f"  ✅ Got {len(option_chain)} strikes")
        msg = format_option_chain_message(option_chain)
        await send_telegram_message(msg)
        await asyncio.sleep(2)
    else:
        await send_telegram_message("⚠️ Option chain unavailable (market closed or auth issue)")
    
    # 3. Nifty 50 Stocks
    print("\n📈 Fetching Nifty 50 Stocks...")
    
    successful_charts = 0
    
    for idx, (name, key) in enumerate(list(NIFTY_50_STOCKS.items())[:10], 1):
        print(f"  [{idx}/10] {name}...", end=" ")
        
        # Try intraday first
        candles = get_intraday_candles_v3(key, "5", "minute")
        
        # If no intraday, try historical
        if not candles or len(candles) < 5:
            candles = get_historical_candles_v3(key, "5", "minute", 5)
        
        if candles and len(candles) > 5:
            chart = create_candlestick_chart(candles, f"{name} - 5 Minute Chart", show_volume=True)
            if chart:
                await send_telegram_photo(chart, f"📊 *{name}*\n{len(candles)} candles (5-min)")
                successful_charts += 1
                print(f"✅ {len(candles)} candles")
                await asyncio.sleep(3)
            else:
                print("❌ Chart error")
        else:
            print("⚠️ No data")
    
    # Summary
    summary = f"\n✅ *Update Complete!*\n\n"
    summary += f"{market_status}\n\n"
    summary += f"📊 Processed:\n"
    summary += f"  • NIFTY 50 Index\n"
    summary += f"  • Option Chain ({len(option_chain)} strikes)\n"
    summary += f"  • {successful_charts}/10 Stock Charts\n\n"
    summary += f"⏰ {datetime.now().strftime('%H:%M:%S')}\n"
    summary += f"🔄 Using Upstox V3 API (5-min candles)"
    
    await send_telegram_message(summary)
    
    print("\n" + "="*70)
    print(f"✅ COMPLETED! Sent {successful_charts} stock charts")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("🔧 Checking dependencies...")
    try:
        import pandas
        import matplotlib
        from telegram import Bot
        print("✅ All dependencies loaded!\n")
    except ImportError as e:
        print(f"❌ Missing: {e}")
        exit(1)
    
    asyncio.run(main())
