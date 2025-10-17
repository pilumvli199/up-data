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
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "tumcha_upstox_access_token_yethetaka")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "tumcha_telegram_bot_token_yethetaka")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "tumcha_telegram_chat_id_yethetaka")

# Timezone Configuration
IST = pytz.timezone('Asia/Kolkata')

# Upstox API Configuration
BASE_URL = "https://api.upstox.com/v2" # V2 base URL
HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
}

# ======================== INSTRUMENT KEYS ========================
NIFTY_50_STOCKS = {
    "HDFCBANK": "NSE_EQ|INE040A01034", "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029", "INFY": "NSE_EQ|INE009A01021",
    "ICICIBANK": "NSE_EQ|INE090A01021", "BHARTIARTL": "NSE_EQ|INE397D01024",
    "HINDUNILVR": "NSE_EQ|INE030A01027", "ITC": "NSE_EQ|INE154A01025",
    "SBIN": "NSE_EQ|INE062A01020", "BAJFINANCE": "NSE_EQ|INE296A01024",
    # तुम्ही येथे अधिक स्टॉक्स ऍड करू शकता
}
NIFTY_INDEX_KEY = "NSE_INDEX|NIFTY 50"

# ======================== HELPER FUNCTIONS ========================

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def resample_to_5min(candles_1min):
    """Convert 1-minute candles to 5-minute candles."""
    if not candles_1min:
        return []
    try:
        df = pd.DataFrame(candles_1min, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to 5 minutes using IST
        resampled = df.resample('5T', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'oi': 'last'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        return resampled.values.tolist()
    except Exception as e:
        print(f"  Error resampling: {str(e)}")
        return []

def get_historical_candles(instrument_key, days=15):
    """Fetch historical 1-minute candles for the last few days and resample to 5-minute."""
    try:
        to_date = get_ist_now().strftime('%Y-%m-%d')
        from_date = (get_ist_now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}"
        
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
            candles_1min = data['data']['candles']
            candles_5min = resample_to_5min(candles_1min)
            return candles_5min[-500:] # Return last 500 5-min candles
        return []
    except Exception as e:
        print(f"  Error fetching historical candles for {instrument_key}: {str(e)}")
        return []

def get_next_thursday_expiry():
    """Get the upcoming Thursday as the expiry date."""
    today = get_ist_now()
    # Wednesday is 2, Thursday is 3
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0: # If today is Thursday or later in the week
        days_ahead += 7
    next_thursday = today + timedelta(days=days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

# === नवीन फंक्शन: ऑप्शन चेन मिळवण्यासाठी ===
def get_option_chain_data():
    """Fetch complete option chain using the direct /option/chain endpoint."""
    try:
        expiry_date = get_next_thursday_expiry()
        print(f"  Fetching option chain for expiry: {expiry_date}")
        
        url = f"{BASE_URL}/option/chain"
        params = {
            "instrument_key": NIFTY_INDEX_KEY,
            "expiry_date": expiry_date
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data:
            print(f"  ✅ Successfully fetched {len(data['data'])} strikes.")
            return data['data']
        else:
            print(f"  ⚠️ API Error: {data.get('errors')}")
            return []
            
    except requests.exceptions.HTTPError as http_err:
        print(f"  ❌ HTTP error fetching option chain: {http_err} - {http_err.response.text}")
        return []
    except Exception as e:
        print(f"  ❌ Error building option chain: {str(e)}")
        return []

# ... (तुमचे create_candlestick_chart फंक्शन जसे आहे तसे ठेवा) ...
def create_candlestick_chart(candles, title, show_volume=False):
    if not candles or len(candles) == 0:
        return None
    try:
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor='white', gridspec_kw={'height_ratios': [3, 1]}) if show_volume else plt.subplots(figsize=(16, 8), facecolor='white')
        ax2 = ax2 if show_volume else None
        ax1.set_facecolor('white')
        
        for idx, row in df.iterrows():
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1.2)
            ax1.add_patch(mpatches.Rectangle((idx - 0.3, min(row['open'], row['close'])), 0.6, abs(row['open'] - row['close']), facecolor=color))
        
        y_margin = (df['high'].max() - df['low'].min()) * 0.05
        ax1.set_ylim(df['low'].min() - y_margin, df['high'].max() + y_margin)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (₹)')
        ax1.grid(True, alpha=0.2)
        
        step = max(len(df) // 10, 1)
        xticks = df.index[::step]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks.strftime('%d-%b %H:%M'), rotation=30, ha='right')
        
        if ax2 and 'volume' in df.columns:
            colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(df['open'], df['close'])]
            ax2.bar(df.index, df['volume'], color=colors, width=0.002, alpha=0.7)
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        print(f"  Error creating chart: {str(e)}")
        return None


# === सुधारित फंक्शन: ऑप्शन चेन मेसेज फॉरमॅट करण्यासाठी (ग्रीक्सशिवाय) ===
def format_option_chain_message(option_data):
    """Format option chain data without Greeks."""
    if not option_data:
        return "❌ Option chain data not available"

    # Find the strike closest to the underlying LTP for ATM
    underlying_ltp = option_data[0].get('underlying_ltp', 0)
    closest_strike_data = min(option_data, key=lambda x: abs(x['strike_price'] - underlying_ltp))
    atm_strike = closest_strike_data['strike_price']
    
    # Filter to show ATM ± 5 strikes
    filtered_data = [
        strike for strike in option_data 
        if atm_strike - 250 <= strike['strike_price'] <= atm_strike + 250
    ]
    
    now_ist = get_ist_now()
    text = f"📊 *NIFTY 50 OPTION CHAIN* 📊\n"
    text += f"Spot Price: *₹{underlying_ltp:.2f}*\n"
    text += f"📅 Expiry: {get_next_thursday_expiry()}\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    for strike_data in sorted(filtered_data, key=lambda x: x['strike_price']):
        strike = strike_data.get('strike_price', 'N/A')
        
        text += f"Strikes: *{strike}*\n"
        
        # Call Option (CE)
        call = strike_data.get('CE')
        if call:
            text += f"📞 *CALL*\n"
            text += f"  LTP: ₹{call.get('last_price', 0):.2f}\n"
            text += f"  OI: {call.get('open_interest', 0):,}\n"
            text += f"  Vol: {call.get('volume', 0):,}\n"
        
        # Put Option (PE)
        put = strike_data.get('PE')
        if put:
            text += f"📉 *PUT*\n"
            text += f"  LTP: ₹{put.get('last_price', 0):.2f}\n"
            text += f"  OI: {put.get('open_interest', 0):,}\n"
            text += f"  Vol: {put.get('volume', 0):,}\n"
        
        text += "--------------------------------\n"
    
    return text

# ... (तुमचे बाकीचे सर्व फंक्शन्स जसे आहेत तसे ठेवा: format_market_status, send_telegram_message, send_telegram_photo) ...
async def send_telegram_message(message, bot):
    try:
        if len(message) > 4096:
            for i in range(0, len(message), 4096):
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message[i:i+4096], parse_mode='Markdown')
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
    except Exception as e:
        print(f"Telegram error: {e}")

async def send_telegram_photo(photo, caption, bot):
    try:
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode='Markdown')
    except Exception as e:
        print(f"Telegram photo error: {e}")

# ======================== MAIN FUNCTION ========================
async def main():
    print("🚀 UPSTOX MARKET DATA BOT STARTED 🚀")
    
    if "your_access_token" in UPSTOX_ACCESS_TOKEN or "your_telegram" in TELEGRAM_BOT_TOKEN:
        print("❌ कृपया UPSTOX_ACCESS_TOKEN आणि TELEGRAM_BOT_TOKEN योग्यरित्या सेट करा!")
        return
        
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await send_telegram_message("🤖 Bot starting... Fetching market data.", bot)
    
    # 1. NIFTY 50 Index
    print("📈 Fetching NIFTY 50 Index data...")
    nifty_candles = get_historical_candles(NIFTY_INDEX_KEY)
    if nifty_candles:
        print(f"  ✅ Got {len(nifty_candles)} 5-min candles for Nifty 50")
        chart = create_candlestick_chart(nifty_candles, "NIFTY 50 - 5 Minute Chart", show_volume=True)
        if chart:
            await send_telegram_photo(chart, "📊 *NIFTY 50 Index* (5-min)", bot)
            await asyncio.sleep(2)
    else:
        await send_telegram_message("⚠️ NIFTY 50 chart data unavailable.", bot)
        
    # 2. Option Chain
    print("📊 Fetching Option Chain...")
    option_chain_data = get_option_chain_data()
    if option_chain_data:
        msg = format_option_chain_message(option_chain_data)
        await send_telegram_message(msg, bot)
        await asyncio.sleep(2)
    else:
        await send_telegram_message("⚠️ NIFTY 50 Option Chain data unavailable.", bot)

    # 3. Nifty 50 Stocks (Top 10)
    print("📈 Fetching Nifty 50 Stocks...")
    for name, key in list(NIFTY_50_STOCKS.items())[:10]:
        print(f"  Fetching {name}...")
        candles = get_historical_candles(key)
        if candles:
            print(f"    ✅ Got {len(candles)} candles")
            chart = create_candlestick_chart(candles, f"{name} - 5 Minute Chart", show_volume=True)
            if chart:
                await send_telegram_photo(chart, f"📊 *{name}* (5-min)", bot)
                await asyncio.sleep(3) # Rate limit
        else:
            print(f"    ⚠️ No data for {name}")

    await send_telegram_message("✅ All tasks completed!", bot)
    print("✅ BOT FINISHED ✅")

if __name__ == "__main__":
    asyncio.run(main())
