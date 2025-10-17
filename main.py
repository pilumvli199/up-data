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

# Timezone Configuration
IST = pytz.timezone('Asia/Kolkata')

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

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def resample_to_5min(candles_1min):
    """
    Convert 1-minute candles to 5-minute candles
    candles format: [[timestamp, open, high, low, close, volume, oi], ...]
    """
    if not candles_1min or len(candles_1min) == 0:
        return []
    
    try:
        # Create DataFrame
        df = pd.DataFrame(candles_1min, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        # Resample to 5 minutes
        resampled = df.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'oi': 'last'
        }).dropna()
        
        # Convert back to list format
        resampled.reset_index(inplace=True)
        candles_5min = resampled.values.tolist()
        
        return candles_5min
    except Exception as e:
        print(f"  Error resampling: {str(e)}")
        return []

def get_intraday_candles(instrument_key):
    """
    Fetch intraday 1-minute candles and convert to 5-minute
    Uses V2 API (no auth required)
    """
    try:
        # V2 Intraday API (supports only 1minute and 30minute)
        url = f"{BASE_URL}/v2/historical-candle/intraday/{instrument_key}/1minute"
        
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
            candles_1min = data['data']['candles']
            
            # Convert to 5-minute candles
            candles_5min = resample_to_5min(candles_1min)
            
            # Limit to 500 candles
            return candles_5min[:500] if len(candles_5min) > 500 else candles_5min
        return []
    except Exception as e:
        print(f"  Error fetching intraday candles: {str(e)}")
        return []

def get_historical_candles(instrument_key, interval="30minute", days=5):
    """
    Fetch historical candles using V2 API
    For 5-minute data: Use 1minute interval and resample
    """
    try:
        now_ist = get_ist_now()
        to_date = now_ist.strftime('%Y-%m-%d')
        from_date = (now_ist - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # V2 API: Use 1minute for better granularity
        url = f"{BASE_URL}/v2/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}"
        
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
            candles_1min = data['data']['candles']
            
            # Convert to 5-minute
            candles_5min = resample_to_5min(candles_1min)
            
            return candles_5min[:500] if len(candles_5min) > 500 else candles_5min
        return []
    except Exception as e:
        print(f"  Error fetching historical candles: {str(e)}")
        return []

def get_option_contracts():
    """
    Fetch all option contracts for NIFTY (all strikes)
    Returns list of instrument keys
    """
    try:
        url = f"{BASE_URL}/v2/option/contract"
        
        params = {
            "instrument_key": NIFTY_INDEX_KEY,
            "expiry_date": get_next_expiry()
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success':
            return data.get('data', [])
        return []
    except Exception as e:
        print(f"  Error fetching option contracts: {str(e)}")
        return []

def get_option_greeks(instrument_keys):
    """
    Fetch option Greeks for given instrument keys
    Max 50 instruments per request
    Returns: {instrument_key: {ltp, delta, theta, gamma, vega, iv, oi, volume}}
    """
    try:
        if not instrument_keys:
            return {}
        
        url = f"{BASE_URL}/v2/option/greek"
        
        # Join first 50 keys
        keys_str = ",".join(instrument_keys[:50])
        
        params = {
            "instrument_key": keys_str
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'success':
            return data.get('data', {})
        return {}
    except Exception as e:
        print(f"  Error fetching Greeks: {str(e)}")
        return {}

def get_option_chain_data():
    """
    Build complete option chain using Contract API + Greeks API
    """
    try:
        print("  Getting option contracts...")
        contracts = get_option_contracts()
        
        if not contracts or len(contracts) == 0:
            print("  ⚠️ No contracts found")
            return []
        
        print(f"  Got {len(contracts)} contracts")
        
        # Extract instrument keys
        instrument_keys = [c.get('instrument_key') for c in contracts if c.get('instrument_key')]
        
        if not instrument_keys:
            print("  ⚠️ No instrument keys")
            return []
        
        # Get Greeks for all instruments (in batches of 50)
        print(f"  Fetching Greeks for {len(instrument_keys)} instruments...")
        
        all_greeks = {}
        for i in range(0, len(instrument_keys), 50):
            batch = instrument_keys[i:i+50]
            greeks = get_option_greeks(batch)
            all_greeks.update(greeks)
            time.sleep(0.5)  # Rate limiting
        
        print(f"  Got Greeks for {len(all_greeks)} instruments")
        
        # Combine contracts with Greeks
        option_chain = []
        
        for contract in contracts:
            key = contract.get('instrument_key')
            strike = contract.get('strike_price')
            option_type = contract.get('option_type')  # CE or PE
            
            if key in all_greeks:
                greek_data = all_greeks[key]
                
                option_info = {
                    'strike_price': strike,
                    'instrument_key': key,
                    'option_type': option_type,
                    'last_price': greek_data.get('last_price', 0),
                    'oi': greek_data.get('oi', 0),
                    'volume': greek_data.get('volume', 0),
                    'delta': greek_data.get('delta', 0),
                    'theta': greek_data.get('theta', 0),
                    'gamma': greek_data.get('gamma', 0),
                    'vega': greek_data.get('vega', 0),
                    'iv': greek_data.get('iv', 0)
                }
                
                option_chain.append(option_info)
        
        # Group by strike price
        strikes_dict = {}
        for opt in option_chain:
            strike = opt['strike_price']
            if strike not in strikes_dict:
                strikes_dict[strike] = {'strike_price': strike, 'call': None, 'put': None}
            
            if opt['option_type'] == 'CE':
                strikes_dict[strike]['call'] = opt
            elif opt['option_type'] == 'PE':
                strikes_dict[strike]['put'] = opt
        
        # Convert to sorted list
        result = sorted(strikes_dict.values(), key=lambda x: x['strike_price'])
        
        return result
        
    except Exception as e:
        print(f"  Error building option chain: {str(e)}")
        return []

def get_next_expiry():
    """Get next Thursday expiry date (in IST)"""
    today = get_ist_now()
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
        # Handle different formats
        if len(candles[0]) >= 6:
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
        else:
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        df = pd.DataFrame(candles, columns=cols[:len(candles[0])])
        
        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
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
                height = row['high'] * 0.0001  # Tiny height for doji
            
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
        print(f"  Error creating chart: {str(e)}")
        return None

def format_option_chain_message(option_data):
    """Format option chain data with Greeks"""
    if not option_data or len(option_data) == 0:
        return "❌ Option chain data not available"
    
    now_ist = get_ist_now()
    text = "📊 *NIFTY 50 OPTION CHAIN* 📊\n\n"
    text += f"⏰ {now_ist.strftime('%d %b %Y, %I:%M:%S %p IST')}\n"
    text += f"📅 Expiry: {get_next_expiry()}\n"
    text += f"📈 Total Strikes: {len(option_data)}\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Show ATM ± 10 strikes (total 20 strikes)
    mid_index = len(option_data) // 2
    start = max(0, mid_index - 10)
    end = min(len(option_data), mid_index + 10)
    
    for strike_data in option_data[start:end]:
        strike = strike_data.get('strike_price', 'N/A')
        
        text += f"*Strike: {strike}*\n"
        
        # Call Option (CE)
        call = strike_data.get('call')
        if call:
            text += f"📞 *CALL*\n"
            text += f"  LTP: ₹{call.get('last_price', 0):.2f}\n"
            text += f"  OI: {call.get('oi', 0):,}\n"
            text += f"  Vol: {call.get('volume', 0):,}\n"
            text += f"  𝛿: {call.get('delta', 0):.3f} | "
            text += f"𝜃: {call.get('theta', 0):.2f}\n"
            text += f"  𝛄: {call.get('gamma', 0):.5f} | "
            text += f"𝜈: {call.get('vega', 0):.2f}\n"
            text += f"  IV: {call.get('iv', 0):.2%}\n"
        
        # Put Option (PE)
        put = strike_data.get('put')
        if put:
            text += f"📉 *PUT*\n"
            text += f"  LTP: ₹{put.get('last_price', 0):.2f}\n"
            text += f"  OI: {put.get('oi', 0):,}\n"
            text += f"  Vol: {put.get('volume', 0):,}\n"
            text += f"  𝛿: {put.get('delta', 0):.3f} | "
            text += f"𝜃: {put.get('theta', 0):.2f}\n"
            text += f"  𝛄: {put.get('gamma', 0):.5f} | "
            text += f"𝜈: {put.get('vega', 0):.2f}\n"
            text += f"  IV: {put.get('iv', 0):.2%}\n"
        
        text += "\n"
    
    return text

def format_market_status():
    """Check if market is open (IST timezone)"""
    now_ist = get_ist_now()
    weekday = now_ist.weekday()
    time_now = now_ist.time()
    
    market_open_time = datetime.strptime("09:15", "%H:%M").time()
    market_close_time = datetime.strptime("15:30", "%H:%M").time()
    
    market_open = time_now >= market_open_time
    market_close = time_now <= market_close_time
    is_weekday = weekday < 5
    
    is_open = is_weekday and market_open and market_close
    
    status = "🟢 MARKET OPEN" if is_open else "🔴 MARKET CLOSED"
    time_str = now_ist.strftime('%I:%M %p IST')
    
    if not is_weekday:
        return f"{status}\n⚠️ Weekend\n🕐 {time_str}"
    elif not market_open:
        return f"{status}\n⚠️ Pre-market (Opens 9:15 AM)\n🕐 {time_str}"
    elif not market_close:
        return f"{status}\n⚠️ After hours\n🕐 {time_str}"
    
    return f"{status}\n🕐 {time_str}"

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

async def send_telegram_photo(photo, caption, retries=3):
    """Send photo to Telegram with retry logic"""
    for attempt in range(retries):
        try:
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID, 
                photo=photo, 
                caption=caption, 
                parse_mode='Markdown',
                read_timeout=30,
                write_timeout=30,
                connect_timeout=30
            )
            return True
        except TelegramError as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1}/{retries} after timeout...")
                await asyncio.sleep(2)
            else:
                print(f"  Telegram photo error after {retries} attempts: {str(e)}")
                return False
        except Exception as e:
            print(f"  Error sending photo: {str(e)}")
            return False
    return False

# ======================== MAIN FUNCTION ========================

async def main():
    print("\n" + "="*70)
    print("🚀 UPSTOX MARKET DATA BOT")
    print("="*70 + "\n")
    
    # Check credentials
    if UPSTOX_ACCESS_TOKEN == "your_access_token":
        print("❌ Set UPSTOX_ACCESS_TOKEN!")
        return
    
    if TELEGRAM_BOT_TOKEN == "your_telegram_bot_token":
        print("❌ Set TELEGRAM_BOT_TOKEN!")
        return
    
    # Market status
    market_status = format_market_status()
    print(market_status)
    print()
    
    # Welcome message
    now_ist = get_ist_now()
    welcome = f"🎯 *Market Data Update*\n\n{market_status}\n⏰ {now_ist.strftime('%d %b %Y, %I:%M:%S %p')}\n\nFetching Upstox V2 API data..."
    await send_telegram_message(welcome)
    
    # 1. NIFTY 50 Index
    print("📈 Fetching NIFTY 50 Index data...")
    
    nifty_candles = get_intraday_candles(NIFTY_INDEX_KEY)
    
    if not nifty_candles or len(nifty_candles) < 5:
        print("  No intraday, trying historical...")
        nifty_candles = get_historical_candles(NIFTY_INDEX_KEY, "1minute", 3)
    
    if nifty_candles and len(nifty_candles) > 0:
        print(f"  ✅ Got {len(nifty_candles)} 5-min candles")
        chart = create_candlestick_chart(nifty_candles, "NIFTY 50 - 5 Minute Chart", show_volume=True)
        if chart:
            await send_telegram_photo(chart, f"📊 *NIFTY 50 Index*\n{len(nifty_candles)} candles (5-min)")
            await asyncio.sleep(2)
    else:
        await send_telegram_message("⚠️ NIFTY 50 data unavailable")
    
    # 2. Option Chain
    print("\n📊 Fetching Option Chain...")
    option_chain = get_option_chain_data()
    
    if option_chain and len(option_chain) > 0:
        print(f"  ✅ Got {len(option_chain)} strikes")
        msg = format_option_chain_message(option_chain)
        await send_telegram_message(msg)
        await asyncio.sleep(2)
    else:
        print("  ⚠️ Option chain unavailable")
    
    # 3. Nifty 50 Stocks
    print("\n📈 Fetching Nifty 50 Stocks...")
    
    successful_charts = 0
    
    for idx, (name, key) in enumerate(list(NIFTY_50_STOCKS.items())[:10], 1):
        print(f"  [{idx}/10] {name}...", end=" ")
        
        candles = get_intraday_candles(key)
        
        if not candles or len(candles) < 5:
            candles = get_historical_candles(key, "1minute", 3)
        
        if candles and len(candles) > 5:
            chart = create_candlestick_chart(candles, f"{name} - 5 Minute Chart", show_volume=True)
            if chart:
                await send_telegram_photo(chart, f"📊 *{name}*\n{len(candles)} candles")
                successful_charts += 1
                print(f"✅ {len(candles)} candles")
                await asyncio.sleep(3)
            else:
                print("❌ Chart failed")
        else:
            print("⚠️ No data")
    
    # Summary
    now_ist = get_ist_now()
    summary = f"\n✅ *Update Complete!*\n\n"
    summary += f"{market_status}\n\n"
    summary += f"📊 Results:\n"
    summary += f"  • NIFTY 50: {'✅' if nifty_candles else '❌'}\n"
    summary += f"  • Option Chain: {len(option_chain)} strikes\n"
    summary += f"  • Stocks: {successful_charts}/10 charts\n\n"
    summary += f"⏰ {now_ist.strftime('%I:%M:%S %p IST')}\n"
    summary += f"🔄 V2 API (1min→5min resampled)"
    
    await send_telegram_message(summary)
    
    print("\n" + "="*70)
    print(f"✅ COMPLETED! {successful_charts} charts sent")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("🔧 Checking dependencies...")
    try:
        import pandas
        import matplotlib
        from telegram import Bot
        import pytz
        print("✅ All dependencies loaded!\n")
    except ImportError as e:
        print(f"❌ Missing: {e}")
        exit(1)
    
    asyncio.run(main())
