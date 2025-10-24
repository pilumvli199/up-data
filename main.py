#!/usr/bin/env python3
# main.py - NIFTY 50 Option Chain + Professional Charts (FIXED)

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta
import pytz
from telegram import Bot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import io

# ==================== CONFIG ====================
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# ==================== NIFTY 50 STOCKS ====================
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE081A01020": "TATASTEEL",
    "NSE_EQ|INE155A01022": "TATAMOTORS",
    "NSE_EQ|INE860A01027": "HCLTECH",
    "NSE_EQ|INE238A01034": "AXISBANK",
    "NSE_EQ|INE397D01024": "BHARTIARTL",
    "NSE_EQ|INE101A01026": "MARUTI",
    "NSE_EQ|INE237A01028": "KOTAKBANK",
    "NSE_EQ|INE044A01036": "SUNPHARMA",
    "NSE_EQ|INE280A01028": "TITAN",
    "NSE_EQ|INE481G01011": "ULTRACEMCO",
    "NSE_EQ|INE742F01042": "ADANIPORTS",
    "NSE_EQ|INE423A01024": "ADANIENT",
    "NSE_EQ|INE192A01025": "TATACONSUM",
    "NSE_EQ|INE752E01010": "POWERGRID",
    "NSE_EQ|INE018A01030": "LT",
    "NSE_EQ|INE154A01025": "ITC",
    "NSE_EQ|INE030A01027": "HINDUNILVR",
    "NSE_EQ|INE522F01014": "COALINDIA",
    "NSE_EQ|INE095A01012": "INDUSINDBK",
    "NSE_EQ|INE628A01036": "UPL",
    "NSE_EQ|INE129A01019": "GAIL",
    "NSE_EQ|INE918I01018": "BAJAJFINSV",
    "NSE_EQ|INE917I01010": "BAJAJAUTO",
    "NSE_EQ|INE059B01024": "CIPLA",
    "NSE_EQ|INE089A01023": "DRREDDY",
    "NSE_EQ|INE066A01021": "EICHERMOT",
    "NSE_EQ|INE361B01024": "DIVISLAB",
    "NSE_EQ|INE021A01026": "ASIANPAINT",
    "NSE_EQ|INE528G01035": "NYKAA",
    "NSE_EQ|INE192R01011": "DMART",
}

print("\n" + "="*70)
print("🚀 NIFTY 50 - UPSTOX MONITOR (FIXED VERSION)")
print("="*70)
print("📊 Option Chain: Real-time data with PCR analysis")
print("📈 Charts: 15-minute candlestick (7 Days)")
print("🎨 Style: Professional TradingView theme")
print("🔑 Source: 100% Upstox API v2 (CORRECTED)")
print("⏰ Interval: Every 5 minutes")
print("📱 Output: Telegram notifications")
print("="*70 + "\n")

# ==================== UPSTOX API FUNCTIONS ====================

def get_expiries(instrument_key):
    """Get available expiry dates"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            contracts = data.get('data', [])
            
            expiries = set()
            for c in contracts:
                if 'expiry' in c:
                    expiries.add(c['expiry'])
            
            return sorted(list(expiries))
    except Exception as e:
        print(f"⚠️ Expiry error: {e}")
    
    return []

def get_next_expiry(instrument_key):
    """Get next available expiry"""
    expiries = get_expiries(instrument_key)
    if not expiries:
        today = datetime.now(IST)
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    today = datetime.now(IST).date()
    future = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
    
    if future:
        return min(future)
    return expiries[0]

def get_option_chain(instrument_key, expiry):
    """Get option chain data"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('data', [])
    except Exception as e:
        print(f"⚠️ Chain error: {e}")
    
    return []

def get_spot_price(instrument_key):
    """Get current spot price"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            quote_data = data.get('data', {})
            if quote_data:
                first_key = list(quote_data.keys())[0]
                ltp = quote_data[first_key].get('last_price', 0)
                return float(ltp) if ltp else 0
    except Exception as e:
        print(f"⚠️ Spot error: {e}")
    
    return 0

def split_30min_to_15min(candles_30min):
    """Split 30-minute candles into 15-minute candles for better visualization"""
    if not candles_30min:
        return []
    
    candles_15min = []
    
    for candle in candles_30min:
        # Format: [timestamp, open, high, low, close, volume, oi]
        timestamp = candle[0]
        open_price = candle[1]
        high_price = candle[2]
        low_price = candle[3]
        close_price = candle[4]
        volume = candle[5]
        oi = candle[6]
        
        # Parse timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            continue
        
        # Create first 15min candle (first half)
        candles_15min.append([
            timestamp,
            open_price,
            high_price,
            low_price,
            (open_price + close_price) / 2,
            volume // 2,
            oi
        ])
        
        # Create second 15min candle (second half - 15 minutes later)
        dt_next = dt + timedelta(minutes=15)
        candles_15min.append([
            dt_next.isoformat(),
            (open_price + close_price) / 2,
            high_price,
            low_price,
            close_price,
            volume - (volume // 2),
            oi
        ])
    
    return candles_15min

def aggregate_1min_to_15min(candles_1min):
    """Aggregate 1-minute candles to 15-minute candles"""
    if not candles_1min or len(candles_1min) < 15:
        return []
    
    candles_15min = []
    
    for i in range(0, len(candles_1min), 15):
        batch = candles_1min[i:i+15]
        if len(batch) < 15:
            continue
        
        timestamp = batch[0][0]
        open_price = batch[0][1]
        high_price = max(c[2] for c in batch)
        low_price = min(c[3] for c in batch)
        close_price = batch[-1][4]
        volume = sum(c[5] for c in batch)
        oi = batch[-1][6]
        
        candles_15min.append([
            timestamp, open_price, high_price, low_price, close_price, volume, oi
        ])
    
    return candles_15min

def get_historical_candles(instrument_key, symbol):
    """Get historical candle data - Combine INTRADAY + HISTORICAL with AUTH"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    today_candles = []
    historical_candles = []
    
    # Step 1: Get TODAY'S LIVE data (1-minute intraday with AUTH)
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        print(f"  🔍 V2 Intraday (1min, TODAY with AUTH)...")
        
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles_1min = data.get('data', {}).get('candles', [])
                if candles_1min and len(candles_1min) >= 15:
                    print(f"  ✅ Today: {len(candles_1min)} 1min candles (LIVE)")
                    today_candles = aggregate_1min_to_15min(candles_1min)
                    print(f"  📊 Today: {len(today_candles)} 15min candles")
        else:
            print(f"  ⚠️ Intraday HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  ⚠️ Intraday 1min error: {e}")
    
    # Step 2: Get RECENT DAYS data (30-minute with AUTH, last 7 days)
    try:
        # Use today as end date, 7 days back as start
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=7)
        
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_str}/{from_str}"
        print(f"  🔍 V2 Historical (30min, 7 days with AUTH)...")
        
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                if candles and len(candles) > 0:
                    print(f"  ✅ History: {len(candles)} 30min candles")
                    historical_candles = split_30min_to_15min(candles)
                    print(f"  📊 History: {len(historical_candles)} 15min candles")
        else:
            print(f"  ⚠️ Historical HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  ⚠️ Historical error: {e}")
    
    # Step 3: COMBINE historical + today
    if historical_candles and today_candles:
        combined = historical_candles + today_candles
        print(f"  ✅ COMBINED: {len(combined)} total candles (History + Today)")
        return combined
    elif today_candles:
        print(f"  ⚠️ Only today: {len(today_candles)} candles")
        return today_candles
    elif historical_candles:
        print(f"  ⚠️ Only history: {len(historical_candles)} candles")
        return historical_candles
    
    # Fallback: Try 30-minute intraday with AUTH
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/30minute"
        print(f"  🔍 Fallback: Intraday 30min with AUTH...")
        
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                if candles and len(candles) > 0:
                    print(f"  ✅ Got {len(candles)} 30min candles")
                    candles_15min = split_30min_to_15min(candles)
                    print(f"  📊 Split to {len(candles_15min)} 15min candles")
                    return candles_15min
    except Exception as e:
        print(f"  ⚠️ Fallback error: {e}")
    
    print(f"  ❌ {symbol}: No candle data available")
    return []

# ==================== CHART CREATION ====================

def create_candlestick_chart(candles, symbol, spot_price):
    """Create TradingView-style professional candlestick chart (Market hours only)"""
    if not candles or len(candles) < 3:  # Reduced from 10 to 3 minimum
        return None
    
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for candle in reversed(candles):
        try:
            timestamp = datetime.fromisoformat(candle[0].replace('Z', '+00:00'))
            timestamp = timestamp.astimezone(IST)
            
            if timestamp.weekday() >= 5:
                continue
            
            hour = timestamp.hour
            minute = timestamp.minute
            
            if hour < 9 or (hour == 9 and minute < 15):
                continue
            if hour > 15 or (hour == 15 and minute > 30):
                continue
            
            dates.append(timestamp)
            opens.append(float(candle[1]))
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
            closes.append(float(candle[4]))
            volumes.append(int(candle[5]) if candle[5] else 0)
        except Exception as e:
            continue
    
    if len(dates) < 3:  # Reduced minimum requirement
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), 
                                     gridspec_kw={'height_ratios': [4, 1]})
    
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#fafafa')
    
    indices = range(len(dates))
    
    for i in indices:
        is_bullish = closes[i] >= opens[i]
        body_color = '#089981' if is_bullish else '#f23645'
        wick_color = body_color
        
        ax1.plot([i, i], [lows[i], highs[i]], 
                color=wick_color, linewidth=1.0, alpha=1.0, 
                solid_capstyle='round', zorder=2)
        
        height = abs(closes[i] - opens[i])
        bottom = min(opens[i], closes[i])
        
        if height > 0.001:
            rect = Rectangle((i - 0.4, bottom), 0.8, height, 
                           facecolor=body_color, 
                           edgecolor=body_color, 
                           alpha=1.0,
                           linewidth=0,
                           zorder=3)
            ax1.add_patch(rect)
        else:
            ax1.plot([i - 0.4, i + 0.4], [opens[i], opens[i]], 
                    color=body_color, linewidth=1.5, solid_capstyle='butt', zorder=3)
    
    ax1.axhline(y=spot_price, color='#2962FF', linestyle='--', 
               linewidth=1.5, alpha=0.85, zorder=4)
    
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_yticks([spot_price])
    ax1_right.set_yticklabels([f'₹{spot_price:.2f}'], fontsize=10, 
                              fontweight='600', color='#2962FF')
    ax1_right.tick_params(colors='#2962FF', length=0)
    
    ax1.set_ylabel('Price (₹)', color='#787B86', fontsize=11, fontweight='500')
    ax1.tick_params(axis='y', colors='#787B86', labelsize=9.5)
    ax1.tick_params(axis='x', colors='#787B86', labelsize=9)
    
    ax1.grid(True, alpha=0.12, color='#D1D4DC', linestyle='-', linewidth=0.5, zorder=1)
    ax1.set_axisbelow(True)
    
    title = f'{symbol}  •  15 Min (30min data)  •  Market Hours (9:15 AM - 3:30 PM)'
    ax1.set_title(title, color='#131722', fontsize=16, fontweight='600', 
                 pad=20, loc='left')
    
    colors_vol = []
    for i in indices:
        if closes[i] >= opens[i]:
            colors_vol.append('#08998166')
        else:
            colors_vol.append('#f2364566')
    
    ax2.bar(indices, volumes, color=colors_vol, width=0.8, 
           alpha=1.0, edgecolor='none', zorder=2)
    
    ax2.set_ylabel('Volume', color='#787B86', fontsize=11, fontweight='500')
    ax2.tick_params(axis='y', colors='#787B86', labelsize=9.5)
    ax2.tick_params(axis='x', colors='#787B86', labelsize=9)
    ax2.grid(True, alpha=0.12, color='#D1D4DC', linestyle='-', linewidth=0.5, zorder=1)
    ax2.set_axisbelow(True)
    
    step = max(1, len(dates) // 10)
    tick_positions = list(range(0, len(dates), step))
    tick_labels = [dates[i].strftime('%d %b\n%H:%M') for i in tick_positions]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-1, len(dates))
        
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_color('#E0E3EB')
            ax.spines[spine].set_linewidth(0.8)
    
    ax1.spines['right'].set_visible(False)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
    
    ax2.set_xlabel('Date & Time (IST)', color='#787B86', fontsize=11, fontweight='500', labelpad=10)
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(hspace=0.05)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='white', 
               edgecolor='none', bbox_inches='tight', pad_inches=0.2)
    buf.seek(0)
    plt.close(fig)
    
    return buf

# ==================== MESSAGE FORMATTING ====================

def format_detailed_message(symbol, spot, expiry, strikes):
    """Format option chain message"""
    if not strikes or len(strikes) < 11:
        return None
    
    atm_index = len(strikes) // 2
    if spot > 0:
        atm_index = min(range(len(strikes)), 
                       key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    
    start = max(0, atm_index - 5)
    end = min(len(strikes), atm_index + 6)
    
    if end - start < 11:
        if start == 0:
            end = min(11, len(strikes))
        else:
            start = max(0, len(strikes) - 11)
    
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol} OPTION CHAIN*\n\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.2f}\n\n"
    
    msg += "```\n"
    msg += "Strike    CE-LTP  CE-Vol   CE-OI    PE-LTP  PE-Vol   PE-OI\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_oi = 0
    total_pe_oi = 0
    total_ce_vol = 0
    total_pe_vol = 0
    
    for i, s in enumerate(selected):
        is_atm = (start + i == atm_index)
        mark = "🔸" if is_atm else "  "
        
        strike = s.get('strike_price', 0)
        
        call = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call.get('ltp', 0)
        ce_vol = call.get('volume', 0)
        ce_oi = call.get('oi', 0)
        
        put = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put.get('ltp', 0)
        pe_vol = put.get('volume', 0)
        pe_oi = put.get('oi', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol
        
        ce_vol_k = ce_vol / 1000 if ce_vol > 0 else 0
        ce_oi_k = ce_oi / 1000 if ce_oi > 0 else 0
        pe_vol_k = pe_vol / 1000 if pe_vol > 0 else 0
        pe_oi_k = pe_oi / 1000 if pe_oi > 0 else 0
        
        msg += f"{mark}{strike:8.2f} {ce_ltp:7.2f} {ce_vol_k:7.1f}K {ce_oi_k:7.1f}K {pe_ltp:7.2f} {pe_vol_k:7.1f}K {pe_oi_k:7.1f}K\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_vol_k = total_ce_vol / 1000
    total_ce_oi_k = total_ce_oi / 1000
    total_pe_vol_k = total_pe_vol / 1000
    total_pe_oi_k = total_pe_oi / 1000
    
    msg += f"TOTAL          {total_ce_vol_k:7.1f}K {total_ce_oi_k:7.1f}K        {total_pe_vol_k:7.1f}K {total_pe_oi_k:7.1f}K\n"
    msg += "```\n\n"
    
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    
    msg += f"📊 *PCR (OI):* {pcr:.3f}\n"
    msg += f"📊 *PCR (Vol):* {pcr_vol:.3f}\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}\n"
    
    return msg

# ==================== TELEGRAM FUNCTIONS ====================

async def send_telegram_text(msg):
    """Send text message"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"❌ Text error: {e}")
        return False

async def send_telegram_photo(photo_buf, caption):
    """Send photo with caption"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_buf, 
                           caption=caption, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"❌ Photo error: {e}")
        return False

# ==================== STOCK PROCESSING ====================

async def process_stock(instrument_key, symbol, idx, total):
    """Process single stock - with validation"""
    print(f"\n[{idx}/{total}] {symbol}")
    print(f"  🔑 Key: {instrument_key}")
    
    try:
        expiry = get_next_expiry(instrument_key)
        spot = get_spot_price(instrument_key)
        
        if spot == 0:
            print(f"  ⚠️ Invalid spot price")
            return False
        
        price_ranges = {
            "TATASTEEL": (150, 250),
            "MARUTI": (3000, 14000),
            "KOTAKBANK": (1500, 2500),
            "LT": (3000, 4500),
            "BAJAJFINSV": (1400, 2000),
            "BHARTIARTL": (1800, 2200),
            "HDFCBANK": (900, 1200),
            "RELIANCE": (1300, 1600),
            "ICICIBANK": (1200, 1500),
            "SBIN": (800, 1000),
        }
        
        if symbol in price_ranges:
            min_price, max_price = price_ranges[symbol]
            if not (min_price <= spot <= max_price):
                print(f"  ⚠️ PRICE MISMATCH! Got ₹{spot:.2f}, expected ₹{min_price}-{max_price}")
                print(f"  ❌ Wrong instrument_key or API data issue!")
                return False
        
        strikes = get_option_chain(instrument_key, expiry)
        
        if not strikes or len(strikes) < 11:
            print(f"  ⚠️ Insufficient strikes ({len(strikes) if strikes else 0})")
            return False
        
        print(f"  ✅ Spot: ₹{spot:.2f} | Strikes: {len(strikes)}")
        
        msg = format_detailed_message(symbol, spot, expiry, strikes)
        
        if not msg:
            print(f"  ⚠️ Message format failed")
            return False
        
        await send_telegram_text(msg)
        print(f"  📤 Option chain sent")
        
        print(f"  📊 Fetching candles...")
        candles = get_historical_candles(instrument_key, symbol)
        
        if candles and len(candles) >= 3:  # Reduced from 10 to 3
            print(f"  📈 Creating chart...")
            chart_buf = create_candlestick_chart(candles, symbol, spot)
            
            if chart_buf:
                caption = f"📈 *{symbol}* - 15min Chart (Market Hours)\n💰 Spot: ₹{spot:.2f}\n⏰ 9:15 AM - 3:30 PM IST"
                await send_telegram_photo(chart_buf, caption)
                print(f"  📤 Chart sent!")
                return True
        else:
            print(f"  ⚠️ No chart data")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== MAIN LOOP ====================

async def fetch_all():
    """Fetch all stocks"""
    print("\n" + "="*60)
    print(f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}")
    print("="*60)
    
    header = f"🚀 *NIFTY 50 - UPSTOX DATA (FIXED)*\n"
    header += f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}\n"
    header += f"📊 Option Chain + 15min Charts (7 Days)\n"
    header += f"📈 {len(NIFTY50_STOCKS)} Stocks\n\n_Starting..._"
    
    await send_telegram_text(header)
    
    success = 0
    total = len(NIFTY50_STOCKS)
    
    for idx, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        result = await process_stock(key, symbol, idx, total)
        if result:
            success += 1
        await asyncio.sleep(3)
    
    summary = f"\n✅ *COMPLETE*\n📊 {success}/{total} stocks\n⏰ {datetime.now(IST).strftime('%I:%M %p')}"
    await send_telegram_text(summary)
    
    print("\n" + "="*60)
    print(f"✅ {success}/{total}")
    print("="*60)

async def monitoring_loop():
    """Main monitoring loop - runs every 5 minutes"""
    print("\n🔄 Starting monitoring loop (5 min interval)...")
    print("🔄 Press Ctrl+C to stop\n")
    
    while True:
        try:
            await fetch_all()
            
            next_time = (datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')
            print(f"\n⏳ Next run: {next_time}\n")
            
            await asyncio.sleep(300)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            await asyncio.sleep(60)

# ==================== ENTRY POINT ====================

async def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🚀 STARTING NIFTY 50 MONITOR (FIXED VERSION)")
    print("="*70)
    print(f"📊 Tracking: {len(NIFTY50_STOCKS)} stocks")
    print("📈 Data: Option Chain + 15min Charts (7 Days)")
    print("🎨 Theme: Professional TradingView style")
    print("🔑 API: Pure Upstox V2 (CORRECTED)")
    print("⏰ Frequency: Every 5 minutes")
    print("="*70 + "\n")
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
