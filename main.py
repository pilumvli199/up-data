#!/usr/bin/env python3
# nifty50_upstox_complete.py - Option Chain + Charts (Pure Upstox API)

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

# Config
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# Nifty 50 Stocks - CORRECT ISIN codes
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE081A01020": "TATASTEEL",  # FIXED
    "NSE_EQ|INE155A01022": "TATAMOTORS",
    "NSE_EQ|INE860A01027": "HCLTECH",
    "NSE_EQ|INE238A01034": "AXISBANK",
    "NSE_EQ|INE397D01024": "BHARTIARTL",  # FIXED
    "NSE_EQ|INE101A01026": "MARUTI",  # FIXED
    "NSE_EQ|INE237A01028": "KOTAKBANK",  # FIXED
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

print("🚀 NIFTY 50 UPSTOX MONITOR")
print(f"📊 {len(NIFTY50_STOCKS)} stocks")
print(f"📈 Upstox API - Option Chain + Charts")
print(f"⏰ Every 5 minutes")

def get_expiries(instrument_key):
    """Get expiries"""
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
    """Get next expiry"""
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
    """Get option chain"""
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
    """Get spot price"""
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

def get_historical_candles(instrument_key, symbol):
    """Get historical candle data using Upstox V3 API (5 minute interval)"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    # Method 1: V3 Intraday API - 5 minute data (Today only)
    try:
        url = f"{BASE_URL}/v3/historical-candle/intraday/{encoded_key}/minutes/5"
        print(f"  🔍 V3 Intraday (5min)...")
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                if candles:
                    print(f"  ✅ Got {len(candles)} candles (today)")
                    return candles
        
        print(f"  ⚠️ V3 Intraday: HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(f"  📄 Response: {resp.text[:300]}")
    except Exception as e:
        print(f"  ⚠️ V3 Intraday error: {e}")
    
    # Method 2: V3 Historical API - 5 minute data (Last 7 days)
    try:
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=7)
        
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v3/historical-candle/{encoded_key}/minutes/5/{to_str}/{from_str}"
        print(f"  🔍 V3 Historical (5min, 7 days)...")
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                if candles:
                    print(f"  ✅ Got {len(candles)} candles (7 days)")
                    return candles
        
        print(f"  ⚠️ V3 Historical: HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(f"  📄 Response: {resp.text[:300]}")
    except Exception as e:
        print(f"  ⚠️ V3 Historical error: {e}")
    
    # Method 3: Fallback to V2 with 1 minute data (aggregate to 5 min)
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        print(f"  🔍 V2 Fallback (1min)...")
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles_1min = data.get('data', {}).get('candles', [])
                if candles_1min:
                    print(f"  ✅ Got {len(candles_1min)} 1min candles")
                    # Aggregate 1min to 5min
                    candles_5min = aggregate_to_5min(candles_1min)
                    print(f"  📊 Aggregated to {len(candles_5min)} 5min candles")
                    return candles_5min
        
        print(f"  ⚠️ V2 Fallback: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ V2 Fallback error: {e}")
    
    print(f"  ❌ {symbol}: No candle data available")
    print(f"  💡 Check: 1) Token valid? 2) Market hours? 3) API subscription?")
    return []

def aggregate_to_5min(candles_1min):
    """Aggregate 1-minute candles to 5-minute candles"""
    if not candles_1min or len(candles_1min) < 5:
        return []
    
    candles_5min = []
    
    # Process in groups of 5
    for i in range(0, len(candles_1min), 5):
        batch = candles_1min[i:i+5]
        if len(batch) < 5:
            continue
        
        # Format: [timestamp, open, high, low, close, volume, oi]
        timestamp = batch[0][0]  # First candle timestamp
        open_price = batch[0][1]  # First candle open
        high_price = max(c[2] for c in batch)  # Highest high
        low_price = min(c[3] for c in batch)  # Lowest low
        close_price = batch[-1][4]  # Last candle close
        volume = sum(c[5] for c in batch)  # Sum of volumes
        oi = batch[-1][6]  # Last OI
        
        candles_5min.append([
            timestamp, open_price, high_price, low_price, close_price, volume, oi
        ])
    
    return candles_5min

def create_candlestick_chart(candles, symbol, spot_price):
    """Create candlestick chart"""
    if not candles or len(candles) < 10:
        return None
    
    # Parse candles: [timestamp, open, high, low, close, volume, oi]
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
            
            dates.append(timestamp)
            opens.append(float(candle[1]))
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
            closes.append(float(candle[4]))
            volumes.append(int(candle[5]) if candle[5] else 0)
        except Exception as e:
            continue
    
    if len(dates) < 10:
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    fig.patch.set_facecolor('#0a0a0a')
    ax1.set_facecolor('#0f0f0f')
    ax2.set_facecolor('#0f0f0f')
    
    # Plot candlesticks
    for i in range(len(dates)):
        color = '#00ff00' if closes[i] >= opens[i] else '#ff0000'
        
        ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], 
                color=color, linewidth=0.8, alpha=0.8)
        
        height = abs(closes[i] - opens[i])
        bottom = min(opens[i], closes[i])
        
        if height > 0:
            rect = Rectangle((mdates.date2num(dates[i]) - 0.0001, bottom),
                           0.0002, height, facecolor=color, 
                           edgecolor=color, alpha=0.9)
            ax1.add_patch(rect)
        else:
            ax1.plot([dates[i], dates[i]], [opens[i], opens[i]], 
                    color=color, linewidth=1.5)
    
    # Spot price line
    ax1.axhline(y=spot_price, color='#ffff00', linestyle='--', 
               linewidth=1.5, label=f'Spot: ₹{spot_price:.2f}', alpha=0.7)
    
    ax1.set_ylabel('Price (₹)', color='white', fontsize=11, fontweight='bold')
    ax1.tick_params(colors='white', labelsize=9)
    ax1.grid(True, alpha=0.2, color='#333333', linestyle=':')
    ax1.legend(loc='upper left', fontsize=10, facecolor='#1a1a1a', 
              edgecolor='#333333', labelcolor='white')
    
    title = f'{symbol} - 5 Minute Candlestick (Upstox Data)'
    ax1.set_title(title, color='white', fontsize=14, fontweight='bold', pad=15)
    
    # Volume bars
    colors_vol = ['#00ff0060' if closes[i] >= opens[i] else '#ff000060' 
                  for i in range(len(dates))]
    ax2.bar(dates, volumes, color=colors_vol, width=0.0002, alpha=0.8)
    
    ax2.set_ylabel('Volume', color='white', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='white', labelsize=9)
    ax2.grid(True, alpha=0.2, color='#333333', linestyle=':')
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M', tz=IST))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax2.set_xlabel('Date & Time (IST)', color='white', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#0a0a0a', 
               edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

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

async def send_telegram_text(msg):
    """Send text"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"❌ Text error: {e}")
        return False

async def send_telegram_photo(photo_buf, caption):
    """Send photo"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_buf, 
                           caption=caption, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"❌ Photo error: {e}")
        return False

async def process_stock(instrument_key, symbol, idx, total):
    """Process stock - WITH VALIDATION"""
    print(f"\n[{idx}/{total}] {symbol}")
    print(f"  🔑 Key: {instrument_key}")
    
    try:
        expiry = get_next_expiry(instrument_key)
        spot = get_spot_price(instrument_key)
        
        if spot == 0:
            print(f"  ⚠️ Invalid spot price")
            return False
        
        # VALIDATION: Check if price matches expected range for stock
        price_ranges = {
            "TATASTEEL": (200, 400),
            "MARUTI": (10000, 14000),
            "KOTAKBANK": (1500, 2000),
            "LT": (3000, 4500),
            "BAJAJFINSV": (1400, 2000),
            "BHARTIARTL": (1500, 2000),
            "HDFCBANK": (1600, 1900),
            "RELIANCE": (1200, 1400),
            "ICICIBANK": (1200, 1500),
            "SBIN": (700, 900),
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
        
        # Double check: Symbol in message should match
        msg = format_detailed_message(symbol, spot, expiry, strikes)
        
        if not msg:
            print(f"  ⚠️ Message format failed")
            return False
        
        await send_telegram_text(msg)
        print(f"  📤 Option chain sent")
        
        # Get chart data
        print(f"  📊 Fetching candles...")
        candles = get_historical_candles(instrument_key, symbol)
        
        if candles and len(candles) >= 10:
            print(f"  📈 Creating chart...")
            chart_buf = create_candlestick_chart(candles, symbol, spot)
            
            if chart_buf:
                caption = f"📈 *{symbol}* - 5min Chart\n💰 Spot: ₹{spot:.2f}\n🔑 {instrument_key.split('|')[1][:12]}"
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

async def fetch_all():
    """Fetch all"""
    print("\n" + "="*60)
    print(f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}")
    print("="*60)
    
    header = f"🚀 *NIFTY 50 - UPSTOX DATA*\n"
    header += f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}\n"
    header += f"📊 Option Chain + 5min Charts\n"
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
    """Main loop"""
    print("\n🔄 Starting loop (5 min)...")
    print("🔄 Ctrl+C to stop\n")
    
    while True:
        try:
            await fetch_all()
            
            next_time = (datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')
            print(f"\n⏳ Next: {next_time}\n")
            
            await asyncio.sleep(300)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            await asyncio.sleep(60)

async def main():
    print("\n" + "="*60)
    print("🚀 NIFTY 50 - PURE UPSTOX API")
    print("="*60)
    print("📊 Option Chain: Full data")
    print("📈 Charts: 5min candlestick")
    print("🔑 Source: 100% Upstox API")
    print("⏰ Every 5 minutes")
    print("="*60)
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
