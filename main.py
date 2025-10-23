#!/usr/bin/env python3
# main.py - NIFTY 50 Option Chain + Professional Charts

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

print("🚀 NIFTY 50 UPSTOX MONITOR - COMPLETE")
print(f"📊 {len(NIFTY50_STOCKS)} stocks")
print(f"📈 Option Chain + Professional Charts")
print(f"⏰ Every 5 minutes")

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

def get_historical_candles(instrument_key, symbol):
    """Get historical candle data - 15 minute timeframe, 7 days"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    # Method 1: V3 Historical API - 15 minute data (Last 7 days)
    try:
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=7)
        
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v3/historical-candle/{encoded_key}/minutes/15/{to_str}/{from_str}"
        print(f"  🔍 V3 Historical (15min, 7 days)...")
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
    
    # Method 2: Fallback - V3 Intraday (Today's 15min data)
    try:
        url = f"{BASE_URL}/v3/historical-candle/intraday/{encoded_key}/minutes/15"
        print(f"  🔍 V3 Intraday (15min, today)...")
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                if candles:
                    print(f"  ✅ Got {len(candles)} candles (today)")
                    return candles
        
        print(f"  ⚠️ V3 Intraday: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ V3 Intraday error: {e}")
    
    # Method 3: Fallback to V2 with 5 minute data (aggregate to 15 min)
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/5minute"
        print(f"  🔍 V2 Fallback (5min → 15min)...")
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles_5min = data.get('data', {}).get('candles', [])
                if candles_5min:
                    print(f"  ✅ Got {len(candles_5min)} 5min candles")
                    candles_15min = aggregate_to_15min(candles_5min)
                    print(f"  📊 Aggregated to {len(candles_15min)} 15min candles")
                    return candles_15min
        
        print(f"  ⚠️ V2 Fallback: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ V2 Fallback error: {e}")
    
    print(f"  ❌ {symbol}: No candle data available")
    print(f"  💡 Check: 1) Token valid? 2) Market hours? 3) API subscription?")
    return []

def aggregate_to_15min(candles_5min):
    """Aggregate 5-minute candles to 15-minute candles"""
    if not candles_5min or len(candles_5min) < 3:
        return []
    
    candles_15min = []
    
    # Process in groups of 3
    for i in range(0, len(candles_5min), 3):
        batch = candles_5min[i:i+3]
        if len(batch) < 3:
            continue
        
        # Format: [timestamp, open, high, low, close, volume, oi]
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

# ==================== CHART CREATION ====================

def create_candlestick_chart(candles, symbol, spot_price):
    """Create TradingView-style professional candlestick chart"""
    if not candles or len(candles) < 10:
        return None
    
    # Parse candles
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
    
    # Create figure - TradingView style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                     gridspec_kw={'height_ratios': [3.5, 1]})
    
    # White background like TradingView
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # Plot candlesticks
    candle_width = 0.0003
    
    for i in range(len(dates)):
        is_bullish = closes[i] >= opens[i]
        
        # TradingView colors
        body_color = '#26a69a' if is_bullish else '#ef5350'
        wick_color = '#26a69a' if is_bullish else '#ef5350'
        
        # Draw wick
        ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], 
                color=wick_color, linewidth=1.2, alpha=0.9, solid_capstyle='round')
        
        # Draw candle body
        height = abs(closes[i] - opens[i])
        bottom = min(opens[i], closes[i])
        
        if height > 0.001:
            rect = Rectangle((mdates.date2num(dates[i]) - candle_width/2, bottom),
                           candle_width, height, 
                           facecolor=body_color, 
                           edgecolor=body_color, 
                           alpha=1.0,
                           linewidth=0)
            ax1.add_patch(rect)
        else:
            # Doji
            ax1.plot([mdates.date2num(dates[i]) - candle_width/2, 
                     mdates.date2num(dates[i]) + candle_width/2], 
                    [opens[i], opens[i]], 
                    color=body_color, linewidth=2, solid_capstyle='butt')
    
    # Spot price line
    ax1.axhline(y=spot_price, color='#2962ff', linestyle='--', 
               linewidth=1.8, label=f'Current: ₹{spot_price:.2f}', alpha=0.8)
    
    # Styling
    ax1.set_ylabel('Price (₹)', color='#131722', fontsize=12, fontweight='600')
    ax1.tick_params(colors='#131722', labelsize=10)
    ax1.grid(True, alpha=0.15, color='#d1d4dc', linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=11, facecolor='white', 
              edgecolor='#e0e3eb', labelcolor='#131722', framealpha=1)
    
    title = f'{symbol} - 15 Minute Chart | Last 7 Days'
    ax1.set_title(title, color='#131722', fontsize=15, fontweight='700', pad=20)
    
    # Volume bars
    colors_vol = []
    for i in range(len(dates)):
        if closes[i] >= opens[i]:
            colors_vol.append('#26a69a80')
        else:
            colors_vol.append('#ef535080')
    
    ax2.bar(dates, volumes, color=colors_vol, width=candle_width, alpha=0.7)
    
    ax2.set_ylabel('Volume', color='#131722', fontsize=12, fontweight='600')
    ax2.tick_params(colors='#131722', labelsize=10)
    ax2.grid(True, alpha=0.15, color='#d1d4dc', linestyle='-', linewidth=0.5)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M', tz=IST))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.spines['top'].set_color('#e0e3eb')
        ax.spines['right'].set_color('#e0e3eb')
        ax.spines['bottom'].set_color('#e0e3eb')
        ax.spines['left'].set_color('#e0e3eb')
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    ax2.set_xlabel('Date & Time (IST)', color='#131722', fontsize=12, fontweight='600')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='white', 
               edgecolor='none', bbox_inches='tight')
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
        
        # Validate price ranges
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
        
        # Send option chain
        msg = format_detailed_message(symbol, spot, expiry, strikes)
        
        if not msg:
            print(f"  ⚠️ Message format failed")
            return False
        
        await send_telegram_text(msg)
        print(f"  📤 Option chain sent")
        
        # Get and send chart
        print(f"  📊 Fetching candles...")
        candles = get_historical_candles(instrument_key, symbol)
        
        if candles and len(candles) >= 10:
            print(f"  📈 Creating chart...")
            chart_buf = create_candlestick_chart(candles, symbol, spot)
            
            if chart_buf:
                caption = f"📈 *{symbol}* - 15min Chart (7 Days)\n💰 Spot: ₹{spot:.2f}\n🔑 {instrument_key.split('|')[1][:12]}"
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
    
    header = f"🚀 *NIFTY 50 - UPSTOX DATA*\n"
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
            
            await asyncio.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            await asyncio.sleep(60)

# ==================== ENTRY POINT ====================

async def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🚀 NIFTY 50 - PURE UPSTOX API")
    print("="*60)
    print("📊 Option Chain: Full data")
    print("📈 Charts: 15min candlestick (7 Days)")
    print("🎨 Style: TradingView Professional")
    print("🔑 Source: 100% Upstox API")
    print("⏰ Every 5 minutes")
    print("="*60)
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
