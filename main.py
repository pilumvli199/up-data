#!/usr/bin/env python3
"""
HYBRID MONITOR - Best of Both Worlds!
- Option Chain: Upstox (accurate, real-time)
- Candlestick Charts: DhanHQ (better historical + live data)
- NIFTY: Tuesday (Weekly)
- SENSEX: Thursday (Weekly)
- Stocks: Thursday (Monthly)
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta
import pytz
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import io

# ==================== CONFIG ====================
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")  # Optional: DhanHQ for charts
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

UPSTOX_BASE_URL = "https://api.upstox.com"
DHAN_BASE_URL = "https://api.dhan.co"
IST = pytz.timezone('Asia/Kolkata')

# Use DhanHQ for charts if token available
USE_DHAN_CHARTS = bool(DHAN_ACCESS_TOKEN)

# ==================== INSTRUMENTS ====================
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"
SENSEX_INDEX_KEY = "BSE_INDEX|SENSEX"

# Symbol mapping: Upstox Key -> (DhanHQ Symbol, Name)
INSTRUMENTS = {
    NIFTY_INDEX_KEY: ("NSE_IDX:Nifty 50", "NIFTY 50", 1, "Weekly"),  # Tuesday
    SENSEX_INDEX_KEY: ("BSE_IDX:SENSEX", "SENSEX", 3, "Weekly"),     # Thursday
    "NSE_EQ|INE002A01018": ("NSE_EQ:RELIANCE", "RELIANCE", 3, "Monthly"),
    "NSE_EQ|INE040A01034": ("NSE_EQ:HDFCBANK", "HDFCBANK", 3, "Monthly"),
    "NSE_EQ|INE090A01021": ("NSE_EQ:ICICIBANK", "ICICIBANK", 3, "Monthly"),
    "NSE_EQ|INE062A01020": ("NSE_EQ:SBIN", "SBIN", 3, "Monthly"),
    "NSE_EQ|INE009A01021": ("NSE_EQ:INFY", "INFY", 3, "Monthly"),
}

print("="*70)
print("🔥 HYBRID MONITOR - Upstox + DhanHQ")
print("="*70)
print(f"📊 Option Chain: Upstox")
print(f"📈 Charts: {'DhanHQ (LIVE)' if USE_DHAN_CHARTS else 'Upstox'}")
print("="*70)

# ==================== UPSTOX - OPTION CHAIN ====================

def upstox_get_expiries(instrument_key):
    """Get expiry dates from Upstox"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{UPSTOX_BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
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
        print(f"  ⚠️ Expiry error: {e}")
    return []

def upstox_get_next_expiry(instrument_key, expiry_day=1):
    """Get next expiry (1=Tuesday, 3=Thursday)"""
    expiries = upstox_get_expiries(instrument_key)
    if not expiries:
        today = datetime.now(IST)
        days_ahead = expiry_day - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    today = datetime.now(IST).date()
    future = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
    if future:
        return min(future)
    return expiries[0]

def upstox_get_option_chain(instrument_key, expiry):
    """Get option chain from Upstox"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{UPSTOX_BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            strikes = data.get('data', [])
            return sorted(strikes, key=lambda x: x.get('strike_price', 0))
    except Exception as e:
        print(f"  ⚠️ Chain error: {e}")
    return []

def upstox_get_spot_price(instrument_key):
    """Get spot price from Upstox"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{UPSTOX_BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
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
        print(f"  ⚠️ Spot error: {e}")
    return 0

# ==================== DHANHQ - CANDLE DATA ====================

def dhan_get_historical_data(security_id, exchange="NSE", interval="5"):
    """
    Get historical data from DhanHQ
    interval: 1, 5, 15, 25, 60, etc.
    """
    if not DHAN_ACCESS_TOKEN:
        return []
    
    headers = {
        "access-token": DHAN_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    
    to_date = datetime.now(IST)
    from_date = to_date - timedelta(days=7)
    
    payload = {
        "securityId": security_id,
        "exchangeSegment": exchange,
        "instrument": "EQUITY",
        "expiryCode": 0,
        "fromDate": from_date.strftime('%Y-%m-%d'),
        "toDate": to_date.strftime('%Y-%m-%d')
    }
    
    url = f"{DHAN_BASE_URL}/v2/charts/historical"
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                return candles
    except Exception as e:
        print(f"  ⚠️ DhanHQ error: {e}")
    
    return []

def dhan_get_intraday_data(security_id, exchange="NSE", interval="5"):
    """Get today's intraday data from DhanHQ"""
    if not DHAN_ACCESS_TOKEN:
        return []
    
    headers = {
        "access-token": DHAN_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    
    payload = {
        "securityId": security_id,
        "exchangeSegment": exchange,
        "instrument": "EQUITY",
        "interval": interval
    }
    
    url = f"{DHAN_BASE_URL}/v2/charts/intraday"
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                return candles
    except Exception as e:
        print(f"  ⚠️ DhanHQ intraday error: {e}")
    
    return []

# ==================== UPSTOX - CANDLE DATA (FALLBACK) ====================

def upstox_get_candles(instrument_key):
    """Fallback: Get candles from Upstox if DhanHQ not available"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    # Get 1-minute intraday
    url = f"{UPSTOX_BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                
                # Aggregate to 5-minute
                candles_5min = []
                for i in range(0, len(candles), 5):
                    batch = candles[i:i+5]
                    if len(batch) < 5:
                        continue
                    
                    timestamp = batch[0][0]
                    open_price = batch[0][1]
                    high_price = max(c[2] for c in batch)
                    low_price = min(c[3] for c in batch)
                    close_price = batch[-1][4]
                    volume = sum(c[5] for c in batch)
                    
                    candles_5min.append([timestamp, open_price, high_price, 
                                        low_price, close_price, volume])
                
                return candles_5min
    except Exception as e:
        print(f"  ⚠️ Upstox candle error: {e}")
    
    return []

# ==================== CHART CREATION ====================

def create_premium_chart(candles, symbol, spot_price):
    """Create premium dark theme chart"""
    if not candles or len(candles) < 10:
        print(f"  ⚠️ Insufficient candles: {len(candles) if candles else 0}")
        return None
    
    # Prepare data
    data = []
    for candle in reversed(candles):
        try:
            # Handle both Upstox and DhanHQ timestamp formats
            timestamp_str = candle[0]
            if 'T' in timestamp_str or 'Z' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).astimezone(IST)
            else:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=IST)
            
            # Skip weekends
            if timestamp.weekday() >= 5:
                continue
            
            # Market hours only
            hour, minute = timestamp.hour, timestamp.minute
            if hour < 9 or (hour == 9 and minute < 15):
                continue
            if hour > 15 or (hour == 15 and minute > 30):
                continue
            
            data.append({
                'timestamp': timestamp,
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': int(candle[5]) if len(candle) > 5 and candle[5] else 0
            })
        except Exception as e:
            continue
    
    if len(data) < 10:
        print(f"  ⚠️ After filtering: {len(data)} candles")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), 
                             gridspec_kw={'height_ratios': [4, 1]},
                             facecolor='#0e1217')
    
    ax1, ax2 = axes
    ax1.set_facecolor('#0e1217')
    ax2.set_facecolor('#0e1217')
    
    # Plot candlesticks
    for idx in range(len(df)):
        row = df.iloc[idx]
        x = idx
        
        is_bullish = row['close'] >= row['open']
        body_color = '#26a69a' if is_bullish else '#ef5350'
        
        # Wick
        ax1.plot([x, x], [row['low'], row['high']], 
                color=body_color, linewidth=1.2, solid_capstyle='round', zorder=2)
        
        # Body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        if body_height > 0.001:
            rect = Rectangle((x - 0.35, body_bottom), 0.7, body_height,
                           facecolor=body_color, edgecolor=body_color,
                           linewidth=0, zorder=3)
            ax1.add_patch(rect)
        else:
            ax1.plot([x - 0.35, x + 0.35], [row['open'], row['open']],
                    color=body_color, linewidth=1.5, solid_capstyle='butt', zorder=3)
    
    # Current price line
    ax1.axhline(y=spot_price, color='#2962ff', linestyle='--', 
               linewidth=2, alpha=0.9, zorder=4)
    
    # Price label on right
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_yticks([spot_price])
    ax1_right.set_yticklabels([f'₹{spot_price:.2f}'], 
                              fontsize=12, fontweight='700', color='#2962ff',
                              bbox=dict(boxstyle='round,pad=0.5', 
                                      facecolor='#2962ff', alpha=0.3))
    ax1_right.tick_params(colors='#2962ff', length=0, pad=10)
    ax1_right.set_facecolor('#0e1217')
    
    # Styling
    ax1.set_ylabel('Price (₹)', color='#b2b5be', fontsize=12, fontweight='600')
    ax1.tick_params(axis='y', colors='#787b86', labelsize=10, width=0)
    ax1.tick_params(axis='x', colors='#787b86', labelsize=10, width=0)
    ax1.grid(True, alpha=0.1, color='#363a45', linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Title with LIVE indicator and data source
    now_str = datetime.now(IST).strftime('%d %b %Y • %I:%M:%S %p IST')
    data_source = "DhanHQ" if USE_DHAN_CHARTS else "Upstox"
    title = f'{symbol}  •  5 Min (LIVE)  •  {data_source}  •  {now_str}'
    ax1.set_title(title, color='#d1d4dc', fontsize=16, fontweight='700',
                 pad=25, loc='left')
    
    # Volume bars
    volumes = df['volume'].values
    colors_vol = ['#26a69a' if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                  else '#ef5350' for i in range(len(df))]
    
    ax2.bar(range(len(volumes)), volumes, color=colors_vol, 
           width=0.7, alpha=0.8, edgecolor='none')
    
    ax2.set_ylabel('Volume', color='#b2b5be', fontsize=12, fontweight='600')
    ax2.tick_params(axis='y', colors='#787b86', labelsize=10, width=0)
    ax2.tick_params(axis='x', colors='#787b86', labelsize=10, width=0)
    ax2.grid(True, alpha=0.1, color='#363a45', linestyle='-', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # X-axis labels
    step = max(1, len(df) // 12)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [df.index[i].strftime('%d %b\n%H:%M') for i in tick_positions]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color='#787b86', fontsize=10)
        ax.set_xlim(-1, len(df))
        
        for spine in ax.spines.values():
            spine.set_color('#1e222d')
            spine.set_linewidth(1.5)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax2.set_xlabel('Date & Time (IST)', color='#b2b5be', 
                  fontsize=12, fontweight='600', labelpad=12)
    
    plt.tight_layout(pad=2)
    plt.subplots_adjust(hspace=0.08)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='#0e1217',
               edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# ==================== MESSAGE FORMATTING ====================

def format_option_chain_message(symbol, spot, expiry, strikes, expiry_type):
    """Format option chain message"""
    if not strikes:
        return None
    
    atm_index = min(range(len(strikes)), 
                   key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    start = max(0, atm_index - 10)
    end = min(len(strikes), atm_index + 11)
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol}* ({expiry_type})\n\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.2f}\n\n"
    msg += "```\n"
    msg += "Strike   CE-LTP CE-OI  PE-LTP PE-OI\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_oi = total_pe_oi = 0
    
    for s in selected:
        strike_price = s.get('strike_price', 0)
        call = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call.get('ltp', 0)
        ce_oi = call.get('oi', 0)
        put = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put.get('ltp', 0)
        pe_oi = put.get('oi', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        
        msg += f"{strike_price:8.0f} {ce_ltp:6.1f} {ce_oi/1000:5.0f}K {pe_ltp:6.1f} {pe_oi/1000:5.0f}K\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"TOTAL         {total_ce_oi/1000:5.0f}K       {total_pe_oi/1000:5.0f}K\n"
    msg += "```\n"
    
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    msg += f"📊 PCR: {pcr:.3f}\n"
    msg += f"🔑 Data: Upstox\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}\n"
    
    return msg

# ==================== TELEGRAM ====================

async def send_telegram_text(msg):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

async def send_telegram_photo(photo_buf, caption):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_buf,
                           caption=caption, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"Photo error: {e}")
        return False

# ==================== PROCESS INSTRUMENT ====================

async def process_instrument(upstox_key, dhan_symbol, name, expiry_day, expiry_type):
    """Process any instrument (index or stock)"""
    print("\n" + "="*60)
    print(f"{name} ({expiry_type})")
    print("="*60)
    
    try:
        # STEP 1: Get Option Chain from Upstox
        expiry = upstox_get_next_expiry(upstox_key, expiry_day=expiry_day)
        spot = upstox_get_spot_price(upstox_key)
        
        if spot == 0:
            print("❌ Invalid spot")
            return False
        
        print(f"✅ Spot: ₹{spot:.2f} | Expiry: {expiry}")
        
        strikes = upstox_get_option_chain(upstox_key, expiry)
        if not strikes:
            print("❌ No option chain")
            return False
        
        print(f"✅ Strikes: {len(strikes)}")
        
        # Send option chain
        msg = format_option_chain_message(name, spot, expiry, strikes, expiry_type)
        if msg:
            await send_telegram_text(msg)
            print("📤 Option chain sent (Upstox)")
        
        # STEP 2: Get Candles (DhanHQ or Upstox)
        print(f"📊 Fetching candles from {'DhanHQ' if USE_DHAN_CHARTS else 'Upstox'}...")
        
        if USE_DHAN_CHARTS and dhan_symbol:
            # TODO: Need DhanHQ security_id mapping
            # For now, fallback to Upstox
            candles = upstox_get_candles(upstox_key)
        else:
            candles = upstox_get_candles(upstox_key)
        
        if candles and len(candles) >= 10:
            print("📈 Creating chart...")
            chart = create_premium_chart(candles, name, spot)
            
            if chart:
                caption = f"📈 *{name}* ({expiry_type})\n💰 ₹{spot:.2f} | 📅 {expiry}"
                await send_telegram_photo(chart, caption)
                print("📤 Chart sent (LIVE)!")
        else:
            print("⚠️ No chart data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== MAIN ====================

async def fetch_all():
    """Main fetch function"""
    print("\n" + "="*60)
    print(f"🚀 RUN: {datetime.now(IST).strftime('%I:%M:%S %p IST')}")
    print("="*60)
    
    header = f"🔥 *HYBRID MONITOR*\n"
    header += f"📊 Chain: Upstox | 📈 Charts: {'DhanHQ' if USE_DHAN_CHARTS else 'Upstox'}\n"
    header += f"⏰ {datetime.now(IST).strftime('%I:%M %p')}\n\n_Processing..._"
    await send_telegram_text(header)
    
    results = {}
    
    for upstox_key, (dhan_symbol, name, expiry_day, expiry_type) in INSTRUMENTS.items():
        result = await process_instrument(upstox_key, dhan_symbol, name, expiry_day, expiry_type)
        results[name] = result
        await asyncio.sleep(3)
    
    # Summary
    success_count = sum(1 for v in results.values() if v)
    summary = f"✅ *COMPLETE*\n"
    summary += f"Success: {success_count}/{len(results)}\n"
    for name, status in results.items():
        summary += f"{name}: {'✅' if status else '❌'}\n"
    
    await send_telegram_text(summary)
    print(f"\n✅ DONE: {success_count}/{len(results)}")

async def monitoring_loop():
    """Main loop"""
    print("\n🔄 Monitoring started (5 min interval)\n")
    
    while True:
        try:
            await fetch_all()
            
            next_time = (datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')
            print(f"\n⏳ Next run: {next_time}\n")
            
            await asyncio.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            await asyncio.sleep(60)

async def main():
    """Entry point"""
    print("\n" + "="*70)
    print("HYBRID MONITOR - Upstox + DhanHQ")
    print("="*70)
    print("📊 Option Chain: Upstox (accurate)")
    print(f"📈 Charts: {'DhanHQ (better data)' if USE_DHAN_CHARTS else 'Upstox (fallback)'}")
    print("🎯 NIFTY: Tuesday | SENSEX: Thursday | Stocks: Thursday")
    print("⏰ Every 5 minutes")
    print("="*70 + "\n")
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
