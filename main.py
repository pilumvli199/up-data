#!/usr/bin/env python3
"""
NIFTY 50 + SENSEX + STOCKS MONITOR
- NIFTY: Tuesday (Weekly)
- SENSEX: Thursday (Weekly)  
- Stocks: Thursday (Monthly)
- LIVE 5min charts + Option Chain
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

# CONFIG
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# INDICES
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"
SENSEX_INDEX_KEY = "BSE_INDEX|SENSEX"

# NIFTY 50 STOCKS
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE009A01021": "INFY",
}

print("="*70)
print("🚀 NIFTY + SENSEX LIVE MONITOR")
print("="*70)

def get_expiries(instrument_key):
    """Get available expiry dates"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
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
        print(f"Expiry error: {e}")
    return []

def get_next_expiry(instrument_key, expiry_day=1):
    """
    Get next expiry
    expiry_day: 1=Tuesday, 3=Thursday
    """
    expiries = get_expiries(instrument_key)
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

def get_option_chain(instrument_key, expiry):
    """Get option chain data"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            strikes = data.get('data', [])
            return sorted(strikes, key=lambda x: x.get('strike_price', 0))
    except Exception as e:
        print(f"Chain error: {e}")
    return []

def get_spot_price(instrument_key):
    """Get current spot/index price"""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
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
        print(f"Spot error: {e}")
    return 0

def split_30min_to_5min(candle_30min):
    """
    Split a 30-minute candle into 6 x 5-minute candles
    Distributes price movement proportionally
    """
    timestamp = candle_30min[0]
    open_price = float(candle_30min[1])
    high_price = float(candle_30min[2])
    low_price = float(candle_30min[3])
    close_price = float(candle_30min[4])
    volume = int(candle_30min[5]) if candle_30min[5] else 0
    oi = int(candle_30min[6]) if len(candle_30min) > 6 and candle_30min[6] else 0
    
    # Parse timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).astimezone(IST)
    except:
        dt = datetime.now(IST)
    
    candles_5min = []
    price_range = close_price - open_price
    vol_per_candle = volume // 6
    
    for i in range(6):
        # Calculate timestamp for each 5-min candle
        candle_time = dt + timedelta(minutes=i*5)
        ts = candle_time.isoformat()
        
        # Distribute price movement
        progress = (i + 1) / 6
        current_close = open_price + (price_range * progress)
        current_open = open_price + (price_range * (i / 6)) if i > 0 else open_price
        
        # Distribute high/low proportionally
        if price_range >= 0:  # Bullish
            current_high = min(high_price, current_close + (high_price - open_price) * 0.3)
            current_low = max(low_price, current_open - (open_price - low_price) * 0.3)
        else:  # Bearish
            current_high = min(high_price, current_open + (high_price - close_price) * 0.3)
            current_low = max(low_price, current_close - (close_price - low_price) * 0.3)
        
        # Last candle gets exact close
        if i == 5:
            current_close = close_price
        
        candles_5min.append([
            ts,
            current_open,
            current_high,
            current_low,
            current_close,
            vol_per_candle,
            oi
        ])
    
    return candles_5min

def get_live_candles(instrument_key, symbol):
    """
    Get combined historical + live candles
    - Historical: 30min data from last 5 days, split into 5min
    - Live: Today's 1min data, aggregated to 5min
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    all_candles_5min = []
    
    # STEP 1: Get HISTORICAL 30min data (last 5 days)
    print(f"  🔍 Fetching historical 30min data...")
    try:
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=5)
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_str}/{from_str}"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                hist_candles_30min = data.get('data', {}).get('candles', [])
                if hist_candles_30min:
                    print(f"  ✅ Historical: {len(hist_candles_30min)} x 30min candles")
                    
                    # Filter out today's data
                    today_date = datetime.now(IST).date()
                    filtered_30min = []
                    for c in hist_candles_30min:
                        try:
                            c_dt = datetime.fromisoformat(c[0].replace('Z', '+00:00')).astimezone(IST)
                            if c_dt.date() < today_date:
                                filtered_30min.append(c)
                        except:
                            pass
                    
                    print(f"  📊 Filtered historical: {len(filtered_30min)} x 30min candles")
                    
                    # Split 30min to 5min
                    print(f"  🔄 Splitting 30min → 5min...")
                    for candle_30 in filtered_30min:
                        split_candles = split_30min_to_5min(candle_30)
                        all_candles_5min.extend(split_candles)
                    
                    print(f"  ✅ After split: {len(all_candles_5min)} x 5min candles")
        else:
            print(f"  ⚠️ Historical HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ Historical error: {e}")
    
    # STEP 2: Get TODAY'S LIVE 1min data
    print(f"  🔍 Fetching TODAY'S LIVE 1min data...")
    today_candles_1min = []
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                today_candles_1min = data.get('data', {}).get('candles', [])
                if today_candles_1min:
                    print(f"  ✅ TODAY LIVE: {len(today_candles_1min)} x 1min candles")
                else:
                    print(f"  ⚠️ No today data")
        else:
            print(f"  ⚠️ Today HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ Today error: {e}")
    
    # STEP 3: Aggregate today's 1min → 5min
    if today_candles_1min and len(today_candles_1min) >= 5:
        print(f"  🔄 Aggregating today's 1min → 5min...")
        
        # Sort by time (reverse because API returns newest first)
        today_candles_1min = sorted(today_candles_1min, 
                                     key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
        
        i = 0
        while i < len(today_candles_1min):
            batch = today_candles_1min[i:i+5]
            
            if len(batch) >= 5:  # Full 5-min batch
                timestamp = batch[0][0]
                open_price = float(batch[0][1])
                high_price = max(float(c[2]) for c in batch)
                low_price = min(float(c[3]) for c in batch)
                close_price = float(batch[-1][4])
                volume = sum(int(c[5]) if c[5] else 0 for c in batch)
                oi = int(batch[-1][6]) if len(batch[-1]) > 6 and batch[-1][6] else 0
                
                all_candles_5min.append([
                    timestamp, open_price, high_price, 
                    low_price, close_price, volume, oi
                ])
            
            i += 5
        
        print(f"  ✅ Today's 5min candles: {(len(today_candles_1min) // 5)}")
    
    # STEP 4: Sort all candles by time
    if all_candles_5min:
        all_candles_5min = sorted(all_candles_5min,
                                  key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
        print(f"  ✅ FINAL TOTAL: {len(all_candles_5min)} x 5min candles (Historical + Today)")
        return all_candles_5min
    
    print(f"  ❌ {symbol}: No data available")
    return []

def create_premium_chart(candles, symbol, spot_price):
    """Create PREMIUM TradingView-style chart WITHOUT mplfinance"""
    if not candles or len(candles) < 10:
        print(f"  ⚠️ Insufficient candles: {len(candles) if candles else 0}")
        return None
    
    # Prepare data
    data = []
    for candle in candles:
        try:
            timestamp = datetime.fromisoformat(candle[0].replace('Z', '+00:00')).astimezone(IST)
            
            # Skip weekends
            if timestamp.weekday() >= 5:
                continue
            
            # Market hours only (9:15 AM to 3:30 PM)
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
                'volume': int(candle[5]) if candle[5] else 0
            })
        except Exception as e:
            continue
    
    if len(data) < 10:
        print(f"  ⚠️ After filtering: {len(data)} candles (need 10+)")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), 
                             gridspec_kw={'height_ratios': [4, 1]},
                             facecolor='#0e1217')
    
    ax1, ax2 = axes
    ax1.set_facecolor('#0e1217')
    ax2.set_facecolor('#0e1217')
    
    # Plot candlesticks
    for idx in range(len(data)):
        row = data[idx]
        x = idx
        
        is_bullish = row['close'] >= row['open']
        body_color = '#26a69a' if is_bullish else '#ef5350'
        
        # Wick (high-low line)
        ax1.plot([x, x], [row['low'], row['high']], 
                color=body_color, linewidth=1.2, solid_capstyle='round', zorder=2)
        
        # Body (open-close rectangle)
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        if body_height > 0.001:
            rect = Rectangle((x - 0.35, body_bottom), 0.7, body_height,
                           facecolor=body_color, edgecolor=body_color,
                           linewidth=0, zorder=3)
            ax1.add_patch(rect)
        else:
            # Doji - flat line
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
    
    # Title with LIVE indicator
    now_str = datetime.now(IST).strftime('%d %b %Y • %I:%M:%S %p IST')
    title = f'{symbol}  •  5 Min Chart (LIVE)  •  {now_str}'
    ax1.set_title(title, color='#d1d4dc', fontsize=16, fontweight='700',
                 pad=25, loc='left')
    
    # Volume bars
    volumes = [d['volume'] for d in data]
    colors_vol = ['#26a69a' if data[i]['close'] >= data[i]['open'] 
                  else '#ef5350' for i in range(len(data))]
    
    ax2.bar(range(len(volumes)), volumes, color=colors_vol, 
           width=0.7, alpha=0.8, edgecolor='none')
    
    ax2.set_ylabel('Volume', color='#b2b5be', fontsize=12, fontweight='600')
    ax2.tick_params(axis='y', colors='#787b86', labelsize=10, width=0)
    ax2.tick_params(axis='x', colors='#787b86', labelsize=10, width=0)
    ax2.grid(True, alpha=0.1, color='#363a45', linestyle='-', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # X-axis labels
    step = max(1, len(data) // 12)
    tick_positions = list(range(0, len(data), step))
    tick_labels = [data[i]['timestamp'].strftime('%d %b\n%H:%M') for i in tick_positions]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color='#787b86', fontsize=10)
        ax.set_xlim(-1, len(data))
        
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

def format_option_chain_message(symbol, spot, expiry, strikes):
    """Format option chain message"""
    if not strikes:
        return None
    
    atm_index = min(range(len(strikes)), 
                   key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    start = max(0, atm_index - 10)
    end = min(len(strikes), atm_index + 11)
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol}*\n\n"
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
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}\n"
    
    return msg

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

async def process_index(index_key, index_name, expiry_day, expiry_type):
    """Process NIFTY or SENSEX index"""
    print("\n" + "="*60)
    print(f"{index_name} ({expiry_type})")
    print("="*60)
    
    try:
        expiry = get_next_expiry(index_key, expiry_day=expiry_day)
        spot = get_spot_price(index_key)
        
        if spot == 0:
            print("❌ Invalid spot price")
            return False
        
        print(f"✅ Spot: ₹{spot:.2f}")
        print(f"📅 Expiry: {expiry}")
        
        strikes = get_option_chain(index_key, expiry)
        if not strikes:
            print("❌ No option chain")
            return False
        
        print(f"✅ Strikes: {len(strikes)}")
        
        # Send option chain
        msg = format_option_chain_message(index_name, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
            print("📤 Option chain sent")
        
        # Send LIVE chart
        print("📊 Fetching LIVE candles (Historical 30min + Today 1min)...")
        candles = get_live_candles(index_key, index_name)
        
        if candles and len(candles) >= 10:
            print("📈 Creating premium chart...")
            chart = create_premium_chart(candles, index_name, spot)
            
            if chart:
                caption = f"📈 *{index_name}* ({expiry_type})\n💰 ₹{spot:.2f} | 📅 {expiry}"
                await send_telegram_photo(chart, caption)
                print("📤 Chart sent (LIVE)!")
                return True
        else:
            print("⚠️ Insufficient candle data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def process_stock(key, symbol, idx, total):
    """Process single stock"""
    print(f"\n[{idx}/{total}] {symbol}")
    
    try:
        expiry = get_next_expiry(key, expiry_day=3)
        spot = get_spot_price(key)
        
        if spot == 0:
            print("  ❌ Invalid spot")
            return False
        
        strikes = get_option_chain(key, expiry)
        if not strikes:
            print("  ❌ No strikes")
            return False
        
        print(f"  ✅ Spot: ₹{spot:.2f} | Strikes: {len(strikes)}")
        
        # Send option chain
        msg = format_option_chain_message(symbol, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
            print("  📤 Chain sent")
        
        # Send LIVE chart
        candles = get_live_candles(key, symbol)
        if candles and len(candles) >= 10:
            chart = create_premium_chart(candles, symbol, spot)
            if chart:
                caption = f"📈 *{symbol}* (Monthly)\n💰 ₹{spot:.2f}"
                await send_telegram_photo(chart, caption)
                print("  📤 Chart sent (LIVE)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def fetch_all():
    """Main fetch function"""
    print("\n" + "="*60)
    print(f"🚀 RUN: {datetime.now(IST).strftime('%I:%M:%S %p IST')}")
    print("="*60)
    
    header = f"🚀 *MARKET MONITOR*\n⏰ {datetime.now(IST).strftime('%I:%M %p')}\n\n_Processing..._"
    await send_telegram_text(header)
    
    # NIFTY (Tuesday Weekly)
    nifty_ok = await process_index(NIFTY_INDEX_KEY, "NIFTY 50", 
                                   expiry_day=1, expiry_type="Weekly Tuesday")
    await asyncio.sleep(3)
    
    # SENSEX (Thursday Weekly)
    sensex_ok = await process_index(SENSEX_INDEX_KEY, "SENSEX", 
                                    expiry_day=3, expiry_type="Weekly Thursday")
    await asyncio.sleep(3)
    
    # Stocks (Thursday Monthly)
    success = 0
    total = len(NIFTY50_STOCKS)
    
    for idx, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        result = await process_stock(key, symbol, idx, total)
        if result:
            success += 1
        await asyncio.sleep(3)
    
    summary = f"✅ *COMPLETE*\n"
    summary += f"NIFTY: {'✅' if nifty_ok else '❌'}\n"
    summary += f"SENSEX: {'✅' if sensex_ok else '❌'}\n"
    summary += f"Stocks: {success}/{total}"
    await send_telegram_text(summary)
    
    print(f"\n✅ DONE: NIFTY={nifty_ok} | SENSEX={sensex_ok} | Stocks={success}/{total}")

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
            print("\n🛑 Stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            await asyncio.sleep(60)

async def main():
    """Entry point"""
    print("\n" + "="*70)
    print("NIFTY + SENSEX + STOCKS LIVE MONITOR")
    print("="*70)
    print("📊 NIFTY: Tuesday (Weekly)")
    print("📊 SENSEX: Thursday (Weekly)")
    print("📈 Stocks: Thursday (Monthly)")
    print("🎨 Premium dark theme charts (NO mplfinance)")
    print("📦 Historical 30min → 5min split")
    print("⏰ LIVE 1min → 5min aggregation")
    print("🔄 Combined charts every 5 minutes")
    print("="*70 + "\n")
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
