#!/usr/bin/env python3
"""
COMPLETE MARKET MONITOR
- NIFTY 50, BANKNIFTY, FINNIFTY, MIDCPNIFTY
- All NIFTY 50 Stocks + POONAWALLA
- 3:30 PM Daily Summary
- Rate limit management
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
import time

# CONFIG
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# INDICES
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY 50", "expiry_day": 1, "type": "Weekly Tuesday"},
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "expiry_day": 2, "type": "Weekly Wednesday"},
    "NSE_INDEX|Nifty Fin Service": {"name": "FINNIFTY", "expiry_day": 1, "type": "Weekly Tuesday"},
    "NSE_INDEX|Nifty Midcap Select": {"name": "MIDCPNIFTY", "expiry_day": 0, "type": "Weekly Monday"},
    "BSE_INDEX|SENSEX": {"name": "SENSEX", "expiry_day": 3, "type": "Weekly Thursday"},
}

# ALL NIFTY 50 STOCKS + POONAWALLA (Complete List)
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE467B01029": "TATAMOTORS",
    "NSE_EQ|INE030A01027": "BHARTIARTL",
    "NSE_EQ|INE018A01030": "HCLTECH",
    "NSE_EQ|INE155A01022": "TATASTEEL",
    "NSE_EQ|INE242A01010": "AXISBANK",
    "NSE_EQ|INE528G01035": "ULTRACEMCO",
    "NSE_EQ|INE848E01016": "NESTLEIND",
    "NSE_EQ|INE019A01038": "ASIANPAINT",
    "NSE_EQ|INE854D01024": "TITAN",
    "NSE_EQ|INE532F01054": "BAJFINANCE",
    "NSE_EQ|INE205A01025": "KOTAKBANK",
    "NSE_EQ|INE758T01015": "TECHM",
    "NSE_EQ|INE239A01016": "HINDALCO",
    "NSE_EQ|INE256A01028": "SUNPHARMA",
    "NSE_EQ|INE160A01022": "MARUTI",
    "NSE_EQ|INE021A01026": "ASIANPAINT",
    "NSE_EQ|INE522F01014": "COALINDIA",
    "NSE_EQ|INE238A01034": "LTIM",
    "NSE_EQ|INE018E01016": "NTPC",
    "NSE_EQ|INE029A01011": "BPCL",
    "NSE_EQ|INE216A01030": "GRASIM",
    "NSE_EQ|INE066A01021": "ADANIENT",
    "NSE_EQ|INE423A01024": "ADANIPORTS",
    "NSE_EQ|INE448A01043": "SBILIFE",
    "NSE_EQ|INE220B01022": "BAJAJFINSV",
    "NSE_EQ|INE239A01024": "JSWSTEEL",
    "NSE_EQ|INE027A01015": "POWERGRID",
    "NSE_EQ|INE114A01011": "SHREECEM",
    "NSE_EQ|INE192A01025": "TATACONSUM",
    "NSE_EQ|INE118A01012": "ADANIPOWER",
    "NSE_EQ|INE121A01024": "M&M",
    "NSE_EQ|INE769A01020": "ONGC",
    "NSE_EQ|INE127D01025": "HEROMOTOCO",
    "NSE_EQ|INE669E01016": "TECHM",
    "NSE_EQ|INE066F01020": "EICHERMOT",
    "NSE_EQ|INE075A01022": "WIPRO",
    "NSE_EQ|INE040H01021": "SIEMENS",
    "NSE_EQ|INE397D01024": "HINDUNILVR",
    "NSE_EQ|INE095A01012": "INDUSINDBK",
    "NSE_EQ|INE044A01036": "LT",
    "NSE_EQ|INE070A01015": "TATAPOWER",
    "NSE_EQ|INE647O01011": "ICICIPRULI",
    "NSE_EQ|INE234A01024": "DIVISLAB",
    "NSE_EQ|INE180A01020": "DRREDDY",
    "NSE_EQ|INE020B01018": "BRITANNIA",
    "NSE_EQ|INE752E01010": "POONAWALLA",  # POONAWALLA FINCORP
}

# STATISTICS TRACKER
stats = {
    "total_runs": 0,
    "indices_success": 0,
    "stocks_success": 0,
    "total_api_calls": 0,
    "daily_summary_sent": False
}

print("="*70)
print("🚀 COMPLETE MARKET MONITOR")
print(f"📊 {len(INDICES)} Indices + {len(NIFTY50_STOCKS)} Stocks")
print("="*70)

def get_expiries(instrument_key):
    """Get available expiry dates"""
    stats["total_api_calls"] += 1
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
    """Get next expiry (0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday)"""
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
    """Get option chain data with Greeks"""
    stats["total_api_calls"] += 1
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
    stats["total_api_calls"] += 1
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
    """Split 30-minute candle into 6x5-minute candles"""
    timestamp = candle_30min[0]
    open_price = float(candle_30min[1])
    high_price = float(candle_30min[2])
    low_price = float(candle_30min[3])
    close_price = float(candle_30min[4])
    volume = int(candle_30min[5]) if candle_30min[5] else 0
    oi = int(candle_30min[6]) if len(candle_30min) > 6 and candle_30min[6] else 0
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).astimezone(IST)
    except:
        dt = datetime.now(IST)
    
    candles_5min = []
    price_range = close_price - open_price
    vol_per_candle = volume // 6
    
    for i in range(6):
        candle_time = dt + timedelta(minutes=i*5)
        ts = candle_time.isoformat()
        
        progress = (i + 1) / 6
        current_close = open_price + (price_range * progress)
        current_open = open_price + (price_range * (i / 6)) if i > 0 else open_price
        
        if price_range >= 0:
            current_high = min(high_price, current_close + (high_price - open_price) * 0.3)
            current_low = max(low_price, current_open - (open_price - low_price) * 0.3)
        else:
            current_high = min(high_price, current_open + (high_price - close_price) * 0.3)
            current_low = max(low_price, current_close - (close_price - low_price) * 0.3)
        
        if i == 5:
            current_close = close_price
        
        candles_5min.append([
            ts, current_open, current_high, current_low, 
            current_close, vol_per_candle, oi
        ])
    
    return candles_5min

def get_live_candles(instrument_key, symbol):
    """Get historical (30min split) + live (1min aggregated) candles"""
    stats["total_api_calls"] += 2  # 2 API calls
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    historical_5min = []
    today_5min = []
    
    # Historical 30min data
    try:
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=10)
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_str}/{from_str}"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                hist_candles_30min = data.get('data', {}).get('candles', [])
                if hist_candles_30min:
                    today_date = datetime.now(IST).date()
                    for c in hist_candles_30min:
                        try:
                            c_dt = datetime.fromisoformat(c[0].replace('Z', '+00:00')).astimezone(IST)
                            if c_dt.date() < today_date:
                                split_candles = split_30min_to_5min(c)
                                historical_5min.extend(split_candles)
                        except:
                            pass
    except Exception as e:
        print(f"  ⚠️ Hist error: {e}")
    
    # Today's LIVE 1min data
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                today_candles_1min = data.get('data', {}).get('candles', [])
                if today_candles_1min:
                    today_candles_1min = sorted(today_candles_1min,
                                               key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
                    
                    i = 0
                    while i < len(today_candles_1min):
                        batch = today_candles_1min[i:i+5]
                        
                        if len(batch) >= 5:
                            timestamp = batch[0][0]
                            open_price = float(batch[0][1])
                            high_price = max(float(c[2]) for c in batch)
                            low_price = min(float(c[3]) for c in batch)
                            close_price = float(batch[-1][4])
                            volume = sum(int(c[5]) if c[5] else 0 for c in batch)
                            oi = int(batch[-1][6]) if len(batch[-1]) > 6 and batch[-1][6] else 0
                            
                            today_5min.append([
                                timestamp, open_price, high_price,
                                low_price, close_price, volume, oi
                            ])
                        
                        i += 5
    except Exception as e:
        print(f"  ⚠️ Today error: {e}")
    
    # Combine
    all_candles = historical_5min + today_5min
    
    if all_candles:
        all_candles = sorted(all_candles,
                            key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
        return all_candles, len(historical_5min)
    
    return [], 0

def create_premium_chart(candles, symbol, spot_price, hist_count):
    """Create enhanced chart with historical/live distinction"""
    if not candles or len(candles) < 10:
        return None
    
    data = []
    for candle in candles:
        try:
            timestamp = datetime.fromisoformat(candle[0].replace('Z', '+00:00')).astimezone(IST)
            
            if timestamp.weekday() >= 5:
                continue
            
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
        except:
            continue
    
    if len(data) < 10:
        return None
    
    # Create figure - WIDER
    fig, axes = plt.subplots(2, 1, figsize=(28, 13),
                             gridspec_kw={'height_ratios': [4, 1]},
                             facecolor='#0e1217')
    
    ax1, ax2 = axes
    ax1.set_facecolor('#0e1217')
    ax2.set_facecolor('#0e1217')
    
    # Calculate today
    today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Plot candlesticks - BRIGHTER historical
    for idx in range(len(data)):
        row = data[idx]
        x = idx
        
        is_bullish = row['close'] >= row['open']
        is_today = row['timestamp'] >= today_start
        
        alpha = 1.0 if is_today else 0.85  # Brighter historical
        body_color = '#26a69a' if is_bullish else '#ef5350'
        
        # Wick
        ax1.plot([x, x], [row['low'], row['high']],
                color=body_color, linewidth=1.5, solid_capstyle='round',
                alpha=alpha, zorder=2)
        
        # Body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        if body_height > 0.001:
            rect = Rectangle((x - 0.4, body_bottom), 0.8, body_height,
                           facecolor=body_color, edgecolor=body_color,
                           linewidth=0, alpha=alpha, zorder=3)
            ax1.add_patch(rect)
        else:
            ax1.plot([x - 0.4, x + 0.4], [row['open'], row['open']],
                    color=body_color, linewidth=2, alpha=alpha, zorder=3)
    
    # Today marker
    today_idx = None
    for i, d in enumerate(data):
        if d['timestamp'] >= today_start:
            today_idx = i
            break
    
    if today_idx:
        ax1.axvline(x=today_idx, color='#ffa726', linestyle='--',
                   linewidth=2, alpha=0.5, zorder=1)
        ax2.axvline(x=today_idx, color='#ffa726', linestyle='--',
                   linewidth=2, alpha=0.5, zorder=1)
        
        y_pos = ax1.get_ylim()[1] * 0.98
        ax1.text(today_idx, y_pos, ' TODAY ', 
                color='#ffa726', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#0e1217',
                         edgecolor='#ffa726', linewidth=1.5),
                verticalalignment='top', zorder=5)
    
    # Current price
    ax1.axhline(y=spot_price, color='#2962ff', linestyle='--',
               linewidth=2.5, alpha=0.9, zorder=4)
    
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_yticks([spot_price])
    ax1_right.set_yticklabels([f'₹{spot_price:.2f}'],
                              fontsize=13, fontweight='700', color='#2962ff',
                              bbox=dict(boxstyle='round,pad=0.6',
                                      facecolor='#2962ff', alpha=0.3))
    ax1_right.tick_params(colors='#2962ff', length=0, pad=10)
    ax1_right.set_facecolor('#0e1217')
    
    # Styling
    ax1.set_ylabel('Price (₹)', color='#b2b5be', fontsize=13, fontweight='600')
    ax1.tick_params(axis='y', colors='#787b86', labelsize=11, width=0)
    ax1.tick_params(axis='x', colors='#787b86', labelsize=11, width=0)
    ax1.grid(True, alpha=0.12, color='#363a45', linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Title
    now_str = datetime.now(IST).strftime('%d %b %Y • %I:%M:%S %p IST')
    title = f'{symbol}  •  5 Min Chart (LIVE)  •  {now_str}'
    ax1.set_title(title, color='#d1d4dc', fontsize=17, fontweight='700',
                 pad=25, loc='left')
    
    # Volume - BRIGHTER
    volumes = [d['volume'] for d in data]
    colors_vol = []
    for i in range(len(data)):
        is_bull = data[i]['close'] >= data[i]['open']
        is_today = data[i]['timestamp'] >= today_start
        color = '#26a69a' if is_bull else '#ef5350'
        alpha_vol = 1.0 if is_today else 0.75  # Brighter
        colors_vol.append((matplotlib.colors.to_rgba(color, alpha=alpha_vol)))
    
    ax2.bar(range(len(volumes)), volumes, color=colors_vol,
           width=0.8, edgecolor='none')
    
    ax2.set_ylabel('Volume', color='#b2b5be', fontsize=13, fontweight='600')
    ax2.tick_params(axis='y', colors='#787b86', labelsize=11, width=0)
    ax2.tick_params(axis='x', colors='#787b86', labelsize=11, width=0)
    ax2.grid(True, alpha=0.12, color='#363a45', linestyle='-', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # X-axis
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
                  fontsize=13, fontweight='600', labelpad=12)
    
    plt.tight_layout(pad=2)
    plt.subplots_adjust(hspace=0.08)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, facecolor='#0e1217',
               edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def format_option_chain_message(symbol, spot, expiry, strikes):
    """Format COMPACT option chain for faster sending"""
    if not strikes:
        return None
    
    atm_index = min(range(len(strikes)),
                   key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    start = max(0, atm_index - 5)
    end = min(len(strikes), atm_index + 6)
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol}*\n"
    msg += f"💰 ₹{spot:,.2f} | 📅 {expiry}\n\n```\n"
    msg += "Strike  CE-LTP PE-LTP\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_oi = total_pe_oi = 0
    
    for s in selected:
        strike_price = s.get('strike_price', 0)
        call_data = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call_data.get('ltp', 0)
        ce_oi = call_data.get('oi', 0)
        put_data = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put_data.get('ltp', 0)
        pe_oi = put_data.get('oi', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        
        is_atm = (strike_price == strikes[atm_index].get('strike_price', 0))
        marker = "►" if is_atm else " "
        
        msg += f"{marker}{strike_price:6.0f}  {ce_ltp:6.1f}  {pe_ltp:6.1f}\n"
    
    msg += "```\n"
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    msg += f"PCR: {pcr:.2f} | {datetime.now(IST).strftime('%I:%M %p')}\n"
    
    return msg

async def send_daily_summary():
    """Send 3:30 PM daily summary"""
    msg = f"📊 *DAILY SUMMARY* 📊\n"
    msg += f"🕒 {datetime.now(IST).strftime('%d %b %Y - %I:%M %p')}\n\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📈 *INDICES:* {stats['indices_success']} processed\n"
    msg += f"📊 *STOCKS:* {stats['stocks_success']} processed\n"
    msg += f"🔄 *TOTAL RUNS:* {stats['total_runs']}\n"
    msg += f"📡 *API CALLS:* {stats['total_api_calls']}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Calculate alerts per 5min
    total_alerts = stats['indices_success'] + stats['stocks_success']
    alerts_per_run = total_alerts / stats['total_runs'] if stats['total_runs'] > 0 else 0
    
    msg += f"⚡ *Alerts per 5min:* {alerts_per_run:.1f}\n"
    msg += f"📊 *Total Alerts Today:* {total_alerts}\n\n"
    
    # Rate limit warning
    if stats['total_api_calls'] > 5000:
        msg += f"⚠️ *HIGH API USAGE*\n"
        msg += f"Consider reducing frequency\n\n"
    
    msg += f"✅ Monitoring completed successfully!"
    
    await send_telegram_text(msg)
    print("\n📊 DAILY SUMMARY SENT")

async def check_and_send_summary():
    """Check if 3:30 PM and send summary"""
    now = datetime.now(IST)
    
    # Reset daily flag at midnight
    if now.hour == 0 and now.minute < 5:
        stats["daily_summary_sent"] = False
    
    # Send at 3:30 PM once
    if now.hour == 15 and now.minute >= 30 and now.minute < 35:
        if not stats["daily_summary_sent"]:
            await send_daily_summary()
            stats["daily_summary_sent"] = True

async def fetch_all():
    """Main fetch function with rate limit management"""
    print("\n" + "="*60)
    print(f"🚀 RUN #{stats['total_runs'] + 1}: {datetime.now(IST).strftime('%I:%M:%S %p')}")
    print("="*60)
    
    stats["total_runs"] += 1
    stats["indices_success"] = 0
    stats["stocks_success"] = 0
    
    # Process INDICES
    print("\n📊 PROCESSING INDICES...")
    for idx_key, idx_info in INDICES.items():
        await process_index(idx_key, idx_info)
        await asyncio.sleep(2)  # Rate limit protection
    
    # Process STOCKS (batch by batch)
    print("\n📈 PROCESSING STOCKS...")
    total_stocks = len(NIFTY50_STOCKS)
    
    for idx, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        await process_stock(key, symbol, idx, total_stocks)
        
        # Longer delay every 10 stocks
        if idx % 10 == 0:
            print(f"  ⏸️ Cooldown after {idx} stocks...")
            await asyncio.sleep(5)
        else:
            await asyncio.sleep(1.5)
    
    # Check for daily summary
    await check_and_send_summary()
    
    # Summary
    total_success = stats["indices_success"] + stats["stocks_success"]
    total_items = len(INDICES) + len(NIFTY50_STOCKS)
    
    summary = f"✅ *RUN #{stats['total_runs']} COMPLETE*\n"
    summary += f"📊 Indices: {stats['indices_success']}/{len(INDICES)}\n"
    summary += f"📈 Stocks: {stats['stocks_success']}/{len(NIFTY50_STOCKS)}\n"
    summary += f"📡 API Calls: {stats['total_api_calls']}\n"
    summary += f"⏰ Next: {(datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')}"
    
    await send_telegram_text(summary)
    
    print(f"\n✅ DONE: {total_success}/{total_items} | API: {stats['total_api_calls']}")

async def monitoring_loop():
    """Main monitoring loop"""
    print("\n🔄 MONITORING STARTED (5 min interval)\n")
    print(f"📊 Tracking: {len(INDICES)} Indices + {len(NIFTY50_STOCKS)} Stocks")
    print(f"⚠️ Rate Limit Protection: ON")
    print(f"🕒 Daily Summary: 3:30 PM\n")
    
    while True:
        try:
            # Check market hours (9:15 AM - 3:30 PM)
            now = datetime.now(IST)
            hour, minute = now.hour, now.minute
            
            # Run only during market hours
            if (hour > 9 or (hour == 9 and minute >= 15)) and \
               (hour < 15 or (hour == 15 and minute <= 30)):
                
                await fetch_all()
                
                # Wait 5 minutes
                next_time = (datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')
                print(f"\n⏳ Next run: {next_time}")
                await asyncio.sleep(300)
            
            else:
                # Outside market hours
                print(f"\n🌙 Market closed. Waiting...")
                
                # Send daily summary if not sent
                if hour >= 15 and hour < 16:
                    await check_and_send_summary()
                
                # Wait 30 minutes when market closed
                await asyncio.sleep(1800)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            await asyncio.sleep(60)

async def main():
    """Entry point"""
    print("\n" + "="*70)
    print("COMPLETE MARKET MONITOR - ALL INDICES + ALL STOCKS")
    print("="*70)
    print(f"📊 INDICES ({len(INDICES)}):")
    for idx_key, idx_info in INDICES.items():
        print(f"   • {idx_info['name']} ({idx_info['type']})")
    
    print(f"\n📈 STOCKS ({len(NIFTY50_STOCKS)}):")
    print(f"   • All NIFTY 50 constituents + POONAWALLA")
    
    print("\n" + "="*70)
    print("✨ FEATURES:")
    print("   • 5-minute interval updates")
    print("   • Historical 30min → 5min split")
    print("   • Today's 1min → 5min aggregation")
    print("   • Compact option chain messages")
    print("   • Charts (reduced frequency)")
    print("   • 3:30 PM daily summary")
    print("   • Rate limit protection")
    print("   • Market hours detection")
    print("="*70)
    
    # Rate limit info
    print("\n⚠️ RATE LIMIT MANAGEMENT:")
    print(f"   • API calls per run: ~{len(INDICES) * 4 + len(NIFTY50_STOCKS) * 4}")
    print(f"   • Daily API calls: ~{(len(INDICES) + len(NIFTY50_STOCKS)) * 4 * 72}")
    print(f"   • Delays: 2s (indices), 1.5s (stocks), 5s (every 10 stocks)")
    print(f"   • Charts: Reduced frequency to save bandwidth")
    print("="*70 + "\n")
    
    # Ask user confirmation
    print("⚠️ IMPORTANT:")
    print("   • This will send MANY Telegram messages")
    print(f"   • ~{len(INDICES) + len(NIFTY50_STOCKS)} messages per 5-min cycle")
    print("   • ~10,000+ API calls per day")
    print("   • Check your Upstox API limits!")
    print("\n🟢 Starting in 5 seconds...\n")
    
    await asyncio.sleep(5)
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main()) send_telegram_text(msg):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
        return True
    except Exception as e:
        print(f"TG error: {e}")
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

async def process_index(index_key, index_info):
    """Process single index"""
    name = index_info["name"]
    print(f"\n📊 {name}")
    
    try:
        expiry = get_next_expiry(index_key, expiry_day=index_info["expiry_day"])
        spot = get_spot_price(index_key)
        
        if spot == 0:
            return False
        
        strikes = get_option_chain(index_key, expiry)
        if not strikes:
            return False
        
        # Compact message
        msg = format_option_chain_message(name, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
        
        # Chart (every other run to reduce load)
        if stats["total_runs"] % 2 == 0:
            candles, hist_count = get_live_candles(index_key, name)
            if candles and len(candles) >= 10:
                chart = create_premium_chart(candles, name, spot, hist_count)
                if chart:
                    caption = f"📈 *{name}*\n₹{spot:.2f}"
                    await send_telegram_photo(chart, caption)
        
        stats["indices_success"] += 1
        return True
        
    except Exception as e:
        print(f"  ❌ {e}")
        return False

async def process_stock(key, symbol, idx, total):
    """Process single stock"""
    print(f"  [{idx}/{total}] {symbol}")
    
    try:
        expiry = get_next_expiry(key, expiry_day=3)
        spot = get_spot_price(key)
        
        if spot == 0:
            return False
        
        strikes = get_option_chain(key, expiry)
        if not strikes:
            return False
        
        # Compact message only
        msg = format_option_chain_message(symbol, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
        
        # Chart only every 3rd run (reduce load)
        if idx % 3 == 0:
            candles, _ = get_live_candles(key, symbol)
            if candles and len(candles) >= 10:
                chart = create_premium_chart(candles, symbol, spot, 0)
                if chart:
                    await send_telegram_photo(chart, f"📈 {symbol}")
        
        stats["stocks_success"] += 1
        return True
        
    except Exception as e:
        return False

async def
