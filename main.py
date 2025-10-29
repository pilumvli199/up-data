#!/usr/bin/env python3
"""
NIFTY 50 + SENSEX + STOCKS MONITOR - ENHANCED VERSION
- Clear historical + live data separation
- Full option chain: Volume, OI, OI Changes, Greeks
- Premium TradingView-style charts
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
print("🚀 NIFTY + SENSEX LIVE MONITOR - ENHANCED")
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
    """Get next expiry (1=Tuesday, 3=Thursday)"""
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
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    historical_5min = []
    today_5min = []
    
    # STEP 1: Historical 30min data (last 10 days)
    print(f"  🔍 Fetching historical 30min data...")
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
                    print(f"  ✅ Historical: {len(hist_candles_30min)} x 30min candles")
                    
                    today_date = datetime.now(IST).date()
                    for c in hist_candles_30min:
                        try:
                            c_dt = datetime.fromisoformat(c[0].replace('Z', '+00:00')).astimezone(IST)
                            if c_dt.date() < today_date:
                                split_candles = split_30min_to_5min(c)
                                historical_5min.extend(split_candles)
                        except:
                            pass
                    
                    print(f"  📊 Historical 5min candles: {len(historical_5min)}")
    except Exception as e:
        print(f"  ⚠️ Historical error: {e}")
    
    # STEP 2: Today's LIVE 1min data
    print(f"  🔍 Fetching TODAY'S LIVE 1min data...")
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                today_candles_1min = data.get('data', {}).get('candles', [])
                if today_candles_1min:
                    print(f"  ✅ TODAY LIVE: {len(today_candles_1min)} x 1min candles")
                    
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
                    
                    print(f"  ✅ Today's 5min candles: {len(today_5min)}")
    except Exception as e:
        print(f"  ⚠️ Today error: {e}")
    
    # STEP 3: Combine
    all_candles = historical_5min + today_5min
    
    if all_candles:
        all_candles = sorted(all_candles,
                            key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
        print(f"  ✅ TOTAL: {len(all_candles)} x 5min (Hist: {len(historical_5min)} + Today: {len(today_5min)})")
        return all_candles, len(historical_5min)
    
    print(f"  ❌ {symbol}: No data")
    return [], 0

def create_premium_chart(candles, symbol, spot_price, hist_count):
    """Create enhanced chart with historical/live distinction"""
    if not candles or len(candles) < 10:
        print(f"  ⚠️ Insufficient candles: {len(candles) if candles else 0}")
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
        print(f"  ⚠️ After filtering: {len(data)} candles")
        return None
    
    # Create figure - WIDER for more horizontal space
    fig, axes = plt.subplots(2, 1, figsize=(28, 13),
                             gridspec_kw={'height_ratios': [4, 1]},
                             facecolor='#0e1217')
    
    ax1, ax2 = axes
    ax1.set_facecolor('#0e1217')
    ax2.set_facecolor('#0e1217')
    
    # Calculate historical cutoff
    today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Plot candlesticks with historical/live distinction
    for idx in range(len(data)):
        row = data[idx]
        x = idx
        
        is_bullish = row['close'] >= row['open']
        is_today = row['timestamp'] >= today_start
        
        # Different opacity for historical vs today
        alpha = 1.0 if is_today else 0.6
        body_color = '#26a69a' if is_bullish else '#ef5350'
        
        # Wick
        ax1.plot([x, x], [row['low'], row['high']],
                color=body_color, linewidth=1.3, solid_capstyle='round',
                alpha=alpha, zorder=2)
        
        # Body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        if body_height > 0.001:
            rect = Rectangle((x - 0.35, body_bottom), 0.7, body_height,
                           facecolor=body_color, edgecolor=body_color,
                           linewidth=0, alpha=alpha, zorder=3)
            ax1.add_patch(rect)
        else:
            ax1.plot([x - 0.35, x + 0.35], [row['open'], row['open']],
                    color=body_color, linewidth=1.5, alpha=alpha, zorder=3)
    
    # Mark today's start with vertical line
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
        
        # Add "TODAY" label
        y_pos = ax1.get_ylim()[1] * 0.98
        ax1.text(today_idx, y_pos, ' TODAY ', 
                color='#ffa726', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#0e1217',
                         edgecolor='#ffa726', linewidth=1.5),
                verticalalignment='top', zorder=5)
    
    # Current price line
    ax1.axhline(y=spot_price, color='#2962ff', linestyle='--',
               linewidth=2.5, alpha=0.9, zorder=4)
    
    # Price label
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
    
    # Volume bars
    volumes = [d['volume'] for d in data]
    colors_vol = []
    for i in range(len(data)):
        is_bull = data[i]['close'] >= data[i]['open']
        is_today = data[i]['timestamp'] >= today_start
        color = '#26a69a' if is_bull else '#ef5350'
        alpha_vol = 0.9 if is_today else 0.5
        colors_vol.append((matplotlib.colors.to_rgba(color, alpha=alpha_vol)))
    
    ax2.bar(range(len(volumes)), volumes, color=colors_vol,
           width=0.7, edgecolor='none')
    
    ax2.set_ylabel('Volume', color='#b2b5be', fontsize=13, fontweight='600')
    ax2.tick_params(axis='y', colors='#787b86', labelsize=11, width=0)
    ax2.tick_params(axis='x', colors='#787b86', labelsize=11, width=0)
    ax2.grid(True, alpha=0.12, color='#363a45', linestyle='-', linewidth=0.8)
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
                  fontsize=13, fontweight='600', labelpad=12)
    
    plt.tight_layout(pad=2)
    plt.subplots_adjust(hspace=0.08)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor='#0e1217',
               edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def format_option_chain_message(symbol, spot, expiry, strikes):
    """Format ENHANCED option chain with Volume, OI, OI Changes, Greeks"""
    if not strikes:
        return None
    
    atm_index = min(range(len(strikes)),
                   key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    start = max(0, atm_index - 8)
    end = min(len(strikes), atm_index + 9)
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol} - OPTION CHAIN*\n\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.2f}\n\n"
    
    # Part 1: LTP & Volume
    msg += "```\n"
    msg += "═══ CALLS ═══════════════════ PUTS ═══\n"
    msg += "Vol   LTP  Strike  LTP   Vol\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_vol = total_pe_vol = 0
    total_ce_oi = total_pe_oi = 0
    
    for s in selected:
        strike_price = s.get('strike_price', 0)
        
        call_data = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call_data.get('ltp', 0)
        ce_vol = call_data.get('volume', 0)
        ce_oi = call_data.get('oi', 0)
        
        put_data = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put_data.get('ltp', 0)
        pe_vol = put_data.get('volume', 0)
        pe_oi = put_data.get('oi', 0)
        
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        
        # Format with K/L suffix
        ce_vol_str = f"{ce_vol/1000:.0f}K" if ce_vol >= 1000 else f"{ce_vol:.0f}"
        pe_vol_str = f"{pe_vol/1000:.0f}K" if pe_vol >= 1000 else f"{pe_vol:.0f}"
        
        is_atm = (strike_price == strikes[atm_index].get('strike_price', 0))
        marker = "►" if is_atm else " "
        
        msg += f"{ce_vol_str:>5} {ce_ltp:6.1f} {marker}{strike_price:6.0f} {pe_ltp:6.1f} {pe_vol_str:>5}\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"TOTAL VOL: {total_ce_vol/1000:.0f}K        {total_pe_vol/1000:.0f}K\n"
    msg += "```\n\n"
    
    # Part 2: OI & OI Change
    msg += "```\n"
    msg += "═══ OPEN INTEREST & CHANGES ═══\n"
    msg += "CE-OI ΔOI Strike ΔOI  PE-OI\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for s in selected:
        strike_price = s.get('strike_price', 0)
        
        call_data = s.get('call_options', {}).get('market_data', {})
        ce_oi = call_data.get('oi', 0)
        ce_oi_change = call_data.get('oi_day_high', 0) - call_data.get('oi_day_low', 0)
        
        put_data = s.get('put_options', {}).get('market_data', {})
        pe_oi = put_data.get('oi', 0)
        pe_oi_change = put_data.get('oi_day_high', 0) - put_data.get('oi_day_low', 0)
        
        ce_oi_str = f"{ce_oi/1000:.0f}K"
        pe_oi_str = f"{pe_oi/1000:.0f}K"
        ce_chg_str = f"{ce_oi_change/1000:+.0f}K" if abs(ce_oi_change) >= 1000 else f"{ce_oi_change:+.0f}"
        pe_chg_str = f"{pe_oi_change/1000:+.0f}K" if abs(pe_oi_change) >= 1000 else f"{pe_oi_change:+.0f}"
        
        is_atm = (strike_price == strikes[atm_index].get('strike_price', 0))
        marker = "►" if is_atm else " "
        
        msg += f"{ce_oi_str:>5} {ce_chg_str:>5} {marker}{strike_price:6.0f} {pe_chg_str:>5} {pe_oi_str:>5}\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"TOTAL OI: {total_ce_oi/1000:.0f}K        {total_pe_oi/1000:.0f}K\n"
    msg += "```\n\n"
    
    # Part 3: Greeks (ATM Strike)
    atm_strike = strikes[atm_index]
    call_greeks = atm_strike.get('call_options', {}).get('greeks', {})
    put_greeks = atm_strike.get('put_options', {}).get('greeks', {})
    
    if call_greeks or put_greeks:
        msg += "```\n"
        msg += f"═══ GREEKS (ATM: {atm_strike.get('strike_price', 0):.0f}) ═══\n"
        msg += "         CALL    PUT\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        ce_delta = call_greeks.get('delta', 0)
        pe_delta = put_greeks.get('delta', 0)
        msg += f"Delta:  {ce_delta:6.3f} {pe_delta:7.3f}\n"
        
        ce_gamma = call_greeks.get('gamma', 0)
        pe_gamma = put_greeks.get('gamma', 0)
        msg += f"Gamma:  {ce_gamma:6.4f} {pe_gamma:7.4f}\n"
        
        ce_theta = call_greeks.get('theta', 0)
        pe_theta = put_greeks.get('theta', 0)
        msg += f"Theta:  {ce_theta:6.2f} {pe_theta:7.2f}\n"
        
        ce_vega = call_greeks.get('vega', 0)
        pe_vega = put_greeks.get('vega', 0)
        msg += f"Vega:   {ce_vega:6.2f} {pe_vega:7.2f}\n"
        
        ce_iv = call_greeks.get('iv', 0)
        pe_iv = put_greeks.get('iv', 0)
        msg += f"IV:     {ce_iv:6.1f}% {pe_iv:6.1f}%\n"
        msg += "```\n\n"
    
    # Summary
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    msg += f"📊 *PCR (OI):* {pcr:.3f}\n"
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
        
        # Send ENHANCED option chain
        msg = format_option_chain_message(index_name, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
            print("📤 Enhanced option chain sent (Vol, OI, Greeks)")
        
        # Send LIVE chart with historical distinction
        print("📊 Fetching candles (Historical 30min + Today 1min)...")
        candles, hist_count = get_live_candles(index_key, index_name)
        
        if candles and len(candles) >= 10:
            print("📈 Creating enhanced chart...")
            chart = create_premium_chart(candles, index_name, spot, hist_count)
            
            if chart:
                caption = f"📈 *{index_name}* ({expiry_type})\n💰 ₹{spot:.2f} | 📅 {expiry}\n🔸 Historical + Today's LIVE data"
                await send_telegram_photo(chart, caption)
                print("📤 Chart sent (Historical + LIVE)!")
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
        
        # Send ENHANCED option chain
        msg = format_option_chain_message(symbol, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
            print("  📤 Enhanced chain sent")
        
        # Send LIVE chart
        candles, hist_count = get_live_candles(key, symbol)
        if candles and len(candles) >= 10:
            chart = create_premium_chart(candles, symbol, spot, hist_count)
            if chart:
                caption = f"📈 *{symbol}* (Monthly)\n💰 ₹{spot:.2f}\n🔸 Hist + Today"
                await send_telegram_photo(chart, caption)
                print("  📤 Chart sent")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def fetch_all():
    """Main fetch function"""
    print("\n" + "="*60)
    print(f"🚀 RUN: {datetime.now(IST).strftime('%I:%M:%S %p IST')}")
    print("="*60)
    
    header = f"🚀 *MARKET MONITOR - ENHANCED*\n⏰ {datetime.now(IST).strftime('%I:%M %p')}\n\n_Processing with Vol, OI, Greeks..._"
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
    summary += f"Stocks: {success}/{total}\n\n"
    summary += f"📊 Enhanced with Volume, OI, Greeks"
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
    print("NIFTY + SENSEX + STOCKS MONITOR - ENHANCED VERSION")
    print("="*70)
    print("📊 NIFTY: Tuesday (Weekly)")
    print("📊 SENSEX: Thursday (Weekly)")
    print("📈 Stocks: Thursday (Monthly)")
    print("="*70)
    print("✨ ENHANCEMENTS:")
    print("  • Clear historical vs live data separation")
    print("  • Historical 30min → 5min split")
    print("  • Today's 1min → 5min aggregation")
    print("  • Option Volume + OI + OI Changes")
    print("  • Greeks (Delta, Gamma, Theta, Vega, IV)")
    print("  • Premium TradingView-style charts")
    print("  • Updates every 5 minutes")
    print("="*70 + "\n")
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
