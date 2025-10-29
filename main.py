#!/usr/bin/env python3
"""
NIFTY 50 + SENSEX + STOCKS MONITOR - VISUAL OPTION CHAIN
- Wide charts with clear historical data
- Visual option chain images (Upstox-style format)
- Full Greeks, Volume, OI data
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
print("🚀 NIFTY + SENSEX LIVE MONITOR - VISUAL OPTION CHAIN")
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
    
    # Historical 30min data
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
                    print(f"  ✅ Historical: {len(hist_candles_30min)} x 30min")
                    
                    today_date = datetime.now(IST).date()
                    for c in hist_candles_30min:
                        try:
                            c_dt = datetime.fromisoformat(c[0].replace('Z', '+00:00')).astimezone(IST)
                            if c_dt.date() < today_date:
                                split_candles = split_30min_to_5min(c)
                                historical_5min.extend(split_candles)
                        except:
                            pass
                    
                    print(f"  📊 Historical 5min: {len(historical_5min)}")
    except Exception as e:
        print(f"  ⚠️ Historical error: {e}")
    
    # Today's LIVE 1min data
    print(f"  🔍 Fetching TODAY'S LIVE 1min...")
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        resp = requests.get(url, headers=headers, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                today_candles_1min = data.get('data', {}).get('candles', [])
                if today_candles_1min:
                    print(f"  ✅ TODAY: {len(today_candles_1min)} x 1min")
                    
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
                    
                    print(f"  ✅ Today 5min: {len(today_5min)}")
    except Exception as e:
        print(f"  ⚠️ Today error: {e}")
    
    # Combine
    all_candles = historical_5min + today_5min
    
    if all_candles:
        all_candles = sorted(all_candles,
                            key=lambda x: datetime.fromisoformat(x[0].replace('Z', '+00:00')))
        print(f"  ✅ TOTAL: {len(all_candles)} x 5min")
        return all_candles, len(historical_5min)
    
    return [], 0

def create_wide_chart(candles, symbol, spot_price, hist_count):
    """Create WIDER chart with clearer historical data"""
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
    
    # WIDER figure - 28 inches wide!
    fig, axes = plt.subplots(2, 1, figsize=(28, 13),
                             gridspec_kw={'height_ratios': [4, 1]},
                             facecolor='#0e1217')
    
    ax1, ax2 = axes
    ax1.set_facecolor('#0e1217')
    ax2.set_facecolor('#0e1217')
    
    today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Plot candlesticks - BRIGHTER historical!
    for idx in range(len(data)):
        row = data[idx]
        x = idx
        
        is_bullish = row['close'] >= row['open']
        is_today = row['timestamp'] >= today_start
        
        # INCREASED alpha for historical: 0.85 (was 0.6)
        alpha = 1.0 if is_today else 0.85
        body_color = '#26a69a' if is_bullish else '#ef5350'
        
        # Thicker wicks
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
                    color=body_color, linewidth=1.8, alpha=alpha, zorder=3)
    
    # TODAY marker
    today_idx = None
    for i, d in enumerate(data):
        if d['timestamp'] >= today_start:
            today_idx = i
            break
    
    if today_idx:
        ax1.axvline(x=today_idx, color='#ffa726', linestyle='--',
                   linewidth=2.5, alpha=0.7, zorder=1)
        ax2.axvline(x=today_idx, color='#ffa726', linestyle='--',
                   linewidth=2.5, alpha=0.7, zorder=1)
        
        y_pos = ax1.get_ylim()[1] * 0.97
        ax1.text(today_idx, y_pos, ' TODAY ', 
                color='#ffa726', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#0e1217',
                         edgecolor='#ffa726', linewidth=2),
                verticalalignment='top', zorder=5)
    
    # Current price
    ax1.axhline(y=spot_price, color='#2962ff', linestyle='--',
               linewidth=2.5, alpha=0.95, zorder=4)
    
    # Price label
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_yticks([spot_price])
    ax1_right.set_yticklabels([f'₹{spot_price:.2f}'],
                              fontsize=14, fontweight='700', color='#2962ff',
                              bbox=dict(boxstyle='round,pad=0.7',
                                      facecolor='#2962ff', alpha=0.35))
    ax1_right.tick_params(colors='#2962ff', length=0, pad=12)
    ax1_right.set_facecolor('#0e1217')
    
    # Styling
    ax1.set_ylabel('Price (₹)', color='#b2b5be', fontsize=14, fontweight='600')
    ax1.tick_params(axis='y', colors='#8a8d96', labelsize=11, width=0)
    ax1.tick_params(axis='x', colors='#8a8d96', labelsize=11, width=0)
    ax1.grid(True, alpha=0.15, color='#363a45', linestyle='-', linewidth=0.9)
    ax1.set_axisbelow(True)
    
    # Title
    now_str = datetime.now(IST).strftime('%d %b %Y • %I:%M:%S %p IST')
    title = f'{symbol}  •  5 Min Chart (LIVE)  •  {now_str}'
    ax1.set_title(title, color='#e1e4ec', fontsize=18, fontweight='700',
                 pad=28, loc='left')
    
    # Volume bars - BRIGHTER historical
    volumes = [d['volume'] for d in data]
    colors_vol = []
    for i in range(len(data)):
        is_bull = data[i]['close'] >= data[i]['open']
        is_today = data[i]['timestamp'] >= today_start
        color = '#26a69a' if is_bull else '#ef5350'
        alpha_vol = 0.95 if is_today else 0.7  # Increased from 0.5
        colors_vol.append((matplotlib.colors.to_rgba(color, alpha=alpha_vol)))
    
    ax2.bar(range(len(volumes)), volumes, color=colors_vol,
           width=0.8, edgecolor='none')
    
    ax2.set_ylabel('Volume', color='#b2b5be', fontsize=14, fontweight='600')
    ax2.tick_params(axis='y', colors='#8a8d96', labelsize=11, width=0)
    ax2.tick_params(axis='x', colors='#8a8d96', labelsize=11, width=0)
    ax2.grid(True, alpha=0.15, color='#363a45', linestyle='-', linewidth=0.9)
    ax2.set_axisbelow(True)
    
    # X-axis labels
    step = max(1, len(data) // 15)
    tick_positions = list(range(0, len(data), step))
    tick_labels = [data[i]['timestamp'].strftime('%d %b\n%H:%M') for i in tick_positions]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color='#8a8d96', fontsize=10)
        ax.set_xlim(-1, len(data))
        
        for spine in ax.spines.values():
            spine.set_color('#1e222d')
            spine.set_linewidth(1.5)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax2.set_xlabel('Date & Time (IST)', color='#b2b5be',
                  fontsize=14, fontweight='600', labelpad=14)
    
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(hspace=0.1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=170, facecolor='#0e1217',
               edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def create_option_chain_image(symbol, spot, expiry, strikes):
    """Create VISUAL option chain image - Upstox style"""
    if not strikes:
        return None
    
    atm_index = min(range(len(strikes)),
                   key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    start = max(0, atm_index - 10)
    end = min(len(strikes), atm_index + 11)
    selected = strikes[start:end]
    
    # Create figure - WIDE format
    fig = plt.figure(figsize=(20, 14), facecolor='#0a0e12')
    
    # Main title
    fig.suptitle(f'{symbol} - OPTION CHAIN', 
                fontsize=22, fontweight='bold', color='#e8eaf0', y=0.98)
    
    # Spot info
    spot_text = f'Spot: ₹{spot:,.2f}  |  Expiry: {expiry}  |  ATM: ₹{strikes[atm_index].get("strike_price", 0):,.2f}'
    fig.text(0.5, 0.94, spot_text, ha='center', fontsize=14, color='#b0b3ba')
    
    # Create 3 subplots
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1], hspace=0.35,
                         left=0.05, right=0.95, top=0.90, bottom=0.05)
    
    ax1 = fig.add_subplot(gs[0])  # Price & Volume
    ax2 = fig.add_subplot(gs[1])  # OI & Changes
    ax3 = fig.add_subplot(gs[2])  # Greeks
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0a0e12')
        ax.axis('off')
    
    # === PANEL 1: PRICE & VOLUME ===
    ax1.text(0.5, 0.95, 'PRICE & VOLUME', ha='center', fontsize=16,
            fontweight='bold', color='#ffa726', transform=ax1.transAxes)
    
    # Headers
    header_y = 0.85
    ax1.text(0.08, header_y, 'CE Vol', ha='center', fontsize=11, 
            fontweight='bold', color='#26a69a', transform=ax1.transAxes)
    ax1.text(0.20, header_y, 'CE LTP', ha='center', fontsize=11,
            fontweight='bold', color='#26a69a', transform=ax1.transAxes)
    ax1.text(0.50, header_y, 'STRIKE', ha='center', fontsize=12,
            fontweight='bold', color='#ffa726', transform=ax1.transAxes)
    ax1.text(0.80, header_y, 'PE LTP', ha='center', fontsize=11,
            fontweight='bold', color='#ef5350', transform=ax1.transAxes)
    ax1.text(0.92, header_y, 'PE Vol', ha='center', fontsize=11,
            fontweight='bold', color='#ef5350', transform=ax1.transAxes)
    
    # Data rows
    y_start = 0.75
    y_step = 0.75 / (len(selected) + 2)
    
    total_ce_vol = total_pe_vol = 0
    total_ce_oi = total_pe_oi = 0
    
    for i, s in enumerate(selected):
        y_pos = y_start - (i * y_step)
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
        
        is_atm = (strike_price == strikes[atm_index].get('strike_price', 0))
        
        # ATM highlight
        if is_atm:
            rect = Rectangle((0.0, y_pos - y_step/3), 1.0, y_step*1.1,
                           facecolor='#ffa726', alpha=0.15, transform=ax2.transAxes)
            ax2.add_patch(rect)
        
        # CE OI
        ce_oi_str = f"{ce_oi/1000:.0f}K" if ce_oi >= 1000 else f"{ce_oi:.0f}"
        ax2.text(0.08, y_pos, ce_oi_str, ha='center', fontsize=10,
                color='#26a69a', transform=ax2.transAxes, fontweight='600')
        
        # CE OI Change
        ce_chg_str = f"{ce_oi_chg/1000:+.1f}K" if abs(ce_oi_chg) >= 1000 else f"{ce_oi_chg:+.0f}"
        chg_color = '#26a69a' if ce_oi_chg > 0 else '#ef5350' if ce_oi_chg < 0 else '#787b86'
        ax2.text(0.20, y_pos, ce_chg_str, ha='center', fontsize=9,
                color=chg_color, transform=ax2.transAxes, fontweight='600')
        
        # STRIKE
        strike_color = '#ffa726' if is_atm else '#e8eaf0'
        strike_weight = '800' if is_atm else '600'
        ax2.text(0.50, y_pos, f"{strike_price:.0f}", ha='center', fontsize=11,
                color=strike_color, transform=ax2.transAxes, fontweight=strike_weight)
        
        # PE OI Change
        pe_chg_str = f"{pe_oi_chg/1000:+.1f}K" if abs(pe_oi_chg) >= 1000 else f"{pe_oi_chg:+.0f}"
        chg_color = '#26a69a' if pe_oi_chg > 0 else '#ef5350' if pe_oi_chg < 0 else '#787b86'
        ax2.text(0.80, y_pos, pe_chg_str, ha='center', fontsize=9,
                color=chg_color, transform=ax2.transAxes, fontweight='600')
        
        # PE OI
        pe_oi_str = f"{pe_oi/1000:.0f}K" if pe_oi >= 1000 else f"{pe_oi:.0f}"
        ax2.text(0.92, y_pos, pe_oi_str, ha='center', fontsize=10,
                color='#ef5350', transform=ax2.transAxes, fontweight='600')
    
    # Total OI
    ax2.text(0.08, total_y, f"{total_ce_oi/1000:.0f}K", ha='center', fontsize=11,
            color='#26a69a', transform=ax2.transAxes, fontweight='bold')
    ax2.text(0.15, total_y, 'TOTAL', ha='center', fontsize=10,
            color='#b0b3ba', transform=ax2.transAxes)
    ax2.text(0.92, total_y, f"{total_pe_oi/1000:.0f}K", ha='center', fontsize=11,
            color='#ef5350', transform=ax2.transAxes, fontweight='bold')
    
    # PCR
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    ax2.text(0.50, total_y, f"PCR: {pcr:.3f}", ha='center', fontsize=12,
            color='#ffa726', transform=ax2.transAxes, fontweight='bold')
    
    # === PANEL 3: GREEKS (ATM Strike) ===
    ax3.text(0.5, 0.95, f'OPTION GREEKS - ATM Strike ({strikes[atm_index].get("strike_price", 0):.0f})',
            ha='center', fontsize=16, fontweight='bold', color='#ffa726', transform=ax3.transAxes)
    
    atm_strike = strikes[atm_index]
    call_greeks = atm_strike.get('call_options', {}).get('greeks', {})
    put_greeks = atm_strike.get('put_options', {}).get('greeks', {})
    
    if call_greeks or put_greeks:
        # Headers
        greek_header_y = 0.75
        ax3.text(0.35, greek_header_y, 'CALL', ha='center', fontsize=13,
                fontweight='bold', color='#26a69a', transform=ax3.transAxes)
        ax3.text(0.50, greek_header_y, 'GREEK', ha='center', fontsize=13,
                fontweight='bold', color='#ffa726', transform=ax3.transAxes)
        ax3.text(0.65, greek_header_y, 'PUT', ha='center', fontsize=13,
                fontweight='bold', color='#ef5350', transform=ax3.transAxes)
        
        greeks_data = [
            ('Delta', call_greeks.get('delta', 0), put_greeks.get('delta', 0), '.4f'),
            ('Gamma', call_greeks.get('gamma', 0), put_greeks.get('gamma', 0), '.5f'),
            ('Theta', call_greeks.get('theta', 0), put_greeks.get('theta', 0), '.2f'),
            ('Vega', call_greeks.get('vega', 0), put_greeks.get('vega', 0), '.2f'),
            ('IV', call_greeks.get('iv', 0), put_greeks.get('iv', 0), '.2f')
        ]
        
        y_greek = 0.60
        y_greek_step = 0.12
        
        for name, ce_val, pe_val, fmt in greeks_data:
            # Greek name
            ax3.text(0.50, y_greek, name, ha='center', fontsize=12,
                    color='#e8eaf0', transform=ax3.transAxes, fontweight='600')
            
            # CE value
            if name == 'IV':
                ce_str = f"{ce_val:{fmt}}%"
            else:
                ce_str = f"{ce_val:{fmt}}"
            ax3.text(0.35, y_greek, ce_str, ha='center', fontsize=11,
                    color='#26a69a', transform=ax3.transAxes, fontweight='700')
            
            # PE value
            if name == 'IV':
                pe_str = f"{pe_val:{fmt}}%"
            else:
                pe_str = f"{pe_val:{fmt}}"
            ax3.text(0.65, y_greek, pe_str, ha='center', fontsize=11,
                    color='#ef5350', transform=ax3.transAxes, fontweight='700')
            
            y_greek -= y_greek_step
    else:
        ax3.text(0.5, 0.5, 'Greeks data not available', ha='center', fontsize=12,
                color='#787b86', transform=ax3.transAxes, style='italic')
    
    # Timestamp
    now_str = datetime.now(IST).strftime('%I:%M:%S %p IST • %d %b %Y')
    fig.text(0.5, 0.015, now_str, ha='center', fontsize=11, color='#787b86')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='#0a0e12',
               edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

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
        
        # Send VISUAL option chain image
        print("📊 Creating option chain image...")
        oc_image = create_option_chain_image(index_name, spot, expiry, strikes)
        if oc_image:
            caption = f"📊 *{index_name}* Option Chain\n💰 ₹{spot:.2f} | 📅 {expiry}"
            await send_telegram_photo(oc_image, caption)
            print("📤 Option chain image sent")
        
        await asyncio.sleep(2)
        
        # Send WIDE chart
        print("📈 Fetching candles...")
        candles, hist_count = get_live_candles(index_key, index_name)
        
        if candles and len(candles) >= 10:
            print("📈 Creating wide chart...")
            chart = create_wide_chart(candles, index_name, spot, hist_count)
            
            if chart:
                caption = f"📈 *{index_name}* ({expiry_type})\n💰 ₹{spot:.2f} | 📅 {expiry}"
                await send_telegram_photo(chart, caption)
                print("📤 Wide chart sent!")
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
        
        # Send option chain image
        oc_image = create_option_chain_image(symbol, spot, expiry, strikes)
        if oc_image:
            caption = f"📊 *{symbol}* Option Chain\n💰 ₹{spot:.2f}"
            await send_telegram_photo(oc_image, caption)
            print("  📤 Option chain sent")
        
        await asyncio.sleep(2)
        
        # Send chart
        candles, hist_count = get_live_candles(key, symbol)
        if candles and len(candles) >= 10:
            chart = create_wide_chart(candles, symbol, spot, hist_count)
            if chart:
                caption = f"📈 *{symbol}* (Monthly)\n💰 ₹{spot:.2f}"
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
    
    header = f"🚀 *MARKET MONITOR*\n⏰ {datetime.now(IST).strftime('%I:%M %p')}\n\n_Processing with visual option chains..._"
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
    summary += f"📊 Wide charts + Visual option chains"
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
    print("NIFTY + SENSEX + STOCKS - VISUAL OPTION CHAIN MONITOR")
    print("="*70)
    print("📊 NIFTY: Tuesday (Weekly)")
    print("📊 SENSEX: Thursday (Weekly)")
    print("📈 Stocks: Thursday (Monthly)")
    print("="*70)
    print("✨ FEATURES:")
    print("  • WIDE charts (28 inches) - clear historical data")
    print("  • Historical data: 85% opacity (bright & visible)")
    print("  • Visual option chain images (Upstox-style)")
    print("  • 3 panels: Price/Vol, OI/Changes, Greeks")
    print("  • Updates every 5 minutes")
    print("="*70 + "\n")
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
        bg_alpha = 0.3 if is_atm else 0
        
        # ATM highlight
        if is_atm:
            rect = Rectangle((0.0, y_pos - y_step/3), 1.0, y_step*1.1,
                           facecolor='#ffa726', alpha=0.15, transform=ax1.transAxes)
            ax1.add_patch(rect)
        
        # CE Volume
        ce_vol_str = f"{ce_vol/1000:.1f}K" if ce_vol >= 1000 else f"{ce_vol:.0f}"
        ax1.text(0.08, y_pos, ce_vol_str, ha='center', fontsize=10,
                color='#26a69a', transform=ax1.transAxes, fontweight='600')
        
        # CE LTP
        ax1.text(0.20, y_pos, f"{ce_ltp:.2f}", ha='center', fontsize=10,
                color='#26a69a', transform=ax1.transAxes, fontweight='700')
        
        # STRIKE
        strike_color = '#ffa726' if is_atm else '#e8eaf0'
        strike_weight = '800' if is_atm else '600'
        ax1.text(0.50, y_pos, f"{strike_price:.0f}", ha='center', fontsize=11,
                color=strike_color, transform=ax1.transAxes, fontweight=strike_weight)
        
        # PE LTP
        ax1.text(0.80, y_pos, f"{pe_ltp:.2f}", ha='center', fontsize=10,
                color='#ef5350', transform=ax1.transAxes, fontweight='700')
        
        # PE Volume
        pe_vol_str = f"{pe_vol/1000:.1f}K" if pe_vol >= 1000 else f"{pe_vol:.0f}"
        ax1.text(0.92, y_pos, pe_vol_str, ha='center', fontsize=10,
                color='#ef5350', transform=ax1.transAxes, fontweight='600')
    
    # Total Volume
    total_y = y_start - (len(selected) + 1) * y_step
    ax1.text(0.08, total_y, f"{total_ce_vol/1000:.0f}K", ha='center', fontsize=11,
            color='#26a69a', transform=ax1.transAxes, fontweight='bold')
    ax1.text(0.15, total_y, 'TOTAL', ha='center', fontsize=10,
            color='#b0b3ba', transform=ax1.transAxes)
    ax1.text(0.92, total_y, f"{total_pe_vol/1000:.0f}K", ha='center', fontsize=11,
            color='#ef5350', transform=ax1.transAxes, fontweight='bold')
    
    # === PANEL 2: OI & CHANGES ===
    ax2.text(0.5, 0.95, 'OPEN INTEREST & CHANGES', ha='center', fontsize=16,
            fontweight='bold', color='#ffa726', transform=ax2.transAxes)
    
    # Headers
    ax2.text(0.08, header_y, 'CE OI', ha='center', fontsize=11,
            fontweight='bold', color='#26a69a', transform=ax2.transAxes)
    ax2.text(0.20, header_y, 'ΔOI', ha='center', fontsize=11,
            fontweight='bold', color='#26a69a', transform=ax2.transAxes)
    ax2.text(0.50, header_y, 'STRIKE', ha='center', fontsize=12,
            fontweight='bold', color='#ffa726', transform=ax2.transAxes)
    ax2.text(0.80, header_y, 'ΔOI', ha='center', fontsize=11,
            fontweight='bold', color='#ef5350', transform=ax2.transAxes)
    ax2.text(0.92, header_y, 'PE OI', ha='center', fontsize=11,
            fontweight='bold', color='#ef5350', transform=ax2.transAxes)
    
    # Data rows
    for i, s in enumerate(selected):
        y_pos = y_start - (i * y_step)
        strike_price = s.get('strike_price', 0)
        
        call_data = s.get('call_options', {}).get('market_data', {})
        ce_oi = call_data.get('oi', 0)
        ce_oi_chg = call_data.get('oi_day_high', 0) - call_data.get('oi_day_low', 0)
        
        put_data = s.get('put_options', {}).get('market_data', {})
        pe_oi = put_data.get('oi', 0)
        pe_oi_chg = put_data.get('oi_day_high', 0) - put_data.get('oi_day_low', 0)
        
        is_atm = (strike_price == strikes[atm_index].get('strike_price', 0))
