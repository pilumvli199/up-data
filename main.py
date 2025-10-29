#!/usr/bin/env python3
"""
COMPLETE MARKET MONITOR - v6 (Professional Candlestick Charts)
- ALL NIFTY 50 Stocks + POONAWALLA
- NIFTY 50, BANK NIFTY, FIN NIFTY, MIDCAP NIFTY
- FIX: Complete overhaul of chart rendering for professional, clear candlesticks with distinct wicks and bodies.
- Enhanced option chain with OI, Volume, and GREEKS.
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import time as time_sleep
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

# INDICES - ALL 4
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY 50", "expiry_day": 1},
    "NSE_INDEX|Nifty Bank": {"name": "BANK NIFTY", "expiry_day": 2},
    "NSE_INDEX|Nifty Fin Service": {"name": "FIN NIFTY", "expiry_day": 1},
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCAP NIFTY", "expiry_day": 0}
}

# COMPLETE NIFTY 50 STOCKS + POONAWALLA (ALL 51)
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE", "NSE_EQ|INE467B01029": "TATAMOTORS", "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK", "NSE_EQ|INE062A01020": "SBIN", "NSE_EQ|INE009A01021": "INFY", "NSE_EQ|INE854D01024": "TCS",
    "NSE_EQ|INE594E01019": "HINDUNILVR", "NSE_EQ|INE030A01027": "BHARTIARTL", "NSE_EQ|INE238A01034": "AXISBANK",
    "NSE_EQ|INE192A01025": "KOTAKBANK", "NSE_EQ|INE155A01022": "TATASTEEL", "NSE_EQ|INE047A01021": "HCLTECH",
    "NSE_EQ|INE742F01042": "ADANIENT", "NSE_EQ|INE012A01025": "WIPRO", "NSE_EQ|INE018A01030": "LT",
    "NSE_EQ|INE019A01038": "ASIANPAINT", "NSE_EQ|INE205A01025": "MARUTI", "NSE_EQ|INE795G01014": "ADANIPORTS",
    "NSE_EQ|INE001A01036": "ULTRACEMCO", "NSE_EQ|INE021A01026": "M&M", "NSE_EQ|INE245A01021": "SUNPHARMA",
    "NSE_EQ|INE114A01011": "TITAN", "NSE_EQ|INE758T01015": "TECHM", "NSE_EQ|INE522F01014": "COALINDIA",
    "NSE_EQ|INE066F01012": "JSWSTEEL", "NSE_EQ|INE216A01030": "NTPC", "NSE_EQ|INE029A01011": "POWERGRID",
    "NSE_EQ|INE101D01020": "NESTLEIND", "NSE_EQ|INE123W01016": "BAJFINANCE", "NSE_EQ|INE296A01024": "ONGC",
    "NSE_EQ|INE044A01036": "HINDALCO", "NSE_EQ|INE242A01010": "ITC", "NSE_EQ|INE860A01027": "HDFCLIFE",
    "NSE_EQ|INE075A01022": "SBILIFE", "NSE_EQ|INE213A01029": "EICHERMOT", "NSE_EQ|INE129A01019": "GRASIM",
    "NSE_EQ|INE180A01020": "INDUSINDBK", "NSE_EQ|INE481G01011": "BAJAJFINSV", "NSE_EQ|INE217A01012": "HEROMOTOCO",
    "NSE_EQ|INE239A01016": "DIVISLAB", "NSE_EQ|INE009A01011": "CIPLA", "NSE_EQ|INE131A01031": "APOLLOHOSP",
    "NSE_EQ|INE040H01021": "ADANIGREEN", "NSE_EQ|INE032A01023": "BPCL", "NSE_EQ|INE030E01023": "BRITANNIA",
    "NSE_EQ|INE758E01017": "LTIM", "NSE_EQ|INE093I01010": "TRENT", "NSE_EQ|INE752E01010": "SHRIRAMFIN",
    "NSE_EQ|INE196A01026": "BEL", "NSE_EQ|INE511C01022": "POONAWALLA",
}

# Global tracking
DAILY_STATS = {"total_alerts": 0, "indices_count": 0, "stocks_count": 0, "start_time": None}

print("="*70); print("🚀 COMPLETE MARKET MONITOR - v6 (Professional Charts)"); print("="*70)

def get_expiries(instrument_key):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            contracts = resp.json().get('data', [])
            return sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
    except Exception as e:
        print(f"Expiry error: {e}")
    return []

def get_next_expiry(instrument_key, expiry_day=1):
    expiries = get_expiries(instrument_key)
    if not expiries:
        today = datetime.now(IST)
        days_ahead = expiry_day - today.weekday()
        if days_ahead <= 0: days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    today = datetime.now(IST).date()
    future = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
    return min(future) if future else expiries[0]

def get_option_chain(instrument_key, expiry):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            strikes = resp.json().get('data', [])
            return sorted(strikes, key=lambda x: x.get('strike_price', 0))
    except Exception as e:
        print(f"Chain error: {e}")
    return []

def get_spot_price(instrument_key):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                quote_data = resp.json().get('data', {})
                if quote_data:
                    ltp = quote_data[list(quote_data.keys())[0]].get('last_price', 0)
                    if ltp: return float(ltp)
            print(f"  ⚠️ Spot price attempt {attempt + 1} failed for {instrument_key.split('|')[1]}. Retrying...")
            time_sleep.sleep(3)
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Spot price network error: {e}. Retrying...")
            time_sleep.sleep(3)
    print(f"  ❌ Failed to get spot price for {instrument_key.split('|')[1]} after 3 attempts.")
    return 0

def split_30min_to_5min(candle_30min):
    try:
        ts_str, o, h, l, c, v, oi = candle_30min
        dt_start = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(IST)
        
        candles_5min = []
        for i in range(6):
            c_time = dt_start + timedelta(minutes=i * 5)
            # This is a visual approximation, not for technical analysis
            c_open = o + (c - o) * (i / 6)
            c_close = o + (c - o) * ((i + 1) / 6)
            c_high = max(h, c_open, c_close)
            c_low = min(l, c_open, c_close)
            
            candles_5min.append([c_time.isoformat(), c_open, c_high, c_low, c_close, v / 6, oi])
        return candles_5min
    except Exception:
        return []

def get_live_candles(instrument_key, symbol):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    historical_5min_candles = []
    try:
        to_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')
        from_date = (datetime.now(IST) - timedelta(days=15)).strftime('%Y-%m-%d')
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_date}/{from_date}"
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200 and resp.json().get('status') == 'success':
            for candle in resp.json().get('data', {}).get('candles', []):
                historical_5min_candles.extend(split_30min_to_5min(candle))
    except Exception as e:
        print(f"  ⚠️ Historical candle error: {e}")

    intraday_5min_candles = []
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200 and resp.json().get('status') == 'success':
            candles_1min = resp.json().get('data', {}).get('candles', [])
            if candles_1min:
                df = pd.DataFrame(candles_1min, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
                df['ts'] = pd.to_datetime(df['ts'])
                df = df.set_index('ts').astype(float)
                df_resampled = df.resample('5min').agg({'o':'first', 'h':'max', 'l':'min', 'c':'last', 'v':'sum', 'oi':'last'}).dropna()
                intraday_5min_candles = [[idx.isoformat(), r['o'], r['h'], r['l'], r['c'], r['v'], r['oi']] for idx, r in df_resampled.iterrows()]
    except Exception as e:
        print(f"  ⚠️ Intraday candle error: {e}")
        
    all_candles = sorted(historical_5min_candles + intraday_5min_candles, key=lambda x: x[0])
    
    today = datetime.now(IST).date()
    hist_count = len([c for c in all_candles if datetime.fromisoformat(c[0]).astimezone(IST).date() < today])

    return all_candles, hist_count


def create_premium_chart(candles, symbol, spot_price, hist_count):
    if not candles or len(candles) < 2: return None
    
    data = []
    for c in candles:
        try:
            ts = datetime.fromisoformat(c[0].replace("Z", "+00:00")).astimezone(IST)
            if time(9, 15) <= ts.time() <= time(15, 30):
                data.append({'ts': ts, 'o': float(c[1]), 'h': float(c[2]), 'l': float(c[3]), 'c': float(c[4]), 'v': int(c[5])})
        except (ValueError, TypeError): continue
    
    if not data: return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 13), gridspec_kw={'height_ratios': [4, 1]}, facecolor='#0e1217')
    for ax in [ax1, ax2]: ax.set_facecolor('#0e1217')
    
    # --- PROFESSIONAL CANDLESTICK RENDERING LOGIC ---
    candle_body_width = 0.6
    
    for i, row in enumerate(data):
        is_today = i >= hist_count
        alpha = 1.0 if is_today else 0.8
        color = '#26a69a' if row['c'] >= row['o'] else '#ef5350'
        
        # 1. Draw the Wick (thin line from High to Low)
        ax1.plot([i, i], [row['l'], row['h']], color=color, linewidth=1.2, alpha=alpha, zorder=1)
        
        # 2. Draw the Body (thicker rectangle from Open to Close)
        body_height = abs(row['c'] - row['o'])
        body_bottom = min(row['o'], row['c'])
        rect = Rectangle((i - candle_body_width/2, body_bottom), candle_body_width, body_height, facecolor=color, alpha=alpha, zorder=2)
        ax1.add_patch(rect)

        # 3. Volume Bar
        ax2.bar(i, row['v'], width=0.7, color=color, alpha=alpha)

    # --- END OF RENDERING LOGIC ---

    if hist_count > 0 and hist_count < len(data):
        ax1.axvline(x=hist_count - 0.5, color='#ffa726', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=hist_count - 0.5, color='#ffa726', linestyle='--', linewidth=1.5, alpha=0.7)

    ax1.axhline(y=spot_price, color='#2962ff', linestyle='--', linewidth=2.5, alpha=0.9)
    ax1_right = ax1.twinx(); ax1_right.set_ylim(ax1.get_ylim()); ax1_right.set_yticks([spot_price])
    ax1_right.set_yticklabels([f'₹{spot_price:.2f}'], fontsize=13, fontweight='700', color='#2962ff', bbox=dict(facecolor='#2962ff', alpha=0.3))
    
    ax1.set_title(f'{symbol} • 5 Min Chart (LIVE) • {datetime.now(IST).strftime("%d %b %Y • %I:%M:%S %p IST")}', color='#d1d4dc', fontsize=17, fontweight='700', loc='left')
    ax1.set_ylabel('Price (₹)', color='#b2b5be'); ax2.set_ylabel('Volume', color='#b2b5be')
    
    tick_positions = []
    tick_labels = []
    last_date_label = None
    for i, row in enumerate(data):
        current_date = row['ts'].date()
        if current_date != last_date_label:
            tick_positions.append(i)
            tick_labels.append(row['ts'].strftime('%d %b'))
            last_date_label = current_date
    
    # Simplify labels if there are too many days shown
    if len(tick_positions) > 10:
        step = max(1, len(tick_positions) // 7)
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.12, color='#363a45'); ax.tick_params(axis='y', colors='#787b86', labelsize=11)
        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, color='#787b86', fontsize=10)
        ax.set_xlim(-1, len(data)); [spine.set_color('#1e222d') for spine in ax.spines.values()]
    ax2.set_xlabel('Date & Time (IST)', color='#b2b5be')

    plt.tight_layout(pad=2); plt.subplots_adjust(hspace=0.08)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=160, facecolor='#0e1217'); buf.seek(0); plt.close(fig)
    return buf

def format_value(num):
    if num >= 10000000: return f"{num/10000000:.1f}Cr"
    if num >= 100000: return f"{num/100000:.1f}L"
    if num >= 1000: return f"{num/1000:.1f}K"
    return str(int(num))

def format_option_chain_message(symbol, spot, expiry, strikes):
    if not strikes: return None
    atm_strike = min(strikes, key=lambda x: abs(x.get('strike_price', 0) - spot))
    atm_index = strikes.index(atm_strike)
    selected = strikes[max(0, atm_index - 5) : min(len(strikes), atm_index + 6)]

    msg = f"📊 *{symbol} - OPTION CHAIN*\n\n"
    msg += f"💰 Spot: `₹{spot:,.2f}`\n"
    msg += f"📅 Expiry: `{expiry}`\n"
    msg += f"🎯 ATM: `₹{atm_strike.get('strike_price', 0):,.0f}`\n\n"
    msg += "```\n"
    msg += "═══ CALLS ══════════════| STRIKE |══════════════ PUTS ═══\n"
    msg += "   OI    Vol     LTP    |        |    LTP     Vol     OI  \n"
    msg += "  Δ / Θ                 |        |                Δ / Θ   \n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_oi = sum(s.get('call_options', {}).get('market_data', {}).get('oi', 0) for s in strikes)
    total_pe_oi = sum(s.get('put_options', {}).get('market_data', {}).get('oi', 0) for s in strikes)

    for s in selected:
        strike_price = s.get('strike_price', 0)
        marker = "►" if strike_price == atm_strike.get('strike_price', 0) else " "
        
        ce = s.get('call_options', {}); ce_md = ce.get('market_data', {}); ce_g = ce.get('greeks', {})
        pe = s.get('put_options', {}); pe_md = pe.get('market_data', {}); pe_g = pe.get('greeks', {})
        
        ce_oi_str = format_value(ce_md.get('oi', 0)); ce_vol_str = format_value(ce_md.get('volume', 0))
        pe_oi_str = format_value(pe_md.get('oi', 0)); pe_vol_str = format_value(pe_md.get('volume', 0))

        msg += f"{ce_oi_str:>5} {ce_vol_str:>6} {ce_md.get('ltp', 0):>7.1f} |{marker}{strike_price:<8.0f}| {pe_md.get('ltp', 0):>7.1f} {pe_vol_str:>6} {pe_oi_str:>5}\n"
        
        ce_d = f"Δ{ce_g.get('delta', 0):.1f}"; ce_t = f"Θ{ce_g.get('theta', 0):.1f}"
        pe_d = f"Δ{pe_g.get('delta', 0):.1f}"; pe_t = f"Θ{pe_g.get('theta', 0):.1f}"
        msg += f" {ce_d:<5} {ce_t:<11} |        | {pe_d:<11} {pe_t:<5}\n"

    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "```\n"
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    msg += f"📊 *Total PCR (OI):* `{pcr:.3f}` ({format_value(total_pe_oi)} / {format_value(total_ce_oi)})\n"
    msg += f"⏰ `{datetime.now(IST).strftime('%I:%M:%S %p IST')}`\n"
    return msg

async def send_telegram_message(bot, text=None, photo=None, caption=None):
    try:
        if photo:
            await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode='Markdown')
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown')
        DAILY_STATS["total_alerts"] += 1
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

async def process_instrument(bot, key, name, expiry_day, is_stock=False, idx=0, total=0):
    prefix = f"[{idx}/{total}] STOCK:" if is_stock else "INDEX:"
    print(f"\n{prefix} {name}")
    try:
        spot = get_spot_price(key)
        if spot == 0: return False
        print(f"  ✅ Spot: ₹{spot:.2f}")

        expiry = get_next_expiry(key, expiry_day=expiry_day)
        strikes = get_option_chain(key, expiry)
        if strikes:
            msg = format_option_chain_message(name, spot, expiry, strikes)
            if msg:
                await send_telegram_message(bot, text=msg)
                print("    📤 Chain sent")
        else:
             print("    ⚠️ No option chain data found.")

        candles, hist_count = get_live_candles(key, name)
        if candles:
            chart = create_premium_chart(candles, name, spot, hist_count)
            if chart:
                caption = f"📈 *{name}*\n💰 `₹{spot:,.2f}`"
                await send_telegram_message(bot, photo=chart, caption=caption)
                print("    📤 Chart sent")
        else:
            print("    ⚠️ No candle data for chart.")
        
        if is_stock: DAILY_STATS["stocks_count"] += 1
        else: DAILY_STATS["indices_count"] += 1
        return True
    except Exception as e:
        print(f"  ❌ Processing Error for {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def fetch_all(bot):
    now = datetime.now(IST)
    print(f"\n{'='*60}\n🚀 RUN: {now.strftime('%I:%M:%S %p IST')}\n{'='*60}")
    
    header = f"🚀 *MARKET UPDATE* @ `{now.strftime('%I:%M %p')}`\n_Processing 4 indices + 51 stocks..._"
    await send_telegram_message(bot, text=header)
    
    DAILY_STATS["indices_count"] = 0
    for key, info in INDICES.items():
        await process_instrument(bot, key, info["name"], info["expiry_day"])
        await asyncio.sleep(2)

    DAILY_STATS["stocks_count"] = 0
    for i, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        await process_instrument(bot, key, symbol, 3, is_stock=True, idx=i, total=len(NIFTY50_STOCKS))
        await asyncio.sleep(2)
        
    summary = (f"✅ *UPDATE COMPLETE*\n\n"
               f"📊 Indices: {DAILY_STATS['indices_count']}/4\n"
               f"📈 Stocks: {DAILY_STATS['stocks_count']}/{len(NIFTY50_STOCKS)}\n"
               f"📡 Total Alerts Today: {DAILY_STATS['total_alerts']}\n\n"
               f"Next update in 5 minutes...")
    await send_telegram_message(bot, text=summary)
    print(f"\n✅ CYCLE DONE: Indices={DAILY_STATS['indices_count']}/4 | Stocks={DAILY_STATS['stocks_count']}/{len(NIFTY50_STOCKS)}")

async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    while True:
        now = datetime.now(IST)
        is_market_hours = (now.weekday() < 5) and (time(9, 15) <= now.time() <= time(15, 35))
        
        if is_market_hours:
            if DAILY_STATS["start_time"] is None: DAILY_STATS["start_time"] = now
            await fetch_all(bot)
            print(f"\n⏳ Next run in 5 minutes...")
            await asyncio.sleep(300)
        else:
            print(f"\n💤 Market closed. Current time: {now.strftime('%I:%M %p')}. Checking again in 15 mins.")
            if now.hour >= 16 and DAILY_STATS["start_time"] is not None:
                 print("🔄 Resetting daily stats for next day...")
                 DAILY_STATS["total_alerts"] = 0; DAILY_STATS["indices_count"] = 0; DAILY_STATS["stocks_count"] = 0
                 DAILY_STATS["start_time"] = None
            await asyncio.sleep(900)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
