# upstox_market_bot_full.py
# Full script: improved, robust, preserves original behavior + fixes
# Supports: intraday 1min->5min resample, historical fetch, option chain (contracts + greeks batching),
# candlestick chart generation, telegram photo/text sending with retries.
#
# Notes: Set UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID as env vars or edit defaults below.

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
import random
import math
from time import sleep

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

# ======================== HELPERS & RETRIES ========================

def get_ist_now():
    return datetime.now(IST)

def http_get(url, headers=None, params=None, timeout=12, retries=3, backoff=1.5):
    """GET with retries and exponential-ish backoff. Returns parsed JSON or None."""
    for attempt in range(1, retries+1):
        try:
            resp = requests.get(url, headers=headers or {"Accept":"application/json"}, params=params, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError:
                # not json
                return None
        except Exception as e:
            wait = (backoff ** (attempt-1)) + random.random()*0.3
            print(f"  Request error {e} (attempt {attempt}/{retries}). Retrying in {wait:.2f}s...")
            if attempt < retries:
                time.sleep(wait)
            else:
                print("  Max retries reached for:", url)
                return None

# ======================== RESAMPLE / CANDLES ========================

def resample_to_5min(candles_1min):
    """
    Convert 1-minute candles to 5-minute candles
    candles format: [[timestamp, open, high, low, close, volume, oi], ...]
    or list of lists where timestamp is parseable
    """
    if not candles_1min:
        return []
    try:
        # Build DataFrame robustly: support both tuple/list with various lengths
        # Normalize each row to at least [timestamp, open, high, low, close, volume, oi]
        rows = []
        for r in candles_1min:
            # r may be list like [ts, o, h, l, c, v, oi] or dict
            if isinstance(r, dict):
                ts = r.get('timestamp') or r.get('time') or r.get('date')
                o = r.get('open')
                h = r.get('high')
                l = r.get('low')
                c = r.get('close') or r.get('ltp')
                v = r.get('volume') or r.get('vol')
                oi = r.get('oi') or r.get('open_interest')
                rows.append([ts, o, h, l, c, v, oi])
            elif isinstance(r, (list, tuple)):
                # adapt by position
                # common positions: [ts, o, h, l, c, v, oi]
                # if length 6 -> no oi
                if len(r) >= 7:
                    ts, o, h, l, c, v, oi = r[:7]
                elif len(r) == 6:
                    ts, o, h, l, c, v = r[:6]
                    oi = None
                elif len(r) == 5:
                    ts, o, h, l, c = r[:5]
                    v = None
                    oi = None
                else:
                    continue
                rows.append([ts, o, h, l, c, v, oi])
            else:
                continue

        df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume','oi'])
        # parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        # Resample to 5T
        resampled = df.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'oi': 'last'
        }).dropna()
        resampled.reset_index(inplace=True)
        # convert to list format expected by charting
        out = resampled[['timestamp','open','high','low','close','volume','oi']].values.tolist()
        return out
    except Exception as e:
        print("  Error resampling:", e)
        return []

# ======================== INTRADAY / HISTORICAL ========================

def get_intraday_candles(instrument_key):
    """
    Fetch intraday 1-minute candles and convert to 5-minute
    Uses V2 API (path: /v2/historical-candle/{instrument_key}/1minute)
    """
    try:
        url = f"{BASE_URL}/v2/historical-candle/{instrument_key}/1minute"
        data = http_get(url, headers={"Accept":"application/json"}, timeout=12, retries=2)
        if not data:
            return []
        # typical shape: {'status': 'success', 'data': {'candles': [...]} }
        candles_1min = None
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], dict) and 'candles' in data['data']:
                candles_1min = data['data']['candles']
            elif 'candles' in data:
                candles_1min = data['candles']
            else:
                # maybe data itself is list
                if isinstance(data.get('data'), list):
                    candles_1min = data['data']
        elif isinstance(data, list):
            candles_1min = data

        if not candles_1min:
            return []
        return resample_to_5min(candles_1min)[:500]
    except Exception as e:
        print("  Error fetching intraday candles:", e)
        return []

def get_historical_candles(instrument_key, interval="1minute", days=5):
    """
    Fetch historical candles with from_date then to_date (fixed ordering)
    interval param kept for compatibility but will use 1minute for resample
    """
    try:
        now = get_ist_now()
        to_date = now.strftime('%Y-%m-%d')
        from_date = (now - timedelta(days=days)).strftime('%Y-%m-%d')
        url = f"{BASE_URL}/v2/historical-candle/{instrument_key}/1minute/{from_date}/{to_date}"
        data = http_get(url, headers={"Accept":"application/json"}, timeout=20, retries=3)
        if not data:
            return []
        candles_1min = None
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], dict) and 'candles' in data['data']:
                candles_1min = data['data']['candles']
            elif 'candles' in data:
                candles_1min = data['candles']
            elif isinstance(data.get('data'), list):
                candles_1min = data['data']
        elif isinstance(data, list):
            candles_1min = data
        if not candles_1min:
            return []
        return resample_to_5min(candles_1min)[:500]
    except Exception as e:
        print("  Error fetching historical candles:", e)
        return []

# ======================== OPTION CHAIN & GREEKS ========================

def get_next_expiry():
    """Get next Thursday expiry date (in IST). Keeps original simple behavior."""
    today = get_ist_now()
    # Thursday is weekday() == 3 (Mon=0)
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days=days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

def get_option_contracts():
    """
    Fetch option contracts for NIFTY for the chosen expiry.
    Tries pagination if supported.
    """
    try:
        expiry = get_next_expiry()
        url = f"{BASE_URL}/v2/option/contract"
        all_contracts = []
        page = 1
        page_size = 200
        while True:
            params = {"instrument_key": NIFTY_INDEX_KEY, "expiry_date": expiry, "page": page, "page_size": page_size}
            data = http_get(url, headers=HEADERS, params=params, timeout=15, retries=3)
            if not data:
                break
            payload = None
            if isinstance(data, dict):
                # common shape
                if data.get('status') and isinstance(data.get('data'), list):
                    payload = data['data']
                elif isinstance(data.get('data'), dict) and 'contracts' in data['data']:
                    payload = data['data']['contracts']
                elif isinstance(data.get('data'), list):
                    payload = data['data']
                elif isinstance(data.get('data'), dict):
                    # sometimes single contract
                    payload = [data['data']]
                else:
                    # fallback: entire response might be list
                    for k in ['data','contracts','result']:
                        if k in data and isinstance(data[k], list):
                            payload = data[k]
                            break
            elif isinstance(data, list):
                payload = data
            if not payload:
                break
            all_contracts.extend(payload)
            # pagination heuristics
            if len(payload) < page_size:
                break
            page += 1
            time.sleep(0.15)
        # sanitize
        all_contracts = [c for c in all_contracts if isinstance(c, dict) and c.get('instrument_key')]
        return all_contracts
    except Exception as e:
        print("  Error fetching option contracts:", e)
        return []

def get_option_greeks(instrument_keys):
    """
    Fetch option Greeks for up to 50 keys at once.
    Returns dict: instrument_key -> greek-data
    """
    try:
        if not instrument_keys:
            return {}
        url = f"{BASE_URL}/v2/option/greek"
        keys_str = ",".join(instrument_keys[:50])
        params = {"instrument_key": keys_str}
        data = http_get(url, headers=HEADERS, params=params, timeout=15, retries=3)
        if not data:
            return {}
        result = {}
        if isinstance(data, dict):
            payload = data.get('data') or data
            if isinstance(payload, dict):
                # payload may be mapping
                for k, v in payload.items():
                    if isinstance(v, dict):
                        # normalize known fields
                        normalized = dict(v)
                        # add possible aliases
                        if 'ltp' in v and 'last_price' not in v:
                            normalized['last_price'] = v['ltp']
                        if 'open_interest' in v and 'oi' not in v:
                            normalized['oi'] = v['open_interest']
                        if 'implied_volatility' in v and 'iv' not in v:
                            normalized['iv'] = v['implied_volatility']
                        result[k] = normalized
            elif isinstance(payload, list):
                for item in payload:
                    key = item.get('instrument_key') or item.get('instrumentKey') or item.get('instrument')
                    if key:
                        result[key] = item
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    key = item.get('instrument_key') or item.get('instrumentKey') or item.get('instrument')
                    if key:
                        result[key] = item
        return result
    except Exception as e:
        print("  Error fetching Greeks:", e)
        return {}

def get_instrument_ltp(instrument_key):
    """Fetch LTP/quote for instrument if possible."""
    try:
        url = f"{BASE_URL}/v2/market/quote/{instrument_key}"
        data = http_get(url, headers=HEADERS, timeout=10, retries=2)
        if not data:
            return None
        if isinstance(data, dict):
            d = data.get('data') or data
            if isinstance(d, dict):
                for k in ('ltp','last_price','lastTradedPrice','last'):
                    if k in d and d[k] is not None:
                        return float(d[k])
                # nested
                if 'market_data' in d and isinstance(d['market_data'], dict):
                    md = d['market_data']
                    for k in ('ltp','last_price'):
                        if k in md and md[k] is not None:
                            return float(md[k])
        return None
    except Exception as e:
        print("  Error fetching LTP:", e)
        return None

def get_option_chain_data():
    """
    Build option chain with combined Greeks.
    Returns a list of strike dicts like original: [{strike_price, call, put}, ...]
    """
    try:
        print("  Getting option contracts...")
        contracts = get_option_contracts()
        if not contracts:
            print("  ⚠️ No contracts found")
            return []
        instrument_keys = [c.get('instrument_key') for c in contracts if c.get('instrument_key')]
        all_greeks = {}
        print(f"  Fetching Greeks in batches for {len(instrument_keys)} instruments...")
        for i in range(0, len(instrument_keys), 50):
            batch = instrument_keys[i:i+50]
            greeks = get_option_greeks(batch)
            if greeks:
                all_greeks.update(greeks)
            time.sleep(0.25)
        # build option entries
        option_chain = []
        for c in contracts:
            key = c.get('instrument_key')
            strike = c.get('strike_price') or c.get('strike') or c.get('strikePrice')
            option_type = (c.get('option_type') or c.get('optionType') or '').upper()
            base = {
                'strike_price': float(strike) if strike is not None else None,
                'instrument_key': key,
                'option_type': 'CE' if option_type in ('CE','CALL') else ('PE' if option_type in ('PE','PUT') else option_type)
            }
            g = all_greeks.get(key, {})
            # merge
            opt = dict(base)
            opt.update({
                'last_price': float(g.get('last_price') or g.get('ltp') or 0),
                'oi': int(g.get('oi') or g.get('open_interest') or 0),
                'volume': int(g.get('volume') or 0),
                'delta': float(g.get('delta') or 0),
                'theta': float(g.get('theta') or 0),
                'gamma': float(g.get('gamma') or 0),
                'vega': float(g.get('vega') or 0),
                'iv': float(g.get('iv') or g.get('implied_volatility') or 0)
            })
            option_chain.append(opt)
        # group by strike
        strikes = {}
        for opt in option_chain:
            s = opt['strike_price']
            if s is None:
                continue
            if s not in strikes:
                strikes[s] = {'strike_price': s, 'call': None, 'put': None}
            if opt['option_type'] == 'CE':
                strikes[s]['call'] = opt
            elif opt['option_type'] == 'PE':
                strikes[s]['put'] = opt
        result = sorted(strikes.values(), key=lambda x: x['strike_price'])
        return result
    except Exception as e:
        print("  Error building option chain:", e)
        return []

# ======================== CHARTING ========================

def create_candlestick_chart(candles, title, show_volume=False):
    """
    Create a candlestick PNG in-memory (BytesIO).
    candles: list of [timestamp, open, high, low, close, volume, oi]
    Returns BytesIO or None.
    """
    if not candles:
        return None
    try:
        # infer columns
        has_oi = len(candles[0]) >= 7
        has_vol = len(candles[0]) >= 6
        cols = ['timestamp','open','high','low','close']
        if has_vol:
            cols.append('volume')
        if has_oi:
            cols.append('oi')
        df = pd.DataFrame(candles, columns=cols[:len(candles[0])])
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        # figure size
        if show_volume and 'volume' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,10), gridspec_kw={'height_ratios':[3,1]})
        else:
            fig, ax1 = plt.subplots(figsize=(16,8))
            ax2 = None
        ax1.set_facecolor('white')
        for idx, row in df.iterrows():
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            height = max(abs(row['close'] - row['open']), (row['high']*0.00001) if row['high'] else 0.0001)
            bottom = min(row['open'], row['close'])
            rect = mpatches.Rectangle((idx - 0.3, bottom), 0.6, height, facecolor=color, edgecolor=color, linewidth=1.2)
            ax1.add_patch(rect)
            ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1.2, solid_capstyle='round')
        ax1.set_xlim(-1, len(df))
        y_margin = (df['high'].max() - df['low'].min()) * 0.05 if len(df) > 1 else 1
        ax1.set_ylim(df['low'].min() - y_margin, df['high'].max() + y_margin)
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        # xticks
        step = max(len(df)//12, 1)
        xticks = list(range(0, len(df), step))
        xticklabels = [df.iloc[i]['timestamp'].strftime('%d %b\n%H:%M') for i in xticks]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=9)
        ax1.tick_params(axis='y', labelsize=10)
        if ax2 is not None and 'volume' in df.columns:
            ax2.set_facecolor('white')
            for idx, row in df.iterrows():
                color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                ax2.bar(idx, row['volume'] or 0, color=color, width=0.8, alpha=0.5)
            ax2.set_xlim(-1, len(df))
            ax2.set_ylabel('Volume', fontsize=11)
            ax2.set_xlabel('Time', fontsize=11)
            ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticklabels, fontsize=9)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        print("  Error creating chart:", e)
        return None

# ======================== FORMATTING / MARKET STATUS ========================

def format_market_status():
    now = get_ist_now()
    weekday = now.weekday()
    time_now = now.time()
    market_open_time = datetime.strptime("09:15", "%H:%M").time()
    market_close_time = datetime.strptime("15:30", "%H:%M").time()
    is_weekday = weekday < 5
    market_open = time_now >= market_open_time
    market_close = time_now <= market_close_time
    is_open = is_weekday and market_open and market_close
    status = "🟢 MARKET OPEN" if is_open else "🔴 MARKET CLOSED"
    time_str = now.strftime('%I:%M %p IST')
    if not is_weekday:
        return f"{status}\n⚠️ Weekend\n🕐 {time_str}"
    elif not market_open:
        return f"{status}\n⚠️ Pre-market (Opens 9:15 AM)\n🕐 {time_str}"
    elif not market_close:
        return f"{status}\n⚠️ After hours\n🕐 {time_str}"
    return f"{status}\n🕐 {time_str}"

def format_option_chain_message(option_data):
    """
    Accepts both old-style list (strikes list) and new dict optionally.
    If input is list of strike dicts (old), show ATM as middle.
    """
    if not option_data:
        return "❌ Option chain data not available"
    now = get_ist_now()
    text = "📊 *NIFTY 50 OPTION CHAIN* 📊\n\n"
    text += f"⏰ {now.strftime('%d %b %Y, %I:%M:%S %p IST')}\n"
    text += f"📅 Expiry: {get_next_expiry()}\n"
    # detect structure
    strikes = None
    atm_idx = None
    underlying = None
    if isinstance(option_data, dict) and 'strikes' in option_data:
        strikes = option_data['strikes']
        atm_idx = option_data.get('atm_index', len(strikes)//2)
        underlying = option_data.get('underlying_ltp')
    elif isinstance(option_data, list):
        strikes = option_data
        atm_idx = len(strikes)//2
    else:
        return "❌ Unexpected option data shape"

    text += f"📈 Total Strikes: {len(strikes)}\n"
    if underlying:
        text += f"📌 Underlying LTP: ₹{underlying:.2f}\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    start = max(0, atm_idx - 10)
    end = min(len(strikes), atm_idx + 11)
    for i in range(start, end):
        s = strikes[i]
        strike = s.get('strike_price', 'N/A')
        is_atm = (i == atm_idx)
        text += f"*Strike: {strike}* {'🔹 ATM' if is_atm else ''}\n"
        call = s.get('call') or s.get('CE') or s.get('Call')
        put = s.get('put') or s.get('PE') or s.get('Put')
        if call:
            try:
                iv_c = float(call.get('iv', 0)) * 100
            except:
                iv_c = float(call.get('iv', 0) or 0)
            text += f"📞 *CALL*\n  LTP: ₹{call.get('last_price', 0):.2f}\n  OI: {call.get('oi', 0):,}\n  Vol: {call.get('volume', 0):,}\n  𝛿: {call.get('delta', 0):.3f} | 𝜃: {call.get('theta', 0):.2f}\n  𝛄: {call.get('gamma', 0):.5f} | 𝜈: {call.get('vega', 0):.2f}\n  IV: {iv_c:.2f}%\n"
        if put:
            try:
                iv_p = float(put.get('iv', 0)) * 100
            except:
                iv_p = float(put.get('iv', 0) or 0)
            text += f"📉 *PUT*\n  LTP: ₹{put.get('last_price', 0):.2f}\n  OI: {put.get('oi', 0):,}\n  Vol: {put.get('volume', 0):,}\n  𝛿: {put.get('delta', 0):.3f} | 𝜃: {put.get('theta', 0):.2f}\n  𝛄: {put.get('gamma', 0):.5f} | 𝜈: {put.get('vega', 0):.2f}\n  IV: {iv_p:.2f}%\n"
        text += "\n"
    return text

# ======================== TELEGRAM SENDING (async) ========================

async def send_telegram_message(message):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        # Telegram max 4096 per message
        if len(message) > 4096:
            parts = [message[i:i+4096] for i in range(0, len(message), 4096)]
            for part in parts:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=part, parse_mode='Markdown')
                await asyncio.sleep(0.6)
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        return True
    except Exception as e:
        print("Telegram error:", e)
        return False

async def send_telegram_photo(photo, caption, retries=3):
    for attempt in range(retries):
        try:
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            # telegram python-bot expects file-like or bytes
            if isinstance(photo, io.BytesIO):
                photo.seek(0)
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode='Markdown')
            else:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode='Markdown')
            return True
        except TelegramError as e:
            print(f"  TelegramError sending photo (attempt {attempt+1}):", e)
            if attempt < retries-1:
                await asyncio.sleep(1.5)
            else:
                return False
        except Exception as e:
            print("  Error sending photo:", e)
            return False
    return False

# ======================== MAIN ========================

async def main():
    print("\n" + "="*60)
    print("🚀 UPSTOX MARKET DATA BOT (FULL) - START")
    print("="*60 + "\n")
    # Credentials check
    if UPSTOX_ACCESS_TOKEN == "your_access_token":
        print("❌ Set UPSTOX_ACCESS_TOKEN environment var or edit file.")
        return
    if TELEGRAM_BOT_TOKEN == "your_telegram_bot_token":
        print("❌ Set TELEGRAM_BOT_TOKEN environment var or edit file.")
        return
    market_status = format_market_status()
    print(market_status, "\n")
    now = get_ist_now()
    welcome = f"🎯 *Market Data Update*\n\n{market_status}\n⏰ {now.strftime('%d %b %Y, %I:%M:%S %p')}\n\nFetching Upstox V2 API data..."
    await send_telegram_message(welcome)

    # 1) NIFTY Index
    print("📈 Fetching NIFTY 50 Index data...")
    nifty_candles = get_intraday_candles(NIFTY_INDEX_KEY)
    if not nifty_candles or len(nifty_candles) < 5:
        print("  No intraday, trying historical...")
        nifty_candles = get_historical_candles(NIFTY_INDEX_KEY, "1minute", 3)
    if nifty_candles:
        print(f"  ✅ Got {len(nifty_candles)} 5-min candles")
        chart = create_candlestick_chart(nifty_candles, "NIFTY 50 - 5 Minute Chart", show_volume=True)
        if chart:
            await send_telegram_photo(chart, f"📊 *NIFTY 50 Index*\n{len(nifty_candles)} candles (5-min)")
            await asyncio.sleep(1.5)
    else:
        await send_telegram_message("⚠️ NIFTY 50 data unavailable")

    # 2) Option Chain
    print("\n📊 Fetching Option Chain...")
    option_chain = get_option_chain_data()
    if option_chain and len(option_chain) > 0:
        print(f"  ✅ Got {len(option_chain)} strikes")
        msg = format_option_chain_message(option_chain)
        await send_telegram_message(msg)
        await asyncio.sleep(1.0)
    else:
        print("  ⚠️ Option chain unavailable")
        await send_telegram_message("⚠️ Option chain data unavailable or empty")

    # 3) Nifty 50 Stocks (first 10)
    print("\n📈 Fetching Nifty 50 Stocks (first 10)...")
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
                await asyncio.sleep(1.5)
            else:
                print("❌ Chart failed")
        else:
            print("⚠️ No data")

    # Summary
    now = get_ist_now()
    summary = f"\n✅ *Update Complete!*\n\n"
    summary += f"{market_status}\n\n"
    summary += f"📊 Results:\n  • NIFTY 50: {'✅' if nifty_candles else '❌'}\n"
    summary += f"  • Option Chain: {len(option_chain) if option_chain else 0} strikes\n"
    summary += f"  • Stocks: {successful_charts}/10 charts\n\n"
    summary += f"⏰ {now.strftime('%I:%M:%S %p IST')}\n"
    summary += f"🔄 V2 API (1min→5min resampled)"
    await send_telegram_message(summary)
    print("\n" + "="*60)
    print(f"✅ COMPLETED! {successful_charts} charts sent")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("🔧 Checking dependencies...")
    try:
        import pandas
        import matplotlib
        from telegram import Bot
        import pytz
        print("✅ All dependencies loaded!\n")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        exit(1)
    asyncio.run(main())
