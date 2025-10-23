#!/usr/bin/env python3
# main.py - UPSTOX OPTION CHAIN BOT (FIXED - Auto Expiry Detection)
# Environment Variables: UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

import os
import io
import time
import random
import asyncio
import requests
import urllib.parse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from telegram import Bot
from telegram.error import TelegramError
import pytz
import logging

# ======================== LOGGING ========================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ======================== CONFIGURATION ========================
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Timezone
IST = pytz.timezone('Asia/Kolkata')

# API base
BASE_URL = "https://api.upstox.com"

# Stock/Index List with Instrument Keys
STOCKS_INDICES = {
    # Indices
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    
    # Top Stocks
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "INFY": "NSE_EQ|INE009A01021",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "SBIN": "NSE_EQ|INE062A01020",
    "LT": "NSE_EQ|INE018A01030",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "AXISBANK": "NSE_EQ|INE238A01034"
}

# ======================== UTILITIES ========================

def get_ist_now():
    """Current IST time"""
    return datetime.now(IST)

def is_market_open():
    """Check if market is open"""
    now = get_ist_now()
    weekday = now.weekday()
    time_now = now.time()
    market_open = datetime.strptime("09:15", "%H:%M").time()
    market_close = datetime.strptime("15:30", "%H:%M").time()
    return weekday < 5 and market_open <= time_now <= market_close

def http_get_with_retry(url, headers=None, timeout=12, retries=2):
    """HTTP GET with retry"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.ok:
                return resp.json()
            logger.warning(f"HTTP {resp.status_code} for {url[:80]}")
            resp.raise_for_status()
        except Exception as e:
            wait = (1.5 ** attempt) + random.random()
            logger.warning(f"Attempt {attempt}/{retries} failed: {str(e)[:100]}")
            if attempt < retries:
                time.sleep(wait)
    logger.error(f"Max retries reached for: {url[:80]}")
    return None

# ======================== MARKET DATA ========================

def get_intraday_candles(instrument_key):
    """Fetch intraday candles - PUBLIC endpoint (NO Authorization!)"""
    try:
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
        
        headers = {"Accept": "application/json"}
        
        logger.info(f"Fetching: {url[:100]}")
        data = http_get_with_retry(url, headers=headers, timeout=15, retries=2)
        
        if not data:
            logger.warning("No data returned from API")
            return []
        
        # Extract candles
        candles_raw = []
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], dict) and 'candles' in data['data']:
                candles_raw = data['data']['candles']
            elif 'candles' in data:
                candles_raw = data['candles']
        elif isinstance(data, list):
            candles_raw = data
        
        if not candles_raw:
            logger.warning("No candles in response")
            return []
        
        # Convert to 30min candles
        candles_30min = []
        for i in range(0, len(candles_raw), 30):
            batch = candles_raw[i:i+30]
            if len(batch) < 5:
                continue
            
            timestamp = batch[0][0]
            opens = [c[1] for c in batch]
            highs = [c[2] for c in batch]
            lows = [c[3] for c in batch]
            closes = [c[4] for c in batch]
            volumes = [c[5] if len(c) > 5 else 0 for c in batch]
            
            candles_30min.append({
                'timestamp': timestamp,
                'open': opens[0],
                'high': max(highs),
                'low': min(lows),
                'close': closes[-1],
                'volume': sum(volumes)
            })
        
        logger.info(f"Converted {len(candles_raw)} 1-min → {len(candles_30min)} 30-min candles")
        return candles_30min
        
    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        return []

def get_spot_price(instrument_key):
    """Get current spot price"""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }
        
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
        
        data = http_get_with_retry(url, headers=headers, timeout=10, retries=2)
        
        if data and isinstance(data, dict) and 'data' in data:
            quote_data = data['data']
            if isinstance(quote_data, dict):
                first_key = list(quote_data.keys())[0] if quote_data else None
                if first_key:
                    quote = quote_data[first_key]
                    for field in ['last_price', 'ltp']:
                        if field in quote and quote[field]:
                            return float(quote[field])
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting spot price: {e}")
        return None

def get_next_weekly_expiry():
    """Calculate next Thursday (FALLBACK)"""
    today = get_ist_now()
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days=days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

# ======================== AUTO EXPIRY DETECTION ========================

def get_available_expiries(instrument_key="NSE_INDEX|Nifty 50"):
    """🆕 Fetch all available expiry dates from contracts"""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }
        
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
        
        logger.info(f"📡 Fetching contracts: {url[:120]}")
        data = http_get_with_retry(url, headers=headers, timeout=15, retries=2)
        
        if not data:
            logger.warning("❌ No data from contracts API")
            return []
        
        logger.info(f"✅ Got response, type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
        
        # Extract contracts
        contracts = []
        if isinstance(data, dict) and 'data' in data:
            contracts = data['data']
            logger.info(f"✅ Found 'data' key with {len(contracts) if isinstance(contracts, list) else 'NON-LIST'} items")
        elif isinstance(data, list):
            contracts = data
            logger.info(f"✅ Response is list with {len(contracts)} items")
        
        if not contracts:
            logger.warning("❌ No contracts found in response")
            return []
        
        # Extract unique expiries
        expiries = set()
        sample_contract = None
        for contract in contracts[:5]:  # Check first 5
            if isinstance(contract, dict):
                logger.info(f"Sample contract keys: {list(contract.keys())[:10]}")
                if 'expiry' in contract:
                    expiries.add(contract['expiry'])
                    sample_contract = contract
        
        # Try all contracts
        for contract in contracts:
            if isinstance(contract, dict) and 'expiry' in contract:
                expiries.add(contract['expiry'])
        
        expiries_list = sorted(list(expiries))
        
        if expiries_list:
            logger.info(f"✅ Found {len(expiries_list)} expiries: {expiries_list[:8]}")
        else:
            logger.warning(f"❌ No expiries found! Sample contract: {sample_contract}")
        
        return expiries_list
        
    except Exception as e:
        logger.error(f"❌ get_available_expiries error: {e}", exc_info=True)
        return []

def get_next_valid_expiry(instrument_key="NSE_INDEX|Nifty 50"):
    """🆕 Get next valid expiry from actual exchange data"""
    try:
        expiries = get_available_expiries(instrument_key)
        
        if not expiries:
            logger.warning("⚠️ No expiries from API, using fallback")
            return get_next_weekly_expiry()
        
        today = get_ist_now().date()
        
        # Find next expiry >= today
        future_expiries = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
        
        if future_expiries:
            next_expiry = min(future_expiries)
            logger.info(f"🎯 Next valid expiry: {next_expiry}")
            return next_expiry
        
        # If all past, return latest
        if expiries:
            latest = max(expiries)
            logger.warning(f"All past, using latest: {latest}")
            return latest
        
        return get_next_weekly_expiry()
        
    except Exception as e:
        logger.error(f"Error getting expiry: {e}")
        return get_next_weekly_expiry()

# ======================== OPTION CHAIN ========================

def get_option_chain_data(instrument_key="NSE_INDEX|Nifty 50"):
    """🔧 FIXED: Uses auto-detected valid expiry"""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }
        
        # 🆕 Get VALID expiry
        expiry = get_next_valid_expiry(instrument_key)
        logger.info(f"📅 Using expiry: {expiry}")
        
        # Get spot price
        spot_price = get_spot_price(instrument_key)
        if spot_price:
            logger.info(f"💰 Spot: {spot_price}")
        else:
            logger.warning("⚠️ No spot price")
            spot_price = 0
        
        # Fetch option chain
        logger.info("🔍 Fetching option chain...")
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
        
        data = http_get_with_retry(url, headers=headers, timeout=20, retries=3)
        
        if data and isinstance(data, dict):
            logger.info(f"📦 Keys: {list(data.keys())}")
            
            # Extract strikes
            strikes_data = []
            if 'data' in data:
                payload = data['data']
                if isinstance(payload, list) and len(payload) > 0:
                    strikes_data = payload
                    logger.info(f"✅ {len(strikes_data)} strikes in data")
                elif isinstance(payload, dict):
                    for key in ['data', 'strikes', 'option_chain']:
                        if key in payload and isinstance(payload[key], list):
                            strikes_data = payload[key]
                            logger.info(f"✅ {len(strikes_data)} in data.{key}")
                            break
            
            if strikes_data and len(strikes_data) > 0:
                logger.info("✅ Processing...")
                return process_option_chain_strikes(strikes_data, spot_price, expiry)
        
        logger.warning("⚠️ Empty response")
        return None
        
    except Exception as e:
        logger.error(f"❌ Option chain error: {e}", exc_info=True)
        return None

def process_option_chain_strikes(strikes_data, spot_price, expiry):
    """Process strikes from API response"""
    try:
        strikes_dict = {}
        processed = 0
        
        for item in strikes_data:
            if not isinstance(item, dict):
                continue
            
            # Get strike
            strike_price = None
            for key in ['strike_price', 'strike', 'strikePrice']:
                if key in item:
                    strike_price = float(item[key])
                    break
            
            if not strike_price:
                continue
            
            if strike_price not in strikes_dict:
                strikes_dict[strike_price] = {'strike_price': strike_price, 'call': None, 'put': None}
            
            # Process CE/PE
            call_data = item.get('call_options', item.get('call', item.get('CE')))
            put_data = item.get('put_options', item.get('put', item.get('PE')))
            
            if call_data and isinstance(call_data, dict):
                md = call_data.get('market_data', call_data)
                greeks = call_data.get('option_greeks', {})
                
                strikes_dict[strike_price]['call'] = {
                    'last_price': float(md.get('ltp', md.get('last_price', 0))),
                    'oi': int(md.get('oi', md.get('open_interest', 0))),
                    'volume': int(md.get('volume', 0)),
                    'delta': float(greeks.get('delta', 0)),
                    'theta': float(greeks.get('theta', 0)),
                    'iv': float(greeks.get('iv', greeks.get('implied_volatility', 0)))
                }
                processed += 1
            
            if put_data and isinstance(put_data, dict):
                md = put_data.get('market_data', put_data)
                greeks = put_data.get('option_greeks', {})
                
                strikes_dict[strike_price]['put'] = {
                    'last_price': float(md.get('ltp', md.get('last_price', 0))),
                    'oi': int(md.get('oi', md.get('open_interest', 0))),
                    'volume': int(md.get('volume', 0)),
                    'delta': float(greeks.get('delta', 0)),
                    'theta': float(greeks.get('theta', 0)),
                    'iv': float(greeks.get('iv', greeks.get('implied_volatility', 0)))
                }
                processed += 1
        
        logger.info(f"✅ Processed {processed} options, {len(strikes_dict)} strikes")
        
        if not strikes_dict:
            return None
        
        sorted_strikes = sorted(strikes_dict.values(), key=lambda x: x['strike_price'])
        
        # Find ATM
        atm_index = len(sorted_strikes) // 2
        if spot_price and sorted_strikes:
            atm_index = min(range(len(sorted_strikes)), 
                          key=lambda i: abs(sorted_strikes[i]['strike_price'] - spot_price))
        
        return {
            'strikes': sorted_strikes,
            'atm_index': atm_index,
            'spot_price': spot_price,
            'expiry': expiry
        }
        
    except Exception as e:
        logger.error(f"Process error: {e}")
        return None

# ======================== CHART CREATION ========================

def create_candlestick_chart(candles, title):
    """Create candlestick chart"""
    if not candles or len(candles) < 2:
        return None
    
    try:
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.set_facecolor('#1e1e1e')
        ax2.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        for idx, row in df.iterrows():
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            height = abs(row['close'] - row['open'])
            if height == 0:
                height = row['high'] * 0.0001
            bottom = min(row['open'], row['close'])
            
            rect = mpatches.Rectangle((idx - 0.3, bottom), 0.6, height, 
                                     facecolor=color, edgecolor=color, linewidth=1.5)
            ax1.add_patch(rect)
            ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1.5)
        
        ax1.set_xlim(-1, len(df))
        y_margin = (df['high'].max() - df['low'].min()) * 0.05
        ax1.set_ylim(df['low'].min() - y_margin, df['high'].max() + y_margin)
        ax1.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15)
        ax1.set_ylabel('Price (₹)', fontsize=12, color='white')
        ax1.grid(True, alpha=0.15, linestyle='--', color='#666')
        ax1.tick_params(colors='white')
        
        for idx, row in df.iterrows():
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            ax2.bar(idx, row['volume'], color=color, width=0.8, alpha=0.6)
        
        ax2.set_xlim(-1, len(df))
        ax2.set_ylabel('Volume', fontsize=11, color='white')
        ax2.grid(True, alpha=0.15, linestyle='--', color='#666')
        ax2.tick_params(colors='white')
        
        step = max(len(df) // 12, 1)
        xticks = list(range(0, len(df), step))
        xticklabels = [df.iloc[i]['timestamp'].strftime('%d %b\n%H:%M') for i in xticks]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=9, color='white')
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticklabels, fontsize=9, color='white')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor='#1e1e1e', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None

# ======================== FORMATTING ========================

def format_option_chain_message(oc_data, symbol="NIFTY 50"):
    """Format option chain message"""
    if not oc_data or not oc_data.get('strikes'):
        return "❌ Option chain unavailable"
    
    try:
        strikes = oc_data['strikes']
        atm_index = oc_data['atm_index']
        spot = oc_data['spot_price']
        expiry = oc_data['expiry']
        
        msg = f"📊 *{symbol} OPTION CHAIN*\n\n"
        msg += f"📅 Expiry: {expiry}\n"
        msg += f"💰 Spot: ₹{spot:,.2f}\n"
        msg += f"🎯 ATM: ₹{strikes[atm_index]['strike_price']:,.0f}\n\n"
        
        start = max(0, atm_index - 5)
        end = min(len(strikes), atm_index + 6)
        selected = strikes[start:end]
        
        msg += "```\n"
        msg += "Strike    CE-LTP  CE-OI   CE-Vol  PE-LTP  PE-OI   PE-Vol\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for i, s in enumerate(selected):
            is_atm = (start + i == atm_index)
            mark = "🔸" if is_atm else "  "
            
            sp = s['strike_price']
            call = s.get('call', {})
            put = s.get('put', {})
            
            ce_ltp = call.get('last_price', 0) if call else 0
            ce_oi = call.get('oi', 0) if call else 0
            ce_vol = call.get('volume', 0) if call else 0
            
            pe_ltp = put.get('last_price', 0) if put else 0
            pe_oi = put.get('oi', 0) if put else 0
            pe_vol = put.get('volume', 0) if put else 0
            
            msg += f"{mark}{sp:7.0f}  {ce_ltp:7.2f} {ce_oi/1000:7.1f}K {ce_vol/1000:6.1f}K {pe_ltp:7.2f} {pe_oi/1000:7.1f}K {pe_vol/1000:6.1f}K\n"
        
        msg += "```\n\n"
        
        atm = strikes[atm_index]
        call_atm = atm.get('call', {})
        put_atm = atm.get('put', {})
        
        if call_atm or put_atm:
            msg += "📈 *ATM Greeks:*\n"
            if call_atm:
                iv = call_atm.get('iv', 0) * 100 if call_atm.get('iv', 0) < 10 else call_atm.get('iv', 0)
                msg += f"CE: Δ={call_atm.get('delta', 0):.3f} | Θ={call_atm.get('theta', 0):.2f} | IV={iv:.1f}%\n"
            if put_atm:
                iv = put_atm.get('iv', 0) * 100 if put_atm.get('iv', 0) < 10 else put_atm.get('iv', 0)
                msg += f"PE: Δ={put_atm.get('delta', 0):.3f} | Θ={put_atm.get('theta', 0):.2f} | IV={iv:.1f}%\n"
        
        total_ce_oi = sum(s.get('call', {}).get('oi', 0) for s in selected if s.get('call'))
        total_pe_oi = sum(s.get('put', {}).get('oi', 0) for s in selected if s.get('put'))
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        msg += f"\n📊 *PCR:* {pcr:.3f}"
        
        return msg
        
    except Exception as e:
        logger.error(f"Format error: {e}")
        return "❌ Error formatting"

# ======================== TELEGRAM ========================

async def send_message(text):
    """Send Telegram message"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        if len(text) > 4096:
            parts = [text[i:i+4096] for i in range(0, len(text), 4096)]
            for part in parts:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=part, parse_mode='Markdown')
                await asyncio.sleep(0.5)
        else:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown')
        return True
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False

async def send_photo(photo_buf, caption):
    """Send chart"""
    try:
        if not photo_buf:
            return False
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        photo_buf.seek(0)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_buf, caption=caption, parse_mode='Markdown')
        return True
    except Exception as e:
        logger.error(f"Photo error: {e}")
        return False

# ======================== MAIN ========================

async def process_symbol(symbol, instrument_key):
    """Process symbol"""
    try:
        logger.info(f"Processing {symbol}...")
        
        candles = get_intraday_candles(instrument_key)
        
        if not candles or len(candles) < 2:
            logger.warning(f"{symbol}: Insufficient ({len(candles) if candles else 0})")
            return False
        
        logger.info(f"{symbol}: Got {len(candles)} candles")
        
        chart = create_candlestick_chart(candles, f"{symbol} - 30 Min")
        
        if not chart:
            logger.warning(f"{symbol}: Chart failed")
            return False
        
        caption = f"📊 *{symbol}*\n📈 {len(candles)} candles\n⏰ {get_ist_now().strftime('%I:%M %p IST')}"
        success = await send_photo(chart, caption)
        
        if success:
            logger.info(f"✅ {symbol} sent")
        
        return success
        
    except Exception as e:
        logger.error(f"Error {symbol}: {e}")
        return False

async def run_bot_cycle():
    """Main cycle"""
    try:
        logger.info("="*60)
        logger.info("🚀 BOT CYCLE START")
        logger.info("="*60)
        
        market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
        now = get_ist_now()
        
        welcome = f"🎯 *Market Update*\n\n{market_status}\n⏰ {now.strftime('%d %b, %I:%M %p IST')}\n\n📊 Processing {len(STOCKS_INDICES)} symbols..."
        await send_message(welcome)
        await asyncio.sleep(2)
        
        symbols = list(STOCKS_INDICES.items())
        batch_size = 5
        successful = 0
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"\n📦 Batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            for symbol, key in batch:
                if await process_symbol(symbol, key):
                    successful += 1
                await asyncio.sleep(3)
            
            if i + batch_size < len(symbols):
                logger.info("⏳ 5 sec...")
                await asyncio.sleep(5)
        
        # OPTION CHAIN WITH DETAILED LOGGING
        logger.info("\n" + "="*60)
        logger.info("📊 STARTING OPTION CHAIN FETCH")
        logger.info("="*60)
        
        oc_data = None
        try:
            logger.info("Step 1: Calling get_option_chain_data()...")
            oc_data = get_option_chain_data("NSE_INDEX|Nifty 50")
            
            logger.info(f"Step 2: Received oc_data: {type(oc_data)}")
            
            if oc_data:
                logger.info(f"Step 3: oc_data keys: {oc_data.keys() if isinstance(oc_data, dict) else 'NOT A DICT'}")
                
                if oc_data.get('strikes'):
                    logger.info(f"Step 4: Found {len(oc_data['strikes'])} strikes")
                    msg = format_option_chain_message(oc_data, "NIFTY 50")
                    logger.info(f"Step 5: Message formatted, length: {len(msg)}")
                    await send_message(msg)
                    logger.info("Step 6: ✅ Option chain message sent!")
                    await asyncio.sleep(1)
                else:
                    logger.warning("⚠️ oc_data has NO 'strikes' key")
                    await send_message("⚠️ Option chain: No strikes data")
            else:
                logger.warning("⚠️ oc_data is None/False")
                await send_message("⚠️ Option chain: API returned empty")
                
        except Exception as e:
            logger.error(f"❌ OPTION CHAIN EXCEPTION: {type(e).__name__}: {str(e)}", exc_info=True)
            error_msg = f"❌ Option chain error:\n{type(e).__name__}: {str(e)[:200]}"
            await send_message(error_msg)
        
        await asyncio.sleep(2)
        
        summary = f"\n✅ *Complete!*\n\n{market_status}\n📊 Charts: {successful}/{len(STOCKS_INDICES)}\n📈 Options: {'✅' if oc_data else '❌'}\n⏰ {get_ist_now().strftime('%I:%M %p')}"
        await send_message(summary)
        
        logger.info("="*60)
        logger.info(f"✅ DONE! {successful} charts")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Cycle error: {e}")
        await send_message(f"❌ Error: {str(e)[:200]}")

async def run_bot_loop():
    """Continuous loop"""
    logger.info("🤖 Bot started!")
    
    while True:
        try:
            await run_bot_cycle()
            logger.info("\n⏳ Waiting 5 minutes...\n")
            await asyncio.sleep(300)
        except KeyboardInterrupt:
            logger.info("Bot stopped")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(60)

async def send_startup():
    """Startup message"""
    try:
        msg = "🤖 *UPSTOX BOT STARTED!* v2.0\n\n"
        msg += f"📊 {len(STOCKS_INDICES)} symbols\n"
        msg += "⏱️ Updates every 5 min\n\n"
        msg += "📈 Features:\n"
        msg += "  • 30-min Charts\n"
        msg += "  • Volume Analysis\n"
        msg += "  • Option Chain (Auto-Expiry)\n"
        msg += "  • Greeks & IV\n\n"
        msg += "🔧 *What's Fixed:*\n"
        msg += "  • Auto-detects valid expiry dates\n"
        msg += "  • Uses exchange-recognized dates\n"
        msg += "  • No manual expiry calculation\n\n"
        msg += f"🕐 {get_ist_now().strftime('%I:%M %p IST')}"
        await send_message(msg)
        logger.info("✅ Startup sent")
    except Exception as e:
        logger.error(f"Startup error: {e}")

async def main():
    """Entry point"""
    try:
        if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
            logger.error("❌ Missing environment variables!")
            return
        
        logger.info("✅ Environment validated")
        logger.info(f"📱 Chat ID: {TELEGRAM_CHAT_ID}")
        
        await send_startup()
        await asyncio.sleep(2)
        await run_bot_loop()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    logger.info("🔧 Starting Upstox Bot v2.0 (Auto-Expiry)...")
    logger.info("📦 Checking dependencies...")
    
    try:
        import pandas
        import matplotlib
        import pytz
        from telegram import Bot
        logger.info("✅ All dependencies loaded!")
    except ImportError as e:
        logger.error(f"❌ Missing: {e}")
        exit(1)
    
    asyncio.run(main())
