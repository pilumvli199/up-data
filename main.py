#!/usr/bin/env python3
# main.py - UPSTOX OPTION CHAIN BOT (Complete with Volume & Greeks)
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
HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
}

# Stock/Index List with Instrument Keys
STOCKS_INDICES = {
    # Indices
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "SENSEX": "BSE_INDEX|SENSEX",
    
    # Top Stocks
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
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "WIPRO": "NSE_EQ|INE075A01022",
    "TECHM": "NSE_EQ|INE669C01036",
    "DMART": "NSE_EQ|INE192R01011",
    "TRENT": "NSE_EQ|INE849A01020"
}

# ======================== UTILITIES ========================

def get_ist_now():
    """Current IST time return करतो"""
    return datetime.now(IST)

def is_market_open():
    """Check market hours"""
    now = get_ist_now()
    weekday = now.weekday()
    time_now = now.time()
    market_open_time = datetime.strptime("09:15", "%H:%M").time()
    market_close_time = datetime.strptime("15:30", "%H:%M").time()
    is_weekday = weekday < 5
    return is_weekday and market_open_time <= time_now <= market_close_time

def http_get_with_retry(url, headers=None, params=None, timeout=12, retries=3):
    """HTTP GET with exponential backoff retry"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers or HEADERS, params=params, timeout=timeout)
            if resp.ok:
                return resp.json()
            logger.warning(f"HTTP {resp.status_code} for {url[:100]}")
            resp.raise_for_status()
        except Exception as e:
            wait = (1.5 ** attempt) + random.random()
            logger.warning(f"Request error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(wait)
    logger.error(f"Max retries reached for: {url[:100]}")
    return None

# ======================== MARKET DATA FUNCTIONS ========================

def get_spot_price(instrument_key):
    """Get current spot/LTP price"""
    try:
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
        data = http_get_with_retry(url, timeout=10, retries=2)
        
        if data and isinstance(data, dict):
            if 'data' in data:
                quote_data = data['data']
                # Handle both dict and nested dict structures
                if isinstance(quote_data, dict):
                    # If quote_data has instrument_key as key
                    if instrument_key in quote_data:
                        quote = quote_data[instrument_key]
                    else:
                        # Otherwise take first value
                        quote = list(quote_data.values())[0] if quote_data else {}
                    
                    # Try different LTP field names
                    for field in ['last_price', 'ltp', 'last_traded_price']:
                        if field in quote and quote[field]:
                            return float(quote[field])
                    
                    # Check nested market_data
                    if 'market_data' in quote:
                        md = quote['market_data']
                        for field in ['last_price', 'ltp']:
                            if field in md and md[field]:
                                return float(md[field])
        
        logger.warning(f"Could not extract LTP from response for {instrument_key[:20]}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting spot price: {e}")
        return None

def get_intraday_candles(instrument_key, interval="30minute"):
    """Fetch intraday candles (current day) - NO Authorization needed!"""
    try:
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/{interval}"
        
        # Intraday endpoint ला Authorization नाही लागत!
        headers_no_auth = {"Accept": "application/json"}
        data = http_get_with_retry(url, headers=headers_no_auth, timeout=15, retries=2)
        
        if not data:
            return []
        
        # Extract candles from response
        candles = []
        if isinstance(data, dict):
            if 'data' in data and 'candles' in data['data']:
                candles = data['data']['candles']
            elif 'candles' in data:
                candles = data['candles']
            elif isinstance(data.get('data'), list):
                candles = data['data']
        elif isinstance(data, list):
            candles = data
        
        if not candles:
            return []
        
        # Convert to standard format: [timestamp, open, high, low, close, volume, oi]
        formatted_candles = []
        for candle in candles:
            if isinstance(candle, (list, tuple)) and len(candle) >= 5:
                formatted_candles.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': int(candle[5]) if len(candle) > 5 else 0,
                    'oi': int(candle[6]) if len(candle) > 6 else 0
                })
        
        return formatted_candles[:200]  # Limit to 200 candles
        
    except Exception as e:
        logger.error(f"Error fetching intraday candles: {e}")
        return []

def get_next_weekly_expiry():
    """Calculate next Thursday expiry"""
    today = get_ist_now()
    days_ahead = 3 - today.weekday()  # Thursday = 3
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days=days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

def get_option_chain_data(instrument_key="NSE_INDEX|Nifty 50"):
    """
    Fetch complete option chain with CE/PE data, Greeks, OI, Volume
    Returns: dict with strikes, ATM info, underlying price
    """
    try:
        expiry = get_next_weekly_expiry()
        logger.info(f"Fetching option chain for {instrument_key[:20]}, expiry: {expiry}")
        
        # Get underlying spot price
        spot_price = get_spot_price(instrument_key)
        if not spot_price:
            logger.warning("Could not fetch underlying spot price")
            spot_price = 0
        
        # Try option chain endpoint (v2/option/chain)
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
        
        data = http_get_with_retry(url, timeout=20, retries=2)
        
        if not data:
            logger.warning("Option chain endpoint returned no data")
            return None
        
        # Parse response structure
        strikes_data = []
        
        if isinstance(data, dict):
            payload = data.get('data', data)
            
            # Check if payload has 'data' array
            if isinstance(payload, dict) and 'data' in payload:
                strikes_data = payload['data']
            elif isinstance(payload, list):
                strikes_data = payload
            # Some APIs return strikes directly
            elif 'strikes' in payload:
                strikes_data = payload['strikes']
        elif isinstance(data, list):
            strikes_data = data
        
        if not strikes_data:
            logger.warning("No strikes data found in option chain response")
            return None
        
        # Process strikes into organized format
        strikes_dict = {}
        
        for item in strikes_data:
            if not isinstance(item, dict):
                continue
            
            strike_price = item.get('strike_price', item.get('strike'))
            if not strike_price:
                continue
            
            strike_price = float(strike_price)
            
            if strike_price not in strikes_dict:
                strikes_dict[strike_price] = {
                    'strike_price': strike_price,
                    'call': None,
                    'put': None
                }
            
            # Check if this is call or put data
            call_data = item.get('call_options', item.get('call'))
            put_data = item.get('put_options', item.get('put'))
            
            # Process CALL data
            if call_data and isinstance(call_data, dict):
                market_data = call_data.get('market_data', call_data)
                option_greeks = call_data.get('option_greeks', {})
                
                strikes_dict[strike_price]['call'] = {
                    'instrument_key': call_data.get('instrument_key', ''),
                    'last_price': float(market_data.get('ltp', market_data.get('last_price', 0))),
                    'oi': int(market_data.get('oi', market_data.get('open_interest', 0))),
                    'volume': int(market_data.get('volume', 0)),
                    'bid_price': float(market_data.get('bid_price', 0)),
                    'ask_price': float(market_data.get('ask_price', 0)),
                    'delta': float(option_greeks.get('delta', 0)),
                    'theta': float(option_greeks.get('theta', 0)),
                    'gamma': float(option_greeks.get('gamma', 0)),
                    'vega': float(option_greeks.get('vega', 0)),
                    'iv': float(option_greeks.get('iv', option_greeks.get('implied_volatility', 0)))
                }
            
            # Process PUT data
            if put_data and isinstance(put_data, dict):
                market_data = put_data.get('market_data', put_data)
                option_greeks = put_data.get('option_greeks', {})
                
                strikes_dict[strike_price]['put'] = {
                    'instrument_key': put_data.get('instrument_key', ''),
                    'last_price': float(market_data.get('ltp', market_data.get('last_price', 0))),
                    'oi': int(market_data.get('oi', market_data.get('open_interest', 0))),
                    'volume': int(market_data.get('volume', 0)),
                    'bid_price': float(market_data.get('bid_price', 0)),
                    'ask_price': float(market_data.get('ask_price', 0)),
                    'delta': float(option_greeks.get('delta', 0)),
                    'theta': float(option_greeks.get('theta', 0)),
                    'gamma': float(option_greeks.get('gamma', 0)),
                    'vega': float(option_greeks.get('vega', 0)),
                    'iv': float(option_greeks.get('iv', option_greeks.get('implied_volatility', 0)))
                }
        
        # Sort strikes
        sorted_strikes = sorted(strikes_dict.values(), key=lambda x: x['strike_price'])
        
        # Find ATM strike
        atm_index = 0
        if spot_price and sorted_strikes:
            atm_index = min(
                range(len(sorted_strikes)),
                key=lambda i: abs(sorted_strikes[i]['strike_price'] - spot_price)
            )
        
        logger.info(f"Option chain: {len(sorted_strikes)} strikes, ATM: {sorted_strikes[atm_index]['strike_price'] if sorted_strikes else 'N/A'}")
        
        return {
            'strikes': sorted_strikes,
            'atm_index': atm_index,
            'spot_price': spot_price,
            'expiry': expiry,
            'total_strikes': len(sorted_strikes)
        }
        
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return None

# ======================== CHART CREATION ========================

def create_candlestick_chart(candles, title, show_volume=True):
    """Create professional candlestick chart with volume"""
    if not candles or len(candles) < 2:
        logger.warning(f"Not enough candles for chart: {len(candles) if candles else 0}")
        return None
    
    try:
        # Prepare DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create figure
        if show_volume and 'volume' in df.columns:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(16, 10),
                gridspec_kw={'height_ratios': [3, 1]}
            )
        else:
            fig, ax1 = plt.subplots(figsize=(16, 8))
            ax2 = None
        
        # Set dark theme
        ax1.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        # Plot candlesticks
        for idx, row in df.iterrows():
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            
            # Candle body
            height = abs(row['close'] - row['open'])
            if height == 0:
                height = row['high'] * 0.0001  # Tiny height for doji
            bottom = min(row['open'], row['close'])
            
            rect = mpatches.Rectangle(
                (idx - 0.3, bottom), 0.6, height,
                facecolor=color, edgecolor=color, linewidth=1.5
            )
            ax1.add_patch(rect)
            
            # Wick
            ax1.plot(
                [idx, idx], [row['low'], row['high']],
                color=color, linewidth=1.5, solid_capstyle='round'
            )
        
        # Configure price axis
        ax1.set_xlim(-1, len(df))
        y_margin = (df['high'].max() - df['low'].min()) * 0.05
        ax1.set_ylim(df['low'].min() - y_margin, df['high'].max() + y_margin)
        
        ax1.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15)
        ax1.set_ylabel('Price (₹)', fontsize=12, color='white')
        ax1.grid(True, alpha=0.15, linestyle='--', color='#666666')
        
        # X-axis labels
        step = max(len(df) // 12, 1)
        xticks = list(range(0, len(df), step))
        xticklabels = [df.iloc[i]['timestamp'].strftime('%d %b\n%H:%M') for i in xticks]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels, fontsize=9, color='white')
        ax1.tick_params(axis='y', labelsize=10, colors='white')
        
        # Volume subplot
        if ax2 is not None and 'volume' in df.columns:
            ax2.set_facecolor('#1e1e1e')
            
            for idx, row in df.iterrows():
                color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                ax2.bar(idx, row['volume'], color=color, width=0.8, alpha=0.6)
            
            ax2.set_xlim(-1, len(df))
            ax2.set_ylabel('Volume', fontsize=11, color='white')
            ax2.set_xlabel('Time', fontsize=11, color='white')
            ax2.grid(True, alpha=0.15, linestyle='--', color='#666666')
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticklabels, fontsize=9, color='white')
            ax2.tick_params(colors='white')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor='#1e1e1e', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return None

# ======================== MESSAGE FORMATTING ========================

def format_option_chain_message(oc_data, symbol="NIFTY 50"):
    """Format option chain data into Telegram message"""
    if not oc_data or not oc_data.get('strikes'):
        return "❌ Option chain data unavailable"
    
    try:
        strikes = oc_data['strikes']
        atm_index = oc_data['atm_index']
        spot_price = oc_data['spot_price']
        expiry = oc_data['expiry']
        
        # Build message
        msg = f"📊 *{symbol} OPTION CHAIN*\n\n"
        msg += f"📅 Expiry: {expiry}\n"
        msg += f"💰 Spot: ₹{spot_price:,.2f}\n"
        msg += f"🎯 ATM: ₹{strikes[atm_index]['strike_price']:,.0f}\n"
        msg += f"📈 Total Strikes: {len(strikes)}\n\n"
        
        # Select strikes around ATM (±5)
        start_idx = max(0, atm_index - 5)
        end_idx = min(len(strikes), atm_index + 6)
        selected_strikes = strikes[start_idx:end_idx]
        
        msg += "```\n"
        msg += "Strike    CE-LTP  CE-OI   CE-Vol  PE-LTP  PE-OI   PE-Vol\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for i, strike in enumerate(selected_strikes):
            actual_idx = start_idx + i
            is_atm = (actual_idx == atm_index)
            atm_mark = "🔸" if is_atm else "  "
            
            strike_price = strike['strike_price']
            call = strike.get('call', {})
            put = strike.get('put', {})
            
            ce_ltp = call.get('last_price', 0) if call else 0
            ce_oi = call.get('oi', 0) if call else 0
            ce_vol = call.get('volume', 0) if call else 0
            
            pe_ltp = put.get('last_price', 0) if put else 0
            pe_oi = put.get('oi', 0) if put else 0
            pe_vol = put.get('volume', 0) if put else 0
            
            msg += f"{atm_mark}{strike_price:7.0f}  {ce_ltp:7.2f} {ce_oi/1000:7.1f}K {ce_vol/1000:6.1f}K {pe_ltp:7.2f} {pe_oi/1000:7.1f}K {pe_vol/1000:6.1f}K\n"
        
        msg += "```\n\n"
        
        # ATM Greeks
        atm_strike = strikes[atm_index]
        call_atm = atm_strike.get('call', {})
        put_atm = atm_strike.get('put', {})
        
        if call_atm or put_atm:
            msg += "📈 *ATM Greeks & IV:*\n"
            
            if call_atm:
                ce_iv = call_atm.get('iv', 0) * 100 if call_atm.get('iv', 0) < 10 else call_atm.get('iv', 0)
                msg += f"CE: Δ={call_atm.get('delta', 0):.3f} | Θ={call_atm.get('theta', 0):.2f} | IV={ce_iv:.1f}%\n"
            
            if put_atm:
                pe_iv = put_atm.get('iv', 0) * 100 if put_atm.get('iv', 0) < 10 else put_atm.get('iv', 0)
                msg += f"PE: Δ={put_atm.get('delta', 0):.3f} | Θ={put_atm.get('theta', 0):.2f} | IV={pe_iv:.1f}%\n"
        
        # PCR calculation
        total_ce_oi = sum(s.get('call', {}).get('oi', 0) for s in selected_strikes if s.get('call'))
        total_pe_oi = sum(s.get('put', {}).get('oi', 0) for s in selected_strikes if s.get('put'))
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        msg += f"\n📊 *PCR (Put/Call Ratio):* {pcr:.3f}\n"
        msg += f"_Range: {selected_strikes[0]['strike_price']:.0f} - {selected_strikes[-1]['strike_price']:.0f}_"
        
        return msg
        
    except Exception as e:
        logger.error(f"Error formatting option chain: {e}")
        return "❌ Error formatting option chain data"

# ======================== TELEGRAM FUNCTIONS ========================

async def send_telegram_message(message):
    """Send text message to Telegram"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Split long messages
        if len(message) > 4096:
            parts = [message[i:i+4096] for i in range(0, len(message), 4096)]
            for part in parts:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=part,
                    parse_mode='Markdown'
                )
                await asyncio.sleep(0.5)
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
        
        return True
        
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return False

async def send_telegram_photo(photo_buf, caption):
    """Send chart image to Telegram"""
    try:
        if not photo_buf:
            return False
        
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        photo_buf.seek(0)
        
        await bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID,
            photo=photo_buf,
            caption=caption,
            parse_mode='Markdown'
        )
        
        return True
        
    except TelegramError as e:
        logger.error(f"Telegram error sending photo: {e}")
        return False
    except Exception as e:
        logger.error(f"Error sending photo: {e}")
        return False

# ======================== MAIN BOT LOGIC ========================

async def process_symbol(symbol, instrument_key):
    """Process single symbol - fetch candles and send chart"""
    try:
        logger.info(f"Processing {symbol}...")
        
        # Fetch intraday candles
        candles = get_intraday_candles(instrument_key, "5minute")
        
        if not candles or len(candles) < 2:
            logger.warning(f"{symbol}: Insufficient candle data")
            return False
        
        logger.info(f"{symbol}: Got {len(candles)} candles")
        
        # Create chart
        chart = create_candlestick_chart(
            candles,
            f"{symbol} - 5 Minute Chart",
            show_volume=True
        )
        
        if not chart:
            logger.warning(f"{symbol}: Chart creation failed")
            return False
        
        # Send to Telegram
        caption = f"📊 *{symbol}*\n📈 {len(candles)} candles (5-min)\n⏰ {get_ist_now().strftime('%I:%M %p IST')}"
        success = await send_telegram_photo(chart, caption)
        
        if success:
            logger.info(f"✅ {symbol} chart sent successfully")
        
        return success
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return False

async def run_bot_cycle():
    """Main bot cycle - process all symbols"""
    try:
        logger.info("="*60)
        logger.info("🚀 UPSTOX OPTION CHAIN BOT - START")
        logger.info("="*60)
        
        # Market status
        market_status = "🟢 MARKET OPEN" if is_market_open() else "🔴 MARKET CLOSED"
        now = get_ist_now()
        
        welcome_msg = f"🎯 *Upstox Market Data Update*\n\n"
        welcome_msg += f"{market_status}\n"
        welcome_msg += f"⏰ {now.strftime('%d %b %Y, %I:%M:%S %p IST')}\n\n"
        welcome_msg += f"📊 Tracking {len(STOCKS_INDICES)} symbols"
        
        await send_telegram_message(welcome_msg)
        await asyncio.sleep(2)
        
        # Process symbols in batches
        symbols = list(STOCKS_INDICES.items())
        batch_size = 5
        successful_charts = 0
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"\n📦 Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            for symbol, instrument_key in batch:
                success = await process_symbol(symbol, instrument_key)
                if success:
                    successful_charts += 1
                
                # Rate limiting
                await asyncio.sleep(3)
            
            # Batch delay
            if i + batch_size < len(symbols):
                logger.info("⏳ Waiting 5 seconds before next batch...")
                await asyncio.sleep(5)
        
        # Option Chain (NIFTY 50)
        logger.info("\n📊 Fetching NIFTY 50 Option Chain...")
        oc_data = get_option_chain_data("NSE_INDEX|Nifty 50")
        
        if oc_data and oc_data.get('strikes'):
            logger.info(f"✅ Got option chain: {oc_data['total_strikes']} strikes")
            oc_message = format_option_chain_message(oc_data, "NIFTY 50")
            await send_telegram_message(oc_message)
        else:
            logger.warning("⚠️ Option chain data unavailable")
            await send_telegram_message("⚠️ NIFTY 50 Option Chain: Data unavailable")
        
        await asyncio.sleep(2)
        
        # Summary
        summary = f"\n✅ *Update Complete!*\n\n"
        summary += f"{market_status}\n"
        summary += f"📊 Charts sent: {successful_charts}/{len(STOCKS_INDICES)}\n"
        summary += f"📈 Option Chain: {'✅' if oc_data else '❌'}\n"
        summary += f"⏰ {get_ist_now().strftime('%I:%M:%S %p IST')}"
        
        await send_telegram_message(summary)
        
        logger.info("="*60)
        logger.info(f"✅ CYCLE COMPLETED! {successful_charts} charts sent")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in bot cycle: {e}")
        await send_telegram_message(f"❌ Bot cycle error: {str(e)[:200]}")

async def run_bot_loop():
    """Continuous loop - run every 5 minutes"""
    logger.info("🤖 Bot started! Running continuous loop...")
    
    while True:
        try:
            await run_bot_cycle()
            
            # Wait 5 minutes
            wait_seconds = 300
            logger.info(f"\n⏳ Waiting {wait_seconds//60} minutes for next cycle...\n")
            await asyncio.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

# ======================== STARTUP MESSAGE ========================

async def send_startup_message():
    """Send bot startup notification"""
    try:
        msg = "🤖 *UPSTOX OPTION CHAIN BOT STARTED!*\n\n"
        msg += f"📊 Tracking {len(STOCKS_INDICES)} stocks/indices\n"
        msg += "⏱️ Updates every 5 minutes\n\n"
        msg += "📈 *Features:*\n"
        msg += "  • Candlestick Charts (5-min)\n"
        msg += "  • Volume Analysis\n"
        msg += "  • Option Chain Data\n"
        msg += "  • CE/PE: LTP, OI, Volume\n"
        msg += "  • Greeks & IV (Delta, Theta)\n"
        msg += "  • PCR (Put-Call Ratio)\n\n"
        msg += "✅ *Powered by:*\n"
        msg += "  • Upstox API v2\n"
        msg += "  • Real-time Market Data\n\n"
        msg += f"🕐 Started at: {get_ist_now().strftime('%I:%M %p IST')}\n"
        msg += "_Market Hours: 9:15 AM - 3:30 PM (Mon-Fri)_"
        
        await send_telegram_message(msg)
        logger.info("✅ Startup message sent")
        
    except Exception as e:
        logger.error(f"Error sending startup message: {e}")

# ======================== ENTRYPOINT ========================

async def main():
    """Main entry point"""
    try:
        # Validate environment variables
        if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
            logger.error("❌ Missing environment variables!")
            logger.error("Required: UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
            return
        
        if UPSTOX_ACCESS_TOKEN == "your_access_token":
            logger.error("❌ Please set valid UPSTOX_ACCESS_TOKEN")
            return
        
        logger.info("✅ Environment variables validated")
        logger.info(f"📱 Telegram Chat ID: {TELEGRAM_CHAT_ID}")
        
        # Send startup notification
        await send_startup_message()
        await asyncio.sleep(2)
        
        # Start bot loop
        await run_bot_loop()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    logger.info("🔧 Starting Upstox Option Chain Bot...")
    logger.info("📦 Checking dependencies...")
    
    try:
        import pandas
        import matplotlib
        import pytz
        from telegram import Bot
        logger.info("✅ All dependencies loaded!")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Run: pip install -r requirements.txt")
        exit(1)
    
    # Run the bot
    asyncio.run(main())
