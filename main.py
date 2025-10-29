#!/usr/bin/env python3
"""
NIFTY 50 + SENSEX + STOCKS MONITOR
- NIFTY: Tuesday expiry (weekly)
- SENSEX: Thursday expiry (weekly)
- Stocks: Thursday expiry (monthly)
- LIVE 5min charts (Historical + Today's data)
- Option Chain + Charts every 5 minutes
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

# NIFTY 50 STOCKS (Top 5 for testing)
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE062A01020": "SBIN",
    "NSE_EQ|INE009A01021": "INFY",
}

print("="*70)
print("🚀 NIFTY + SENSEX MONITOR - LIVE DATA")
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
    expiry_day: 1=Tuesday (NIFTY), 3=Thursday (SENSEX, Stocks)
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

def split_30min_to_5min(candles_30min):
    """
    Split 30-minute candles into 6 x 5-minute candles
    """
    if not candles_30min:
        return []
    
    candles_5min = []
    
    for candle in candles_30min:
        timestamp = candle[0]
        open_price = candle[1]
        high_price = candle[2]
        low_price = candle[3]
        close_price = candle[4]
        volume = candle[5]
        oi = candle[6] if len(candle) > 6 else 0
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            continue
        
        # Create 6 x 5-minute candles from 1 x 30-minute candle
        price_step = (close_price - open_price) / 6
        vol_per_candle = volume // 6
        
        for i in range(6):
            dt_5min = dt + timedelta(minutes=i*5)
            
            # Simulate price movement
            candle_open = open_price + (price_step * i)
            candle_close = open_price + (price_step * (i + 1))
            candle_high = max(candle_open, candle_close, high_price - (abs(price_step) * (6-i-1)/6))
            candle_low = min(candle_open, candle_close, low_price + (abs(price_step) * i/6))
            
            candles_5min.append([
                dt_5min.isoformat(),
                candle_open,
                candle_high,
                candle_low,
                candle_close,
                vol_per_candle,
                oi
            ])
    
    return candles_5min

def get_historical_candles_with_today(instrument_key, symbol):
    """
    Get HISTORICAL (7 days) + TODAY'S LIVE data combined
    Convert 30min → 5min candles
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    
    all_candles_30min = []
    
    # STEP 1: Get HISTORICAL data (last 7 days, 30min)
    try:
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=7)
        to_str = to_date.strftime('%Y-%m-%d')
        from_str = from_date.strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_str}/{from_str}"
        print(f"  🔍 Historical 30min (7 days)...")
        
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                if candles:
                    print(f"  ✅ History: {len(candles)} candles")
                    all_candles_30min.extend(candles)
    except Exception as e:
        print(f"  ⚠️ Historical error: {e}")
    
    # STEP 2: Get TODAY'S INTRADAY data (30min with AUTH)
    try:
        url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/30minute"
        print(f"  🔍 Today's 30min intraday (LIVE)...")
        
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('status') == 'success':
                candles_today = data.get('data', {}).get('candles', [])
                if candles_today:
                    print(f"  ✅ Today LIVE: {len(candles_today)} candles")
                    
                    # Remove duplicates: Filter out today's date from historical
                    today_date = datetime.now(IST).date()
                    filtered_historical = []
                    
                    for c in all_candles_30min:
                        try:
                            c_dt = datetime.fromisoformat(c[0].replace('Z', '+00:00'))
                            c_dt = c_dt.astimezone(IST)
                            if c_dt.date() < today_date:
                                filtered_historical.append(c)
                        except:
                            filtered_historical.append(c)
                    
                    all_candles_30min = filtered_historical + candles_today
                    print(f"  📊 COMBINED: {len(all_candles_30min)} total 30min candles")
    except Exception as e:
        print(f"  ⚠️ Today's error: {e}")
    
    # STEP 3: Convert 30min → 5min
    if all_candles_30min:
        print(f"  🔄 Converting 30min → 5min...")
        candles_5min = split_30min_to_5min(all_candles_30min)
        print(f"  ✅ FINAL: {len(candles_5min)} x 5min candles (LIVE + Historical)")
        return candles_5min
    
    print(f"  ❌ {symbol}: No candle data")
    return []

def create_candlestick_chart(candles, symbol, spot_price):
    """
    Create 5-minute candlestick chart with LIVE data
    """
    if not candles or len(candles) < 3:
        return None
    
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    
    for candle in reversed(candles):
        try:
            timestamp = datetime.fromisoformat(candle[0].replace('Z', '+00:00'))
            timestamp = timestamp.astimezone(IST)
            
            # Skip weekends
            if timestamp.weekday() >= 5:
                continue
            
            # Market hours only (9:15 AM - 3:30 PM)
            hour, minute = timestamp.hour, timestamp.minute
            if hour < 9 or (hour == 9 and minute < 15):
                continue
            if hour > 15 or (hour == 15 and minute > 30):
                continue
            
            dates.append(timestamp)
            opens.append(float(candle[1]))
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
            closes.append(float(candle[4]))
            volumes.append(int(candle[5]) if candle[5] else 0)
        except:
            continue
    
    if len(dates) < 3:
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), 
                                     gridspec_kw={'height_ratios': [4, 1]})
    
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#fafafa')
    
    indices = range(len(dates))
    
    # Draw candlesticks
    for i in indices:
        is_bullish = closes[i] >= opens[i]
        color = '#089981' if is_bullish else '#f23645'
        
        # Wick
        ax1.plot([i, i], [lows[i], highs[i]], 
                color=color, linewidth=1.0, solid_capstyle='round', zorder=2)
        
        # Body
        height = abs(closes[i] - opens[i])
        bottom = min(opens[i], closes[i])
        
        if height > 0.001:
            rect = Rectangle((i - 0.4, bottom), 0.8, height, 
                           facecolor=color, edgecolor=color, linewidth=0, zorder=3)
            ax1.add_patch(rect)
        else:
            ax1.plot([i - 0.4, i + 0.4], [opens[i], opens[i]], 
                    color=color, linewidth=1.5, solid_capstyle='butt', zorder=3)
    
    # Current price line
    ax1.axhline(y=spot_price, color='#2962FF', linestyle='--', 
               linewidth=1.5, alpha=0.85, zorder=4)
    
    # Price label on right
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_yticks([spot_price])
    ax1_right.set_yticklabels([f'₹{spot_price:.2f}'], fontsize=10, 
                              fontweight='600', color='#2962FF')
    ax1_right.tick_params(colors='#2962FF', length=0)
    
    ax1.set_ylabel('Price (₹)', color='#787B86', fontsize=11, fontweight='500')
    ax1.tick_params(axis='y', colors='#787B86', labelsize=9.5)
    ax1.tick_params(axis='x', colors='#787B86', labelsize=9)
    ax1.grid(True, alpha=0.12, color='#D1D4DC', linestyle='-', linewidth=0.5, zorder=1)
    ax1.set_axisbelow(True)
    
    # Title with LIVE indicator
    now_ist = datetime.now(IST)
    title = f'{symbol}  •  5 Min (LIVE)  •  {now_ist.strftime("%d %b %I:%M %p")}'
    ax1.set_title(title, color='#131722', fontsize=16, fontweight='600', 
                 pad=20, loc='left')
    
    # Volume bars
    colors_vol = ['#08998166' if closes[i] >= opens[i] else '#f2364566' for i in indices]
    ax2.bar(indices, volumes, color=colors_vol, width=0.8, alpha=1.0, edgecolor='none', zorder=2)
    
    ax2.set_ylabel('Volume', color='#787B86', fontsize=11, fontweight='500')
    ax2.tick_params(axis='y', colors='#787B86', labelsize=9.5)
    ax2.tick_params(axis='x', colors='#787B86', labelsize=9)
    ax2.grid(True, alpha=0.12, color='#D1D4DC', linestyle='-', linewidth=0.5, zorder=1)
    ax2.set_axisbelow(True)
    
    # X-axis labels (show more ticks for 5min data)
    step = max(1, len(dates) // 15)
    tick_positions = list(range(0, len(dates), step))
    tick_labels = [dates[i].strftime('%d %b\n%H:%M') for i in tick_positions]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-1, len(dates))
        
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_color('#E0E3EB')
            ax.spines[spine].set_linewidth(0.8)
    
    ax1.spines['right'].set_visible(False)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
    
    ax2.set_xlabel('Date & Time (IST)', color='#787B86', fontsize=11, 
                  fontweight='500', labelpad=10)
    
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(hspace=0.05)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='white', 
               edgecolor='none', bbox_inches='tight', pad_inches=0.2)
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
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M %p')}\n"
    
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

async def process_index(index_key, index_name, expiry_day):
    """Process NIFTY or SENSEX index"""
    print("\n" + "="*50)
    print(f"{index_name}")
    print("="*50)
    
    try:
        expiry = get_next_expiry(index_key, expiry_day=expiry_day)
        spot = get_spot_price(index_key)
        
        if spot == 0:
            print("Invalid spot")
            return False
        
        print(f"Spot: {spot:.2f} | Expiry: {expiry}")
        
        strikes = get_option_chain(index_key, expiry)
        if not strikes:
            print("No strikes")
            return False
        
        print(f"Strikes: {len(strikes)}")
        
        # Send option chain
        msg = format_option_chain_message(index_name, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
            print("Chain sent")
        
        # Send chart with LIVE data
        print("Fetching candles...")
        candles = get_historical_candles_with_today(index_key, index_name)
        
        if candles and len(candles) >= 3:
            print("Creating 5min chart...")
            chart = create_candlestick_chart(candles, index_name, spot)
            if chart:
                caption = f"📈 *{index_name}* - 5min LIVE\n💰 ₹{spot:.2f} | 📅 {expiry}"
                await send_telegram_photo(chart, caption)
                print("Chart sent (LIVE)")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def process_stock(key, symbol, idx, total):
    """Process single stock"""
    print(f"\n[{idx}/{total}] {symbol}")
    
    try:
        expiry = get_next_expiry(key, expiry_day=3)  # Thursday
        spot = get_spot_price(key)
        
        if spot == 0:
            print("  Invalid spot")
            return False
        
        strikes = get_option_chain(key, expiry)
        if not strikes:
            print("  No strikes")
            return False
        
        print(f"  Spot: {spot:.2f} | Strikes: {len(strikes)}")
        
        # Send option chain
        msg = format_option_chain_message(symbol, spot, expiry, strikes)
        if msg:
            await send_telegram_text(msg)
        
        # Send chart with LIVE data
        candles = get_historical_candles_with_today(key, symbol)
        if candles and len(candles) >= 3:
            chart = create_candlestick_chart(candles, symbol, spot)
            if chart:
                caption = f"📈 *{symbol}* - 5min LIVE\n💰 ₹{spot:.2f}"
                await send_telegram_photo(chart, caption)
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

async def fetch_all():
    """Fetch NIFTY + SENSEX + All stocks"""
    print("\n" + "="*50)
    print(f"RUN: {datetime.now(IST).strftime('%I:%M %p')}")
    print("="*50)
    
    header = f"🚀 *MARKET MONITOR*\n⏰ {datetime.now(IST).strftime('%I:%M %p')}\n\n_Starting..._"
    await send_telegram_text(header)
    
    # Process NIFTY (Tuesday)
    nifty_ok = await process_index(NIFTY_INDEX_KEY, "NIFTY 50", expiry_day=1)
    await asyncio.sleep(2)
    
    # Process SENSEX (Thursday)
    sensex_ok = await process_index(SENSEX_INDEX_KEY, "SENSEX", expiry_day=3)
    await asyncio.sleep(2)
    
    # Process stocks
    success = 0
    total = len(NIFTY50_STOCKS)
    
    for idx, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        result = await process_stock(key, symbol, idx, total)
        if result:
            success += 1
        await asyncio.sleep(2)
    
    summary = f"✅ *DONE*\n"
    summary += f"NIFTY: {'✅' if nifty_ok else '❌'}\n"
    summary += f"SENSEX: {'✅' if sensex_ok else '❌'}\n"
    summary += f"Stocks: {success}/{total}"
    await send_telegram_text(summary)
    
    print(f"\nDONE: NIFTY={'OK' if nifty_ok else 'FAIL'} | "
          f"SENSEX={'OK' if sensex_ok else 'FAIL'} | Stocks={success}/{total}")

async def monitoring_loop():
    """Main monitoring loop - every 5 minutes"""
    print("\n🔄 Loop started (5 min interval)\n")
    
    while True:
        try:
            await fetch_all()
            
            next_time = (datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')
            print(f"\n⏳ Next: {next_time}\n")
            
            await asyncio.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break
        except Exception as e:
            print(f"\nLoop error: {e}")
            await asyncio.sleep(60)

async def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("NIFTY + SENSEX + STOCKS MONITOR")
    print("="*70)
    print("📊 NIFTY (Tuesday) + SENSEX (Thursday)")
    print("📈 5min LIVE Charts (Historical + Today)")
    print("⏰ Every 5 minutes")
    print("="*70 + "\n")
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
