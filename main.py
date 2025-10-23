#!/usr/bin/env python3
# nifty50_all_options.py - Get option chain for ALL Nifty 50 stocks every 5 minutes

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta
import pytz
from telegram import Bot

# Config
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# Nifty 50 Stocks (instrument keys for Upstox)
NIFTY50_STOCKS = [
    "NSE_EQ|INE040A01034",  # HDFCBANK
    "NSE_EQ|INE002A01018",  # RELIANCE
    "NSE_EQ|INE467B01029",  # TATASTEEL
    "NSE_EQ|INE009A01021",  # INFY
    "NSE_EQ|INE256A01028",  # ZEEL
    "NSE_EQ|INE155A01022",  # TATAMOTORS
    "NSE_EQ|INE018A01030",  # HCLTECH
    "NSE_EQ|INE123W01016",  # WIPRO
    "NSE_EQ|INE021A01026",  # ASIANPAINT
    "NSE_EQ|INE854D01024",  # UNITECH
    "NSE_EQ|INE019A01038",  # AXISBANK
    "NSE_EQ|INE114A01011",  # BHARTIARTL
    "NSE_EQ|INE090A01021",  # ICICIBANK
    "NSE_EQ|INE081A01012",  # BAJFINANCE
    "NSE_EQ|INE397D01024",  # MARUTI
    "NSE_EQ|INE030A01027",  # SBIN
    "NSE_EQ|INE216A01030",  # KOTAKBANK
    "NSE_EQ|INE192A01025",  # SUNPHARMA
    "NSE_EQ|INE238A01034",  # TITAN
    "NSE_EQ|INE205A01025",  # ULTRACEMCO
    "NSE_EQ|INE752E01010",  # ADANIENT
    "NSE_EQ|INE423A01024",  # ADANIPORTS
    "NSE_EQ|INE848E01016",  # NYKAA
    "NSE_EQ|INE758T01015",  # TATACONSUM
    "NSE_EQ|INE001A01036",  # NESTLEIND
    "NSE_EQ|INE062A01020",  # POWERGRID
    "NSE_EQ|INE084A01016",  # LT
    "NSE_EQ|INE242A01010",  # ITC
    "NSE_EQ|INE860A01027",  # HUL
    "NSE_EQ|INE101D01020",  # TECHM
    "NSE_EQ|INE239A01016",  # HINDALCO
    "NSE_EQ|INE059A01026",  # COALINDIA
    "NSE_EQ|INE121A01024",  # INDUSINDBK
    "NSE_EQ|INE093A01033",  # ONGC
    "NSE_EQ|INE669E01016",  # DMART
    "NSE_EQ|INE628A01036",  # UPL
    "NSE_EQ|INE129A01019",  # GAIL
    "NSE_EQ|INE002B01027",  # M&M
    "NSE_EQ|INE070A01015",  # BAJAJFINSV
    "NSE_EQ|INE118A01012",  # SBILIFE
    "NSE_EQ|INE758E01017",  # BAJAJ-AUTO
    "NSE_EQ|INE009A09029",  # NTPC
    "NSE_EQ|INE220B01022",  # JSWSTEEL
    "NSE_EQ|INE467B01029",  # TATAPOWER
    "NSE_EQ|INE079A01024",  # CIPLA
    "NSE_EQ|INE154A01025",  # DRREDDY
    "NSE_EQ|INE066A01021",  # EICHERMOT
    "NSE_EQ|INE885A01032",  # BPCL
    "NSE_EQ|INE075A01022",  # DIVISLAB
    "NSE_EQ|INE523A01015",  # BRITANNIA
]

# Symbol mapping (you'll need to get proper symbols from Upstox API)
STOCK_NAMES = {
    "INE040A01034": "HDFCBANK",
    "INE002A01018": "RELIANCE",
    "INE009A01021": "INFY",
    "INE030A01027": "SBIN",
    "INE090A01021": "ICICIBANK",
    # Add more mappings
}

print("🚀 NIFTY 50 OPTIONS MONITOR")
print(f"📊 Tracking {len(NIFTY50_STOCKS)} stocks")
print(f"⏰ Updates every 5 minutes")
print(f"📱 Sending to Telegram: {TELEGRAM_CHAT_ID}")

def get_stock_symbol(instrument_key):
    """Extract stock symbol from instrument key"""
    isin = instrument_key.split('|')[1]
    return STOCK_NAMES.get(isin, isin[-10:])

def get_expiries(instrument_key):
    """Get expiries for a stock"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
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
        print(f"⚠️ Expiry fetch error: {e}")
    
    return []

def get_next_expiry(instrument_key):
    """Get next valid expiry"""
    expiries = get_expiries(instrument_key)
    if not expiries:
        # Fallback to next Thursday
        today = datetime.now(IST)
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    today = datetime.now(IST).date()
    future = [e for e in expiries if datetime.strptime(e, '%Y-%m-%d').date() >= today]
    
    if future:
        return min(future)
    return expiries[0]

def get_option_chain(instrument_key, expiry):
    """Get option chain data for a stock"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('data', [])
    except Exception as e:
        print(f"⚠️ Chain fetch error: {e}")
    
    return []

def get_spot_price(instrument_key):
    """Get spot price for a stock"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            quote_data = data.get('data', {})
            first_key = list(quote_data.keys())[0]
            return float(quote_data[first_key].get('last_price', 0))
    except Exception as e:
        print(f"⚠️ Spot price error: {e}")
    
    return 0

def format_compact_message(stock_data_list):
    """Format compact message for multiple stocks"""
    msg = f"📊 *NIFTY 50 OPTION CHAINS*\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}\n"
    msg += f"📈 {len(stock_data_list)} Stocks\n\n"
    
    for stock_data in stock_data_list[:10]:  # First 10 stocks per message
        symbol = stock_data['symbol']
        spot = stock_data['spot']
        expiry = stock_data['expiry']
        strikes = stock_data['strikes']
        
        if not strikes:
            continue
        
        # Find ATM
        atm_index = len(strikes) // 2
        if spot:
            atm_index = min(range(len(strikes)), 
                           key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
        
        # Get ATM strike data
        atm_strike = strikes[atm_index]
        strike_price = atm_strike.get('strike_price', 0)
        
        call = atm_strike.get('call_options', {}).get('market_data', {})
        put = atm_strike.get('put_options', {}).get('market_data', {})
        
        ce_ltp = call.get('ltp', 0)
        pe_ltp = put.get('ltp', 0)
        
        # Calculate PCR
        total_ce_oi = sum(s.get('call_options', {}).get('market_data', {}).get('oi', 0) for s in strikes)
        total_pe_oi = sum(s.get('put_options', {}).get('market_data', {}).get('oi', 0) for s in strikes)
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        msg += f"*{symbol}* ₹{spot:.2f}\n"
        msg += f"  ATM {strike_price:.0f} | CE:{ce_ltp:.2f} PE:{pe_ltp:.2f} | PCR:{pcr:.2f}\n\n"
    
    return msg

def format_detailed_message(stock_data):
    """Format detailed message for single stock"""
    symbol = stock_data['symbol']
    spot = stock_data['spot']
    expiry = stock_data['expiry']
    strikes = stock_data['strikes']
    
    if not strikes:
        return f"❌ *{symbol}* - No data available"
    
    # Find ATM
    atm_index = len(strikes) // 2
    if spot:
        atm_index = min(range(len(strikes)), 
                       key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    
    # Select 5 strikes around ATM
    start = max(0, atm_index - 2)
    end = min(len(strikes), atm_index + 3)
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol} OPTION CHAIN*\n\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.0f}\n\n"
    
    msg += "```\n"
    msg += "Strike   CE-LTP  PE-LTP\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for i, s in enumerate(selected):
        is_atm = (start + i == atm_index)
        mark = "🔸" if is_atm else "  "
        
        strike = s.get('strike_price', 0)
        
        call = s.get('call_options', {}).get('market_data', {})
        put = s.get('put_options', {}).get('market_data', {})
        
        ce_ltp = call.get('ltp', 0)
        pe_ltp = put.get('ltp', 0)
        
        msg += f"{mark}{strike:7.0f} {ce_ltp:7.2f} {pe_ltp:7.2f}\n"
    
    msg += "```\n"
    
    return msg

async def send_telegram(msg):
    """Send message to Telegram"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=msg, 
            parse_mode='Markdown'
        )
        print("✅ Message sent!")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

async def fetch_all_stocks_data():
    """Fetch option chain data for all Nifty 50 stocks"""
    print("\n" + "="*60)
    print(f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}")
    print("📊 Fetching data for all stocks...")
    
    stock_data_list = []
    
    for idx, instrument_key in enumerate(NIFTY50_STOCKS, 1):
        symbol = get_stock_symbol(instrument_key)
        print(f"[{idx}/{len(NIFTY50_STOCKS)}] Fetching {symbol}...")
        
        try:
            # Get expiry
            expiry = get_next_expiry(instrument_key)
            
            # Get spot price
            spot = get_spot_price(instrument_key)
            
            # Get option chain
            strikes = get_option_chain(instrument_key, expiry)
            
            if strikes:
                stock_data_list.append({
                    'symbol': symbol,
                    'spot': spot,
                    'expiry': expiry,
                    'strikes': strikes
                })
                print(f"  ✅ {symbol}: ₹{spot:.2f} | {len(strikes)} strikes")
            else:
                print(f"  ⚠️ {symbol}: No option data")
            
            # Rate limiting
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
            continue
    
    print(f"\n📊 Successfully fetched {len(stock_data_list)} stocks")
    return stock_data_list

async def send_batch_messages(stock_data_list):
    """Send data in batches to Telegram"""
    # Send compact summary
    if stock_data_list:
        # Split into batches of 10 stocks
        batch_size = 10
        for i in range(0, len(stock_data_list), batch_size):
            batch = stock_data_list[i:i+batch_size]
            msg = format_compact_message(batch)
            await send_telegram(msg)
            await asyncio.sleep(2)  # Avoid rate limits

async def monitoring_loop():
    """Main monitoring loop - runs every 5 minutes"""
    print("\n🔄 Starting monitoring loop...")
    
    while True:
        try:
            # Fetch all data
            stock_data_list = await fetch_all_stocks_data()
            
            # Send to Telegram
            if stock_data_list:
                await send_batch_messages(stock_data_list)
                print("✅ Batch sent to Telegram!")
            else:
                await send_telegram("⚠️ No option chain data available")
            
            # Wait 5 minutes
            print("\n⏳ Waiting 5 minutes for next update...")
            await asyncio.sleep(300)  # 5 minutes = 300 seconds
            
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
            print("⏳ Retrying in 1 minute...")
            await asyncio.sleep(60)

async def main():
    print("\n" + "="*60)
    print("🚀 STARTING NIFTY 50 OPTIONS MONITOR")
    print("="*60)
    
    # Run monitoring loop
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
