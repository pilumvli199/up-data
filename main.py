#!/usr/bin/env python3
# nifty50_all_options.py - Complete option chain with 11 strikes (ATM ±5)

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

# Nifty 50 Stocks - ONLY stocks with active options
NIFTY50_STOCKS = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE090A01021": "ICICIBANK",
    "NSE_EQ|INE030A01027": "SBIN",
    "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE467B01029": "TATASTEEL",
    "NSE_EQ|INE155A01022": "TATAMOTORS",
    "NSE_EQ|INE018A01030": "HCLTECH",
    "NSE_EQ|INE019A01038": "AXISBANK",
    "NSE_EQ|INE114A01011": "BHARTIARTL",
    "NSE_EQ|INE397D01024": "MARUTI",
    "NSE_EQ|INE216A01030": "KOTAKBANK",
    "NSE_EQ|INE192A01025": "SUNPHARMA",
    "NSE_EQ|INE238A01034": "TITAN",
    "NSE_EQ|INE205A01025": "ULTRACEMCO",
    "NSE_EQ|INE423A01024": "ADANIPORTS",
    "NSE_EQ|INE752E01010": "ADANIENT",
    "NSE_EQ|INE758T01015": "TATACONSUM",
    "NSE_EQ|INE062A01020": "POWERGRID",
    "NSE_EQ|INE084A01016": "LT",
    "NSE_EQ|INE242A01010": "ITC",
    "NSE_EQ|INE860A01027": "HINDUNILVR",
    "NSE_EQ|INE059A01026": "COALINDIA",
    "NSE_EQ|INE121A01024": "INDUSINDBK",
    "NSE_EQ|INE628A01036": "UPL",
    "NSE_EQ|INE129A01019": "GAIL",
    "NSE_EQ|INE070A01015": "BAJAJFINSV",
    "NSE_EQ|INE758E01017": "BAJAJAUTO",
    "NSE_EQ|INE079A01024": "CIPLA",
    "NSE_EQ|INE154A01025": "DRREDDY",
    "NSE_EQ|INE066A01021": "EICHERMOT",
    "NSE_EQ|INE075A01022": "DIVISLAB",
    "NSE_EQ|INE021A01026": "ASIANPAINT",
    "NSE_EQ|INE848E01016": "NYKAA",
    "NSE_EQ|INE669E01016": "DMART",
}

print("🚀 NIFTY 50 OPTIONS MONITOR")
print(f"📊 Tracking {len(NIFTY50_STOCKS)} stocks")
print(f"⏰ Updates every 5 minutes")
print(f"📱 Sending to Telegram: {TELEGRAM_CHAT_ID}")

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
            if quote_data:
                first_key = list(quote_data.keys())[0]
                ltp = quote_data[first_key].get('last_price', 0)
                return float(ltp) if ltp else 0
    except Exception as e:
        print(f"⚠️ Spot price error: {e}")
    
    return 0

def format_detailed_message(symbol, spot, expiry, strikes):
    """Format detailed message with 11 strikes (ATM ±5) and full data"""
    if not strikes or len(strikes) < 11:
        return None
    
    # Find ATM strike
    atm_index = len(strikes) // 2
    if spot > 0:
        atm_index = min(range(len(strikes)), 
                       key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    
    # Select 11 strikes: ATM + 5 up + 5 down
    start = max(0, atm_index - 5)
    end = min(len(strikes), atm_index + 6)
    
    # Adjust if we don't have enough strikes on either side
    if end - start < 11:
        if start == 0:
            end = min(11, len(strikes))
        else:
            start = max(0, len(strikes) - 11)
    
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol}*\n\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.2f}\n\n"
    
    msg += "```\n"
    msg += "Strike    CE-LTP  CE-Vol   CE-OI    PE-LTP  PE-Vol   PE-OI\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_oi = 0
    total_pe_oi = 0
    total_ce_vol = 0
    total_pe_vol = 0
    
    for i, s in enumerate(selected):
        is_atm = (start + i == atm_index)
        mark = "🔸" if is_atm else "  "
        
        strike = s.get('strike_price', 0)
        
        # Call data
        call = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call.get('ltp', 0)
        ce_vol = call.get('volume', 0)
        ce_oi = call.get('oi', 0)
        
        # Put data
        put = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put.get('ltp', 0)
        pe_vol = put.get('volume', 0)
        pe_oi = put.get('oi', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol
        
        # Format with K for thousands
        ce_vol_k = ce_vol / 1000 if ce_vol > 0 else 0
        ce_oi_k = ce_oi / 1000 if ce_oi > 0 else 0
        pe_vol_k = pe_vol / 1000 if pe_vol > 0 else 0
        pe_oi_k = pe_oi / 1000 if pe_oi > 0 else 0
        
        msg += f"{mark}{strike:8.2f} {ce_ltp:7.2f} {ce_vol_k:7.1f}K {ce_oi_k:7.1f}K {pe_ltp:7.2f} {pe_vol_k:7.1f}K {pe_oi_k:7.1f}K\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    # Totals
    total_ce_vol_k = total_ce_vol / 1000
    total_ce_oi_k = total_ce_oi / 1000
    total_pe_vol_k = total_pe_vol / 1000
    total_pe_oi_k = total_pe_oi / 1000
    
    msg += f"TOTAL          {total_ce_vol_k:7.1f}K {total_ce_oi_k:7.1f}K        {total_pe_vol_k:7.1f}K {total_pe_oi_k:7.1f}K\n"
    msg += "```\n\n"
    
    # PCR Calculation
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    
    msg += f"📊 *PCR (OI):* {pcr:.3f}\n"
    msg += f"📊 *PCR (Vol):* {pcr_vol:.3f}\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}\n"
    
    return msg

async def send_telegram(msg):
    """Send message to Telegram"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        # Split long messages if needed (Telegram limit: 4096 chars)
        if len(msg) > 4000:
            # Send in chunks
            chunks = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
            for chunk in chunks:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID, 
                    text=chunk, 
                    parse_mode='Markdown'
                )
                await asyncio.sleep(1)
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=msg, 
                parse_mode='Markdown'
            )
        return True
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

async def fetch_and_send_stock(instrument_key, symbol, idx, total):
    """Fetch and send option chain for a single stock"""
    print(f"[{idx}/{total}] Fetching {symbol}...")
    
    try:
        # Get expiry
        expiry = get_next_expiry(instrument_key)
        
        # Get spot price
        spot = get_spot_price(instrument_key)
        
        if spot == 0:
            print(f"  ⚠️ {symbol}: Invalid spot price")
            return False
        
        # Get option chain
        strikes = get_option_chain(instrument_key, expiry)
        
        if not strikes or len(strikes) < 11:
            print(f"  ⚠️ {symbol}: Insufficient strikes ({len(strikes)})")
            return False
        
        print(f"  ✅ {symbol}: ₹{spot:.2f} | {len(strikes)} strikes")
        
        # Format message
        msg = format_detailed_message(symbol, spot, expiry, strikes)
        
        if msg:
            # Send to Telegram
            success = await send_telegram(msg)
            if success:
                print(f"  📤 {symbol}: Sent to Telegram!")
                return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return False

async def fetch_all_stocks():
    """Fetch and send option chain for all stocks"""
    print("\n" + "="*60)
    print(f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}")
    print("📊 Fetching option chains for all stocks...")
    print("="*60 + "\n")
    
    # Send header message
    header_msg = f"🚀 *NIFTY 50 OPTION CHAINS - UPDATE*\n"
    header_msg += f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}\n"
    header_msg += f"📊 {len(NIFTY50_STOCKS)} Stocks\n"
    header_msg += f"📈 11 Strikes per stock (ATM ±5)\n\n"
    header_msg += f"_Starting data fetch..._"
    
    await send_telegram(header_msg)
    
    success_count = 0
    total = len(NIFTY50_STOCKS)
    
    for idx, (instrument_key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        success = await fetch_and_send_stock(instrument_key, symbol, idx, total)
        
        if success:
            success_count += 1
        
        # Rate limiting - wait between requests
        await asyncio.sleep(2)
    
    # Send summary
    summary_msg = f"\n✅ *UPDATE COMPLETE*\n"
    summary_msg += f"📊 Successfully sent: {success_count}/{total} stocks\n"
    summary_msg += f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}"
    
    await send_telegram(summary_msg)
    
    print("\n" + "="*60)
    print(f"✅ Sent {success_count}/{total} stocks to Telegram")
    print("="*60)

async def monitoring_loop():
    """Main monitoring loop - runs every 5 minutes"""
    print("\n🔄 Starting monitoring loop...")
    print("🔄 Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Fetch and send all stocks
            await fetch_all_stocks()
            
            # Wait 5 minutes
            print(f"\n⏳ Next update in 5 minutes...")
            print(f"⏳ Next run at: {(datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p IST')}\n")
            
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
    print("🚀 NIFTY 50 OPTIONS MONITOR - FULL DATA")
    print("="*60)
    print(f"📊 11 Strikes per stock (ATM + 5 up + 5 down)")
    print(f"📈 Complete Option Chain: LTP, Volume, OI, PCR")
    print(f"⏰ Updates every 5 minutes")
    print(f"📱 Telegram: {TELEGRAM_CHAT_ID}")
    print("="*60)
    
    # Run monitoring loop
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
