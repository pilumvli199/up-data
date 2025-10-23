#!/usr/bin/env python3
# quick_test.py - Send option chain DIRECTLY to Telegram (NO charts)

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

print("🚀 QUICK OPTION CHAIN TEST")
print(f"Token: {UPSTOX_ACCESS_TOKEN[:20]}...")
print(f"Chat ID: {TELEGRAM_CHAT_ID}")

def get_expiries():
    """Get expiries from contracts"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    instrument_key = "NSE_INDEX|Nifty 50"
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
    
    print(f"📡 Fetching contracts...")
    resp = requests.get(url, headers=headers, timeout=15)
    
    if resp.status_code == 200:
        data = resp.json()
        contracts = data.get('data', [])
        
        expiries = set()
        for c in contracts:
            if 'expiry' in c:
                expiries.add(c['expiry'])
        
        expiries_list = sorted(list(expiries))
        print(f"✅ Found {len(expiries_list)} expiries")
        return expiries_list
    
    return []

def get_next_expiry():
    """Get next valid expiry"""
    expiries = get_expiries()
    if not expiries:
        # Fallback
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

def get_option_chain(expiry):
    """Get option chain data"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    instrument_key = "NSE_INDEX|Nifty 50"
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
    
    print(f"📊 Fetching option chain for {expiry}...")
    resp = requests.get(url, headers=headers, timeout=20)
    
    if resp.status_code == 200:
        data = resp.json()
        strikes = data.get('data', [])
        print(f"✅ Got {len(strikes)} strikes")
        return strikes
    
    print(f"❌ HTTP {resp.status_code}")
    return []

def get_spot_price():
    """Get spot price"""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }
    
    instrument_key = "NSE_INDEX|Nifty 50"
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
    
    resp = requests.get(url, headers=headers, timeout=10)
    
    if resp.status_code == 200:
        data = resp.json()
        quote_data = data.get('data', {})
        first_key = list(quote_data.keys())[0]
        return float(quote_data[first_key].get('last_price', 0))
    
    return 0

def format_message(strikes, expiry, spot):
    """Format option chain message"""
    if not strikes:
        return "❌ No strikes data"
    
    # Find ATM
    atm_index = len(strikes) // 2
    if spot:
        atm_index = min(range(len(strikes)), 
                       key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    
    # Select strikes around ATM
    start = max(0, atm_index - 5)
    end = min(len(strikes), atm_index + 6)
    selected = strikes[start:end]
    
    msg = f"📊 *NIFTY 50 OPTION CHAIN*\n\n"
    msg += f"📅 Expiry: {expiry}\n"
    msg += f"💰 Spot: ₹{spot:,.2f}\n"
    msg += f"🎯 ATM: ₹{strikes[atm_index].get('strike_price', 0):,.0f}\n\n"
    
    msg += "```\n"
    msg += "Strike    CE-LTP  CE-OI   PE-LTP  PE-OI\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for i, s in enumerate(selected):
        is_atm = (start + i == atm_index)
        mark = "🔸" if is_atm else "  "
        
        strike = s.get('strike_price', 0)
        
        call = s.get('call_options', {}).get('market_data', {})
        put = s.get('put_options', {}).get('market_data', {})
        
        ce_ltp = call.get('ltp', 0)
        ce_oi = call.get('oi', 0) / 1000
        
        pe_ltp = put.get('ltp', 0)
        pe_oi = put.get('oi', 0) / 1000
        
        msg += f"{mark}{strike:7.0f}  {ce_ltp:7.2f} {ce_oi:7.1f}K {pe_ltp:7.2f} {pe_oi:7.1f}K\n"
    
    msg += "```\n\n"
    
    # PCR
    total_ce_oi = sum(s.get('call_options', {}).get('market_data', {}).get('oi', 0) for s in selected)
    total_pe_oi = sum(s.get('put_options', {}).get('market_data', {}).get('oi', 0) for s in selected)
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    msg += f"📊 *PCR:* {pcr:.3f}\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}"
    
    return msg

async def send_telegram(msg):
    """Send to Telegram"""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
    print("✅ Message sent to Telegram!")

async def main():
    print("\n" + "="*60)
    
    # Step 1: Get expiry
    expiry = get_next_expiry()
    print(f"📅 Using expiry: {expiry}")
    
    # Step 2: Get spot
    spot = get_spot_price()
    print(f"💰 Spot: ₹{spot:,.2f}")
    
    # Step 3: Get option chain
    strikes = get_option_chain(expiry)
    
    if not strikes:
        print("❌ No strikes data!")
        await send_telegram("❌ Option chain fetch failed!")
        return
    
    # Step 4: Format message
    print(f"📝 Formatting message...")
    msg = format_message(strikes, expiry, spot)
    
    # Step 5: Send to Telegram
    print(f"📤 Sending to Telegram...")
    await send_telegram(msg)
    
    print("="*60)
    print("✅ COMPLETE!")

if __name__ == "__main__":
    asyncio.run(main())
