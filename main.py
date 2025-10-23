#!/usr/bin/env python3
# test_option_chain.py - QUICK TEST SCRIPT
# Just tests option chain fetching

import os
import requests
import urllib.parse
from datetime import datetime, timedelta
import pytz

# Config
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

def log(msg):
    print(f"[{datetime.now(IST).strftime('%H:%M:%S')}] {msg}")

def get_next_thursday():
    """Fallback expiry calculation"""
    today = datetime.now(IST)
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = today + timedelta(days=days_ahead)
    return next_thursday.strftime('%Y-%m-%d')

def test_contracts_api():
    """Test 1: Can we fetch option contracts?"""
    log("="*60)
    log("TEST 1: Fetching option contracts")
    log("="*60)
    
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }
        
        instrument_key = "NSE_INDEX|Nifty 50"
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
        
        log(f"URL: {url}")
        log("Making request...")
        
        resp = requests.get(url, headers=headers, timeout=15)
        
        log(f"Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            log(f"✅ SUCCESS!")
            log(f"Response type: {type(data)}")
            log(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            if isinstance(data, dict) and 'data' in data:
                contracts = data['data']
                log(f"Number of contracts: {len(contracts)}")
                
                if contracts:
                    # Sample first contract
                    sample = contracts[0]
                    log(f"Sample contract keys: {list(sample.keys())}")
                    log(f"Sample contract: {sample}")
                    
                    # Extract expiries
                    expiries = set()
                    for c in contracts:
                        if 'expiry' in c:
                            expiries.add(c['expiry'])
                    
                    expiries_list = sorted(list(expiries))
                    log(f"✅ Found {len(expiries_list)} expiries:")
                    log(f"   {expiries_list[:10]}")
                    
                    return expiries_list
                else:
                    log("❌ No contracts in data")
            else:
                log("❌ No 'data' key in response")
        else:
            log(f"❌ HTTP Error: {resp.status_code}")
            log(f"Response: {resp.text[:500]}")
        
        return []
        
    except Exception as e:
        log(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def test_option_chain_api(expiry):
    """Test 2: Can we fetch option chain for expiry?"""
    log("="*60)
    log(f"TEST 2: Fetching option chain for expiry: {expiry}")
    log("="*60)
    
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }
        
        instrument_key = "NSE_INDEX|Nifty 50"
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
        
        log(f"URL: {url}")
        log("Making request...")
        
        resp = requests.get(url, headers=headers, timeout=20)
        
        log(f"Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            log(f"✅ SUCCESS!")
            log(f"Response type: {type(data)}")
            log(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            if isinstance(data, dict) and 'data' in data:
                payload = data['data']
                log(f"Payload type: {type(payload)}")
                
                if isinstance(payload, list):
                    log(f"Number of strikes: {len(payload)}")
                    if payload:
                        sample = payload[0]
                        log(f"Sample strike keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                        log(f"Sample strike: {sample}")
                        log(f"✅ Option chain has {len(payload)} strikes!")
                        return True
                    else:
                        log("❌ Empty strikes list")
                elif isinstance(payload, dict):
                    log(f"Payload dict keys: {list(payload.keys())}")
                else:
                    log(f"❌ Unexpected payload type: {type(payload)}")
            else:
                log("❌ No 'data' key in response")
        else:
            log(f"❌ HTTP Error: {resp.status_code}")
            log(f"Response: {resp.text[:500]}")
        
        return False
        
    except Exception as e:
        log(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_spot_price():
    """Test 3: Can we get spot price?"""
    log("="*60)
    log("TEST 3: Fetching spot price")
    log("="*60)
    
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
        }
        
        instrument_key = "NSE_INDEX|Nifty 50"
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
        
        log(f"URL: {url}")
        log("Making request...")
        
        resp = requests.get(url, headers=headers, timeout=10)
        
        log(f"Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            log(f"✅ SUCCESS!")
            log(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            if 'data' in data:
                quote_data = data['data']
                log(f"Quote data keys: {list(quote_data.keys())}")
                
                first_key = list(quote_data.keys())[0]
                quote = quote_data[first_key]
                
                ltp = quote.get('last_price', quote.get('ltp'))
                log(f"✅ Spot Price: ₹{ltp}")
                return ltp
        else:
            log(f"❌ HTTP Error: {resp.status_code}")
            log(f"Response: {resp.text[:500]}")
        
        return None
        
    except Exception as e:
        log(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    log("🔧 OPTION CHAIN TESTING SCRIPT")
    log(f"Token: {UPSTOX_ACCESS_TOKEN[:20]}..." if UPSTOX_ACCESS_TOKEN else "❌ NO TOKEN")
    log("")
    
    if not UPSTOX_ACCESS_TOKEN:
        log("❌ UPSTOX_ACCESS_TOKEN not set!")
        return
    
    # Test 1: Get contracts and expiries
    expiries = test_contracts_api()
    log("")
    
    if not expiries:
        log("⚠️ No expiries found, trying fallback...")
        expiries = [get_next_thursday()]
        log(f"Using fallback expiry: {expiries[0]}")
    
    # Test 2: Get option chain for first expiry
    if expiries:
        test_option_chain_api(expiries[0])
        log("")
    
    # Test 3: Get spot price
    test_spot_price()
    log("")
    
    log("="*60)
    log("✅ TESTING COMPLETE!")
    log("="*60)

if __name__ == "__main__":
    main()
