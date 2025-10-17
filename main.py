import math
import random
from time import sleep

# ======== Shared request helper with retries ========
def http_get(url, headers=None, params=None, timeout=10, retries=3, backoff=1.2):
    """Simple GET with retries and exponential backoff. Returns JSON or None."""
    for attempt in range(1, retries+1):
        try:
            resp = requests.get(url, headers=headers or {"Accept":"application/json"}, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = backoff ** (attempt-1) + random.random()*0.2
            print(f"  Request error (attempt {attempt}/{retries}): {e}. retrying in {wait:.2f}s...")
            if attempt < retries:
                sleep(wait)
            else:
                print("  Max retries reached.")
                return None

# ======== Fix: historical candles (from_date then to_date) ========
def get_historical_candles(instrument_key, interval="30minute", days=5):
    """
    Fetch historical candles using V2 API.
    For 5-minute data: Use 1minute interval and resample.
    Corrected date ordering and robust handling.
    """
    try:
        now_ist = get_ist_now()
        to_date = now_ist.strftime('%Y-%m-%d')
        from_date = (now_ist - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # V2 API expects: .../{instrument_key}/{interval}/{from_date}/{to_date}
        url = f"{BASE_URL}/v2/historical-candle/{instrument_key}/1minute/{from_date}/{to_date}"
        
        data = http_get(url, headers={"Accept":"application/json"}, timeout=15, retries=3)
        if not data:
            print("  No response for historical candles.")
            return []
        
        # Support different shapes: data -> candles or response directly is candles
        candles_1min = None
        if isinstance(data, dict):
            # typical API: {'status': 'success', 'data': {'candles': [...]}}
            if data.get('status') in ('success', True) and 'data' in data and 'candles' in data['data']:
                candles_1min = data['data']['candles']
            elif 'candles' in data:
                candles_1min = data['candles']
        elif isinstance(data, list):
            candles_1min = data
        
        if not candles_1min:
            print("  No 1min candles found in response.")
            return []
        
        candles_5min = resample_to_5min(candles_1min)
        return candles_5min[:500] if len(candles_5min) > 500 else candles_5min
    except Exception as e:
        print(f"  Error fetching historical candles: {str(e)}")
        return []

# ======== Utility: get index/instrument LTP ========
def get_instrument_ltp(instrument_key):
    """
    Fetch last traded price (LTP) for a given instrument using quote endpoint.
    Returns float or None.
    """
    try:
        url = f"{BASE_URL}/v2/market/quote/{instrument_key}"
        data = http_get(url, headers=HEADERS, timeout=10, retries=2)
        if not data:
            return None
        
        # Different APIs return different shapes. Try to find ltp/last_price
        if isinstance(data, dict):
            # common shape: {'status':'success','data':{'ltp': 12345.0, ...}}
            d = data.get('data') or data
            if isinstance(d, dict):
                for k in ('ltp', 'last_price', 'lastTradedPrice', 'last'):
                    if k in d and d[k] is not None:
                        return float(d[k])
                
                # sometimes inside nested 'market_data'
                if 'market_data' in d and isinstance(d['market_data'], dict):
                    md = d['market_data']
                    for k in ('ltp','last_price'):
                        if k in md and md[k] is not None:
                            return float(md[k])
        return None
    except Exception as e:
        print(f"  Error fetching LTP for {instrument_key}: {e}")
        return None

# ======== Fix: get_option_contracts with expiry param and pagination/retries ========
def get_option_contracts():
    """
    Fetch all option contracts for NIFTY (all strikes) for the nearest expiry.
    Returns list of contract dicts.
    """
    try:
        expiry = get_next_expiry()
        print(f"  Looking up option contracts for expiry {expiry} ...")
        
        # Some Upstox setups allow pagination; we'll try to fetch all results if paginated.
        url = f"{BASE_URL}/v2/option/contract"
        all_contracts = []
        page = 1
        page_size = 200  # try larger page
        while True:
            params = {
                "instrument_key": NIFTY_INDEX_KEY,
                "expiry_date": expiry,
                "page": page,
                "page_size": page_size
            }
            data = http_get(url, headers=HEADERS, params=params, timeout=12, retries=3)
            if not data:
                break
            # Accept shape: {'status':'success','data': [...], 'meta':{...}}
            payload = data.get('data') if isinstance(data, dict) else data
            if not payload:
                break
            if isinstance(payload, list):
                batch = payload
            elif isinstance(payload, dict) and 'contracts' in payload:
                batch = payload['contracts']
            else:
                # if data is a dict of instruments keyed, convert to list
                if isinstance(payload, dict):
                    batch = list(payload.values())
                else:
                    batch = []
            
            if not batch:
                break
            all_contracts.extend(batch)
            
            # Pagination break conditions
            meta = data.get('meta') if isinstance(data, dict) else None
            if meta and meta.get('has_more') is True:
                page += 1
                sleep(0.2)
                continue
            # if returned less than page_size, assume last page
            if len(batch) < page_size:
                break
            page += 1
            sleep(0.2)
        
        # final sanitization: ensure instrument_key present
        all_contracts = [c for c in all_contracts if c and c.get('instrument_key')]
        return all_contracts
    except Exception as e:
        print(f"  Error fetching option contracts: {str(e)}")
        return []

# ======== Fix: get_option_greeks robust handling ========
def get_option_greeks(instrument_keys):
    """
    Fetch option Greeks for given instrument keys (max 50 per request).
    Returns dict mapping instrument_key -> greek_data
    """
    try:
        if not instrument_keys:
            return {}
        
        url = f"{BASE_URL}/v2/option/greek"
        keys_str = ",".join(instrument_keys[:50])
        params = {"instrument_key": keys_str}
        
        data = http_get(url, headers=HEADERS, params=params, timeout=12, retries=3)
        if not data:
            return {}
        
        # Response shapes vary: sometimes {'status':'success','data':{key: {...}}}
        result = {}
        if isinstance(data, dict):
            payload = data.get('data') or data
            # If payload is dict mapping keys -> values
            if isinstance(payload, dict):
                # Normalize numeric fields to expected names if necessary
                for k, v in payload.items():
                    if not isinstance(v, dict):
                        continue
                    # Common alias mapping
                    normalized = {
                        'last_price': v.get('last_price') or v.get('ltp') or v.get('lastPrice') or v.get('last'),
                        'delta': v.get('delta'),
                        'theta': v.get('theta'),
                        'gamma': v.get('gamma'),
                        'vega': v.get('vega'),
                        'iv': v.get('iv') or v.get('implied_volatility'),
                        'oi': v.get('oi') or v.get('open_interest'),
                        'volume': v.get('volume')
                    }
                    # copy original fields too so calling code can use whichever
                    normalized.update(v)
                    result[k] = normalized
            elif isinstance(payload, list):
                # sometimes data is a list of dicts each containing instrument_key
                for item in payload:
                    key = item.get('instrument_key') or item.get('instrumentKey') or item.get('instrument')
                    if key:
                        result[key] = item
        elif isinstance(data, list):
            for item in data:
                key = item.get('instrument_key') or item.get('instrumentKey') or item.get('instrument')
                if key:
                    result[key] = item
        
        return result
    except Exception as e:
        print(f"  Error fetching Greeks: {str(e)}")
        return {}

# ======== Improved: get_option_chain_data now computes ATM via index LTP and shows nearby strikes ========
def get_option_chain_data():
    """
    Build complete option chain using Contract API + Greeks API.
    Now computes ATM using index LTP and returns strikes sorted,
    with call/put data combined. Robust to missing Greeks.
    """
    try:
        print("  Getting option contracts...")
        contracts = get_option_contracts()
        
        if not contracts:
            print("  ⚠️ No contracts found")
            return []
        
        print(f"  Got {len(contracts)} contracts")
        instrument_keys = [c.get('instrument_key') for c in contracts if c.get('instrument_key')]
        
        if not instrument_keys:
            print("  ⚠️ No instrument keys")
            return []
        
        # Fetch Greeks in batches
        print(f"  Fetching Greeks for {len(instrument_keys)} instruments (batches)...")
        all_greeks = {}
        for i in range(0, len(instrument_keys), 50):
            batch = instrument_keys[i:i+50]
            greeks = get_option_greeks(batch)
            all_greeks.update(greeks or {})
            sleep(0.25)  # respectful pause
        
        print(f"  Got Greeks for {len(all_greeks)} instruments")
        
        # merge contracts + greeks
        option_chain = []
        for c in contracts:
            key = c.get('instrument_key')
            strike = c.get('strike_price') or c.get('strike') or c.get('strikePrice')
            option_type = (c.get('option_type') or c.get('optionType') or '').upper()
            if key in all_greeks:
                g = all_greeks[key]
                option_info = {
                    'strike_price': float(strike) if strike is not None else None,
                    'instrument_key': key,
                    'option_type': 'CE' if option_type in ('CE','CALL') else ('PE' if option_type in ('PE','PUT') else option_type),
                    'last_price': float(g.get('last_price') or g.get('ltp') or g.get('last_price', 0) or 0),
                    'oi': int(g.get('oi') or g.get('open_interest') or 0),
                    'volume': int(g.get('volume') or 0),
                    'delta': float(g.get('delta') or 0),
                    'theta': float(g.get('theta') or 0),
                    'gamma': float(g.get('gamma') or 0),
                    'vega': float(g.get('vega') or 0),
                    'iv': float(g.get('iv') or g.get('implied_volatility') or 0)
                }
                option_chain.append(option_info)
            else:
                # still keep the contract (without Greeks) in case user wants it
                option_info = {
                    'strike_price': float(strike) if strike is not None else None,
                    'instrument_key': key,
                    'option_type': 'CE' if option_type in ('CE','CALL') else ('PE' if option_type in ('PE','PUT') else option_type),
                    'last_price': 0,
                    'oi': 0,
                    'volume': 0,
                    'delta': 0,
                    'theta': 0,
                    'gamma': 0,
                    'vega': 0,
                    'iv': 0
                }
                option_chain.append(option_info)
        
        # Group by strike
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
        
        sorted_strikes = sorted(strikes.values(), key=lambda x: x['strike_price'])
        
        # Compute ATM using underlying index LTP if possible
        underlying_ltp = get_instrument_ltp(NIFTY_INDEX_KEY)
        if underlying_ltp is not None:
            # find nearest strike to underlying
            closest_idx = min(range(len(sorted_strikes)), key=lambda i: abs(sorted_strikes[i]['strike_price'] - underlying_ltp))
            # re-order to center at ATM for user convenience
            # but still return full sorted list — we will use ATM index in formatting
            atm_index = closest_idx
            print(f"  Underlying LTP: {underlying_ltp:.2f} — ATM approx strike: {sorted_strikes[atm_index]['strike_price']}")
        else:
            atm_index = len(sorted_strikes) // 2
            print("  Could not get underlying LTP — using mid-index as ATM fallback.")
        
        # attach metadata
        result = {
            "strikes": sorted_strikes,
            "atm_index": atm_index,
            "underlying_ltp": underlying_ltp
        }
        return result
    except Exception as e:
        print(f"  Error building option chain: {str(e)}")
        return {}

# ======== Update: format_option_chain_message to accept new structure ========
def format_option_chain_message(option_payload):
    """
    option_payload: { 'strikes': [...], 'atm_index': int, 'underlying_ltp': float }
    Shows ATM ± 10 strikes (or available)
    """
    if not option_payload or 'strikes' not in option_payload:
        return "❌ Option chain data not available"
    
    strikes = option_payload['strikes']
    atm_index = option_payload.get('atm_index', len(strikes)//2)
    underlying_ltp = option_payload.get('underlying_ltp')
    
    now_ist = get_ist_now()
    text = "📊 *NIFTY 50 OPTION CHAIN* 📊\n\n"
    text += f"⏰ {now_ist.strftime('%d %b %Y, %I:%M:%S %p IST')}\n"
    text += f"📅 Expiry: {get_next_expiry()}\n"
    text += f"📈 Total Strikes: {len(strikes)}\n"
    if underlying_ltp:
        text += f"📌 Underlying LTP: ₹{underlying_ltp:.2f}\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # define window around ATM
    start = max(0, atm_index - 10)
    end = min(len(strikes), atm_index + 11)
    
    for i in range(start, end):
        strike_data = strikes[i]
        strike = strike_data.get('strike_price', 'N/A')
        is_atm = (i == atm_index)
        text += f"*Strike: {strike}* {'🔹 ATM' if is_atm else ''}\n"
        
        call = strike_data.get('call')
        if call:
            text += f"📞 *CALL*\n"
            text += f"  LTP: ₹{call.get('last_price', 0):.2f}\n"
            text += f"  OI: {call.get('oi', 0):,}\n"
            text += f"  Vol: {call.get('volume', 0):,}\n"
            text += f"  𝛿: {call.get('delta', 0):.3f} | 𝜃: {call.get('theta', 0):.2f}\n"
            text += f"  𝛄: {call.get('gamma', 0):.5f} | 𝜈: {call.get('vega', 0):.2f}\n"
            try:
                ivpct = float(call.get('iv', 0)) * 100
            except:
                ivpct = float(call.get('iv', 0) or 0)
            text += f"  IV: {ivpct:.2f}%\n"
        
        put = strike_data.get('put')
        if put:
            text += f"📉 *PUT*\n"
            text += f"  LTP: ₹{put.get('last_price', 0):.2f}\n"
            text += f"  OI: {put.get('oi', 0):,}\n"
            text += f"  Vol: {put.get('volume', 0):,}\n"
            text += f"  𝛿: {put.get('delta', 0):.3f} | 𝜃: {put.get('theta', 0):.2f}\n"
            text += f"  𝛄: {put.get('gamma', 0):.5f} | 𝜈: {put.get('vega', 0):.2f}\n"
            try:
                ivpct = float(put.get('iv', 0)) * 100
            except:
                ivpct = float(put.get('iv', 0) or 0)
            text += f"  IV: {ivpct:.2f}%\n"
        
        text += "\n"
    
    return text
