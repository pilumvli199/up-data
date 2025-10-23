#!/usr/bin/env python3
# nifty50_options_with_charts.py - Option Chain + 5min Candlestick Charts (7 days)

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta
import pytz
from telegram import Bot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import io

# Config
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# Nifty 50 Stocks with active options
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

print("🚀 NIFTY 50 OPTIONS + CHARTS MONITOR")
print(f"📊 Tracking {len(NIFTY50_STOCKS)} stocks")
print(f"📈 5min Candlestick Charts (7 days)")
print(f"⏰ Updates every 5 minutes")

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
        print(f"⚠️ Expiry error: {e}")
    
    return []

def get_next_expiry(instrument_key):
    """Get next valid expiry"""
    expiries = get_expiries(instrument_key)
    if not expiries:
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
    """Get option chain data"""
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
        print(f"⚠️ Chain error: {e}")
    
    return []

def get_spot_price(instrument_key):
    """Get spot price"""
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
        print(f"⚠️ Spot error: {e}")
    
    return 0

def get_historical_data_yfinance(symbol):
    """Get historical data from Yahoo Finance as fallback"""
    try:
        import yfinance as yf
        
        # Yahoo Finance symbol format for NSE stocks
        ticker = f"{symbol}.NS"
        
        # Get last 7 days of 5min data
        stock = yf.Ticker(ticker)
        
        # Download 5min data for last 7 days
        hist = stock.history(period="7d", interval="5m")
        
        if hist.empty:
            return []
        
        # Convert to format similar to Upstox
        candles = []
        for idx, row in hist.iterrows():
            # Format: [timestamp, open, high, low, close, volume, oi]
            candle = [
                idx.isoformat(),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume'],
                0  # OI not available in yfinance
            ]
            candles.append(candle)
        
        return candles
    except ImportError:
        print("  ⚠️ yfinance not installed. Run: pip install yfinance")
        return []
    except Exception as e:
        print(f"  ⚠️ Yahoo Finance error: {e}")
        return []

def create_candlestick_chart(candles, symbol, spot_price):
    """Create candlestick chart from data"""
    if not candles or len(candles) < 10:
        return None
    
    # Parse candle data: [timestamp, open, high, low, close, volume, oi]
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for candle in reversed(candles):  # Reverse for chronological order
        try:
            timestamp = datetime.fromisoformat(candle[0].replace('Z', '+00:00'))
            timestamp = timestamp.astimezone(IST)
            
            dates.append(timestamp)
            opens.append(float(candle[1]))
            highs.append(float(candle[2]))
            lows.append(float(candle[3]))
            closes.append(float(candle[4]))
            volumes.append(int(candle[5]))
        except Exception as e:
            continue
    
    if len(dates) < 10:
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    fig.patch.set_facecolor('#0a0a0a')
    ax1.set_facecolor('#0f0f0f')
    ax2.set_facecolor('#0f0f0f')
    
    # Plot candlesticks
    for i in range(len(dates)):
        color = '#00ff00' if closes[i] >= opens[i] else '#ff0000'
        
        # Draw high-low line
        ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], 
                color=color, linewidth=0.8, alpha=0.8)
        
        # Draw candle body
        height = abs(closes[i] - opens[i])
        bottom = min(opens[i], closes[i])
        
        if height > 0:
            rect = Rectangle((mdates.date2num(dates[i]) - 0.0001, bottom),
                           0.0002, height, facecolor=color, 
                           edgecolor=color, alpha=0.9)
            ax1.add_patch(rect)
        else:
            # Doji - draw horizontal line
            ax1.plot([dates[i], dates[i]], [opens[i], opens[i]], 
                    color=color, linewidth=1.5)
    
    # Add spot price line
    ax1.axhline(y=spot_price, color='#ffff00', linestyle='--', 
               linewidth=1.5, label=f'Spot: ₹{spot_price:.2f}', alpha=0.7)
    
    # Format price axis
    ax1.set_ylabel('Price (₹)', color='white', fontsize=11, fontweight='bold')
    ax1.tick_params(colors='white', labelsize=9)
    ax1.grid(True, alpha=0.2, color='#333333', linestyle=':')
    ax1.legend(loc='upper left', fontsize=10, facecolor='#1a1a1a', 
              edgecolor='#333333', labelcolor='white')
    
    # Title
    title = f'{symbol} - 5 Minute Candlestick Chart (Last 7 Days)'
    ax1.set_title(title, color='white', fontsize=14, fontweight='bold', pad=15)
    
    # Volume bars
    colors_vol = ['#00ff0060' if closes[i] >= opens[i] else '#ff000060' 
                  for i in range(len(dates))]
    ax2.bar(dates, volumes, color=colors_vol, width=0.0002, alpha=0.8)
    
    ax2.set_ylabel('Volume', color='white', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='white', labelsize=9)
    ax2.grid(True, alpha=0.2, color='#333333', linestyle=':')
    
    # Format x-axis for both subplots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M', tz=IST))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax2.set_xlabel('Date & Time (IST)', color='white', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#0a0a0a', 
               edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def format_detailed_message(symbol, spot, expiry, strikes):
    """Format option chain message"""
    if not strikes or len(strikes) < 11:
        return None
    
    # Find ATM
    atm_index = len(strikes) // 2
    if spot > 0:
        atm_index = min(range(len(strikes)), 
                       key=lambda i: abs(strikes[i].get('strike_price', 0) - spot))
    
    # Select 11 strikes
    start = max(0, atm_index - 5)
    end = min(len(strikes), atm_index + 6)
    
    if end - start < 11:
        if start == 0:
            end = min(11, len(strikes))
        else:
            start = max(0, len(strikes) - 11)
    
    selected = strikes[start:end]
    
    msg = f"📊 *{symbol} OPTION CHAIN*\n\n"
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
        
        call = s.get('call_options', {}).get('market_data', {})
        ce_ltp = call.get('ltp', 0)
        ce_vol = call.get('volume', 0)
        ce_oi = call.get('oi', 0)
        
        put = s.get('put_options', {}).get('market_data', {})
        pe_ltp = put.get('ltp', 0)
        pe_vol = put.get('volume', 0)
        pe_oi = put.get('oi', 0)
        
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol
        
        ce_vol_k = ce_vol / 1000 if ce_vol > 0 else 0
        ce_oi_k = ce_oi / 1000 if ce_oi > 0 else 0
        pe_vol_k = pe_vol / 1000 if pe_vol > 0 else 0
        pe_oi_k = pe_oi / 1000 if pe_oi > 0 else 0
        
        msg += f"{mark}{strike:8.2f} {ce_ltp:7.2f} {ce_vol_k:7.1f}K {ce_oi_k:7.1f}K {pe_ltp:7.2f} {pe_vol_k:7.1f}K {pe_oi_k:7.1f}K\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    total_ce_vol_k = total_ce_vol / 1000
    total_ce_oi_k = total_ce_oi / 1000
    total_pe_vol_k = total_pe_vol / 1000
    total_pe_oi_k = total_pe_oi / 1000
    
    msg += f"TOTAL          {total_ce_vol_k:7.1f}K {total_ce_oi_k:7.1f}K        {total_pe_vol_k:7.1f}K {total_pe_oi_k:7.1f}K\n"
    msg += "```\n\n"
    
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    
    msg += f"📊 *PCR (OI):* {pcr:.3f}\n"
    msg += f"📊 *PCR (Vol):* {pcr_vol:.3f}\n"
    msg += f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}\n"
    
    return msg

async def send_telegram_text(msg):
    """Send text message"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=msg, 
            parse_mode='Markdown'
        )
        return True
    except Exception as e:
        print(f"❌ Text error: {e}")
        return False

async def send_telegram_photo(photo_buf, caption):
    """Send photo with caption"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID,
            photo=photo_buf,
            caption=caption,
            parse_mode='Markdown'
        )
        return True
    except Exception as e:
        print(f"❌ Photo error: {e}")
        return False

async def process_stock(instrument_key, symbol, idx, total):
    """Process single stock - fetch data and send"""
    print(f"\n[{idx}/{total}] Processing {symbol}...")
    
    try:
        # Get expiry
        expiry = get_next_expiry(instrument_key)
        
        # Get spot price
        spot = get_spot_price(instrument_key)
        
        if spot == 0:
            print(f"  ⚠️ {symbol}: Invalid spot")
            return False
        
        # Get option chain
        strikes = get_option_chain(instrument_key, expiry)
        
        if not strikes or len(strikes) < 11:
            print(f"  ⚠️ {symbol}: Insufficient strikes")
            return False
        
        print(f"  ✅ {symbol}: ₹{spot:.2f} | {len(strikes)} strikes")
        
        # Format option chain message
        msg = format_detailed_message(symbol, spot, expiry, strikes)
        
        if not msg:
            print(f"  ⚠️ {symbol}: Message format failed")
            return False
        
        # Send option chain first
        success = await send_telegram_text(msg)
        if success:
            print(f"  📤 {symbol}: Option chain sent")
        
        # Try to get chart data from Yahoo Finance
        print(f"  📊 {symbol}: Fetching chart data...")
        candles = get_historical_data_yfinance(symbol)
        
        if candles and len(candles) >= 10:
            # Create chart
            print(f"  📈 {symbol}: Creating chart...")
            chart_buf = create_candlestick_chart(candles, symbol, spot)
            
            if chart_buf:
                # Send chart
                caption = f"📈 *{symbol}* - 5min Chart (7 days)\n💰 Spot: ₹{spot:.2f}"
                success = await send_telegram_photo(chart_buf, caption)
                if success:
                    print(f"  📤 {symbol}: Chart sent!")
                    return True
        else:
            # If no chart data, add TradingView link
            chart_link = f"https://in.tradingview.com/chart/?symbol=NSE%3A{symbol}"
            link_msg = f"📈 [View {symbol} Chart]({chart_link})"
            await send_telegram_text(link_msg)
            print(f"  📤 {symbol}: Chart link sent")
        
        return True
        
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return False

async def fetch_all():
    """Fetch and send all stocks"""
    print("\n" + "="*60)
    print(f"⏰ {datetime.now(IST).strftime('%I:%M:%S %p IST')}")
    print("="*60)
    
    header = f"🚀 *NIFTY 50 UPDATE*\n"
    header += f"⏰ {datetime.now(IST).strftime('%I:%M %p IST')}\n"
    header += f"📊 Option Chain + 5min Charts\n"
    header += f"📈 {len(NIFTY50_STOCKS)} Stocks\n\n_Starting..._"
    
    await send_telegram_text(header)
    
    success = 0
    total = len(NIFTY50_STOCKS)
    
    for idx, (key, symbol) in enumerate(NIFTY50_STOCKS.items(), 1):
        result = await process_stock(key, symbol, idx, total)
        if result:
            success += 1
        await asyncio.sleep(3)  # Rate limit
    
    summary = f"\n✅ *COMPLETE*\n📊 {success}/{total} stocks\n⏰ {datetime.now(IST).strftime('%I:%M %p')}"
    await send_telegram_text(summary)
    
    print("\n" + "="*60)
    print(f"✅ Complete: {success}/{total}")
    print("="*60)

async def monitoring_loop():
    """Main loop - every 5 minutes"""
    print("\n🔄 Starting loop...")
    print("🔄 Press Ctrl+C to stop\n")
    
    while True:
        try:
            await fetch_all()
            
            next_time = (datetime.now(IST) + timedelta(minutes=5)).strftime('%I:%M %p')
            print(f"\n⏳ Next update: {next_time}\n")
            
            await asyncio.sleep(300)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            await asyncio.sleep(60)

async def main():
    print("\n" + "="*60)
    print("🚀 NIFTY 50 OPTIONS + CHARTS MONITOR")
    print("="*60)
    print("📊 Option Chain: 11 strikes, full data")
    print("📈 Charts: 5min candlestick, 7 days")
    print("⏰ Updates: Every 5 minutes")
    print("="*60)
    
    await monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())
