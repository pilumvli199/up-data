import os
import logging
from datetime import datetime, timedelta
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv('UPSTOX_API_KEY')
API_SECRET = os.getenv('UPSTOX_API_SECRET')
ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Upstox API Base URLs
BASE_URL = "https://api.upstox.com"
NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

class UpstoxAPI:
    def __init__(self, access_token):
        self.access_token = access_token
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
    
    def get_historical_candles(self, instrument_key, interval=5, unit='minutes', days_back=30):
        """
        Fetch historical candle data
        interval: 5 for 5-minute candles
        unit: 'minutes', 'hours', 'days'
        """
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        url = f"{BASE_URL}/v3/historical-candle/{instrument_key}/{unit}/{interval}"
        params = {
            'from_date': from_date,
            'to_date': to_date
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success':
                return data['data']['candles']
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def get_intraday_candles(self, instrument_key, interval=5, unit='minutes'):
        """Fetch today's intraday candle data"""
        url = f"{BASE_URL}/v3/intraday-candle/{instrument_key}/{unit}/{interval}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success':
                return data['data']['candles']
            return None
        except Exception as e:
            logger.error(f"Error fetching intraday data: {e}")
            return None
    
    def get_option_chain(self, underlying_key, expiry_date):
        """Fetch option chain data"""
        url = f"{BASE_URL}/v2/option/chain"
        params = {
            'instrument_key': underlying_key,
            'expiry_date': expiry_date
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return None
    
    def get_market_quote(self, instrument_key):
        """Get current market quote to find ATM"""
        url = f"{BASE_URL}/v3/market-quote/quotes"
        params = {'instrument_key': instrument_key}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success':
                return data['data'][instrument_key]['last_price']
            return None
        except Exception as e:
            logger.error(f"Error fetching market quote: {e}")
            return None

def create_candlestick_chart(candles, filename='nifty_chart.png'):
    """
    Create TradingView-style candlestick chart
    candles: List of [timestamp, open, high, low, close, volume, oi]
    """
    # Prepare data
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Take last 2500 candles
    df = df.tail(2500)
    
    # Convert timestamp to matplotlib date format
    df['date_num'] = mdates.date2num(df['timestamp'])
    
    # Prepare data for candlestick
    ohlc = df[['date_num', 'open', 'high', 'low', 'close']].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    ax.set_facecolor('white')
    
    # Draw candlesticks
    for i in range(len(ohlc)):
        date_num, open_price, high, low, close = ohlc[i]
        
        # Determine color
        if close >= open_price:
            color = '#26a69a'  # Green
            body_color = '#26a69a'
        else:
            color = '#ef5350'  # Red
            body_color = '#ef5350'
        
        # Draw high-low line
        ax.plot([date_num, date_num], [low, high], color=color, linewidth=1, solid_capstyle='round')
        
        # Draw body
        height = abs(close - open_price)
        bottom = min(open_price, close)
        
        if height == 0:
            height = 0.01
        
        rect = mpatches.Rectangle((date_num - 0.0003, bottom), 0.0006, height, 
                                   facecolor=body_color, edgecolor=body_color)
        ax.add_patch(rect)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    
    # Styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Date Time', fontsize=10, fontweight='bold')
    ax.set_ylabel('Price', fontsize=10, fontweight='bold')
    ax.set_title('NIFTY 50 - 5 Minute Chart (Last 2500 Candles)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Chart saved: {filename}")
    return filename

def format_option_chain_data(option_data, atm_strike, num_strikes=10):
    """
    Format option chain data for ATM ± num_strikes
    Returns formatted text
    """
    if not option_data or 'data' not in option_data:
        return "❌ Option chain data not available"
    
    strikes = sorted([float(strike) for strike in option_data['data'].keys() if strike != 'underlying_spot_price'])
    
    # Find ATM index
    atm_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm_strike))
    
    # Get ATM ± num_strikes
    start_idx = max(0, atm_index - num_strikes)
    end_idx = min(len(strikes), atm_index + num_strikes + 1)
    selected_strikes = strikes[start_idx:end_idx]
    
    # Format data
    text = "📊 *NIFTY 50 OPTION CHAIN*\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    text += f"🎯 *Spot Price:* {option_data['data'].get('underlying_spot_price', 'N/A')}\n"
    text += f"🎯 *ATM Strike:* {atm_strike}\n\n"
    
    text += "```\n"
    text += "CALL SIDE          STRIKE          PUT SIDE\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += "OI      CHG    |         | OI      CHG\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for strike in selected_strikes:
        strike_str = str(strike)
        strike_data = option_data['data'].get(strike_str, {})
        
        call_data = strike_data.get('call_options', {}).get('market_data', {})
        put_data = strike_data.get('put_options', {}).get('market_data', {})
        
        call_oi = call_data.get('oi', 0)
        call_prev_oi = call_data.get('prev_oi', 0)
        call_chg = call_oi - call_prev_oi
        
        put_oi = put_data.get('oi', 0)
        put_prev_oi = put_data.get('prev_oi', 0)
        put_chg = put_oi - put_prev_oi
        
        # Format numbers
        call_oi_str = f"{call_oi/100000:.1f}L" if call_oi else "0"
        call_chg_str = f"{call_chg/100000:+.1f}L" if call_chg else "0"
        put_oi_str = f"{put_oi/100000:.1f}L" if put_oi else "0"
        put_chg_str = f"{put_chg/100000:+.1f}L" if put_chg else "0"
        
        strike_marker = " 🎯" if abs(strike - atm_strike) < 50 else ""
        
        text += f"{call_oi_str:>6} {call_chg_str:>7} | {strike:>7.0f}{strike_marker:<2} | {put_oi_str:>6} {put_chg_str:>7}\n"
    
    text += "```\n\n"
    
    # Add Greeks summary for ATM
    atm_data = option_data['data'].get(str(atm_strike), {})
    if atm_data:
        text += "📈 *ATM Greeks*\n"
        
        call_greeks = atm_data.get('call_options', {}).get('option_greeks', {})
        put_greeks = atm_data.get('put_options', {}).get('option_greeks', {})
        
        if call_greeks:
            text += f"*CALL:* Δ:{call_greeks.get('delta', 0):.4f} | γ:{call_greeks.get('gamma', 0):.5f} | "
            text += f"θ:{call_greeks.get('theta', 0):.2f} | ν:{call_greeks.get('vega', 0):.2f} | IV:{call_greeks.get('iv', 0):.2f}%\n"
        
        if put_greeks:
            text += f"*PUT:*  Δ:{put_greeks.get('delta', 0):.4f} | γ:{put_greeks.get('gamma', 0):.5f} | "
            text += f"θ:{put_greeks.get('theta', 0):.2f} | ν:{put_greeks.get('vega', 0):.2f} | IV:{put_greeks.get('iv', 0):.2f}%\n"
    
    # PCR calculation
    total_call_oi = sum([option_data['data'].get(str(s), {}).get('call_options', {}).get('market_data', {}).get('oi', 0) for s in selected_strikes])
    total_put_oi = sum([option_data['data'].get(str(s), {}).get('put_options', {}).get('market_data', {}).get('oi', 0) for s in selected_strikes])
    pcr = total_put_oi / total_call_oi if total_call_oi else 0
    
    text += f"\n📊 *PCR (Put/Call Ratio):* {pcr:.2f}\n"
    text += f"⏰ *Time:* {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}\n"
    
    return text

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = """
🎯 *Welcome to Upstox Nifty50 Alert Bot!*

*Available Commands:*
/start - Show this message
/alert - Get latest chart + option chain data
/chart - Get only candlestick chart
/optionchain - Get only option chain data

*Features:*
✅ Last 2500 candles (5-minute timeframe)
✅ TradingView style charts
✅ Option chain with OI & OI Change
✅ ATM ± 10 strikes (21 strikes total)
✅ Greeks data for ATM
✅ PCR calculation

Happy Trading! 📈
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main alert command - sends chart + option chain"""
    await update.message.reply_text("⏳ Fetching data... Please wait...")
    
    try:
        api = UpstoxAPI(ACCESS_TOKEN)
        
        # 1. Fetch historical + intraday data
        historical = api.get_historical_candles(NIFTY_INSTRUMENT_KEY, interval=5, days_back=30)
        intraday = api.get_intraday_candles(NIFTY_INSTRUMENT_KEY, interval=5)
        
        if not historical:
            await update.message.reply_text("❌ Failed to fetch historical data")
            return
        
        # Combine data
        all_candles = historical + (intraday if intraday else [])
        
        # 2. Create chart
        chart_file = create_candlestick_chart(all_candles)
        
        # 3. Get current price for ATM
        current_price = api.get_market_quote(NIFTY_INSTRUMENT_KEY)
        if not current_price:
            await update.message.reply_text("❌ Failed to fetch current price")
            return
        
        # Round to nearest 50 for ATM
        atm_strike = round(current_price / 50) * 50
        
        # 4. Get option chain (need to find nearest expiry)
        # For simplicity, using weekly expiry (Thursday)
        today = datetime.now()
        days_ahead = 3 - today.weekday()  # Thursday
        if days_ahead <= 0:
            days_ahead += 7
        expiry_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        option_data = api.get_option_chain(NIFTY_INSTRUMENT_KEY, expiry_date)
        option_text = format_option_chain_data(option_data, atm_strike, num_strikes=10)
        
        # 5. Send chart
        with open(chart_file, 'rb') as photo:
            await update.message.reply_photo(photo=photo, caption="📊 NIFTY 50 - Last 2500 Candles (5 Min)")
        
        # 6. Send option chain data
        await update.message.reply_text(option_text, parse_mode='Markdown')
        
        # Cleanup
        if os.path.exists(chart_file):
            os.remove(chart_file)
        
        logger.info("Alert sent successfully")
        
    except Exception as e:
        logger.error(f"Error in alert_command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send only chart"""
    await update.message.reply_text("⏳ Creating chart...")
    
    try:
        api = UpstoxAPI(ACCESS_TOKEN)
        
        historical = api.get_historical_candles(NIFTY_INSTRUMENT_KEY, interval=5, days_back=30)
        intraday = api.get_intraday_candles(NIFTY_INSTRUMENT_KEY, interval=5)
        
        if not historical:
            await update.message.reply_text("❌ Failed to fetch data")
            return
        
        all_candles = historical + (intraday if intraday else [])
        chart_file = create_candlestick_chart(all_candles)
        
        with open(chart_file, 'rb') as photo:
            await update.message.reply_photo(photo=photo, caption="📊 NIFTY 50 - Last 2500 Candles (5 Min)")
        
        if os.path.exists(chart_file):
            os.remove(chart_file)
        
    except Exception as e:
        logger.error(f"Error in chart_command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def optionchain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send only option chain"""
    await update.message.reply_text("⏳ Fetching option chain...")
    
    try:
        api = UpstoxAPI(ACCESS_TOKEN)
        
        current_price = api.get_market_quote(NIFTY_INSTRUMENT_KEY)
        if not current_price:
            await update.message.reply_text("❌ Failed to fetch current price")
            return
        
        atm_strike = round(current_price / 50) * 50
        
        today = datetime.now()
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        expiry_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        option_data = api.get_option_chain(NIFTY_INSTRUMENT_KEY, expiry_date)
        option_text = format_option_chain_data(option_data, atm_strike, num_strikes=10)
        
        await update.message.reply_text(option_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in optionchain_command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():
    """Start the bot"""
    # Validate environment variables
    if not all([API_KEY, API_SECRET, ACCESS_TOKEN, TELEGRAM_BOT_TOKEN]):
        logger.error("❌ Missing required environment variables!")
        logger.error("Required: UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN")
        return
    
    logger.info("✅ All environment variables loaded")
    logger.info(f"🤖 Starting bot with token: {TELEGRAM_BOT_TOKEN[:10]}...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("alert", alert_command))
    application.add_handler(CommandHandler("chart", chart_command))
    application.add_handler(CommandHandler("optionchain", optionchain_command))
    
    # Start bot
    logger.info("Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
