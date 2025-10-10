import os
import logging
from datetime import datetime, timedelta
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import io
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
UPSTOX_API_KEY = os.getenv('UPSTOX_API_KEY')
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
BASE_URL = "https://api.upstox.com/v2"

class UpstoxNiftyBot:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
        }
    
    def get_nearest_expiry(self):
        """Get nearest Thursday expiry for Nifty options"""
        today = datetime.now()
        days_ahead = 3 - today.weekday()  # Thursday = 3
        if days_ahead <= 0:
            days_ahead += 7
        next_thursday = today + timedelta(days=days_ahead)
        return next_thursday.strftime('%Y-%m-%d')
    
    def get_option_chain(self, expiry_date=None):
        """Fetch option chain data from Upstox API"""
        try:
            if not expiry_date:
                expiry_date = self.get_nearest_expiry()
            
            url = f"{BASE_URL}/option/chain"
            params = {
                'instrument_key': NIFTY_INSTRUMENT_KEY,
                'expiry_date': expiry_date
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success':
                return data.get('data', [])
            else:
                logger.error(f"API returned error: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return []
    
    def get_atm_strikes(self, option_data, spot_price, num_strikes=10):
        """Get ATM and surrounding strikes (21 strikes total)"""
        if not option_data:
            return []
        
        # Sort by strike price
        sorted_data = sorted(option_data, key=lambda x: x.get('strike_price', 0))
        
        # Find ATM strike (closest to spot)
        atm_idx = min(range(len(sorted_data)), 
                     key=lambda i: abs(sorted_data[i].get('strike_price', 0) - spot_price))
        
        # Get 10 strikes above and below ATM (total 21 including ATM)
        start_idx = max(0, atm_idx - num_strikes)
        end_idx = min(len(sorted_data), atm_idx + num_strikes + 1)
        
        return sorted_data[start_idx:end_idx]
    
    def get_historical_candles(self, interval='5minute', days_back=10):
        """Fetch last 2500 historical candles"""
        try:
            # Calculate from_date to get approximately 2500 candles
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            url = f"{BASE_URL}/historical-candle/{NIFTY_INSTRUMENT_KEY}/{interval}/{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
            
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success':
                candles = data.get('data', {}).get('candles', [])
                # Return last 2500 candles
                return candles[-2500:] if len(candles) > 2500 else candles
            else:
                logger.error(f"Historical data API error: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching historical candles: {e}")
            return []
    
    def create_candlestick_chart(self, candles):
        """Create TradingView-style candlestick chart with white background"""
        try:
            if not candles:
                return None
            
            # Parse candle data: [timestamp, open, high, low, close, volume, oi]
            dates = [datetime.fromisoformat(c[0].replace('Z', '+00:00')) for c in candles]
            opens = [float(c[1]) for c in candles]
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            
            # Create figure with white background
            fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
            ax.set_facecolor('white')
            
            # Plot candlesticks
            width = 0.6
            width2 = 0.05
            
            for i in range(len(dates)):
                color = '#00C805' if closes[i] >= opens[i] else '#FF0000'  # Green/Red
                
                # Candle body
                height = closes[i] - opens[i]
                bottom = opens[i]
                rect = Rectangle((i - width/2, bottom), width, height, 
                               facecolor=color, edgecolor=color, alpha=0.9)
                ax.add_patch(rect)
                
                # Wicks
                ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1, alpha=0.9)
            
            # Formatting
            ax.set_xlim(-0.5, len(dates) - 0.5)
            ax.set_ylim(min(lows) * 0.999, max(highs) * 1.001)
            
            # X-axis formatting
            step = max(1, len(dates) // 10)
            ax.set_xticks(range(0, len(dates), step))
            ax.set_xticklabels([dates[i].strftime('%d-%m %H:%M') for i in range(0, len(dates), step)], 
                              rotation=45, ha='right')
            
            # Grid and labels
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
            ax.set_xlabel('Date & Time', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price', fontsize=12, fontweight='bold')
            ax.set_title(f'NIFTY 50 - Last {len(candles)} Candles (5min)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None
    
    def format_option_chain_message(self, option_data, spot_price):
        """Format option chain data for Telegram message"""
        try:
            if not option_data:
                return "❌ No option chain data available"
            
            msg = f"📊 *NIFTY 50 OPTION CHAIN*\n"
            msg += f"{'=' * 50}\n"
            msg += f"💹 *Spot Price:* ₹{spot_price:,.2f}\n"
            msg += f"📅 *Expiry:* {option_data[0].get('expiry', 'N/A')}\n"
            msg += f"📈 *PCR:* {option_data[0].get('pcr', 0):.2f}\n"
            msg += f"{'=' * 50}\n\n"
            
            # ATM strikes (21 total)
            atm_strikes = self.get_atm_strikes(option_data, spot_price, 10)
            
            msg += f"*{'Strike':<8} | {'CALL':^30} | {'PUT':^30}*\n"
            msg += f"*{'Price':<8} | {'OI':>8} {'Vol':>8} {'IV':>6} | {'OI':>8} {'Vol':>8} {'IV':>6}*\n"
            msg += f"{'-' * 70}\n"
            
            for strike_data in atm_strikes:
                strike = strike_data.get('strike_price', 0)
                
                # Call data
                call_opts = strike_data.get('call_options', {})
                call_mkt = call_opts.get('market_data', {})
                call_greeks = call_opts.get('option_greeks', {})
                call_oi = call_mkt.get('oi', 0)
                call_vol = call_mkt.get('volume', 0)
                call_iv = call_greeks.get('iv', 0)
                
                # Put data
                put_opts = strike_data.get('put_options', {})
                put_mkt = put_opts.get('market_data', {})
                put_greeks = put_opts.get('option_greeks', {})
                put_oi = put_mkt.get('oi', 0)
                put_vol = put_mkt.get('volume', 0)
                put_iv = put_greeks.get('iv', 0)
                
                # Highlight ATM
                prefix = "🔥" if abs(strike - spot_price) < 100 else "  "
                
                msg += f"`{prefix}{strike:<6.0f} | {call_oi:>8.0f} {call_vol:>8.0f} {call_iv:>6.1f} | {put_oi:>8.0f} {put_vol:>8.0f} {put_iv:>6.1f}`\n"
            
            msg += f"\n{'=' * 50}\n"
            msg += f"📌 *Greeks Summary (ATM)*\n"
            
            # ATM Greeks
            atm_data = min(atm_strikes, key=lambda x: abs(x.get('strike_price', 0) - spot_price))
            call_greeks = atm_data.get('call_options', {}).get('option_greeks', {})
            put_greeks = atm_data.get('put_options', {}).get('option_greeks', {})
            
            msg += f"*Call:* Δ={call_greeks.get('delta', 0):.3f} | Γ={call_greeks.get('gamma', 0):.4f} | "
            msg += f"Θ={call_greeks.get('theta', 0):.2f} | V={call_greeks.get('vega', 0):.2f}\n"
            msg += f"*Put:* Δ={put_greeks.get('delta', 0):.3f} | Γ={put_greeks.get('gamma', 0):.4f} | "
            msg += f"Θ={put_greeks.get('theta', 0):.2f} | V={put_greeks.get('vega', 0):.2f}\n"
            
            return msg
            
        except Exception as e:
            logger.error(f"Error formatting message: {e}")
            return f"❌ Error formatting data: {str(e)}"

# Bot instance
bot = UpstoxNiftyBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_msg = """
🚀 *Welcome to Upstox Nifty Bot!*

📊 Commands:
/nifty - Get Nifty 50 option chain + chart
/chart - Get only chart
/optionchain - Get only option chain
/help - Show this message

Made with ❤️ for trading!
    """
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def nifty_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send Nifty chart + option chain"""
    try:
        await update.message.reply_text("⏳ Fetching data from Upstox... Please wait!")
        
        # Fetch option chain
        option_data = bot.get_option_chain()
        if not option_data:
            await update.message.reply_text("❌ Failed to fetch option chain data!")
            return
        
        spot_price = option_data[0].get('underlying_spot_price', 0)
        
        # Fetch and create chart
        candles = bot.get_historical_candles()
        chart_buf = bot.create_candlestick_chart(candles)
        
        if chart_buf:
            await update.message.reply_photo(
                photo=chart_buf,
                caption="📈 NIFTY 50 - Last 2500 Candles (5min)"
            )
        
        # Send option chain
        msg = bot.format_option_chain_message(option_data, spot_price)
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in nifty_command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send only chart"""
    try:
        await update.message.reply_text("📊 Creating chart...")
        
        candles = bot.get_historical_candles()
        chart_buf = bot.create_candlestick_chart(candles)
        
        if chart_buf:
            await update.message.reply_photo(
                photo=chart_buf,
                caption="📈 NIFTY 50 - Historical Chart"
            )
        else:
            await update.message.reply_text("❌ Failed to create chart!")
            
    except Exception as e:
        logger.error(f"Error in chart_command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def optionchain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send only option chain"""
    try:
        await update.message.reply_text("📊 Fetching option chain...")
        
        option_data = bot.get_option_chain()
        if not option_data:
            await update.message.reply_text("❌ Failed to fetch option chain!")
            return
        
        spot_price = option_data[0].get('underlying_spot_price', 0)
        msg = bot.format_option_chain_message(option_data, spot_price)
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in optionchain_command: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    await start(update, context)

def main():
    """Start the bot"""
    try:
        # Create application
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("nifty", nifty_command))
        application.add_handler(CommandHandler("chart", chart_command))
        application.add_handler(CommandHandler("optionchain", optionchain_command))
        application.add_handler(CommandHandler("help", help_command))
        
        # Start bot
        logger.info("🚀 Bot started successfully!")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")

if __name__ == '__main__':
    main()
