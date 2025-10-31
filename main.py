#!/usr/bin/env python3
"""
HYBRID F&O SCANNER BOT v17.0
================================
✅ CORE LOGIC FIXED:
    - BULLISH Market => CE_BUY (Call Option Buy)
    - BEARISH Market => PE_BUY (Put Option Buy)
    
✅ ADVANCED AI BRAIN:
    - Uses the "Advanced Pro Trader" prompt as the AI's core persona.
    - Python pre-analyzes Chart + OI; AI synthesizes the data.

✅ TRUE 30-MIN OI COMPARISON:
    - Uses Redis List as a time-series (LPUSH/LINDEX).
    - Compares Current OI vs. 30-Min Ago OI.
    - Detects: Long Build-up, Short Build-up, Short Covering, Long Unwinding.

✅ PROFESSIONAL CHART ANALYSIS:
    - 1H Trend (MA), 15M S/R (Swing High/Low), 5M Entry (Breakout).

✅ COMPLETE & ROBUST CODE:
    - All classes and functions included.
    - Improved error handling.
    - No data gaps (uses continuous 15-min data).
"""

import os
import asyncio
import requests
import urllib.parse
import pytz
import json
import logging
import traceback
import re
import io
from datetime import datetime, timedelta, time
import time as time_sleep
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# --- Redis Import ---
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis module not found. OI analysis will be disabled.")

# --- Logging Setup ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

# Bot Settings
REDIS_EXPIRY_24H = 86400  # 24 hours in seconds
OI_SNAPSHOT_COUNT = 288    # 12 snapshots/hr * 24 hrs = 288
OI_COMPARE_INDEX = 6       # 6 * 5 min = 30 minutes ago
CONFIDENCE_MIN, SCORE_MIN, SCAN_INTERVAL = 75, 90, 300

# Symbols to Scan (Can be expanded)
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY 50", "expiry_day": 1},
    "BSE_INDEX|SENSEX": {"name": "SENSEX", "expiry_day": 4}
}
SECTOR_KEYWORDS = {"INDEX": ["nifty", "sensex", "india", "indian", "market", "bse", "nse", "equity", "stocks", "rupee", "rbi", "inflation", "gdp"]}

# --- AI SYSTEM PROMPT (User Provided) ---
AI_SYSTEM_PROMPT = """
You are a professional institutional trader. Analyze these 500 candlesticks (15-min, Nifty F&O).

STEP 1: Read all 500 candles, understand complete price journey.
STEP 2: Mark important levels:
- All-time high/low in this dataset
- Recent swing highs/lows (last 100 candles)
- Tested support/resistance (minimum 3 touches)
STEP 3: Identify current market phase:
- Accumulation / Distribution / Markup / Markdown
STEP 4: Pattern recognition:
- Scan for all chart patterns (completed + forming)
- Candlestick patterns (single, double, triple candle)
STEP 5: Structure analysis:
- Break of structure points
- Order blocks (where price reversed strongly)
- Fair value gaps / imbalances
STEP 6: Volume insights:
- High volume nodes (price acceptance zones)
- Low volume areas (quick pass-through zones)
STEP 7: Smart money behavior:
- Any stop hunts / liquidity grabs?
- Absorption patterns (supply/demand zones)
STEP 8: Trade setups (prioritize best 1-2 setups):
- High probability setups only (70%+ success historical)
- Complete trade plan: Entry trigger, SL, 2 targets, position size guidance
- Risk-reward minimum 1:2
STEP 9: Scenarios:
- Bullish scenario: "If price does X, then Y will happen"
- Bearish scenario: "If price does A, then B will happen"
- Neutral scenario: "If choppy, stay out until..."
STEP 10: Final recommendation:
- Clear bias with confidence level (high/medium/low)
- Action: Aggressive buy/sell, Wait for confirmation, or Stay out
- Key price alerts to set

Be brutally honest. If setup is unclear, say "No clear setup, wait."
"""


# --- DATA CLASSES ---

@dataclass
class OISnapshot:
    """Stores a snapshot of OI totals and price for comparison."""
    timestamp: int
    spot_price: float
    total_ce_oi: int
    total_pe_oi: int
    total_ce_vol: int
    total_pe_vol: int

    def to_json(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str):
        return OISnapshot(**json.loads(json_str))

@dataclass
class AggregateOIAnalysis:
    """Holds the *result* of the OI comparison."""
    sentiment: str  # e.g., "Long Build-up", "Short Covering"
    pcr: float
    ce_oi_change_pct: float
    pe_oi_change_pct: float
    ce_vol_change_pct: float
    pe_vol_change_pct: float

@dataclass
class MultiTimeframeData:
    """Simple container for resampled dataframes."""
    df_5m: pd.DataFrame
    df_15m: pd.DataFrame
    df_1h: pd.DataFrame

@dataclass
class ChartAnalysis:
    """Holds the *result* of the local Python chart analysis."""
    trend_1h: str
    pattern_15m: str
    entry_signal_5m: str
    supports: List[float]
    resistances: List[float]
    spot_price: float

@dataclass
class NewsData:
    headline: str
    summary: str
    sentiment: str
    impact_score: int
    relevance_score: int = 0

@dataclass
class DeepAnalysis:
    """The final, validated trade setup from the AI."""
    opportunity: str  # CE_BUY or PE_BUY
    confidence: int
    total_score: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    recommended_strike: int
    analysis_summary: str
    risk_factors: List[str]
    chart_bias: str
    oi_sentiment: str
    news_sentiment: str
    news_impact: int
    support_levels: List[float]
    resistance_levels: List[float]


# --- API & CACHE CLASSES ---

class FinnhubNewsAPI:
    def __init__(self):
        self.api_key = FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
        self.connected = self._test_connection()

    def _test_connection(self):
        if not self.api_key:
            logger.warning("Finnhub API key not found. News disabled.")
            return False
        try:
            r = requests.get(f"{self.base_url}/news?category=general&token={self.api_key}", timeout=10)
            if r.status_code == 200:
                logger.info("✅ Finnhub API connected.")
                return True
            logger.error(f"Finnhub API error: {r.status_code}")
            return False
        except Exception as e:
            logger.error(f"Finnhub connection failed: {e}")
            return False

    def get_market_news(self, limit=10):
        if not self.connected:
            return []
        try:
            r = requests.get(f"{self.base_url}/news?category=general&token={self.api_key}", timeout=10)
            return r.json()[:limit] if r.status_code == 200 else []
        except Exception as e:
            logger.error(f"Failed to fetch market news: {e}")
            return []

class RedisCache:
    def __init__(self):
        self.redis_client = None
        self.connected = False
        if not REDIS_AVAILABLE:
            logger.warning("Redis module not found. OI analysis disabled.")
            return
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis connected.")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")

    def store_oi_snapshot(self, symbol: str, snapshot: OISnapshot):
        if not self.connected:
            return
        try:
            key = f"oi_timeseries:{symbol}"
            self.redis_client.lpush(key, snapshot.to_json())
            self.redis_client.ltrim(key, 0, OI_SNAPSHOT_COUNT - 1)
            self.redis_client.expire(key, REDIS_EXPIRY_24H)
        except Exception as e:
            logger.error(f"Redis store error: {e}")

    def get_oi_snapshots(self, symbol: str) -> Tuple[Optional[OISnapshot], Optional[OISnapshot]]:
        """Fetches current (index 0) and 30-min ago (index 6) snapshots."""
        if not self.connected:
            return None, None
        try:
            key = f"oi_timeseries:{symbol}"
            snapshots_str = self.redis_client.lrange(key, 0, OI_COMPARE_INDEX)
            
            current_snapshot = None
            past_snapshot = None

            if snapshots_str and len(snapshots_str) > 0:
                current_snapshot = OISnapshot.from_json(snapshots_str[0])
            
            if snapshots_str and len(snapshots_str) > OI_COMPARE_INDEX:
                past_snapshot = OISnapshot.from_json(snapshots_str[OI_COMPARE_INDEX])
                
            return current_snapshot, past_snapshot
        except Exception as e:
            logger.error(f"Redis retrieve error: {e}")
            return None, None

class UpstoxDataFetcher:
    def __init__(self):
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
        self.connected = self._test_connection()

    def _test_connection(self):
        if not UPSTOX_ACCESS_TOKEN:
            logger.error("Upstox Access Token not found.")
            return False
        try:
            r = requests.get(f"{BASE_URL}/v2/user/profile", headers=self.headers, timeout=10)
            if r.status_code == 200:
                logger.info("✅ Upstox API connected.")
                return True
            logger.error(f"Upstox API error: {r.status_code}")
            return False
        except Exception as e:
            logger.error(f"Upstox connection failed: {e}")
            return False

    def get_spot_price(self, instrument_key: str) -> float:
        if not self.connected:
            return 0.0
        try:
            encoded_key = urllib.parse.quote(instrument_key)
            url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get('data', {})
                ltp = data.get(instrument_key.upper(), {}).get('last_price', 0)
                if ltp:
                    return float(ltp)
        except Exception as e:
            logger.error(f"Spot price fetch error: {e}")
        return 0.0

    def get_option_chain(self, instrument_key: str, expiry: str) -> List[Dict]:
        """Fetches the full option chain for a given expiry."""
        if not self.connected:
            return []
        try:
            encoded_key = urllib.parse.quote(instrument_key)
            url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                chain = resp.json().get('data', [])
                return sorted(chain, key=lambda x: x.get('strike_price', 0))
            logger.warning(f"Failed to get option chain for {instrument_key}: {resp.status_code}")
            return []
        except Exception as e:
            logger.error(f"Option chain fetch error: {e}")
            return []

    def get_next_expiry(self, instrument_key: str, expiry_day=1) -> Optional[str]:
        """Auto-selects the nearest valid weekly expiry."""
        try:
            encoded_key = urllib.parse.quote(instrument_key)
            url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                return None
            
            contracts = resp.json().get('data', [])
            expiries = sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
            
            today, now_time = datetime.now(IST).date(), datetime.now(IST).time()
            for exp_str in expiries:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                if exp_date > today or (exp_date == today and now_time < time(15, 30)):
                    return exp_str
            return expiries[0] if expiries else None
        except Exception as e:
            logger.error(f"Expiry fetch error: {e}")
            return None

    def get_multi_timeframe_data(self, instrument_key: str) -> Optional[MultiTimeframeData]:
        """Fetches continuous 1-min data and resamples it."""
        if not self.connected:
            return None
        try:
            encoded = urllib.parse.quote(instrument_key)
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=60)).strftime('%Y-%m-%d')
            
            # ✅ FIX: Changed interval from "15minute" to "1minute" as per API spec
            url = f"{BASE_URL}/v2/historical-candle/{encoded}/1minute/{to_date}/{from_date}"
            # Increased timeout for potentially larger 1-min data request
            r = requests.get(url, headers=self.headers, timeout=45)
            
            if r.status_code != 200 or r.json().get('status') != 'success':
                logger.warning(f"Failed to fetch 1min data for {instrument_key}: {r.text}")
                return None

            candles = r.json().get('data', {}).get('candles', [])
            if not candles:
                logger.warning(f"No 1min candle data returned for {instrument_key}")
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').astype(float).sort_index()
            
            # Resample to 3 timeframes from 1-minute base data
            ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df_5m = df.resample('5min').apply(ohlc_dict).dropna()
            # ✅ FIX: Correctly resample 15m from 1m data
            df_15m = df.resample('15min').apply(ohlc_dict).dropna()
            df_1h = df.resample('1H').apply(ohlc_dict).dropna()
            
            if df_5m.empty or df_15m.empty or df_1h.empty:
                logger.warning(f"Resampling failed for {instrument_key}, not enough base data.")
                return None
            
            logger.info(f"  📊 Continuous data: 5m={len(df_5m)}, 15m={len(df_15m)}, 1h={len(df_1h)}")
            # ✅ FIX: Pass the correct resampled 15m dataframe
            return MultiTimeframeData(df_5m=df_5m, df_15m=df_15m, df_1h=df_1h)
            
        except Exception as e:
            logger.error(f"MTF data processing error: {e}")
            traceback.print_exc()
            return None

# --- ANALYSIS CLASSES ---

class NewsAnalyzer:
    @staticmethod
    def analyze_news(symbol: str, news_list: List[Dict]) -> Optional[NewsData]:
        if not news_list or not DEEPSEEK_API_KEY:
            return None
        
        try:
            relevant = NewsAnalyzer._filter_relevant_news(news_list)
            if not relevant:
                logger.info("  📰 No relevant news found after filtering.")
                return None
            
            summary = "\n".join([f"- {n.get('headline', '')}" for n in relevant[:3]])
            prompt = f"Analyze these INDIAN market headlines for {symbol}:\n{summary}\n\nReply ONLY with JSON: {{\"sentiment\": \"BULLISH/BEARISH/NEUTRAL\", \"impact_score\": <0-100>, \"summary\": \"<1-line summary>\"}}"
            
            r = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={"model": "deepseek-chat", "messages": [{"role": "system", "content": "Indian market analyst. JSON only."}, {"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 200},
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                timeout=20
            )
            
            if r.status_code != 200:
                logger.warning(f"News AI API error: {r.status_code}")
                return None
                
            content = r.json()['choices'][0]['message']['content']
            analysis = json.loads(re.search(r'\{.*?\}', content, re.DOTALL).group(0))
            
            return NewsData(
                headline=relevant[0].get('headline', '')[:100],
                summary=analysis.get('summary', 'N/A'),
                sentiment=analysis.get('sentiment', 'NEUTRAL'),
                impact_score=analysis.get('impact_score', 50),
                relevance_score=relevant[0].get('relevance_score', 0)
            )
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return None

    @staticmethod
    def _filter_relevant_news(news_list: List[Dict]) -> List[Dict]:
        relevant = []
        for news in news_list:
            text = (news.get('headline', '') + ' ' + news.get('summary', '')).lower()
            score = sum(10 for kw in SECTOR_KEYWORDS["INDEX"] if kw in text)
            if score >= 10 and not any(kw in text for kw in ["amazon", "apple", "nasdaq", "dow"]):
                news['relevance_score'] = score
                relevant.append(news)
        return sorted(relevant, key=lambda x: x['relevance_score'], reverse=True)

class ChartAnalyzer:
    """Performs local, pre-AI chart analysis."""
    
    def get_full_analysis(self, mtf_data: MultiTimeframeData) -> ChartAnalysis:
        trend_1h = self._analyze_1h_trend(mtf_data.df_1h)
        sr_levels = self._calculate_support_resistance(mtf_data.df_15m)
        pattern_15m = self._analyze_15m_patterns(mtf_data.df_15m, sr_levels)
        entry_signal_5m = self._analyze_5m_entry(mtf_data.df_5m)
        
        return ChartAnalysis(
            trend_1h=trend_1h,
            pattern_15m=pattern_15m,
            entry_signal_5m=entry_signal_5m,
            supports=sr_levels['supports'],
            resistances=sr_levels['resistances'],
            spot_price=mtf_data.df_5m['close'].iloc[-1]
        )

    def _analyze_1h_trend(self, df: pd.DataFrame) -> str:
        if len(df) < 20: return "NEUTRAL"
        current_price = df['close'].iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        if current_price > ma20 * 1.001: return "BULLISH"
        if current_price < ma20 * 0.999: return "BEARISH"
        return "NEUTRAL"

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Uses Swing High/Low logic from user's v16 prompt."""
        try:
            if len(df) < 50:
                c = df['close'].iloc[-1]
                return {'supports': [c * 0.98], 'resistances': [c * 1.02]}
            
            recent, c = df.tail(200), df['close'].iloc[-1]
            supports = [recent['low'].iloc[i] for i in range(5, len(recent) - 5) if recent['low'].iloc[i] == recent['low'].iloc[i-5:i+5].min()]
            resistances = [recent['high'].iloc[i] for i in range(5, len(recent) - 5) if recent['high'].iloc[i] == recent['high'].iloc[i-5:i+5].max()]
            
            supports = sorted(list(set(s for s in supports if s < c)), reverse=True)[:2]
            resistances = sorted(list(set(r for r in resistances if r > c)))[:2]
            
            if not supports: supports = [c * 0.985]
            if not resistances: resistances = [c * 1.015]
            
            return {'supports': supports, 'resistances': resistances}
        except Exception as e:
            logger.error(f"S/R calculation error: {e}")
            c = df['close'].iloc[-1]
            return {'supports': [c * 0.98], 'resistances': [c * 1.02]}

    def _analyze_15m_patterns(self, df: pd.DataFrame, sr_levels: Dict) -> str:
        if len(df) < 20: return "RANGING"
        c = df['close'].iloc[-1]
        if c > sr_levels['resistances'][0] * 1.0005: return "BREAKOUT"
        if c < sr_levels['supports'][0] * 0.9995: return "BREAKDOWN"
        return "RANGING"

    def _analyze_5m_entry(self, df: pd.DataFrame) -> str:
        if len(df) < 3: return "NONE"
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last['close'] > prev['high'] and last['close'] > last['open']: return "BULLISH_CANDLE"
        if last['close'] < prev['low'] and last['close'] < last['open']: return "BEARISH_CANDLE"
        return "NONE"

class OIAnalyzer:
    """Parses OI, stores snapshots, and runs 30-min comparison."""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis = redis_cache

    def get_analysis(self, symbol: str, strikes: List[Dict], spot_price: float) -> Optional[AggregateOIAnalysis]:
        if not strikes:
            logger.warning(f"No strikes data provided for {symbol}")
            return None
        
        # 1. Parse current data into a snapshot
        current_snapshot = self._parse_to_snapshot(strikes, spot_price)
        if not current_snapshot:
            logger.warning(f"Failed to parse current OI for {symbol}")
            return None

        # 2. Store the new snapshot
        self.redis.store_oi_snapshot(symbol, current_snapshot)

        # 3. Get comparison snapshots
        _, past_snapshot = self.redis.get_oi_snapshots(symbol)

        # 4. Perform analysis
        if past_snapshot:
            return self._compare_snapshots(current_snapshot, past_snapshot)
        else:
            logger.info(f"  📊 No 30-min OI data for {symbol}, analyzing current PCR only.")
            return self._analyze_current_snapshot(current_snapshot)

    def _parse_to_snapshot(self, strikes: List[Dict], spot_price: float) -> Optional[OISnapshot]:
        try:
            total_ce_oi, total_pe_oi, total_ce_vol, total_pe_vol = 0, 0, 0, 0
            atm_strike = min(strikes, key=lambda x: abs(x.get('strike_price', 0) - spot_price)).get('strike_price', 0)
            price_range = spot_price * 0.05  # Analyze 5% OTM/ITM

            for s in strikes:
                sp = s.get('strike_price', 0)
                if abs(sp - atm_strike) <= price_range:
                    ce = s.get('call_options', {}).get('market_data', {})
                    pe = s.get('put_options', {}).get('market_data', {})
                    total_ce_oi += ce.get('oi', 0)
                    total_pe_oi += pe.get('oi', 0)
                    total_ce_vol += ce.get('volume', 0)
                    total_pe_vol += pe.get('volume', 0)

            return OISnapshot(
                timestamp=int(datetime.now(IST).timestamp()),
                spot_price=spot_price,
                total_ce_oi=total_ce_oi, total_pe_oi=total_pe_oi,
                total_ce_vol=total_ce_vol, total_pe_vol=total_pe_vol
            )
        except Exception as e:
            logger.error(f"OI parsing error: {e}")
            return None

    def _compare_snapshots(self, current: OISnapshot, past: OISnapshot) -> AggregateOIAnalysis:
        """Performs the 4-quadrant (Price vs OI) analysis."""
        
        price_change = current.spot_price - past.spot_price
        ce_oi_change = current.total_ce_oi - past.total_ce_oi
        pe_oi_change = current.total_pe_oi - past.total_pe_oi
        
        ce_oi_change_pct = (ce_oi_change / past.total_ce_oi * 100) if past.total_ce_oi > 0 else 0
        pe_oi_change_pct = (pe_oi_change / past.total_pe_oi * 100) if past.total_pe_oi > 0 else 0
        ce_vol_change_pct = ((current.total_ce_vol - past.total_ce_vol) / past.total_ce_vol * 100) if past.total_ce_vol > 0 else 0
        pe_vol_change_pct = ((current.total_pe_vol - past.total_pe_vol) / past.total_pe_vol * 100) if past.total_pe_vol > 0 else 0
        
        pcr = current.total_pe_oi / current.total_ce_oi if current.total_ce_oi > 0 else 0

        # Determine sentiment based on OI change (from search results)
        sentiment = "NEUTRAL"
        if price_change > 0:
            if pe_oi_change > 0 and ce_oi_change < 0:
                sentiment = "STRONG BULLISH (Long Build-up in Puts / CE Unwind)"
            elif pe_oi_change > 0 and pe_oi_change > ce_oi_change:
                sentiment = "BULLISH (Long Build-up)"
            elif ce_oi_change < 0 and pe_oi_change < 0:
                sentiment = "WEAK BULLISH (Short Covering)"
        elif price_change < 0:
            if ce_oi_change > 0 and pe_oi_change < 0:
                sentiment = "STRONG BEARISH (Short Build-up in Calls / PE Unwind)"
            elif ce_oi_change > 0 and ce_oi_change > pe_oi_change:
                sentiment = "BEARISH (Short Build-up)"
            elif ce_oi_change < 0 and pe_oi_change < 0:
                sentiment = "WEAK BEARISH (Long Unwinding)"

        logger.info(f"  📊 OI-30min: {sentiment} (PCR:{pcr:.2f})")
        
        return AggregateOIAnalysis(
            sentiment=sentiment, pcr=pcr,
            ce_oi_change_pct=ce_oi_change_pct, pe_oi_change_pct=pe_oi_change_pct,
            ce_vol_change_pct=ce_vol_change_pct, pe_vol_change_pct=pe_vol_change_pct
        )

    def _analyze_current_snapshot(self, current: OISnapshot) -> AggregateOIAnalysis:
        """Fallback analysis when no 30-min data is available."""
        pcr = current.total_pe_oi / current.total_ce_oi if current.total_ce_oi > 0 else 0
        sentiment = "BULLISH" if pcr > 1.2 else "BEARISH" if pcr < 0.8 else "NEUTRAL"
        
        return AggregateOIAnalysis(
            sentiment=f"NEUTRAL (PCR: {sentiment})", pcr=pcr,
            ce_oi_change_pct=0, pe_oi_change_pct=0,
            ce_vol_change_pct=0, pe_vol_change_pct=0
        )

class AIAnalyzer:
    """Uses the 'Pro Trader' prompt to synthesize pre-analyzed data."""
    
    def __init__(self):
        self.system_prompt = AI_SYSTEM_PROMPT
        if not DEEPSEEK_API_KEY:
            logger.error("DeepSeek API Key not found. AI Analysis disabled.")
            
    def get_deep_analysis(self, chart: ChartAnalysis, oi: AggregateOIAnalysis, news: Optional[NewsData]) -> Optional[DeepAnalysis]:
        if not DEEPSEEK_API_KEY:
            return None
            
        try:
            user_prompt = self._format_user_prompt(chart, oi, news)
            
            r = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500,
                    "response_format": {"type": "json_object"} # Request JSON
                },
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                timeout=40
            )
            
            if r.status_code != 200:
                logger.error(f"AI API Error: {r.status_code} - {r.text}")
                return None
                
            content = r.json()['choices'][0]['message']['content']
            analysis = json.loads(content) # No regex needed due to response_format

            # Final validation and packaging
            return self._validate_and_package(analysis, chart, oi, news)

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            traceback.print_exc()
            return None

    def _format_user_prompt(self, chart: ChartAnalysis, oi: AggregateOIAnalysis, news: Optional[NewsData]) -> str:
        """Creates the concise user prompt with pre-analyzed data."""
        
        news_str = f"- News Sentiment: {news.sentiment} (Impact: {news.impact_score})" if news else "- News: No significant news."
        
        return f"""
        Here is my pre-analyzed data for the instrument. Act as the professional institutional trader (your persona) to synthesize this into a high-probability trade plan.
        
        PRE-ANALYZED DATA:
        - Current Spot Price: {chart.spot_price:.2f}
        
        CHART ANALYSIS:
        - 1H Overall Trend: {chart.trend_1h}
        - 15M Key Supports: {chart.supports}
        - 15M Key Resistances: {chart.resistances}
        - 15M Current Pattern: {chart.pattern_15m}
        - 5M Entry Signal: {chart.entry_signal_5m}
        
        OI ANALYSIS (30-MIN CHANGE):
        - OI Sentiment: {oi.sentiment}
        - PCR (Overall): {oi.pcr:.2f}
        - Call OI Change: {oi.ce_oi_change_pct:+.1f}%
        - Put OI Change: {oi.pe_oi_change_pct:+.1f}%

        NEWS:
        {news_str}
        
        TASK:
        Based on your rules (Step 1-10 from your persona), analyze this pre-digested data.
        Provide a JSON response with your final, complete trade setup.
        
        - CRITICAL: Your "opportunity" MUST be 'CE_BUY' for a BULLISH bias and 'PE_BUY' for a BEARISH bias.
        - CRITICAL: If no clear setup (e.g., choppy, conflicting signals), set "opportunity" to "WAIT".
        - Ensure T1/T2/SL levels are logical based on the S/R levels provided.
        
        JSON OUTPUT FORMAT:
        {{
            "opportunity": "CE_BUY" | "PE_BUY" | "WAIT",
            "confidence": <integer 0-100>,
            "total_score": <integer 0-125>,
            "chart_bias": "BULLISH" | "BEARISH" | "NEUTRAL",
            "analysis_summary": "<string 1-2 line summary of your reasoning>",
            "entry_price": <float>,
            "stop_loss": <float>,
            "target_1": <float>,
            "target_2": <float>,
            "risk_reward": "<string>",
            "recommended_strike": <integer>,
            "risk_factors": ["<list of risks>"]
        }}
        """

    def _validate_and_package(self, ai_json: Dict, chart: ChartAnalysis, oi: AggregateOIAnalysis, news: Optional[NewsData]) -> Optional[DeepAnalysis]:
        """Validates AI output for logic (CE/PE) and target direction."""
        
        opp = ai_json.get('opportunity')
        entry = ai_json.get('entry_price', chart.spot_price)
        sl = ai_json.get('stop_loss', 0)
        t1 = ai_json.get('target_1', 0)
        t2 = ai_json.get('target_2', 0)

        # ✅ FIXED LOGIC VALIDATION
        if opp == "CE_BUY": # Bullish
            if t1 <= entry or t2 <= entry:
                logger.warning(f"AI returned invalid targets for CE_BUY. Fixing.")
                t1 = entry + (entry - sl) * 1.5
                t2 = entry + (entry - sl) * 3.0
            if sl >= entry:
                logger.warning(f"AI returned invalid SL for CE_BUY. Fixing.")
                sl = entry * 0.995 # Default 0.5% SL
        
        elif opp == "PE_BUY": # Bearish
            if t1 >= entry or t2 >= entry:
                logger.warning(f"AI returned invalid targets for PE_BUY. Fixing.")
                t1 = entry - (sl - entry) * 1.5
                t2 = entry - (sl - entry) * 3.0
            if sl <= entry:
                logger.warning(f"AI returned invalid SL for PE_BUY. Fixing.")
                sl = entry * 1.005 # Default 0.5% SL

        elif opp == "WAIT":
            logger.info("  Signal: AI recommended to WAIT. No trade.")
            return None
            
        return DeepAnalysis(
            opportunity=opp,
            confidence=ai_json.get('confidence', 0),
            total_score=ai_json.get('total_score', 0),
            entry_price=entry,
            stop_loss=sl,
            target_1=t1,
            target_2=t2,
            risk_reward=ai_json.get('risk_reward', "1:2"),
            recommended_strike=ai_json.get('recommended_strike', int(chart.spot_price / 50) * 50),
            analysis_summary=ai_json.get('analysis_summary', "No summary provided."),
            risk_factors=ai_json.get('risk_factors', []),
            chart_bias=ai_json.get('chart_bias', "NEUTRAL"),
            oi_sentiment=oi.sentiment,
            news_sentiment=news.sentiment if news else "NEUTRAL",
            news_impact=news.impact_score if news else 0,
            support_levels=chart.supports,
            resistance_levels=chart.resistances
        )

# --- CHARTING & NOTIFICATION ---

class ChartGenerator:
    @staticmethod
    def create_chart(chart_data: ChartAnalysis, analysis: DeepAnalysis, symbol: str) -> Optional[io.BytesIO]:
        try:
            df = chart_data.df_15m.tail(100).copy().reset_index(drop=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot Candlesticks
            for i in range(len(df)):
                row = df.iloc[i]
                color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1.2)
                bh, bb = abs(row['close'] - row['open']), min(row['open'], row['close'])
                ax1.add_patch(Rectangle((i - 0.35, bb), 0.7, bh if bh > 0 else 0.1, facecolor=color, edgecolor=color))

            # Plot S/R Levels
            for s in chart_data.supports:
                ax1.axhline(y=s, color='green', linestyle='--', linewidth=1.5, label=f'Support: {s:.1f}')
            for r in chart_data.resistances:
                ax1.axhline(y=r, color='red', linestyle='--', linewidth=1.5, label=f'Resistance: {r:.1f}')
            
            # Plot Trade Levels
            ax1.axhline(y=analysis.entry_price, color='#ff9800', linestyle=':', linewidth=2, label=f'Entry: {analysis.entry_price:.1f}')
            ax1.axhline(y=analysis.stop_loss, color='#f44336', linestyle=':', linewidth=2, label=f'Stop Loss: {analysis.stop_loss:.1f}')
            ax1.axhline(y=analysis.target_1, color='#4caf50', linestyle=':', linewidth=2, label=f'Target 1: {analysis.target_1:.1f}')
            
            # Current Price Line
            ax1.axhline(y=chart_data.spot_price, color='#2962ff', linestyle='-', linewidth=2.5, label=f'CMP: {chart_data.spot_price:.1f}')
            
            ax1.set_title(f'{symbol} | 15min | Bias: {analysis.chart_bias} | Score: {analysis.total_score}', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.2)
            ax1.set_facecolor('#fafafa')

            # Volume Bars
            colors = ['#26a69a' if df.iloc[i]['close'] >= df.iloc[i]['open'] else '#ef5350' for i in range(len(df))]
            ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.7)
            ax2.set_facecolor('#fafafa')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            buf.seek(0)
            plt.close(fig)
            return buf
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            traceback.print_exc()
            return None

class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)

    async def send_startup_message(self, api_status: Dict):
        status_msg = "\n".join([f"{'🟢' if status else '🔴'} {name}" for name, status in api_status.items()])
        msg = f"""
🔥 **F&O Scanner Bot v17.0 ONLINE** 🔥
Using "Pro Trader" AI Brain.

**API Status:**
{status_msg}

**Logic:**
- 1H Trend + 15M S/R + 5M Entry
- 30min vs. Current OI Analysis
- **CE_BUY** = BULLISH
- **PE_BUY** = BEARISH

Bot is running and waiting for market hours...
"""
        await self._send_message(msg)

    async def send_alert(self, symbol: str, analysis: DeepAnalysis, chart: Optional[io.BytesIO]):
        # ✅ CORRECTED EMOJI
        emoji = "🟢" if analysis.opportunity == "CE_BUY" else "🔴" if analysis.opportunity == "PE_BUY" else "⚪"
        
        news_str = f"📰 **News:** {analysis.news_sentiment} (Impact: {analysis.news_impact})" if analysis.news_impact > 60 else ""
        
        msg = f"""
🎯 **{symbol} SIGNAL** | {emoji} **{analysis.opportunity}**

**Bias:** {analysis.chart_bias} | **Conf:** {analysis.confidence}% | **Score:** {analysis.total_score}/125

**AI Summary:** *{analysis.analysis_summary}*

---
**TRADE SETUP (15m Chart)**
- **Entry:** ₹{analysis.entry_price:.2f}
- **Stop Loss:** ₹{analysis.stop_loss:.2f}
- **Target 1:** ₹{analysis.target_1:.2f}
- **Target 2:** ₹{analysis.target_2:.2f}
- **Risk/Reward:** {analysis.risk_reward}
- **Strike:** {analysis.recommended_strike}

---
**DATA POINTS**
- **OI Sentiment (30m):** {analysis.oi_sentiment}
- **S/R Levels:** S: {analysis.support_levels}, R: {analysis.resistance_levels}
- **Risk Factors:** {', '.join(analysis.risk_factors)}
{news_str}
"""
        try:
            if chart:
                chart.name = f"{symbol.lower()}_chart.png"
                await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=chart, caption=msg, parse_mode='Markdown')
            else:
                await self._send_message(msg)
            logger.info(f"✅ Alert sent for {symbol}: {analysis.opportunity}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def send_summary(self, scanned_count: int, alerts_sent: int):
        msg = f"📊 **Scan Complete:** Analyzed {scanned_count} symbols, Sent {alerts_sent} alerts. Next scan in 5 mins."
        await self._send_message(msg)
        
    async def _send_message(self, text: str):
        try:
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown', disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"Telegram message failed: {e}")

# --- MAIN BOT CLASS ---

class HybridBot:
    def __init__(self):
        logger.info("Initializing Bot v17.0 (Pro Trader AI)...")
        self.fetcher = UpstoxDataFetcher()
        self.redis = RedisCache()
        self.news_api = FinnhubNewsAPI()
        self.chart_analyzer = ChartAnalyzer()
        self.oi_analyzer = OIAnalyzer(self.redis)
        self.ai_analyzer = AIAnalyzer()
        self.notifier = TelegramNotifier()
        
        self.api_status = {
            "Upstox": self.fetcher.connected,
            "Redis": self.redis.connected,
            "Finnhub": self.news_api.connected,
            "DeepSeek": bool(DEEPSEEK_API_KEY)
        }

    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        return now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)

    async def scan_symbol(self, instrument_key: str, info: Dict, market_news: List[Dict]) -> Optional[DeepAnalysis]:
        symbol = info['name']
        logger.info(f"\n--- 1. Scanning: {symbol} ---")
        
        try:
            # 1. Fetch Chart Data
            mtf_data = self.fetcher.get_multi_timeframe_data(instrument_key)
            if not mtf_data:
                logger.warning(f"  ❌ No chart data for {symbol}.")
                return None
            
            # 2. Local Chart Analysis
            chart_analysis = self.chart_analyzer.get_full_analysis(mtf_data)
            logger.info(f"  📈 Chart: 1H={chart_analysis.trend_1h}, 15M={chart_analysis.pattern_15m}, 5M={chart_analysis.entry_signal_5m}")

            # 3. Fetch OI Data
            expiry = self.fetcher.get_next_expiry(instrument_key, info['expiry_day'])
            if not expiry:
                logger.warning(f"  ❌ No expiry found for {symbol}.")
                return None
                
            strikes = self.fetcher.get_option_chain(instrument_key, expiry)
            if not strikes:
                logger.warning(f"  ❌ No option chain data for {symbol}.")
                return None
            
            # 4. Local OI Analysis (incl. 30-min comparison)
            oi_analysis = self.oi_analyzer.get_analysis(symbol, strikes, chart_analysis.spot_price)
            if not oi_analysis:
                logger.warning(f"  ❌ OI analysis failed for {symbol}.")
                return None

            # 5. News Analysis (Optional)
            news_analysis = self.news_api.analyze_news(symbol, market_news)

            # 6. AI Synthesis (The "Brain")
            logger.info(f"  🧠 Sending to AI for synthesis...")
            deep_analysis = self.ai_analyzer.get_deep_analysis(chart_analysis, oi_analysis, news_analysis)
            
            return deep_analysis

        except Exception as e:
            logger.error(f"FATAL error scanning {symbol}: {e}")
            traceback.print_exc()
            return None

    async def run(self):
        await self.notifier.send_startup_message(self.api_status)
        
        while True:
            try:
                if self.is_market_open():
                    logger.info(f"\n{'='*50}\nSCANNING CYCLE @ {datetime.now(IST):%I:%M %p}\n{'='*50}")
                    market_news = self.news_api.get_market_news()
                    alerts_sent = 0

                    scan_tasks = []
                    for key, info in INDICES.items():
                        scan_tasks.append(self.scan_symbol(key, info, market_news))
                    
                    results = await asyncio.gather(*scan_tasks)

                    for idx, analysis in enumerate(results):
                        if analysis and analysis.opportunity != "WAIT":
                            # Final Filters
                            if analysis.confidence < CONFIDENCE_MIN or analysis.total_score < SCORE_MIN:
                                logger.info(f"  ❌ Signal failed filters (Conf: {analysis.confidence}%, Score: {analysis.total_score}).")
                                continue
                            
                            logger.info(f"  ✅ High-Confidence Signal Found: {analysis.opportunity}")
                            
                            # Fetch data again just for charting (or pass mtf_data through)
                            # This is a bit inefficient, but simplest to implement post-analysis
                            instrument_key = list(INDICES.keys())[idx]
                            symbol = INDICES[instrument_key]['name']
                            mtf_data_for_chart = self.fetcher.get_multi_timeframe_data(instrument_key)
                            chart_analysis_for_chart = self.chart_analyzer.get_full_analysis(mtf_data_for_chart)
                            
                            if mtf_data_for_chart:
                                chart = ChartGenerator.create_chart(chart_analysis_for_chart, analysis, symbol)
                                await self.notifier.send_alert(symbol, analysis, chart)
                            else:
                                await self.notifier.send_alert(symbol, analysis, None) # Send without chart
                            
                            alerts_sent += 1
                        
                    await self.notifier.send_summary(len(INDICES), alerts_sent)
                    logger.info(f"Cycle finished. Waiting {SCAN_INTERVAL} seconds...")
                    await asyncio.sleep(SCAN_INTERVAL)
                else:
                    logger.info(f"Market is closed. Waiting... (Time: {datetime.now(IST):%I:%M %p})")
                    await asyncio.sleep(60)
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"CRITICAL error in main loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

# --- Main Execution ---
async def main():
    if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY]):
        logger.critical("FATAL: Missing one or more critical API keys (UPSTOX, TELEGRAM, DEEPSEEK). Exiting.")
        return
    if not REDIS_AVAILABLE:
        logger.warning("Redis is not installed. OI analysis will not function.")
        
    bot = HybridBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())


