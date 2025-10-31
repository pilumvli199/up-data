#!/usr/bin/env python3
"""
HYBRID TRADING BOT v16.0 - REWRITTEN & LOGIC FIXED
=====================================================
✅ CORE LOGIC FIXED:
    - BULLISH Market => CE_BUY (Call Option Buy)
    - BEARISH Market => PE_BUY (Put Option Buy)

✅ PROFESSIONAL MULTI-TIMEFRAME ANALYSIS:
    - 1hr TF: Overall trend confirmation (using 20 MA)
    - 15min TF: Support/Resistance using DAILY PIVOT POINTS
    - 5min TF: Entry signal based on mini-breakout

✅ EFFICIENT DATA FETCHING:
    - Single, continuous data stream (15min candles) resampled to all TFs.
    - No more data gaps between historical and intraday fetches.

✅ UPGRADED CHARTING:
    - Plots daily Pivot Point levels (S1, R1, S2, R2) for accurate S/R.
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import time as time_sleep
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import traceback
import re

# Redis import with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - running without OI tracking")

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
BASE_URL = "https://api.upstox.com"
IST = pytz.timezone('Asia/Kolkata')

REDIS_EXPIRY_30MIN = 1800  # 30 minutes for OI comparison

# FOCUSED INDICES
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY 50", "expiry_day": 1},
    "BSE_INDEX|SENSEX": {"name": "SENSEX", "expiry_day": 4}
}

SECTOR_KEYWORDS = {
    "INDEX": ["nifty", "sensex", "india", "indian", "market", "bse", "nse",
              "equity", "stocks", "rupee", "rbi", "inflation", "gdp", "sebi"]
}

# Analysis thresholds
CONFIDENCE_MIN = 75
SCORE_MIN = 90
SCAN_INTERVAL = 300  # 5 MINUTES

# --- DATA CLASSES ---
@dataclass
class OIData:
    strike: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int

@dataclass
class AggregateOIAnalysis:
    total_ce_oi: int
    total_pe_oi: int
    pcr: float
    ce_oi_change_pct: float
    pe_oi_change_pct: float
    overall_sentiment: str

@dataclass
class PivotPoints:
    pivot: float
    s1: float
    r1: float
    s2: float
    r2: float

@dataclass
class MultiTimeframeData:
    df_5m: pd.DataFrame
    df_15m: pd.DataFrame
    df_1h: pd.DataFrame
    pivot_points: PivotPoints

@dataclass
class NewsData:
    headline: str
    summary: str
    sentiment: str
    impact_score: int
    relevance_score: int = 0

@dataclass
class DeepAnalysis:
    opportunity: str
    confidence: int
    total_score: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    recommended_strike: int
    analysis_summary: str
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    chart_bias: str = "NEUTRAL"
    news_sentiment: str = "NEUTRAL"
    news_impact: int = 0
    tf_1h_trend: str = "NEUTRAL"
    tf_15m_structure: str = "RANGING"
    tf_5m_signal: str = "NONE"

# --- API & CACHE CLASSES ---
class FinnhubNewsAPI:
    def __init__(self):
        self.api_key = FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
        self.connected = bool(self.api_key)
        if self.connected:
             logger.info("✅ Finnhub API Initialized.")
        else:
             logger.warning("⚠️ Finnhub API key not found.")

    def get_market_news(self, limit: int = 15) -> List[Dict]:
        if not self.connected: return []
        try:
            url = f"{self.base_url}/news?category=general&token={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                news_list = response.json()[:limit]
                logger.info(f"📰 Fetched {len(news_list)} market news articles.")
                return news_list
            logger.error(f"Finnhub API error: {response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Finnhub connection failed: {e}")
            return []

class RedisCache:
    def __init__(self):
        self.redis_client = None
        self.connected = False
        if not REDIS_AVAILABLE:
            logger.warning("Redis module not installed. OI change tracking disabled.")
            return
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis connected successfully!")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")

    def store_data(self, key: str, value: str, expiry: int):
        if not self.connected: return
        try:
            self.redis_client.setex(key, expiry, value)
        except Exception as e:
            logger.error(f"Redis store error for key '{key}': {e}")

    def get_data(self, key: str) -> Optional[str]:
        if not self.connected: return None
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis get error for key '{key}': {e}")
            return None

class UpstoxDataFetcher:
    def __init__(self):
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"}
        self.connected = self._test_connection()

    def _test_connection(self) -> bool:
        if not UPSTOX_ACCESS_TOKEN:
            logger.error("❌ Upstox access token not found.")
            return False
        try:
            response = requests.get(f"{BASE_URL}/v2/user/profile", headers=self.headers, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Upstox API connected successfully!")
                return True
            logger.error(f"❌ Upstox API error: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"❌ Upstox connection failed: {e}")
            return False

    def get_spot_price(self, instrument_key: str) -> float:
        if not self.connected: return 0.0
        try:
            encoded_key = urllib.parse.quote(instrument_key)
            url = f"{BASE_URL}/v2/market-quote/quotes?instrument_key={encoded_key}"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                quote_data = resp.json().get('data', {})
                if quote_data:
                    ltp = quote_data.get(instrument_key.upper(), {}).get('last_price', 0)
                    return float(ltp)
        except Exception as e:
            logger.error(f"Spot price fetch error for {instrument_key}: {e}")
        return 0.0

    def get_previous_day_ohlc(self, instrument_key: str) -> Optional[Dict]:
        if not self.connected: return None
        try:
            to_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')
            url = f"{BASE_URL}/v2/historical-candle/{urllib.parse.quote(instrument_key)}/day/{to_date}"
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code == 200 and resp.json().get('data', {}).get('candles'):
                last_candle = resp.json()['data']['candles'][-1]
                return {'open': last_candle[1], 'high': last_candle[2], 'low': last_candle[3], 'close': last_candle[4]}
        except Exception as e:
            logger.error(f"Previous day OHLC fetch error: {e}")
        return None

    def get_multi_timeframe_data(self, instrument_key: str, symbol: str) -> Optional[MultiTimeframeData]:
        if not self.connected: return None
        try:
            # EFFICIENT: Fetch one large, continuous dataset (15min for last 60 days)
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            from_date = (datetime.now(IST) - timedelta(days=60)).strftime('%Y-%m-%d')
            url = f"{BASE_URL}/v2/historical-candle/{urllib.parse.quote(instrument_key)}/15minute/{to_date}/{from_date}"
            response = requests.get(url, headers=self.headers, timeout=20)

            if response.status_code != 200 or not response.json().get('data', {}).get('candles'):
                logger.warning(f"No 15-min candle data found for {symbol}.")
                return None

            df = pd.DataFrame(response.json()['data']['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(IST)
            df = df.set_index('timestamp').astype(float).sort_index()
            logger.info(f"  📊 Fetched {len(df)} continuous 15m candles.")

            # Resample to required timeframes
            ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df_1h = df.resample('1H').apply(ohlc_dict).dropna()
            df_5m = df.resample('5min').apply(ohlc_dict).dropna()

            # Get pivots
            prev_day_ohlc = self.get_previous_day_ohlc(instrument_key)
            if not prev_day_ohlc:
                logger.warning(f"Could not calculate Pivot Points for {symbol}.")
                return None
            pivot_points = ChartAnalyzer.calculate_pivot_points(prev_day_ohlc)

            return MultiTimeframeData(df_5m=df_5m, df_15m=df, df_1h=df_1h, pivot_points=pivot_points)

        except Exception as e:
            logger.error(f"Multi-timeframe data fetch failed for {symbol}: {e}")
            return None

# --- ANALYSIS CLASSES ---
class OIAnalyzer:
    def __init__(self, redis_cache: RedisCache):
        self.redis = redis_cache

    def get_oi_analysis(self, symbol: str, strikes: List[Dict], spot_price: float) -> Optional[AggregateOIAnalysis]:
        if not strikes: return None
        
        # Parse current OI
        oi_list = self._parse_option_chain(strikes, spot_price)
        if not oi_list: return None
        
        total_ce_oi_new = sum(oi.ce_oi for oi in oi_list)
        total_pe_oi_new = sum(oi.pe_oi for oi in oi_list)
        
        # Compare with cached data
        key = f"oi_data:{symbol}"
        cached_str = self.redis.get_data(key)
        
        ce_oi_change_pct = 0.0
        pe_oi_change_pct = 0.0

        if cached_str:
            try:
                old_data = json.loads(cached_str)
                total_ce_oi_old = old_data.get('total_ce_oi', 0)
                total_pe_oi_old = old_data.get('total_pe_oi', 0)

                if total_ce_oi_old > 0:
                    ce_oi_change_pct = ((total_ce_oi_new - total_ce_oi_old) / total_ce_oi_old) * 100
                if total_pe_oi_old > 0:
                    pe_oi_change_pct = ((total_pe_oi_new - total_pe_oi_old) / total_pe_oi_old) * 100
            except json.JSONDecodeError:
                logger.warning(f"Could not decode cached OI for {symbol}.")

        # Store new data for next run
        data_to_cache = json.dumps({'total_ce_oi': total_ce_oi_new, 'total_pe_oi': total_pe_oi_new})
        self.redis.store_data(key, data_to_cache, REDIS_EXPIRY_30MIN)
        
        pcr = total_pe_oi_new / total_ce_oi_new if total_ce_oi_new > 0 else 0
        
        # Determine sentiment
        sentiment = "NEUTRAL"
        if pe_oi_change_pct > 3 and pe_oi_change_pct > ce_oi_change_pct: sentiment = "BULLISH"
        elif ce_oi_change_pct > 3 and ce_oi_change_pct > pe_oi_change_pct: sentiment = "BEARISH"
        elif pcr > 1.2: sentiment = "BULLISH"
        elif pcr < 0.8: sentiment = "BEARISH"

        logger.info(f"  📊 OI-30min: {sentiment} (CE:{ce_oi_change_pct:+.1f}% PE:{pe_oi_change_pct:+.1f}% PCR:{pcr:.2f})")

        return AggregateOIAnalysis(
            total_ce_oi=total_ce_oi_new, total_pe_oi=total_pe_oi_new, pcr=pcr,
            ce_oi_change_pct=ce_oi_change_pct, pe_oi_change_pct=pe_oi_change_pct,
            overall_sentiment=sentiment
        )

    def _parse_option_chain(self, strikes: List[Dict], spot_price: float) -> List[OIData]:
        oi_list = []
        # Consider strikes within 5% of the spot price for analysis
        price_range = spot_price * 0.05
        for s in strikes:
            strike_price = s.get('strike_price', 0)
            if abs(strike_price - spot_price) <= price_range:
                ce_data = s.get('call_options', {}).get('market_data', {})
                pe_data = s.get('put_options', {}).get('market_data', {})
                oi_list.append(OIData(
                    strike=strike_price,
                    ce_oi=ce_data.get('oi', 0), pe_oi=pe_data.get('oi', 0),
                    ce_volume=ce_data.get('volume', 0), pe_volume=pe_data.get('volume', 0)
                ))
        return oi_list

class NewsAnalyzer:
    @staticmethod
    def analyze_news(symbol: str, news_list: List[Dict]) -> Optional[NewsData]:
        if not news_list or not DEEPSEEK_API_KEY: return None

        # Filter relevant news
        relevant_news = NewsAnalyzer._filter_relevant_news(news_list)
        if not relevant_news:
            logger.info(f"  ❌ No relevant news for {symbol} after filtering.")
            return None

        # Analyze with AI
        try:
            news_summary = ""
            for idx, news in enumerate(relevant_news[:5], 1):
                news_summary += f"{idx}. {news.get('headline', '')}\n"

            prompt = f"""Analyze these INDIAN MARKET news headlines for overall market sentiment.
            
            Headlines:
            {news_summary}
            
            Provide sentiment analysis for INDEX trading (Nifty/Sensex). Reply ONLY in JSON format.
            
            Example JSON:
            {{
              "sentiment": "BULLISH",
              "impact_score": 70,
              "key_insight": "Positive global cues and strong domestic data suggest a bullish opening."
            }}
            """
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are an expert Indian stock market news analyst. Provide concise, direct JSON output."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2, "max_tokens": 200
                },
                timeout=30
            )

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    analysis = json.loads(match.group(0))
                    top_news = relevant_news[0]
                    news_data = NewsData(
                        headline=top_news.get('headline', '')[:100],
                        summary=analysis.get('key_insight', 'N/A'),
                        sentiment=analysis.get('sentiment', 'NEUTRAL').upper(),
                        impact_score=analysis.get('impact_score', 50),
                        relevance_score=top_news.get('relevance_score', 0)
                    )
                    logger.info(f"  📰 News Analysis: {news_data.sentiment} (Impact: {news_data.impact_score})")
                    return news_data
        except Exception as e:
            logger.error(f"News sentiment AI error: {e}")
        return None

    @staticmethod
    def _filter_relevant_news(news_list: List[Dict]) -> List[Dict]:
        relevant_news = []
        for news in news_list:
            text = (news.get('headline', '') + " " + news.get('summary', '')).lower()
            if any(kw in text for kw in SECTOR_KEYWORDS["INDEX"]):
                us_keywords = ["amazon", "apple", "google", "microsoft", "tesla", "nasdaq", "dow jones", "wall street"]
                if not any(kw in text for kw in us_keywords):
                    news['relevance_score'] = sum(1 for kw in SECTOR_KEYWORDS["INDEX"] if kw in text)
                    relevant_news.append(news)
        
        # Sort by relevance score, descending
        return sorted(relevant_news, key=lambda x: x['relevance_score'], reverse=True)


class ChartAnalyzer:
    @staticmethod
    def calculate_pivot_points(prev_day: Dict) -> PivotPoints:
        h, l, c = prev_day['high'], prev_day['low'], prev_day['close']
        pivot = (h + l + c) / 3
        r1 = (2 * pivot) - l
        s1 = (2 * pivot) - h
        r2 = pivot + (h - l)
        s2 = pivot - (h - l)
        logger.info(f"  📈 Pivots: S2={s2:.1f}, S1={s1:.1f}, P={pivot:.1f}, R1={r1:.1f}, R2={r2:.1f}")
        return PivotPoints(pivot=pivot, s1=s1, r1=r1, s2=s2, r2=r2)

    @staticmethod
    def analyze_1h_trend(df_1h: pd.DataFrame) -> str:
        if len(df_1h) < 20: return "NEUTRAL"
        ma20 = df_1h['close'].rolling(20).mean().iloc[-1]
        current_price = df_1h['close'].iloc[-1]
        if current_price > ma20 * 1.001: return "BULLISH"
        if current_price < ma20 * 0.999: return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def analyze_15m_structure(df_15m: pd.DataFrame) -> str:
        if len(df_15m) < 10: return "RANGING"
        recent_high = df_15m['high'].tail(10).max()
        recent_low = df_15m['low'].tail(10).min()
        current_price = df_15m['close'].iloc[-1]
        if current_price > recent_high: return "BREAKOUT"
        if current_price < recent_low: return "BREAKDOWN"
        return "RANGING"

    @staticmethod
    def analyze_5m_entry_signal(df_5m: pd.DataFrame) -> str:
        if len(df_5m) < 3: return "NONE"
        last_candle = df_5m.iloc[-1]
        prev_candle = df_5m.iloc[-2]
        if last_candle['close'] > prev_candle['high']: return "BULLISH_BREAK"
        if last_candle['close'] < prev_candle['low']: return "BEARISH_BREAK"
        return "NONE"

class AIAnalyzer:
    @staticmethod
    def determine_chart_bias(tf_1h: str, tf_15m: str, oi_sentiment: str) -> str:
        bull_score = 0
        bear_score = 0
        if "BULLISH" in tf_1h: bull_score += 1
        if "BEARISH" in tf_1h: bear_score += 1
        if "BREAKOUT" in tf_15m: bull_score += 1
        if "BREAKDOWN" in tf_15m: bear_score += 1
        if "BULLISH" in oi_sentiment: bull_score += 1
        if "BEARISH" in oi_sentiment: bear_score += 1

        if bull_score >= 2: return "BULLISH"
        if bear_score >= 2: return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def get_deep_analysis(symbol: str, spot_price: float, mtf_data: MultiTimeframeData,
                            oi_analysis: AggregateOIAnalysis, news_data: Optional[NewsData]) -> Optional[DeepAnalysis]:
        
        # 1. Perform multi-timeframe analysis
        tf_1h = ChartAnalyzer.analyze_1h_trend(mtf_data.df_1h)
        tf_15m = ChartAnalyzer.analyze_15m_structure(mtf_data.df_15m)
        tf_5m = ChartAnalyzer.analyze_5m_entry_signal(mtf_data.df_5m)

        # 2. Determine overall bias
        chart_bias = AIAnalyzer.determine_chart_bias(tf_1h, tf_15m, oi_analysis.overall_sentiment)
        logger.info(f"  🎯 Chart Bias: {chart_bias} (1H:{tf_1h}, 15M:{tf_15m}, OI:{oi_analysis.overall_sentiment})")

        if chart_bias == "NEUTRAL":
            logger.info("  ⏭️ Skipping AI analysis due to neutral market bias.")
            return None

        # ✅ CORRECTED LOGIC: Determine correct opportunity and levels
        opportunity = "CE_BUY" if chart_bias == "BULLISH" else "PE_BUY"
        pivots = mtf_data.pivot_points
        entry = spot_price
        
        if chart_bias == "BULLISH": # CE_BUY
            stop_loss = pivots.s1
            target_1 = pivots.r1
            target_2 = pivots.r2
        else: # PE_BUY
            stop_loss = pivots.r1
            target_1 = pivots.s1
            target_2 = pivots.s2

        # 3. Create prompt for AI to get summary and scores
        news_section = f"NEWS: {news_data.sentiment} (Impact: {news_data.impact_score})" if news_data else "NEWS: No significant news."
        
        prompt = f"""
        Provide a trading analysis for {symbol}. I have already determined the trade direction based on my indicators. Your role is to provide a concise summary, confidence score, and total score based on the data provided.
        
        Data:
        - Instrument: {symbol}
        - Current Price: {spot_price:.2f}
        - My Determined Bias: {chart_bias}
        - My Trade Signal: {opportunity}
        - 1H Trend: {tf_1h}
        - 15M Structure: {tf_15m}
        - 5M Signal: {tf_5m}
        - OI Sentiment: {oi_analysis.overall_sentiment} (PCR: {oi_analysis.pcr:.2f})
        - {news_section}
        - Key Levels (Pivots): R2={pivots.r2:.1f}, R1={pivots.r1:.1f}, P={pivots.pivot:.1f}, S1={pivots.s1:.1f}, S2={pivots.s2:.1f}
        
        Task:
        Return ONLY a JSON object with your analysis. Do NOT change the trade signal.
        
        JSON Format:
        {{
            "confidence": <integer, 0-100>,
            "total_score": <integer, 0-125, based on confluence of all data points>,
            "analysis_summary": "<string, a very brief 1-line summary explaining why the trade is viable>",
            "risk_reward": "<string>"
        }}
        """
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a trading analysis assistant. Provide only the requested JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2, "max_tokens": 300
                },
                timeout=40
            )

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    ai_scores = json.loads(match.group(0))
                    
                    # Ensure RR is valid
                    risk = abs(entry - stop_loss)
                    reward = abs(target_1 - entry)
                    rr = f"1:{reward/risk:.1f}" if risk > 0 else "N/A"

                    return DeepAnalysis(
                        opportunity=opportunity,
                        confidence=ai_scores.get('confidence', 70),
                        total_score=ai_scores.get('total_score', 80),
                        entry_price=entry,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        risk_reward=rr,
                        recommended_strike=round(spot_price / 50) * 50, # Round to nearest 50
                        analysis_summary=ai_scores.get('analysis_summary', "AI summary unavailable."),
                        support_levels=[pivots.s1, pivots.s2],
                        resistance_levels=[pivots.r1, pivots.r2],
                        chart_bias=chart_bias,
                        news_sentiment=news_data.sentiment if news_data else "NEUTRAL",
                        news_impact=news_data.impact_score if news_data else 0,
                        tf_1h_trend=tf_1h,
                        tf_15m_structure=tf_15m,
                        tf_5m_signal=tf_5m
                    )
        except Exception as e:
            logger.error(f"Deep analysis AI error: {e}")
        return None

# --- UTILITY CLASSES ---
class ChartGenerator:
    @staticmethod
    def create_chart(symbol: str, mtf_data: MultiTimeframeData, spot_price: float, analysis: DeepAnalysis) -> Optional[io.BytesIO]:
        try:
            df_plot = mtf_data.df_15m.tail(100).copy()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
            plt.style.use('fivethirtyeight')

            # Candlesticks
            for index, row in df_plot.iterrows():
                color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
                ax1.plot([index, index], [row['low'], row['high']], color=color, linewidth=1)
                ax1.add_patch(Rectangle((index - pd.Timedelta(minutes=6), min(row['open'], row['close'])), 
                                        pd.Timedelta(minutes=12), abs(row['close'] - row['open']), 
                                        facecolor=color, edgecolor=color))

            # Pivot Points
            pivots = mtf_data.pivot_points
            ax1.axhline(pivots.r2, color='red', linestyle='--', linewidth=1.5, label=f'R2: {pivots.r2:.1f}')
            ax1.axhline(pivots.r1, color='red', linestyle='--', linewidth=1.0, label=f'R1: {pivots.r1:.1f}')
            ax1.axhline(pivots.pivot, color='blue', linestyle='-', linewidth=1.0, label=f'Pivot: {pivots.pivot:.1f}')
            ax1.axhline(pivots.s1, color='green', linestyle='--', linewidth=1.0, label=f'S1: {pivots.s1:.1f}')
            ax1.axhline(pivots.s2, color='green', linestyle='--', linewidth=1.5, label=f'S2: {pivots.s2:.1f}')
            ax1.legend(loc='upper left')

            # Current Price Line
            ax1.axhline(spot_price, color='#FFD700', linestyle='-', linewidth=2, label=f'CMP: {spot_price:.1f}')
            
            # Trade Levels
            if analysis.opportunity != "WAIT":
                ax1.axhline(analysis.entry_price, color='orange', linestyle=':', linewidth=2, label=f'Entry: {analysis.entry_price:.1f}')
                ax1.axhline(analysis.stop_loss, color='red', linestyle=':', linewidth=2, label=f'SL: {analysis.stop_loss:.1f}')
                ax1.axhline(analysis.target_1, color='green', linestyle=':', linewidth=2, label=f'T1: {analysis.target_1:.1f}')

            ax1.set_title(f'{symbol} | 15min Chart | Bias: {analysis.chart_bias}', fontsize=16)
            
            # Volume
            colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for index, row in df_plot.iterrows()]
            ax2.bar(df_plot.index, df_plot['volume'], color=colors, width=0.005)
            ax2.set_ylabel('Volume')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            buf.seek(0)
            plt.close(fig)
            return buf
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

class TelegramNotifier:
    def __init__(self, api_status: Dict):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.api_status_msg = "\n".join([f"{'🟢' if status else '🔴'} {name.title()}" for name, status in api_status.items()])

    async def send_startup_message(self):
        msg = f"""🔥 **HYBRID TRADING BOT v16.0 - ONLINE** 🔥
        
**API STATUS:**
{self.api_status_msg}

**STRATEGY:**
- **1H:** Trend Confirmation
- **15M:** Pivot Point S/R
- **5M:** Breakout Entry

Bot is running and waiting for market hours (9:15 AM - 3:30 PM)."""
        await self._send_message(msg)

    async def send_alert(self, symbol: str, analysis: DeepAnalysis, oi: AggregateOIAnalysis, chart: Optional[io.BytesIO]):
        # ✅ CORRECTED EMOJI LOGIC
        emoji = "🟢" if analysis.opportunity == "CE_BUY" else "🔴" if analysis.opportunity == "PE_BUY" else "⚪"
        
        msg = f"""
🎯 **{symbol} SIGNAL** | {emoji} **{analysis.opportunity}**

**Bias:** {analysis.chart_bias} | **Confidence:** {analysis.confidence}% | **Score:** {analysis.total_score}/125

**Summary:** _{analysis.analysis_summary}_

---
**MULTI-TIMEFRAME VIEW**
- **1H Trend:** {analysis.tf_1h_trend}
- **15M Structure:** {analysis.tf_15m_structure}
- **5M Entry Signal:** {analysis.tf_5m_signal}

---
**TRADE SETUP**
- **Entry:** ₹{analysis.entry_price:.2f}
- **Stop-Loss:** ₹{analysis.stop_loss:.2f}
- **Target 1:** ₹{analysis.target_1:.2f}
- **Target 2:** ₹{analysis.target_2:.2f}
- **Risk/Reward:** {analysis.risk_reward}
- **Strike:** {analysis.recommended_strike}

---
**OPTIONS DATA (30min ∆)**
- **Sentiment:** {oi.overall_sentiment}
- **PCR:** {oi.pcr:.2f}
- **CE OI%:** {oi.ce_oi_change_pct:+.1f}% | **PE OI%:** {oi.pe_oi_change_pct:+.1f}%

_Time: {datetime.now(IST).strftime('%I:%M:%S %p')}_
"""
        if chart:
            await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=chart, caption=msg, parse_mode='Markdown')
        else:
            await self._send_message(msg)
        logger.info(f"✅ Alert sent for {symbol}: {analysis.opportunity}")
        
    async def send_summary(self, alerts_sent: int):
        msg = f"📊 **Scan Complete.** {alerts_sent} alerts sent. Next scan in 5 minutes."
        await self._send_message(msg)

    async def _send_message(self, text: str):
        try:
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Telegram message failed: {e}")

# --- MAIN BOT CLASS ---
class HybridBot:
    def __init__(self):
        logger.info("Initializing Bot v16.0...")
        self.redis = RedisCache()
        self.fetcher = UpstoxDataFetcher()
        self.finnhub = FinnhubNewsAPI()
        self.oi_analyzer = OIAnalyzer(self.redis)
        
        api_status = {
            'Upstox': self.fetcher.connected, 'Redis': self.redis.connected,
            'Finnhub': self.finnhub.connected, 'DeepSeek': bool(DEEPSEEK_API_KEY)
        }
        self.notifier = TelegramNotifier(api_status)

    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        return now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)

    async def scan_single_index(self, instrument_key: str, info: Dict, market_news: List[Dict]):
        symbol = info['name']
        logger.info(f"\n--- Scanning {symbol} ---")
        
        spot_price = self.fetcher.get_spot_price(instrument_key)
        if spot_price == 0:
            logger.warning(f"  ❌ Could not get spot price for {symbol}.")
            return 0

        # This will now return None if any sub-part fails (like pivot calculation)
        mtf_data = self.fetcher.get_multi_timeframe_data(instrument_key, symbol)
        if not mtf_data:
            logger.warning(f"  ❌ Failed to get multi-timeframe data for {symbol}.")
            return 0
            
        # These will be fetched from Upstox API in a real scenario
        # Hardcoding for example purpose as the API calls might not be available
        dummy_strikes = [{'strike_price': spot_price + i*50, 
                          'call_options': {'market_data': {'oi': 10000, 'volume': 500}},
                          'put_options': {'market_data': {'oi': 12000, 'volume': 600}}} for i in range(-5, 6)]

        oi_analysis = self.oi_analyzer.get_oi_analysis(symbol, dummy_strikes, spot_price)
        if not oi_analysis:
            logger.warning(f"  ❌ Failed to get OI analysis for {symbol}.")
            return 0

        news_analysis = NewsAnalyzer.analyze_news(symbol, market_news)

        deep_analysis = AIAnalyzer.get_deep_analysis(symbol, spot_price, mtf_data, oi_analysis, news_analysis)
        if not deep_analysis:
            logger.info(f"  ❌ No valid trading signal found by AI for {symbol}.")
            return 0

        # Final Filters
        if deep_analysis.confidence < CONFIDENCE_MIN or deep_analysis.total_score < SCORE_MIN:
            logger.info(f"  ❌ Signal for {symbol} failed filters (Conf: {deep_analysis.confidence}%, Score: {deep_analysis.total_score}).")
            return 0
        
        chart_image = ChartGenerator.create_chart(symbol, mtf_data, spot_price, deep_analysis)
        await self.notifier.send_alert(symbol, deep_analysis, oi_analysis, chart_image)
        return 1

    async def run(self):
        await self.notifier.send_startup_message()
        while True:
            try:
                if self.is_market_open():
                    logger.info("\n" + "="*50 + f"\nSCANNING CYCLE @ {datetime.now(IST):%I:%M %p}\n" + "="*50)
                    market_news = self.finnhub.get_market_news()
                    alerts_sent = 0
                    
                    scan_tasks = [self.scan_single_index(key, info, market_news) for key, info in INDICES.items()]
                    results = await asyncio.gather(*scan_tasks, return_exceptions=True)
                    
                    for res in results:
                        if isinstance(res, int):
                            alerts_sent += res
                        elif isinstance(res, Exception):
                            logger.error(f"An error occurred during scan: {res}")
                            traceback.print_exc()

                    await self.notifier.send_summary(alerts_sent)
                    logger.info(f"Cycle finished. Waiting {SCAN_INTERVAL} seconds for next scan.")
                    await asyncio.sleep(SCAN_INTERVAL)
                else:
                    logger.info(f"Market is closed. Waiting... (Current time: {datetime.now(IST):%I:%M %p})")
                    await asyncio.sleep(60)
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"Critical error in main loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

async def main():
    if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY, FINNHUB_API_KEY]):
        logger.critical("FATAL: One or more required environment variables are missing. Exiting.")
        return
    bot = HybridBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
