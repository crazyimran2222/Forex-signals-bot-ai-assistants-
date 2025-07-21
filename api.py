"""
API integrations for market data, news, and AI services
Handles Twelve Data, Gemini, Grok, and News API
"""

import os
import json
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from google import genai
from google.genai import types
from openai import OpenAI

from utils import retry_api_call

logger = logging.getLogger(__name__)

# Initialize clients
gemini_client = None
grok_client = None

def initialize_clients():
    """Initialize AI clients with proper error handling"""
    global gemini_client, grok_client
    
    try:
        if os.getenv("GEMINI_API_KEY"):
            gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")
    
    try:
        if os.getenv("XAI_API_KEY"):
            grok_client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=os.getenv("XAI_API_KEY")
            )
    except Exception as e:
        logger.warning(f"Failed to initialize Grok client: {e}")

# API Keys
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

async def fetch_market_data(pair: str, timeframe: str, start_date: str = None, 
                           end_date: str = None) -> Optional[List[Dict]]:
    """
    Fetch real market data from Twelve Data API
    Returns OHLCV data for the specified pair and timeframe
    """
    try:
        # Handle symbol mappings for Twelve Data API
        if '/' in pair:
            # For forex pairs like EUR/USD
            symbol = pair
        else:
            # For other instruments
            symbol_mappings = {
                'XAUUSD': 'XAU/USD',
                'XAGUSD': 'XAG/USD', 
                'WTIUSD': 'WTI',
                'NAS100': 'NAS100',
                'US30': 'DJI',
                'DXY': 'DXY',
                'BTCUSD': 'BTC/USD',
                'ETHUSD': 'ETH/USD',
                'SOLUSD': 'SOL/USD'
            }
            symbol = symbol_mappings.get(pair, pair)
        
        # Convert timeframe to Twelve Data format
        timeframe_mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1D': '1day'
        }
        api_timeframe = timeframe_mapping.get(timeframe, timeframe)
        
        # Build API URL
        base_url = "https://api.twelvedata.com/time_series"
        # Validate API key
        if not TWELVE_DATA_API_KEY:
            logger.error("TWELVE_DATA_API_KEY not found in environment")
            return None
        
        params = {
            'symbol': symbol,
            'interval': api_timeframe,
            'apikey': TWELVE_DATA_API_KEY,
            'format': 'JSON',
            'outputsize': 100  # Get more data for indicators
        }
        
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'values' in data and data['values']:
                        # Convert to standard format
                        market_data = []
                        for item in reversed(data['values']):  # Reverse to get chronological order
                            try:
                                market_data.append({
                                    'timestamp': item['datetime'],
                                    'open': float(item['open']),
                                    'high': float(item['high']),
                                    'low': float(item['low']),
                                    'close': float(item['close']),
                                    'volume': float(item.get('volume', 0))
                                })
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Skipping malformed data point: {e}")
                                continue
                        
                        logger.info(f"Successfully fetched {len(market_data)} data points for {pair} {timeframe}")
                        return market_data
                    
                    elif 'code' in data and data['code'] == 429:
                        logger.warning("Twelve Data API rate limit reached")
                        return None
                    else:
                        logger.warning(f"No data returned from Twelve Data API: {data}")
                        return None
                else:
                    logger.error(f"Twelve Data API error: {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return None

async def simulate_market_data(pair: str, timeframe: str) -> Optional[List[Dict]]:
    """
    Generate realistic simulated market data using Gemini AI
    Used as fallback when real data is unavailable
    """
    try:
        current_time = datetime.now()
        
        # Create prompt for realistic market data simulation
        prompt = f"""Generate realistic {pair} market data for {timeframe} timeframe.
        Create 50 OHLCV candlesticks with realistic price movements.
        Current time: {current_time.isoformat()}
        
        Requirements:
        - Realistic price ranges for {pair}
        - Natural price movements with volatility
        - Proper OHLC relationships (High >= max(Open,Close), Low <= min(Open,Close))
        - Volume data
        - Timestamp progression for {timeframe}
        
        Return as JSON array with this exact format:
        [
            {{
                "timestamp": "2025-07-21T10:00:00",
                "open": 1.0950,
                "high": 1.0975,
                "low": 1.0940,
                "close": 1.0965,
                "volume": 1500000
            }}
        ]
        
        Make the data unique and varied - no repetitive patterns."""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        if response.text:
            simulated_data = json.loads(response.text)
            
            # Validate and clean the data
            cleaned_data = []
            for item in simulated_data:
                try:
                    cleaned_item = {
                        'timestamp': item['timestamp'],
                        'open': float(item['open']),
                        'high': float(item['high']),
                        'low': float(item['low']),
                        'close': float(item['close']),
                        'volume': float(item.get('volume', 1000000))
                    }
                    
                    # Validate OHLC relationships
                    if (cleaned_item['high'] >= max(cleaned_item['open'], cleaned_item['close']) and
                        cleaned_item['low'] <= min(cleaned_item['open'], cleaned_item['close'])):
                        cleaned_data.append(cleaned_item)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid simulated data point: {e}")
                    continue
            
            logger.info(f"Generated {len(cleaned_data)} simulated data points for {pair} {timeframe}")
            return cleaned_data if cleaned_data else None
            
    except Exception as e:
        logger.error(f"Error generating simulated data: {e}")
        return None

async def fetch_news(pair: str) -> Optional[List[Dict]]:
    """
    Fetch latest economic news related to the trading pair
    """
    try:
        # Extract currencies from pair for news search
        if '/' in pair:
            base_currency = pair.split('/')[0]
            quote_currency = pair.split('/')[1]
            search_terms = f"{base_currency} OR {quote_currency}"
        else:
            search_terms = pair
        
        # Add forex and economic terms
        search_terms += " OR forex OR economy OR trading OR market"
        
        base_url = "https://newsapi.org/v2/everything"
        params = {
            'q': search_terms,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': NEWS_API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'articles' in data:
                        articles = []
                        for article in data['articles']:
                            articles.append({
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'publishedAt': article.get('publishedAt', ''),
                                'source': article.get('source', {}).get('name', '')
                            })
                        
                        logger.info(f"Fetched {len(articles)} news articles for {pair}")
                        return articles
                    else:
                        logger.warning("No articles found in news response")
                        return None
                else:
                    logger.error(f"News API error: {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return None

async def send_to_gemini(payload: Dict, chat_history: List[Dict] = None) -> str:
    """
    Send request to Gemini AI for analysis or chat
    """
    try:
        if chat_history is None:
            chat_history = []
        
        if gemini_client is None:
            initialize_clients()
        
        if gemini_client is None:
            raise Exception("Gemini client not initialized")
        
        # Prepare system instruction based on payload type
        if payload.get('type') == 'chat':
            system_instruction = """You are an expert Forex and Commodities trading advisor AI assistant.
            You MUST respond in Bengali language only.
            Provide helpful, accurate, and contextual responses about trading, technical analysis, and market conditions.
            Use the conversation history to maintain context and avoid repetitive responses.
            Be conversational and educational."""
            
            # Build conversation context
            conversation_text = f"User question: {payload['message']}"
            
            if chat_history:
                recent_history = chat_history[-10:]  # Last 10 messages for context
                context = "Previous conversation:\n"
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    context += f"{role}: {msg['content']}\n"
                conversation_text = context + "\n" + conversation_text
            
        elif payload.get('type') == 'analysis':
            system_instruction = """You are an expert trading analyst. 
            Analyze the provided trading data and generate insights in Bengali.
            Focus on explaining the signals, market conditions, and potential outcomes.
            Be specific and educational."""
            
            analysis_data = payload['data']
            conversation_text = f"""
            Trading Analysis for {analysis_data['pair']}:
            Signal: {analysis_data['signal']}
            Current Price: {analysis_data['current_price']}
            Recommendations: {analysis_data['recommendations']}
            News Sentiment: {analysis_data['news_sentiment']}
            Using Simulated Data: {analysis_data['using_simulated']}
            
            Provide insights and explanation in Bengali about this analysis.
            """
        else:
            conversation_text = str(payload)
            system_instruction = "You are a helpful AI assistant. Respond in Bengali."
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=conversation_text)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        if response.text:
            return response.text.strip()
        else:
            return "দুঃখিত, আমি এই মুহূর্তে উত্তর দিতে পারছি না।"
            
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise e

async def send_to_grok(payload: Dict, chat_history: List[Dict] = None) -> str:
    """
    Send request to Grok AI as fallback
    """
    try:
        if chat_history is None:
            chat_history = []
        
        if grok_client is None:
            initialize_clients()
        
        if grok_client is None:
            raise Exception("Grok client not initialized")
        
        # Prepare messages for Grok
        messages = []
        
        if payload.get('type') == 'chat':
            messages.append({
                "role": "system",
                "content": """You are an expert Forex and Commodities trading advisor AI assistant.
                You MUST respond in Bengali language only.
                Provide helpful, accurate, and contextual responses about trading, technical analysis, and market conditions.
                Use the conversation history to maintain context and avoid repetitive responses.
                Be conversational and educational."""
            })
            
            # Add recent chat history for context
            if chat_history:
                for msg in chat_history[-5:]:  # Last 5 messages
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            messages.append({
                "role": "user",
                "content": payload['message']
            })
            
        elif payload.get('type') == 'analysis':
            messages.append({
                "role": "system",
                "content": "You are an expert trading analyst. Analyze the provided trading data and generate insights in Bengali."
            })
            
            analysis_data = payload['data']
            analysis_text = f"""
            Trading Analysis for {analysis_data['pair']}:
            Signal: {analysis_data['signal']}
            Current Price: {analysis_data['current_price']}
            Recommendations: {analysis_data['recommendations']}
            News Sentiment: {analysis_data['news_sentiment']}
            
            Provide insights and explanation in Bengali about this analysis.
            """
            
            messages.append({
                "role": "user",
                "content": analysis_text
            })
        
        response = grok_client.chat.completions.create(
            model="grok-2-1212",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return "দুঃখিত, আমি এই মুহূর্তে উত্তর দিতে পারছি না।"
            
    except Exception as e:
        logger.error(f"Grok API error: {e}")
        raise e
