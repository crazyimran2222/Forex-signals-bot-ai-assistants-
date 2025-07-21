#!/usr/bin/env python3
"""
Web interface for the Forex & Commodities Trading Bot
Provides a simple web UI to interact with the trading analysis
"""

from flask import Flask, render_template, request, jsonify
import asyncio
import json
from datetime import datetime
import threading
import time

from api import fetch_market_data, simulate_market_data, send_to_gemini, initialize_clients
from indicators import calculate_all_indicators, generate_signals
from utils import load_env, setup_logging

app = Flask(__name__)

# Load environment and setup
load_env()
logger = setup_logging()
initialize_clients()

# Supported instruments
SUPPORTED_PAIRS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'USD/CAD', 'AUD/USD', 'NZD/USD',
    'XAU/USD', 'XAG/USD', 'WTI/USD',
    'BTC/USD', 'ETH/USD', 'SOL/USD',
    'NAS100', 'US30', 'DXY'
]

SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1D']

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', pairs=SUPPORTED_PAIRS, timeframes=SUPPORTED_TIMEFRAMES)

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for market analysis"""
    try:
        data = request.json
        pair = data.get('pair', 'EUR/USD')
        timeframe = data.get('timeframe', '1h')
        
        # Run analysis in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyze_market(pair, timeframe))
        loop.close()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat API endpoint"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Prepare payload for AI
        payload = {
            'type': 'chat',
            'message': message,
            'language': 'bengali',
            'context': 'forex_trading'
        }
        
        # Run chat in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(send_to_gemini(payload, []))
        loop.close()
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

async def analyze_market(pair: str, timeframe: str):
    """Analyze market data for given pair and timeframe"""
    try:
        # Try to fetch real market data first
        market_data = await fetch_market_data(pair, timeframe)
        using_simulated = False
        
        if market_data is None or len(market_data) == 0:
            # Fallback to simulated data
            market_data = await simulate_market_data(pair, timeframe)
            using_simulated = True
        
        if not market_data or len(market_data) == 0:
            return {'error': 'No market data available'}
        
        # Calculate indicators
        indicators = calculate_all_indicators(market_data)
        
        if not indicators:
            return {'error': 'Failed to calculate indicators'}
        
        # Generate signals
        signals = generate_signals(indicators, market_data)
        
        # Determine overall signal
        bullish_count = sum(1 for s in signals.values() if s.get('signal') == 'BUY')
        bearish_count = sum(1 for s in signals.values() if s.get('signal') == 'SELL')
        
        if bullish_count >= 3:
            overall_signal = 'BUY'
        elif bearish_count >= 3:
            overall_signal = 'SELL'
        else:
            overall_signal = 'HOLD'
        
        current_price = indicators.get('current_price', 0)
        
        # Calculate trade recommendations based on current price
        if overall_signal == 'BUY':
            entry = current_price
            # Calculate stop loss based on support level or 1-2% below current price
            support_level = indicators.get('support')
            if support_level and support_level > 0:
                stop_loss = max(support_level * 0.999, current_price * 0.985)
            else:
                stop_loss = current_price * 0.985
            
            # Calculate take profit based on resistance or 2-3% above current price
            resistance_level = indicators.get('resistance')
            if resistance_level and resistance_level > current_price:
                take_profit = min(resistance_level * 0.999, current_price * 1.025)
            else:
                take_profit = current_price * 1.025
                
        elif overall_signal == 'SELL':
            entry = current_price
            # Calculate stop loss based on resistance level or 1-2% above current price
            resistance_level = indicators.get('resistance')
            if resistance_level and resistance_level > current_price:
                stop_loss = min(resistance_level * 1.001, current_price * 1.015)
            else:
                stop_loss = current_price * 1.015
            
            # Calculate take profit based on support or 2-3% below current price
            support_level = indicators.get('support')
            if support_level and support_level > 0:
                take_profit = max(support_level * 1.001, current_price * 0.975)
            else:
                take_profit = current_price * 0.975
        else:
            entry = current_price
            stop_loss = None
            take_profit = None
        
        # Calculate risk-reward ratio
        if stop_loss and take_profit and stop_loss != entry:
            profit_potential = abs(take_profit - entry)
            risk_amount = abs(entry - stop_loss)
            risk_reward = profit_potential / risk_amount if risk_amount > 0 else None
        else:
            risk_reward = None
        
        # Generate AI insights
        analysis_payload = {
            'type': 'analysis',
            'data': {
                'pair': pair,
                'signal': overall_signal,
                'current_price': current_price,
                'recommendations': {
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': risk_reward
                },
                'using_simulated': using_simulated,
                'news_sentiment': 'নিউট্রাল'
            }
        }
        
        ai_insights = await send_to_gemini(analysis_payload, [])
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'signal': overall_signal,
            'current_price': current_price,
            'indicators': {
                'ema_50': indicators.get('ema_50', 0),
                'ema_200': indicators.get('ema_200', 0),
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'support': indicators.get('support'),
                'resistance': indicators.get('resistance')
            },
            'signals': signals,
            'recommendations': {
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward
            },
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'using_simulated': using_simulated,
            'ai_insights': ai_insights,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    print("🚀 Starting Forex Trading Bot Web Interface...")
    print("🌐 ওয়েব অ্যাপ চালু হচ্ছে...")
    app.run(host='0.0.0.0', port=5000, debug=True)