"""
Technical indicators calculation and signal generation
Includes EMA, RSI, MACD, Support/Resistance, and Candlestick patterns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return [None] * len(prices)
    
    ema = [None] * (period - 1)
    sma = sum(prices[:period]) / period
    ema.append(sma)
    
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(prices)):
        ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return [None] * len(prices)
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    rsi_values = [None]  # First value is None
    
    # Calculate initial average gain and loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        rsi_values.extend([100] * period)
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values.extend([None] * (period - 1) + [rsi])
    
    # Calculate RSI for remaining periods
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
    
    return rsi_values

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow:
        return {
            'macd': [None] * len(prices),
            'signal': [None] * len(prices),
            'histogram': [None] * len(prices)
        }
    
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd_line = []
    for i in range(len(prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    # Calculate signal line (EMA of MACD)
    macd_values = [x for x in macd_line if x is not None]
    if len(macd_values) >= signal:
        signal_line = calculate_ema(macd_values, signal)
        # Pad with None values to match original length
        signal_padded = [None] * (len(macd_line) - len(signal_line)) + signal_line
    else:
        signal_padded = [None] * len(macd_line)
    
    # Calculate histogram
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is not None and signal_padded[i] is not None:
            histogram.append(macd_line[i] - signal_padded[i])
        else:
            histogram.append(None)
    
    return {
        'macd': macd_line,
        'signal': signal_padded,
        'histogram': histogram
    }

def calculate_support_resistance(data: List[Dict], lookback: int = 20) -> Dict:
    """Calculate dynamic support and resistance levels"""
    if len(data) < lookback:
        return {'support': None, 'resistance': None}
    
    recent_data = data[-lookback:]
    highs = [candle['high'] for candle in recent_data]
    lows = [candle['low'] for candle in recent_data]
    
    # Find local maxima and minima
    resistance_levels = []
    support_levels = []
    
    for i in range(2, len(recent_data) - 2):
        # Check for local high (resistance)
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            resistance_levels.append(highs[i])
        
        # Check for local low (support)
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            support_levels.append(lows[i])
    
    # Get the most relevant levels
    current_price = data[-1]['close']
    
    # Find nearest support (below current price)
    valid_support = [level for level in support_levels if level < current_price]
    support = max(valid_support) if valid_support else min(lows)
    
    # Find nearest resistance (above current price)
    valid_resistance = [level for level in resistance_levels if level > current_price]
    resistance = min(valid_resistance) if valid_resistance else max(highs)
    
    return {'support': support, 'resistance': resistance}

def detect_candlestick_patterns(data: List[Dict]) -> Dict:
    """Detect common candlestick patterns"""
    if len(data) < 3:
        return {}
    
    patterns = {}
    current = data[-1]
    previous = data[-2] if len(data) > 1 else None
    
    # Current candle properties
    body_size = abs(current['close'] - current['open'])
    upper_shadow = current['high'] - max(current['open'], current['close'])
    lower_shadow = min(current['open'], current['close']) - current['low']
    total_range = current['high'] - current['low']
    
    if total_range == 0:
        return patterns
    
    # Doji pattern
    if body_size / total_range < 0.1:
        patterns['doji'] = 'NEUTRAL'
    
    # Hammer pattern
    if (lower_shadow > body_size * 2 and upper_shadow < body_size and 
        current['close'] > current['open']):
        patterns['hammer'] = 'BUY'
    
    # Shooting Star pattern
    if (upper_shadow > body_size * 2 and lower_shadow < body_size and 
        current['close'] < current['open']):
        patterns['shooting_star'] = 'SELL'
    
    # Engulfing patterns (requires previous candle)
    if previous:
        prev_body = abs(previous['close'] - previous['open'])
        
        # Bullish Engulfing
        if (previous['close'] < previous['open'] and  # Previous bearish
            current['close'] > current['open'] and   # Current bullish
            current['open'] < previous['close'] and  # Current opens below prev close
            current['close'] > previous['open']):    # Current closes above prev open
            patterns['bullish_engulfing'] = 'BUY'
        
        # Bearish Engulfing
        if (previous['close'] > previous['open'] and  # Previous bullish
            current['close'] < current['open'] and   # Current bearish
            current['open'] > previous['close'] and  # Current opens above prev close
            current['close'] < previous['open']):    # Current closes below prev open
            patterns['bearish_engulfing'] = 'SELL'
    
    return patterns

def calculate_all_indicators(data: List[Dict]) -> Dict:
    """Calculate all technical indicators for the given data"""
    if not data or len(data) < 50:
        logger.warning("Insufficient data for indicator calculation")
        return {}
    
    try:
        # Extract close prices
        close_prices = [candle['close'] for candle in data]
        
        # Calculate EMAs
        ema_50 = calculate_ema(close_prices, 50)
        ema_200 = calculate_ema(close_prices, 200)
        
        # Calculate RSI
        rsi = calculate_rsi(close_prices, 14)
        
        # Calculate MACD
        macd_data = calculate_macd(close_prices, 12, 26, 9)
        
        # Calculate Support/Resistance
        sr_levels = calculate_support_resistance(data, 20)
        
        # Detect candlestick patterns
        patterns = detect_candlestick_patterns(data)
        
        # Get latest values
        indicators = {
            'ema_50': ema_50[-1] if ema_50[-1] is not None else 0,
            'ema_200': ema_200[-1] if ema_200[-1] is not None else 0,
            'rsi': rsi[-1] if rsi[-1] is not None else 50,
            'macd': macd_data['macd'][-1] if macd_data['macd'][-1] is not None else 0,
            'macd_signal': macd_data['signal'][-1] if macd_data['signal'][-1] is not None else 0,
            'macd_histogram': macd_data['histogram'][-1] if macd_data['histogram'][-1] is not None else 0,
            'support': sr_levels['support'],
            'resistance': sr_levels['resistance'],
            'patterns': patterns,
            'current_price': close_prices[-1]
        }
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

def generate_signals(indicators: Dict, data: List[Dict]) -> Dict:
    """Generate trading signals based on indicators"""
    signals = {}
    
    try:
        current_price = indicators.get('current_price', 0)
        
        # EMA Cross Signal
        ema_50 = indicators.get('ema_50', 0)
        ema_200 = indicators.get('ema_200', 0)
        
        if ema_50 > ema_200:
            ema_signal = 'BUY' if current_price > ema_50 else 'NEUTRAL'
        elif ema_50 < ema_200:
            ema_signal = 'SELL' if current_price < ema_50 else 'NEUTRAL'
        else:
            ema_signal = 'NEUTRAL'
        
        signals['ema_cross'] = {
            'signal': ema_signal,
            'value': f"EMA50={ema_50:.4f} / EMA200={ema_200:.4f}"
        }
        
        # RSI Signal
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            rsi_signal = 'BUY'  # Oversold
        elif rsi > 70:
            rsi_signal = 'SELL'  # Overbought
        else:
            rsi_signal = 'NEUTRAL'
        
        signals['rsi'] = {
            'signal': rsi_signal,
            'value': f"RSI={rsi:.2f}"
        }
        
        # MACD Signal
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        
        if macd > macd_signal and macd > 0:
            macd_sig = 'BUY'
        elif macd < macd_signal and macd < 0:
            macd_sig = 'SELL'
        else:
            macd_sig = 'NEUTRAL'
        
        signals['macd'] = {
            'signal': macd_sig,
            'value': f"MACD={macd:.5f} / Signal={macd_signal:.5f}"
        }
        
        # Support/Resistance Signal
        support = indicators.get('support')
        resistance = indicators.get('resistance')
        
        if support and resistance:
            if current_price <= support * 1.01:  # Near support
                sr_signal = 'BUY'
            elif current_price >= resistance * 0.99:  # Near resistance
                sr_signal = 'SELL'
            else:
                sr_signal = 'NEUTRAL'
        else:
            sr_signal = 'NEUTRAL'
        
        support_str = f"{support:.4f}" if support else 'N/A'
        resistance_str = f"{resistance:.4f}" if resistance else 'N/A'
        signals['support_resistance'] = {
            'signal': sr_signal,
            'value': f"S={support_str} / R={resistance_str}"
        }
        
        # Candlestick Pattern Signals
        patterns = indicators.get('patterns', {})
        pattern_signals = []
        
        for pattern, signal in patterns.items():
            pattern_signals.append(f"{pattern}={signal}")
            signals[f'pattern_{pattern}'] = {
                'signal': signal,
                'value': pattern.replace('_', ' ').title()
            }
        
        if not pattern_signals:
            signals['patterns'] = {
                'signal': 'NEUTRAL',
                'value': 'কোন প্যাটার্ন সনাক্ত হয়নি'
            }
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return {}
