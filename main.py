#!/usr/bin/env python3
"""
Forex & Commodities AI-Powered Trading Signal Bot
Main CLI interface with argparse for user interaction
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import List, Dict, Any

from api import fetch_market_data, simulate_market_data, fetch_news, send_to_gemini, send_to_grok, initialize_clients
from indicators import calculate_all_indicators, generate_signals
from utils import setup_logging, print_colored, load_env, retry_api_call

# Load environment variables
load_env()
logger = setup_logging()

# Initialize AI clients
initialize_clients()

# Supported instruments
SUPPORTED_PAIRS = [
    # Forex Majors
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'USD/CAD', 'AUD/USD', 'NZD/USD',
    # Commodities
    'XAU/USD', 'XAG/USD', 'WTI/USD',
    # Cryptocurrencies
    'BTC/USD', 'ETH/USD', 'SOL/USD',
    # Indices
    'NAS100', 'US30', 'DXY'
]

SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1D']

def load_chat_history():
    """Load chat history from JSON file"""
    try:
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

def save_chat_history(history):
    """Save chat history to JSON file"""
    try:
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

async def analyze_market(pair: str, timeframes: List[str], start_date: str = None, 
                        end_date: str = None, verbose: bool = False):
    """Analyze market data for given pair and timeframes"""
    
    print_colored(f"\n{pair} বিশ্লেষণ করা হচ্ছে {', '.join(timeframes)} টাইমফ্রেমে", "cyan")
    
    all_signals = {}
    current_price = None
    using_simulated = False
    
    # Fetch market data for all timeframes
    for timeframe in timeframes:
        try:
            print_colored(f"[{timeframe}] ডেটা আনা হচ্ছে...", "yellow")
            
            # Try to fetch real market data first
            market_data = await retry_api_call(
                fetch_market_data, pair, timeframe, start_date, end_date
            )
            
            if market_data is None or len(market_data) == 0:
                print_colored(f"[সতর্কতা: {timeframe} এর জন্য Gemini এর মাধ্যমে সিমুলেটেড ডেটা ব্যবহার করা হচ্ছে]", "red")
                market_data = await retry_api_call(
                    simulate_market_data, pair, timeframe
                )
                using_simulated = True
            
            if market_data and len(market_data) > 0:
                # Calculate indicators
                indicators = calculate_all_indicators(market_data)
                
                if verbose:
                    print_colored(f"[{timeframe}] Raw OHLC (শেষ 3টি ক্যান্ডেল):", "blue")
                    for i, candle in enumerate(market_data[-3:]):
                        print(f"  {i+1}: O={candle.get('open', 'N/A')} H={candle.get('high', 'N/A')} "
                              f"L={candle.get('low', 'N/A')} C={candle.get('close', 'N/A')}")
                    
                    print_colored(f"[{timeframe}] Indicators:", "blue")
                    for key, value in indicators.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
                
                # Generate signals for this timeframe
                signals = generate_signals(indicators, market_data)
                all_signals[timeframe] = signals
                
                # Update current price from latest data
                if market_data and 'close' in market_data[-1]:
                    current_price = market_data[-1]['close']
                
                # Display timeframe analysis
                print_colored(f"[{timeframe}] সিগন্যাল বিশ্লেষণ:", "green")
                for signal_type, signal_data in signals.items():
                    status = "✅" if signal_data['signal'] != 'NEUTRAL' else "➖"
                    print(f"  {signal_type}: {signal_data['value']} → {signal_data['signal']} {status}")
            
            else:
                print_colored(f"[{timeframe}] ডেটা পাওয়া যায়নি", "red")
                
        except Exception as e:
            logger.error(f"Error analyzing {timeframe}: {e}")
            print_colored(f"[{timeframe}] ত্রুটি: {str(e)}", "red")
    
    # Generate overall signal
    overall_signal = aggregate_signals(all_signals)
    
    # Get news sentiment
    try:
        news_data = await retry_api_call(fetch_news, pair)
        news_sentiment = analyze_news_sentiment(news_data) if news_data else "নিউট্রাল"
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        news_sentiment = "নিউট্রাল"
    
    # Calculate trade recommendations
    trade_recommendations = calculate_trade_recommendations(
        current_price, overall_signal, all_signals
    )
    
    # Display results
    display_final_analysis(
        pair, overall_signal, current_price, trade_recommendations, 
        news_sentiment, using_simulated, all_signals
    )
    
    return {
        'pair': pair,
        'signal': overall_signal,
        'current_price': current_price,
        'recommendations': trade_recommendations,
        'signals': all_signals,
        'news_sentiment': news_sentiment,
        'using_simulated': using_simulated
    }

def aggregate_signals(all_signals: Dict) -> Dict:
    """Aggregate signals across all timeframes"""
    bullish_count = 0
    bearish_count = 0
    timeframe_count = len(all_signals)
    
    signal_details = []
    
    for timeframe, signals in all_signals.items():
        tf_bullish = 0
        tf_bearish = 0
        
        for signal_type, signal_data in signals.items():
            if signal_data['signal'] == 'BUY':
                tf_bullish += 1
                signal_details.append(f"{signal_type} ({timeframe})")
            elif signal_data['signal'] == 'SELL':
                tf_bearish += 1
                signal_details.append(f"{signal_type} ({timeframe})")
        
        if tf_bullish >= 3:
            bullish_count += 1
        elif tf_bearish >= 3:
            bearish_count += 1
    
    # Determine overall signal
    if bullish_count >= 2 and timeframe_count >= 2:
        signal = 'BUY'
    elif bearish_count >= 2 and timeframe_count >= 2:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    return {
        'signal': signal,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'signal_details': signal_details
    }

def analyze_news_sentiment(news_data: List) -> str:
    """Analyze sentiment from news headlines"""
    if not news_data:
        return "নিউট্রাল"
    
    positive_words = ['bullish', 'rise', 'gain', 'up', 'strong', 'growth', 'positive']
    negative_words = ['bearish', 'fall', 'loss', 'down', 'weak', 'decline', 'negative']
    
    positive_count = 0
    negative_count = 0
    
    for article in news_data[:5]:  # Check top 5 headlines
        title = article.get('title', '').lower()
        for word in positive_words:
            if word in title:
                positive_count += 1
        for word in negative_words:
            if word in title:
                negative_count += 1
    
    if positive_count > negative_count:
        return "ইতিবাচক"
    elif negative_count > positive_count:
        return "নেতিবাচক"
    else:
        return "নিউট্রাল"

def calculate_trade_recommendations(current_price: float, overall_signal: Dict, 
                                  all_signals: Dict) -> Dict:
    """Calculate entry, stop-loss, and take-profit levels"""
    if not current_price:
        return {
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward': None
        }
    
    # Default risk percentages
    stop_loss_pct = 0.02  # 2%
    take_profit_pct = 0.04  # 4% (1:2 risk/reward)
    
    if overall_signal['signal'] == 'BUY':
        entry = current_price * 1.001  # Slight premium for buy
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
    elif overall_signal['signal'] == 'SELL':
        entry = current_price * 0.999  # Slight discount for sell
        stop_loss = current_price * (1 + stop_loss_pct)
        take_profit = current_price * (1 - take_profit_pct)
    else:
        return {
            'entry': current_price,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward': None
        }
    
    risk_reward = abs(take_profit - entry) / abs(entry - stop_loss)
    
    return {
        'entry': entry,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward
    }

def display_final_analysis(pair: str, overall_signal: Dict, current_price: float,
                          trade_recommendations: Dict, news_sentiment: str,
                          using_simulated: bool, all_signals: Dict):
    """Display final analysis results"""
    
    if using_simulated:
        print_colored("\n⚠️  [সতর্কতা: কিছু ডেটা সিমুলেটেড]", "red")
    
    print_colored(f"\n🎯 {pair} - চূড়ান্ত বিশ্লেষণ", "cyan")
    print_colored("=" * 50, "cyan")
    
    # Signal summary
    signal_color = "green" if overall_signal['signal'] == 'BUY' else "red" if overall_signal['signal'] == 'SELL' else "yellow"
    print_colored(f"➤ সিগন্যাল: {overall_signal['signal']} ", signal_color)
    print(f"  ({overall_signal['bullish_count']} বুলিশ, {overall_signal['bearish_count']} বেয়ারিশ টাইমফ্রেম)")
    
    if current_price:
        print_colored(f"➤ বর্তমান মূল্য: {current_price:.5f}", "white")
    
    # Trade recommendations
    if trade_recommendations['entry']:
        print_colored(f"➤ সম্ভাব্য এন্ট্রি: {trade_recommendations['entry']:.5f}", "blue")
        if trade_recommendations['stop_loss']:
            print_colored(f"➤ সম্ভাব্য স্টপ-লস: {trade_recommendations['stop_loss']:.5f}", "red")
        if trade_recommendations['take_profit']:
            print_colored(f"➤ সম্ভাব্য টেক-প্রফিট: {trade_recommendations['take_profit']:.5f}", "green")
        if trade_recommendations['risk_reward']:
            print_colored(f"➤ ঝুঁকি/পুরস্কার অনুপাত: 1:{trade_recommendations['risk_reward']:.2f}", "yellow")
    
    print_colored(f"➤ নিউজ সেন্টিমেন্ট: {news_sentiment}", "magenta")
    
    # Signal details
    if overall_signal['signal_details']:
        print_colored("➤ সক্রিয় সিগন্যাল:", "white")
        for detail in overall_signal['signal_details'][:5]:  # Show top 5
            print(f"  • {detail}")

async def chat_mode():
    """Interactive AI chat mode in Bengali"""
    print_colored("\n🤖 AI Chat Mode (Bengali) - 'quit' লিখে বের হন", "cyan")
    print_colored("=" * 50, "cyan")
    
    chat_history = load_chat_history()
    
    while True:
        try:
            user_input = input("\n🗣️  আপনি: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'বের', 'q']:
                print_colored("চ্যাট শেষ। ধন্যবাদ!", "green")
                break
            
            if not user_input:
                continue
            
            # Add user message to history
            chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Prepare payload for AI
            payload = {
                'type': 'chat',
                'message': user_input,
                'language': 'bengali',
                'context': 'forex_trading'
            }
            
            print_colored("🤔 AI চিন্তা করছে...", "yellow")
            
            # Try Gemini first, fallback to Grok
            try:
                response = await retry_api_call(send_to_gemini, payload, chat_history)
                ai_source = "Gemini"
            except Exception as e:
                logger.error(f"Gemini failed: {e}")
                try:
                    response = await retry_api_call(send_to_grok, payload, chat_history)
                    ai_source = "Grok"
                except Exception as e2:
                    logger.error(f"Grok also failed: {e2}")
                    response = "দুঃখিত, AI সেবা এই মুহূর্তে উপলব্ধ নেই। পরে আবার চেষ্টা করুন।"
                    ai_source = "Error"
            
            print_colored(f"🤖 AI ({ai_source}): {response}", "green")
            
            # Add AI response to history
            chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat(),
                'source': ai_source
            })
            
            # Keep only last 50 messages to manage memory
            if len(chat_history) > 50:
                chat_history = chat_history[-50:]
            
            # Save chat history
            save_chat_history(chat_history)
            
        except KeyboardInterrupt:
            print_colored("\n\nচ্যাট বন্ধ করা হয়েছে। ধন্যবাদ!", "yellow")
            break
        except EOFError:
            print_colored("\n\nচ্যাট শেষ। ধন্যবাদ!", "yellow")
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print_colored(f"চ্যাট ত্রুটি: {str(e)}", "red")
            break

def validate_arguments(args):
    """Validate command line arguments"""
    if args.pair not in SUPPORTED_PAIRS:
        print_colored(f"ত্রুটি: অসমর্থিত পেয়ার '{args.pair}'", "red")
        print_colored(f"সমর্থিত পেয়ার: {', '.join(SUPPORTED_PAIRS)}", "yellow")
        return False
    
    for tf in args.timeframes:
        if tf not in SUPPORTED_TIMEFRAMES:
            print_colored(f"ত্রুটি: অসমর্থিত টাইমফ্রেম '{tf}'", "red")
            print_colored(f"সমর্থিত টাইমফ্রেম: {', '.join(SUPPORTED_TIMEFRAMES)}", "yellow")
            return False
    
    return True

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Forex & Commodities AI-Powered Trading Signal Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Pairs: {', '.join(SUPPORTED_PAIRS)}
Supported Timeframes: {', '.join(SUPPORTED_TIMEFRAMES)}

Examples:
  python main.py --pair EUR/USD --timeframes 1h,4h
  python main.py --pair BTC/USD --timeframes 1h,4h --chat --verbose
  python main.py --pair XAU/USD --timeframes 15m,1h --start 2025-07-01 --end 2025-07-21
        """
    )
    
    parser.add_argument('--pair', required=True, help='Trading pair (e.g., EUR/USD)')
    parser.add_argument('--timeframes', required=True, 
                       help='Comma-separated timeframes (e.g., 1h,4h)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD) for historical analysis')
    parser.add_argument('--end', help='End date (YYYY-MM-DD) for historical analysis')
    parser.add_argument('--chat', action='store_true', help='Enable AI chat mode')
    parser.add_argument('--verbose', action='store_true', help='Show detailed debug info')
    
    args = parser.parse_args()
    
    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    args.timeframes = timeframes
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    print_colored("🚀 Forex & Commodities AI Trading Signal Bot", "cyan")
    print_colored("=" * 50, "cyan")
    
    try:
        # Perform market analysis
        if not args.chat:  # If not chat-only mode
            analysis_result = await analyze_market(
                args.pair, args.timeframes, args.start, args.end, args.verbose
            )
            
            # Generate AI insights
            if analysis_result:
                print_colored("\n🧠 AI অন্তর্দৃষ্টি তৈরি করা হচ্ছে...", "yellow")
                try:
                    ai_payload = {
                        'type': 'analysis',
                        'data': analysis_result,
                        'language': 'bengali'
                    }
                    ai_insights = await retry_api_call(send_to_gemini, ai_payload, [])
                    print_colored(f"➤ এআই অন্তর্দৃষ্টি: {ai_insights}", "magenta")
                except Exception as e:
                    logger.error(f"AI insights error: {e}")
                    print_colored("AI অন্তর্দৃষ্টি উপলব্ধ নেই।", "red")
        
        # Enter chat mode if requested
        if args.chat:
            await chat_mode()
    
    except KeyboardInterrupt:
        print_colored("\n\nপ্রোগ্রাম বন্ধ করা হয়েছে।", "yellow")
    except Exception as e:
        logger.error(f"Main error: {e}")
        print_colored(f"ত্রুটি: {str(e)}", "red")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
