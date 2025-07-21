#!/usr/bin/env python3
"""
Demo script to test the Trading Bot functionality
Shows examples of different trading pairs and timeframes
"""

import asyncio
import os
from api import send_to_gemini, initialize_clients
from utils import load_env, setup_logging

async def demo_chat():
    """Demo the AI chat functionality with preset questions"""
    
    load_env()
    logger = setup_logging()
    initialize_clients()
    
    print("🤖 AI Trading Chat Demo (Bengali)")
    print("=" * 40)
    
    # Test questions in Bengali
    test_questions = [
        "RSI কী এবং এটি কিভাবে কাজ করে?",
        "MACD ইন্ডিকেটর ব্যাখ্যা করুন",
        "সাপোর্ট এবং রেজিস্ট্যান্স কী?",
        "বিটকয়েনের বর্তমান বাজার পরিস্থিতি কেমন?"
    ]
    
    chat_history = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. প্রশ্ন: {question}")
        
        payload = {
            'type': 'chat',
            'message': question,
            'language': 'bengali',
            'context': 'forex_trading'
        }
        
        try:
            response = await send_to_gemini(payload, chat_history)
            print(f"   উত্তর: {response[:200]}...")  # First 200 chars
            
            # Add to chat history
            chat_history.append({'role': 'user', 'content': question})
            chat_history.append({'role': 'assistant', 'content': response})
            
        except Exception as e:
            print(f"   ত্রুটি: {e}")
    
    print("\n✅ Chat Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_chat())