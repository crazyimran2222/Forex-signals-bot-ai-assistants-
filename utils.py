"""
Utility functions for logging, formatting, retries, and environment management
"""

import os
import time
import logging
import asyncio
from datetime import datetime
from typing import Callable, Any
from dotenv import load_dotenv

# Color codes for CLI output
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'bold': '\033[1m',
    'end': '\033[0m'
}

def load_env():
    """Load environment variables from .env file"""
    try:
        load_dotenv()
        print_colored("✅ Environment variables loaded", "green")
    except Exception as e:
        print_colored(f"⚠️  Warning: Could not load .env file: {e}", "yellow")

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('log.txt', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    return logger

def print_colored(text: str, color: str = 'white', bold: bool = False):
    """Print colored text to console"""
    color_code = COLORS.get(color.lower(), COLORS['white'])
    bold_code = COLORS['bold'] if bold else ''
    end_code = COLORS['end']
    
    print(f"{bold_code}{color_code}{text}{end_code}")

def log_with_timestamp(message: str, level: str = 'INFO'):
    """Log message with timestamp to file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {level}: {message}\n"
    
    try:
        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print_colored(f"Logging error: {e}", "red")

async def retry_api_call(func: Callable, *args, max_retries: int = 3, 
                        base_delay: float = 1.0, **kwargs) -> Any:
    """
    Retry API calls with exponential backoff
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            if result is not None:
                return result
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"API call returned None, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"API call failed: {e}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                logger.error(f"API call failed after {max_retries} attempts: {e}")
                raise e
    
    return None

def validate_api_keys():
    """Validate that required API keys are present"""
    required_keys = [
        'TWELVE_DATA_API_KEY',
        'GEMINI_API_KEY', 
        'XAI_API_KEY',
        'NEWS_API_KEY'
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print_colored(f"⚠️  Missing API keys: {', '.join(missing_keys)}", "red")
        print_colored("Please check your .env file", "yellow")
        return False
    
    print_colored("✅ All API keys found", "green")
    return True

def format_price(price: float, pair: str) -> str:
    """Format price based on currency pair conventions"""
    if price is None:
        return "N/A"
    
    # JPY pairs typically show 3 decimal places
    if 'JPY' in pair:
        return f"{price:.3f}"
    # Most other pairs show 5 decimal places
    else:
        return f"{price:.5f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def format_timeframe_display(timeframe: str) -> str:
    """Format timeframe for display"""
    timeframe_map = {
        '1m': '১ মিনিট',
        '5m': '৫ মিনিট', 
        '15m': '১৫ মিনিট',
        '30m': '৩০ মিনিট',
        '1h': '১ ঘন্টা',
        '4h': '৪ ঘন্টা',
        '1D': '১ দিন'
    }
    return timeframe_map.get(timeframe, timeframe)

def sanitize_pair_for_api(pair: str) -> str:
    """Convert pair format for different APIs"""
    # Remove slash for some APIs
    return pair.replace('/', '')

def is_market_hours() -> bool:
    """Check if it's during market hours (simplified)"""
    # Forex market is open 24/5, this is a simplified check
    current_time = datetime.now()
    weekday = current_time.weekday()
    
    # Monday (0) to Friday (4)
    if 0 <= weekday <= 4:
        return True
    # Sunday evening (start of forex week)
    elif weekday == 6 and current_time.hour >= 17:
        return True
    # Friday evening (end of forex week)
    elif weekday == 4 and current_time.hour < 17:
        return True
    else:
        return False

def get_risk_level(signal_strength: int) -> str:
    """Determine risk level based on signal strength"""
    if signal_strength >= 4:
        return "উচ্চ নিশ্চয়তা"
    elif signal_strength >= 2:
        return "মধ্যম নিশ্চয়তা"
    else:
        return "নিম্ন নিশ্চয়তা"

def format_bengali_number(number: float) -> str:
    """Convert English numbers to Bengali numerals"""
    bengali_digits = {'0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪', 
                     '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'}
    
    number_str = str(number)
    bengali_number = ''
    
    for char in number_str:
        bengali_number += bengali_digits.get(char, char)
    
    return bengali_number

class ProgressBar:
    """Simple progress bar for CLI"""
    
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0
    
    def update(self, step: int = 1):
        self.current += step
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)
        percent = progress * 100
        
        print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete

def create_summary_table(data: dict) -> str:
    """Create a formatted table for summary display"""
    lines = []
    lines.append("+" + "-" * 48 + "+")
    
    for key, value in data.items():
        key_str = str(key)[:20].ljust(20)
        value_str = str(value)[:25].ljust(25)
        lines.append(f"| {key_str} | {value_str} |")
    
    lines.append("+" + "-" * 48 + "+")
    return "\n".join(lines)

# Initialize logging when module is imported
logger = setup_logging()
