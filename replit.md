# Forex & Commodities AI-Powered Trading Signal Bot

## Overview

This is a complete Python-based CLI application that provides AI-powered trading signals for forex, commodities, and cryptocurrencies. The system integrates multiple data sources and AI models to analyze market conditions and generate trading recommendations through technical analysis and market news sentiment. **Status: FULLY OPERATIONAL** - The bot successfully analyzes real market data, generates trading signals, and provides comprehensive AI analysis in Bengali.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (July 21, 2025)

✓ **Complete system implementation finished**
✓ **Real market data integration working** - Successfully connects to Twelve Data API
✓ **AI chat system fully operational** - Gemini AI provides analysis in Bengali
✓ **Multi-timeframe analysis working** - Supports 1m to 1D timeframes
✓ **Technical indicators functional** - EMA, RSI, MACD, Support/Resistance, Candlestick patterns
✓ **Signal generation active** - BUY/SELL/HOLD signals with detailed explanations
✓ **Fallback system implemented** - Uses AI-generated simulated data when needed
✓ **Bengali language support** - Full Bengali interface and AI responses

## System Architecture

### Core Architecture Pattern
- **Modular CLI Application**: Command-line interface using argparse for user interactions
- **Async API Integration**: Asynchronous HTTP clients for handling multiple external API calls efficiently
- **Data Pipeline Architecture**: Sequential flow from data fetching → technical analysis → AI analysis → signal generation

### Key Design Decisions
- **Python-based**: Chosen for extensive financial libraries, AI/ML ecosystem, and rapid development
- **Async/Await Pattern**: Handles multiple API calls concurrently to improve performance
- **JSON File Storage**: Simple persistence for chat history without database overhead
- **Environment-based Configuration**: Secure API key management through environment variables

## Key Components

### 1. API Integration Layer (`api.py`)
**Purpose**: Handles all external service integrations
- **Twelve Data API**: Real-time market data (OHLCV) for forex, commodities, and crypto
- **Google Gemini**: AI model for market analysis and signal interpretation
- **Grok (X.AI)**: Alternative AI model using OpenAI-compatible interface
- **News API**: Market news and sentiment data
- **Retry Mechanism**: Built-in API call reliability with exponential backoff

### 2. Technical Analysis Engine (`indicators.py`)
**Purpose**: Calculates technical indicators and generates trading signals
- **Indicators**: EMA, RSI, MACD, Support/Resistance levels, Candlestick patterns
- **Signal Generation**: Rule-based logic combining multiple indicators
- **NumPy/Pandas Integration**: Efficient numerical computations for large datasets

### 3. CLI Interface (`main.py`)
**Purpose**: User interaction and application orchestration
- **Argparse Integration**: Command-line argument parsing
- **Chat History**: Persistent conversation tracking via JSON
- **Instrument Support**: Pre-defined lists of forex pairs, commodities, and crypto
- **Color-coded Output**: Enhanced user experience with terminal colors

### 4. Utilities (`utils.py`)
**Purpose**: Shared functionality and cross-cutting concerns
- **Logging System**: File and console logging with configurable levels
- **Environment Management**: Secure API key loading from .env files
- **Retry Decorators**: Reusable retry logic for API reliability
- **Terminal Formatting**: Color-coded CLI output utilities

## Data Flow

1. **Input Processing**: User specifies trading pair, timeframe, and analysis preferences via CLI
2. **Data Fetching**: Parallel API calls to fetch market data and news
3. **Technical Analysis**: Calculate indicators (EMA, RSI, MACD) from OHLCV data
4. **Signal Generation**: Apply rule-based logic to generate buy/sell/hold signals
5. **AI Enhancement**: Send technical analysis to Gemini/Grok for market context and validation
6. **Output Generation**: Present signals with explanations and confidence levels
7. **History Tracking**: Save conversation context for continuity

## External Dependencies

### Required APIs
- **Twelve Data**: Market data provider (requires API key)
- **Google Gemini**: AI analysis (requires GEMINI_API_KEY)
- **Grok (X.AI)**: Alternative AI model (requires GROK_API_KEY)
- **News API**: Market news (requires NEWS_API_KEY)

### Python Packages
- **aiohttp**: Async HTTP client for API calls
- **pandas/numpy**: Data manipulation and numerical computations
- **google-genai**: Google's Gemini AI client
- **openai**: OpenAI-compatible interface for Grok
- **python-dotenv**: Environment variable management

### Configuration Requirements
- Environment variables stored in `.env` file
- All API keys must be configured for full functionality
- Optional: Simulation mode available when APIs are unavailable

## Deployment Strategy

### Development Setup
- **Local Development**: Direct Python execution with virtual environment
- **Configuration**: Environment variables via `.env` file
- **Dependencies**: pip install from requirements (implied, not present in repo)

### Production Considerations
- **API Rate Limiting**: Built-in retry mechanisms and respectful API usage
- **Error Handling**: Graceful degradation when services are unavailable
- **Logging**: Comprehensive logging to `log.txt` for debugging
- **Scalability**: Async architecture supports multiple concurrent operations

### Replit-Specific Adaptations
- **Database Integration**: May require Postgres addition for persistent storage beyond JSON files
- **Environment Secrets**: API keys should be configured in Replit's secrets management
- **Package Installation**: Standard pip requirements installation process
- **File Permissions**: Ensure write access for logs and chat history files