from fastapi import FastAPI
from fastmcp import FastMCP
import httpx
import os
import logging
import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
from common.utils import setup_logging
setup_logging(__name__)

app = FastAPI(redirect_slashes=False)
mcp = FastMCP(name="finance")

# Rate limiting and caching
_cache = {}
_cache_duration = 300  # Cache for 5 minutes

class RateLimiter:
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.last_call = 0
    
    async def wait_if_needed(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

finnhub_limiter = RateLimiter(min_interval=1.2)
quandl_limiter = RateLimiter(min_interval=0.5)

def get_cached_data(key: str) -> Dict[str, Any] | None:
    """Get cached data if it exists and is still valid"""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_duration:
            return data
    return None

def cache_data(key: str, data: Dict[str, Any]):
    """Cache the data with current timestamp"""
    _cache[key] = (data, time.time())

# Internal function for fetching a single stock quote - NOT EXPOSED AS A TOOL DIRECTLY
async def _get_stock_quote_internal(symbol: str) -> dict:
    """
    Internal function to fetch real-time stock quote from Finnhub with Quandl fallback.
    This function is called by other tools within this MCP server.
    """
    cache_key = f"quote_{symbol.upper()}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        logger.info(f"Returning cached quote for {symbol}")
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    
    # Try Finnhub first
    if finnhub_key:
        await finnhub_limiter.wait_if_needed()
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://finnhub.io/api/v1/quote",
                    params={"symbol": symbol, "token": finnhub_key}
                )
                response.raise_for_status()
                quote_data = response.json()
                
                if quote_data.get("c") is not None:
                    result = {
                        "status": "success",
                        "source": "Finnhub",
                        "symbol": symbol.upper(),
                        "current_price": quote_data.get("c", 0),
                        "change": quote_data.get("d", 0),
                        "change_percent": quote_data.get("dp", 0),
                        "high": quote_data.get("h", 0),
                        "low": quote_data.get("l", 0),
                        "open": quote_data.get("o", 0),
                        "previous_close": quote_data.get("pc", 0),
                        "timestamp": quote_data.get("t", 0)
                    }
                    cache_data(cache_key, result)
                    return result
                    
        except Exception as e:
            logger.warning(f"Finnhub quote failed for {symbol}: {e}")
    
    # Fallback to Quandl
    quandl_key = os.getenv("QUANDL_API_KEY")
    if quandl_key:
        await quandl_limiter.wait_if_needed()
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"https://www.quandl.com/api/v3/datasets/WIKI/{symbol.upper()}.json",
                    params={"api_key": quandl_key, "limit": 1}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("dataset") and data["dataset"].get("data"):
                    latest_data = data["dataset"]["data"][0]
                    result = {
                        "status": "success",
                        "source": "Quandl",
                        "symbol": symbol.upper(),
                        "date": latest_data[0],
                        "open": latest_data[1],
                        "high": latest_data[2],
                        "low": latest_data[3],
                        "close": latest_data[4],
                        "volume": latest_data[5],
                        "note": "Historical data from Quandl WIKI dataset"
                    }
                    cache_data(cache_key, result)
                    return result
                    
        except Exception as e:
            logger.warning(f"Quandl fallback failed for {symbol}: {e}")
    
    return {
        "status": "error",
        "message": f"Unable to fetch quote for {symbol}. Please check API keys and symbol validity."
    }

# Public tool for fetching a single stock quote
@mcp.tool()
async def get_stock_quote(symbol: str) -> dict:
    """
    Fetches the **real-time current stock price and basic quote information for a single stock ticker symbol**.
    Use this tool when the user asks for the price of one specific company's stock.

    :param symbol: The stock ticker symbol (e.g., "TSLA", "AAPL").
    :returns: A dictionary containing the current stock price, daily change, high, low, open, and previous close.
    """
    return await _get_stock_quote_internal(symbol)


@mcp.tool()
async def get_company_profile(symbol: str) -> dict:
    """
    Retrieves comprehensive company profile information for a given stock ticker symbol.
    This includes details like company name, country, industry, market capitalization, website, and IPO date.
    Use this tool when the user asks for background information about a company.

    :param symbol: The stock ticker symbol (e.g., "MSFT", "AMZN").
    :returns: A dictionary containing detailed company profile data.
    """
    cache_key = f"profile_{symbol.upper()}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/stock/profile2",
                params={"symbol": symbol, "token": finnhub_key}
            )
            response.raise_for_status()
            profile_data = response.json()
            
            if profile_data.get("name"):
                result = {
                    "status": "success",
                    "symbol": symbol.upper(),
                    "name": profile_data.get("name"),
                    "country": profile_data.get("country"),
                    "currency": profile_data.get("currency"),
                    "exchange": profile_data.get("exchange"),
                    "industry": profile_data.get("finnhubIndustry"),
                    "market_cap": profile_data.get("marketCapitalization"),
                    "shares_outstanding": profile_data.get("shareOutstanding"),
                    "website": profile_data.get("weburl"),
                    "logo": profile_data.get("logo"),
                    "phone": profile_data.get("phone"),
                    "ipo_date": profile_data.get("ipo")
                }
                cache_data(cache_key, result)
                return result
            else:
                return {"status": "error", "message": f"No profile data found for {symbol}"}
                
    except Exception as e:
        logger.error(f"Error fetching profile for {symbol}: {e}")
        return {"status": "error", "message": f"Error fetching profile: {e}"}

@mcp.tool()
async def get_stock_metrics(symbol: str) -> dict:
    """
    Fetches a wide range of comprehensive financial metrics and ratios for a specific stock.
    This includes valuation metrics (like P/E ratio), profitability (like net margin), financial strength (like debt-to-equity), and growth metrics.
    Use this tool when the user asks for in-depth financial analysis or specific financial data points for a company.

    :param symbol: The stock ticker symbol (e.g., "NVDA", "TSLA").
    :returns: A dictionary containing various financial metrics.
    """
    cache_key = f"metrics_{symbol.upper()}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/stock/metric",
                params={"symbol": symbol, "metric": "all", "token": finnhub_key}
            )
            response.raise_for_status()
            data = response.json()
            
            metrics = data.get("metric", {})
            if metrics:
                result = {
                    "status": "success",
                    "symbol": symbol.upper(),
                    "valuation_metrics": {
                        "pe_ratio": metrics.get("peBasicExclExtraTTM"),
                        "pe_forward": metrics.get("peNormalizedAnnual"),
                        "price_to_book": metrics.get("pbAnnual"),
                        "price_to_sales": metrics.get("psAnnual"),
                        "price_to_cash_flow": metrics.get("pcfShareTTM"),
                        "enterprise_value": metrics.get("enterpriseValueTTM"),
                        "ev_to_ebitda": metrics.get("evToEbitdaTTM")
                    },
                    "profitability_metrics": {
                        "gross_margin": metrics.get("grossMarginTTM"),
                        "operating_margin": metrics.get("operatingMarginTTM"),
                        "net_margin": metrics.get("netProfitMarginTTM"),
                        "return_on_equity": metrics.get("roeTTM"),
                        "return_on_assets": metrics.get("roaTTM"),
                        "return_on_invested_capital": metrics.get("roicTTM")
                    },
                    "financial_strength": {
                        "debt_to_equity": metrics.get("totalDebt/totalEquityAnnual"),
                        "current_ratio": metrics.get("currentRatioAnnual"),
                        "quick_ratio": metrics.get("quickRatioAnnual"),
                        "cash_ratio": metrics.get("cashRatioAnnual")
                    },
                    "per_share_metrics": {
                        "eps_ttm": metrics.get("epsBasicExclExtraItemsTTM"),
                        "eps_diluted": metrics.get("epsDilutedExclExtraItemsTTM"),
                        "book_value_per_share": metrics.get("bookValuePerShareAnnual"),
                        "tangible_book_value": metrics.get("tangibleBookValuePerShareAnnual")
                    },
                    "growth_metrics": {
                        "revenue_growth_ttm": metrics.get("revenueGrowthTTMYoy"),
                        "eps_growth_ttm": metrics.get("epsGrowthTTMYoy"),
                        "revenue_ttm": metrics.get("revenueTTM")
                    },
                    "market_metrics": {
                        "beta": metrics.get("beta"),
                        "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
                        "52_week_high": metrics.get("52WeekHigh"),
                        "52_week_low": metrics.get("52WeekLow"),
                        "52_week_return": metrics.get("52WeekPriceReturnDaily")
                    }
                }
                cache_data(cache_key, result)
                return result
            else:
                return {"status": "error", "message": f"No metrics data found for {symbol}"}
                
    except Exception as e:
        logger.error(f"Error fetching metrics for {symbol}: {e}")
        return {"status": "error", "message": f"Error fetching metrics: {e}"}

@mcp.tool()
async def get_stock_news(symbol: str, limit: int = 20) -> dict:
    """
    Retrieves recent news articles specifically related to a **single stock ticker symbol**.
    Use this tool when the user asks for news about a particular company (e.g., "What's the latest news on Apple?").
    For general news topics not tied to a specific stock, use the `newsapi_org` tool.

    :param symbol: The stock ticker symbol (e.g., "AAPL", "GOOG").
    :param limit: The maximum number of recent news articles to return (default is 20, max 50).
    :returns: A dictionary containing recent news articles, including headlines, summaries, and URLs.
    """
    cache_key = f"news_{symbol.upper()}_{limit}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    # Get date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "token": finnhub_key
                }
            )
            response.raise_for_status()
            news_data = response.json()
            
            if isinstance(news_data, list):
                limited_news = news_data[:min(limit, len(news_data))]
                
                formatted_news = []
                for article in limited_news:
                    formatted_news.append({
                        "headline": article.get("headline", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "datetime": article.get("datetime", 0),
                        "category": article.get("category", ""),
                        "image": article.get("image", "")
                    })
                
                result = {
                    "status": "success",
                    "symbol": symbol.upper(),
                    "news_count": len(formatted_news),
                    "articles": formatted_news
                }
                cache_data(cache_key, result)
                return result
            else:
                return {
                    "status": "success",
                    "symbol": symbol.upper(),
                    "news_count": 0,
                    "articles": []
                }
                
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return {"status": "error", "message": f"Error fetching news: {e}"}

@mcp.tool()
async def get_market_news(category: str = "general", limit: int = 20) -> dict:
    """
    Fetches general market news headlines from Finnhub, limited to specific, predefined market categories.
    **DO NOT use this tool for general news topics like 'AI', 'technology', or specific company news.**
    Supported categories are: 'general', 'forex', 'crypto', 'merger'.
    For broader news topics, use the `newsapi_org` tool.

    :param category: The news category to fetch (e.g., 'general', 'forex', 'crypto', 'merger'). Defaults to 'general'.
    :param limit: The number of news articles to return (default is 20).
    :returns: A dictionary containing market news articles.
    """
    cache_key = f"market_news_{category}_{limit}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/news",
                params={
                    "category": category,
                    "token": finnhub_key
                }
            )
            response.raise_for_status()
            news_data = response.json()
            
            if isinstance(news_data, list):
                limited_news = news_data[:min(limit, len(news_data))]
                
                formatted_news = []
                for article in limited_news:
                    formatted_news.append({
                        "headline": article.get("headline", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "datetime": article.get("datetime", 0),
                        "category": article.get("category", ""),
                        "image": article.get("image", "")
                    })
                
                result = {
                    "status": "success",
                    "category": category,
                    "news_count": len(formatted_news),
                    "articles": formatted_news
                }
                cache_data(cache_key, result)
                return result
            else:
                return {
                    "status": "success",
                    "category": category,
                    "news_count": 0,
                    "articles": []
                }
                
    except Exception as e:
        logger.error(f"Error fetching market news: {e}")
        return {"status": "error", "message": f"Error fetching market news: {e}"}

@mcp.tool()
async def get_stock_peers(symbol: str) -> dict:
    """
    Identifies and fetches a list of peer companies for a given stock ticker symbol.
    Peer companies are typically in the same industry or have similar market characteristics.
    Use this tool when the user asks for competitors or comparable companies to a specific stock.

    :param symbol: The stock ticker symbol (e.g., "GOOGL").
    :returns: A dictionary containing a list of peer company symbols.
    """
    cache_key = f"peers_{symbol.upper()}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/stock/peers",
                params={"symbol": symbol, "token": finnhub_key}
            )
            response.raise_for_status()
            peers_data = response.json()
            
            if isinstance(peers_data, list):
                result = {
                    "status": "success",
                    "symbol": symbol.upper(),
                    "peers": peers_data,
                    "peer_count": len(peers_data)
                }
                cache_data(cache_key, result)
                return result
            else:
                return {"status": "error", "message": f"No peers data found for {symbol}"}
                
    except Exception as e:
        logger.error(f"Error fetching peers for {symbol}: {e}")
        return {"status": "error", "message": f"Error fetching peers: {e}"}

@mcp.tool()
async def get_stock_recommendations(symbol: str) -> dict:
    """
    Retrieves the latest analyst recommendations (e.g., Strong Buy, Buy, Hold, Sell, Strong Sell) for a specific stock.
    Use this tool when the user asks for analyst opinions or ratings on a company's stock.

    :param symbol: The stock ticker symbol (e.g., "AAPL").
    :returns: A dictionary containing the most recent analyst recommendations and historical data.
    """
    cache_key = f"recommendations_{symbol.upper()}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/stock/recommendation",
                params={"symbol": symbol, "token": finnhub_key}
            )
            response.raise_for_status()
            rec_data = response.json()
            
            if isinstance(rec_data, list) and len(rec_data) > 0:
                # Get the most recent recommendation
                latest_rec = rec_data[0]
                
                result = {
                    "status": "success",
                    "symbol": symbol.upper(),
                    "period": latest_rec.get("period"),
                    "strong_buy": latest_rec.get("strongBuy", 0),
                    "buy": latest_rec.get("buy", 0),
                    "hold": latest_rec.get("hold", 0),
                    "sell": latest_rec.get("sell", 0),
                    "strong_sell": latest_rec.get("strongSell", 0),
                    "total_analysts": (
                        latest_rec.get("strongBuy", 0) + 
                        latest_rec.get("buy", 0) + 
                        latest_rec.get("hold", 0) + 
                        latest_rec.get("sell", 0) + 
                        latest_rec.get("strongSell", 0)
                    ),
                    "historical_data": rec_data
                }
                cache_data(cache_key, result)
                return result
            else:
                return {"status": "error", "message": f"No recommendations data found for {symbol}"}
                
    except Exception as e:
        logger.error(f"Error fetching recommendations for {symbol}: {e}")
        return {"status": "error", "message": f"Error fetching recommendations: {e}"}

@mcp.tool()
async def get_market_status() -> dict:
    """
    Checks the current operating status of the US stock market (e.g., open, closed, extended hours).
    Use this tool when the user asks if the stock market is currently open or for general market hours information.

    :returns: A dictionary indicating whether the market is open, the current session, and timezone.
    """
    cache_key = "market_status"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/stock/market-status",
                params={"exchange": "US", "token": finnhub_key}
            )
            response.raise_for_status()
            status_data = response.json()
            
            result = {
                "status": "success",
                "exchange": "US",
                "is_open": status_data.get("isOpen", False),
                "session": status_data.get("session", ""),
                "timezone": status_data.get("timezone", ""),
                "timestamp": int(time.time())
            }
            cache_data(cache_key, result)
            return result
            
    except Exception as e:
        logger.error(f"Error fetching market status: {e}")
        return {"status": "error", "message": f"Error fetching market status: {e}"}

@mcp.tool()
async def get_multiple_stocks(symbols: List[str]) -> dict:
    """
    **PRIMARY TOOL FOR MULTIPLE STOCK QUOTES.**
    Efficiently fetches current stock quotes for **two or more ticker symbols simultaneously**.
    This is the preferred tool when the user asks for the prices of multiple companies at once (e.g., "What are the stock prices for Google and Apple?", "Give me quotes for TSLA, NVDA, and MSFT").
    It handles rate limiting internally for batch requests.

    :param symbols: A list of stock ticker symbols (e.g., ["GOOGL", "NVDA", "AAPL"]).
    :returns: A dictionary containing the stock quote data for all requested symbols.
    """
    results = {}
    
    for symbol in symbols:
        try:
            # Call the internal function directly, not the decorated tool
            stock_data = await _get_stock_quote_internal(symbol)
            results[symbol.upper()] = stock_data
            # Small delay between requests to respect API limits if not batched by provider
            await asyncio.sleep(0.5)
        except Exception as e:
            results[symbol.upper()] = {
                "status": "error",
                "message": f"Failed to fetch data for {symbol}: {str(e)}"
            }
    
    return {
        "status": "success",
        "data": results,
        "total_symbols": len(symbols)
    }

@mcp.tool()
async def search_stocks(query: str, limit: int = 10) -> dict:
    """
    Searches for stock ticker symbols and company descriptions based on a given query.
    Use this tool when the user provides a company name or a partial symbol and needs to find the correct stock ticker.

    :param query: The search query (e.g., "Tesla", "Microsoft", "GOOG").
    :param limit: The maximum number of search results to return (default is 10).
    :returns: A dictionary containing potential stock symbols and their descriptions.
    """
    cache_key = f"search_{query}_{limit}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        return {"status": "error", "message": "FINNHUB_API_KEY not found"}
    
    await finnhub_limiter.wait_if_needed()
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://finnhub.io/api/v1/search",
                params={"q": query, "token": finnhub_key}
            )
            response.raise_for_status()
            search_data = response.json()
            
            if search_data.get("result"):
                results = search_data["result"][:limit]
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "symbol": result.get("symbol", ""),
                        "description": result.get("description", ""),
                        "display_symbol": result.get("displaySymbol", ""),
                        "type": result.get("type", "")
                    })
                
                result = {
                    "status": "success",
                    "query": query,
                    "count": len(formatted_results),
                    "results": formatted_results
                }
                cache_data(cache_key, result)
                return result
            else:
                return {
                    "status": "success",
                    "query": query,
                    "count": 0,
                    "results": []
                }
                
    except Exception as e:
        logger.error(f"Error searching stocks: {e}")
        return {"status": "error", "message": f"Error searching stocks: {e}"}

http_mcp = mcp.http_app(transport="streamable-http")
app = FastAPI(lifespan=http_mcp.router.lifespan_context)
app.mount("/", http_mcp)
logger.info("Finance MCP server initialized with Finnhub primary and Quandl fallback.")
