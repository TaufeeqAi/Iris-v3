import os
import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load API keys from environment
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
STACKEXCHANGE_KEY = os.getenv("STACKEXCHANGE_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY is required")
if not SERPAPI_KEY:
    raise RuntimeError("SERPAPI_KEY environment variable is required")
if not RAPIDAPI_KEY:
    raise RuntimeError("RAPIDAPI_KEY environment variable is required")
if not STACKEXCHANGE_KEY:
    raise RuntimeError("STACKEXCHANGE_KEY environment variable is required")
if not NEWSAPI_KEY:
    raise RuntimeError("NEWSAPI_KEY environment variable is required")

mcp = FastMCP("multi_search")

@mcp.tool()
async def serpapi_search(query: str, num_results: int = 5) -> dict:
    """
    Performs a live, organic web search via SerpApi, providing comprehensive search engine results.
    This tool is highly effective for general knowledge queries, factual lookups, and finding information
    that might be available on various websites. It returns the top `num_results` from the search engine.
    Use this for broad web searches that are not covered by more specific tools.

    :param query: The search query string (e.g., "latest advancements in quantum computing", "history of the internet").
    :param num_results: The number of top organic search results to return (default is 5).
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results
    }
    url = "https://serpapi.com/search.json"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })
    return {"query": query, "results": results}



@mcp.tool()
async def stackoverflow_search(tag: str, num_results: int = 5) -> dict:
    """
    Searches StackOverflow for questions related to a specific programming or technical tag.
    Use this tool when the user asks for help with coding problems, technical errors,
    or questions related to programming languages, frameworks, or software development.

    :param tag: The tag to search for (e.g., "python", "javascript", "reactjs", "machine-learning").
    :param num_results: The number of top questions to retrieve (default is 5).
    """
    url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "order": "desc",
        "sort": "activity",
        "tagged": tag,
        "site": "stackoverflow",
        "pagesize": num_results,
        "filter": "!nKzQUR3Egv" 
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    questions = [{
        "title": q.get("title"),
        "link": q.get("link")
    } for q in data.get("items", [])]
    return {"tag": tag, "questions": questions}

@mcp.tool()
async def newsapi_org(topic: str, num_results: int = 5) -> dict:
    """
    **PRIMARY TOOL FOR GENERAL NEWS TOPICS.**
    Retrieves the latest news headlines and articles on **any specific topic or keyword** from a wide range of global news sources via NewsAPI.org.
    This tool is ideal for finding news on subjects like 'AI', 'technology trends', 'environmental policy', 'sports events', 'political developments', or specific company names (if not a stock-specific news query).
    It provides a broad overview of recent developments related to the specified topic.

    :param topic: The keyword or phrase for the news search (e.g., "artificial intelligence", "climate change impacts", "Olympics 2028").
    :param num_results: The number of top headlines to fetch (default is 5, max is typically 100).
    """
    params = {"q": topic, "pageSize": num_results, "apiKey": NEWSAPI_KEY}
    url = "https://newsapi.org/v2/everything"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    articles = data.get("articles", [])[:num_results]

    headlines = [{
        "title": a.get("title"),
        "url": a.get("url"),
        "source": a.get("source", {}).get("name")
    } for a in articles]
    return {"topic": topic, "headlines": headlines}



@mcp.tool()
async def get_weather(city: str) -> dict:
    """
    Fetches the current weather conditions for a specified city using OpenWeatherMap.
    Use this tool when the user asks about the weather, temperature, or forecast for a particular location.

    :param city: The name of the city (e.g., "London", "New York", "Noida").
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    return {
        "city": data.get("name"),
        "description": data["weather"][0]["description"],
        "temperature_c": data["main"]["temp"],
        "humidity": data["main"]["humidity"]
    }

@mcp.tool()
async def google_search(query: str, num_results: int = 5) -> dict:
    """
    Performs a live Google Web Search via RapidAPI, providing general web search capabilities.
    This tool is a good alternative for general queries if `serpapi_search` is unavailable or for specific types of information
    that might be better found through a direct Google search.

    :param query: The search query string (e.g., "best restaurants in Tokyo", "how to tie a knot").
    :param num_results: The number of search results to return (default is 5).
    """
    url = "https://google-search72.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "google-search72.p.rapidapi.com"
    }
    params = {
        "q": query,
        "num": num_results
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", [])[:num_results]:
        results.append({
            "title":       item.get("title"),
            "link":        item.get("link"),
            "description": item.get("description")
        })

    return {"query": query, "results": results}

# Mount the MCP server
http_mcp = mcp.http_app(transport="streamable-http")
app = FastAPI(lifespan=http_mcp.router.lifespan_context)
app.mount("/", http_mcp)
