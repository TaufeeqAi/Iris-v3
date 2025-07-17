# fastmcp_core_server.py
import os
from fastapi import FastAPI
from fastmcp import FastMCP
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

try:
    from common.utils import setup_logging
    setup_logging(__name__)
except ImportError:
    # If setup_logging fails, use basicConfig and log a warning
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.warning("Could not import common.utils.setup_logging. Using default logging.")


mcp = FastMCP("core")

# Get the FastAPI app from the FastMCP instance
http_mcp = mcp.http_app(transport="streamable-http")

# Define a simple lifespan for the core MCP server
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Core FastMCP server startup initiated.")
    # FastMCP's own lifespan context handles its internal task group initialization
    async with http_mcp.router.lifespan_context(app) as fastmcp_lifespan_yield:
        yield fastmcp_lifespan_yield
    logger.info("Core FastMCP server shutdown initiated.")

# Create the FastAPI app instance
app = FastAPI(lifespan=lifespan)
# Mount FastMCP's app at the root path "/"
app.mount("/", http_mcp)

logger.info("Core FastMCP server initialized and ready to register tools.")
