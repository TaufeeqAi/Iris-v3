# mcp-servers/discord-mcp/Dockerfile
FROM base-mcp:latest

WORKDIR /app

COPY requirements.txt .          
RUN pip install --no-cache-dir -r requirements.txt

# copy the server code
COPY server.py .      

# Set environment variables for Kubernetes deployment
ENV LOCAL_MODE="false"
ENV FASTMCP_BASE_URL="http://fastmcp-core-svc:9000"
ENV BOT_API_BASE_URL="http://bot-api-svc:8000"

# Standard MCP port in Kubernetes
EXPOSE 9000

CMD ["uvicorn", "mcp-servers.discord-mcp.server:app", "--host", "0.0.0.0", "--port", "9000"]
