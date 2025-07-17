# mcp-servers/web-mcp/Dockerfile
FROM base-mcp:latest

WORKDIR /app

# copy the server code
COPY server.py .      

# Set environment variables for Kubernetes deployment
ENV LOCAL_MODE="false"
ENV FASTMCP_BASE_URL="http://fastmcp-core-svc:9000"

# Standard MCP port in Kubernetes
EXPOSE 9000

CMD ["uvicorn", "mcp-servers.web-mcp.server:app", "--host", "0.0.0.0", "--port", "9000"]
