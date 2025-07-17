# mcp-servers/agent-marketplace-mcp/Dockerfile
FROM base-mcp:latest 

WORKDIR /app

COPY requirements.txt .          
RUN pip install --no-cache-dir -r requirements.txt

# copy the server code
COPY server.py .                


# Set environment variables for internal communication
ENV LOCAL_MODE="true"
ENV FASTMCP_BASE_URL="http://fastmcp-core:9000" 
# ADK's default API server port is 9005
EXPOSE 9005


CMD ["uvicorn", "mcp-servers.agent-marketplace-mcp.server:app", "--host", "0.0.0.0", "--port", "9005"]
