FROM base-mcp:latest

WORKDIR /app

# Copy the bot application code
COPY . .

# Set environment variables for Kubernetes deployment
ENV LOCAL_MODE="false"
ENV FASTMCP_BASE_URL="http://fastmcp-core-svc:9000"
ENV BOT_API_BASE_URL="http://bot-api-svc:8000"
ENV DISCORD_EVENTS_ENDPOINT="http://bot-api-svc:8000/discord/receive_message"

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "bot.bot_api:app", "--host", "0.0.0.0", "--port", "8000"]
