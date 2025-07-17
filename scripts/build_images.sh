#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building Docker images..."

# 1. Build the base-mcp image FIRST. It uses the root requirements.txt and common/
echo "Building base-mcp:latest..."
docker build -f mcp-servers/base-mcp/Dockerfile.base -t base-mcp:latest .

# 2. Build other images that use base-mcp as their base
echo "Building multi-agent-bot-api..."
docker build -f bot/Dockerfile -t multi-agent-bot-api:latest .

echo "Building fastmcp-core-server..."
docker build -f Dockerfile.fastmcp_core -t fastmcp-core-server:latest .

echo "Building telegram-mcp-server..."
docker build -f mcp-servers/telegram-mcp/Dockerfile -t telegram-mcp-server:latest .

echo "Building discord-mcp-server..."
docker build -f mcp-servers/discord-mcp/Dockerfile -t discord-mcp-server:latest .

echo "Building web-mcp-server..."
docker build -f mcp-servers/web-mcp/Dockerfile -t web-mcp-server:latest .

echo "Building finance-mcp-server..."
docker build -f mcp-servers/finance-mcp/Dockerfile -t finance-mcp-server:latest .

echo "Building rag-mcp-server..."
docker build -f mcp-servers/rag-mcp/Dockerfile -t rag-mcp-server:latest .

# Build the RAG data loader image
echo "Building rag-data-loader:latest..."
docker build -f Dockerfile.rag_data_loader -t rag-data-loader:latest .

echo "All Docker images built successfully!"
