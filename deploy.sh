#!/bin/bash

# CultiKure Deployment Script
# This script automates the Docker deployment process

echo "=========================================="
echo "    CultiKure Deployment Automation      "
echo "=========================================="

# 1. Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "❌ Error: Docker is not installed or not in your PATH."
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✅ Docker is installed."

# 2. Check for .env file
if [ ! -f .env ]; then
    echo "⚠️ .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please ensure your ANTHROPIC_API_KEY is set."
fi

# 3. Pull latest changes (optional but good practice)
# echo "Pulling latest changes from git..."
# git pull origin main

# 4. Build and Start the Docker containers
echo "🚀 Building and starting Docker containers in the background..."
docker-compose up --build -d

# 5. Check if it was successful
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ Deployment Successful!"
    echo "🌍 Application is running at: http://localhost:5000"
    echo "📊 Triton Server is running on port 8000"
    echo "=========================================="
    echo "To view live logs, run: docker-compose logs -f"
    echo "To stop the application, run: docker-compose down"
else
    echo "❌ Deployment failed. Check the errors above."
    exit 1
fi
