#!/bin/bash

# MLX Fine-Tuning GUI Startup Script
# This script resolves npm permission issues and starts the complete application

set -e

echo "🚀 Starting MLX Fine-Tuning GUI..."

# Navigate to project directory
cd "$(dirname "$0")"

echo "📂 Project directory: $(pwd)"

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing root dependencies..."
    npm install --cache /tmp/npm-cache --no-optional
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install --cache /tmp/npm-cache --no-optional
    cd ..
fi

# Check if Python dependencies are installed
echo "🐍 Checking Python backend dependencies..."
source "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/activate"

# Test if FastAPI is available
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing backend dependencies..."
    pip install -r backend/requirements.txt
fi

echo "✅ All dependencies installed successfully!"
echo ""
echo "🎯 To start the application, run:"
echo "   npm run dev"
echo ""
echo "🔧 This will:"
echo "   - Start the FastAPI backend server on port 8000"
echo "   - Start the React development server on port 5173"
echo "   - Launch the Electron desktop application"
echo ""
echo "📱 The GUI will automatically open as a desktop application"
echo ""

# Option to start immediately
read -p "Start the application now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting application..."
    npm run dev
fi