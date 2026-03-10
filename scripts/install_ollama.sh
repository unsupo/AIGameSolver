#!/bin/bash

# AutoGamePlayer Ollama Installer for macOS
echo "🦙 Starting Ollama Installation..."

if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed."
else
    echo "📥 Downloading and installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "🚀 Starting Ollama Application..."
# On macOS, the app is usually in /Applications
if [ -d "/Applications/Ollama.app" ]; then
    open -a Ollama
    echo "⏳ Waiting for Ollama server to wake up..."
    sleep 5
else
    # Fallback for CLI-only or linux-style
    ollama serve > /dev/null 2>&1 &
    echo "⏳ Started Ollama background service."
    sleep 5
fi

# Pre-pull the vision model
echo "📥 Pulling Llama 3.2 Vision (this may take a few minutes)..."
ollama pull llama3.2-vision

echo "✨ Ollama is ready! You can now run: poetry run llm_play"
