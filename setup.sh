#!/bin/bash

# Setup script for BERT Attention Visualizer
echo "🔍 Setting up BERT Attention Visualizer..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv bert_env
    source bert_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if font file exists
if [ ! -f "assets/fonts/OpenSans-Regular.ttf" ]; then
    echo "⚠️  Font file not found at assets/fonts/OpenSans-Regular.ttf"
    echo "Please ensure the font file exists for proper visualization"
else
    echo "✅ Font file found"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the application:"
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "1. Activate virtual environment: source bert_env/bin/activate"
fi
echo "2. Run the app: python mask.py"
echo "3. Enter text with [MASK] token when prompted"
echo ""
echo "🔍 Happy attention visualizing!"
