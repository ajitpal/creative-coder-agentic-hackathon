#!/bin/bash

# Healthcare Navigator Setup Script
echo "🏥 Setting up Intelligent Healthcare Navigator..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= 3.8 required)"
else
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv healthcare_navigator_env
source healthcare_navigator_env/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template
if [ ! -f .env ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.template .env
    echo "📝 Please edit .env file with your API keys:"
    echo "   - GEMINI_API_KEY (required)"
    echo "   - WHO_ICD credentials are pre-configured"
    echo "   - OPENFDA_API_KEY (optional but recommended)"
else
    echo "✅ Environment file already exists"
fi

# Create logs directory
mkdir -p logs

# Test installation
echo "🧪 Testing installation..."
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from src.config import Config
    print('✅ Configuration module loaded successfully')
    from src.agent import HealthcareNavigatorAgent
    print('✅ Agent module loaded successfully')
    print('🎉 Setup completed successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo ""
echo "🚀 Setup complete! Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source healthcare_navigator_env/bin/activate"
echo "3. Start CLI: python cli.py"
echo "4. Start Web UI: streamlit run web_app.py"
echo ""
echo "📚 See README.md for detailed usage instructions"