#!/usr/bin/env python3
"""
Setup script for Intelligent Healthcare Navigator
Handles dependency installation and environment configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header():
    """Print setup header"""
    print("Setting up Intelligent Healthcare Navigator...")
    print("=" * 50)

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    # Core dependencies that are essential
    core_deps = [
        "google-generativeai>=0.3.0",
        "requests>=2.28.0", 
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "python-dotenv>=0.19.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "pydantic>=1.10.0",
        "json5>=0.9.0"
    ]
    
    # Optional dependencies (nice to have but not critical)
    optional_deps = [
        "medspacy>=1.0.0",
        "spacy>=3.4.0", 
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "scikit-learn>=1.1.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=22.0.0",
        "flake8>=5.0.0"
    ]
    
    # Upgrade pip first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✓ Pip upgraded")
    except subprocess.CalledProcessError:
        print("⚠️ Could not upgrade pip, continuing...")
    
    # Install core dependencies
    print("\nInstalling core dependencies...")
    failed_core = []
    for dep in core_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {dep}")
        except subprocess.CalledProcessError:
            print(f"❌ {dep}")
            failed_core.append(dep)
    
    # Install optional dependencies
    print("\nInstalling optional dependencies...")
    failed_optional = []
    for dep in optional_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️ {dep} (optional - skipping)")
            failed_optional.append(dep)
    
    # Summary
    if failed_core:
        print(f"\n❌ Critical dependencies failed: {len(failed_core)}")
        print("The app may not work properly without these.")
        for dep in failed_core:
            print(f"   - {dep}")
    else:
        print(f"\n✓ All core dependencies installed successfully!")
    
    if failed_optional:
        print(f"\n⚠️ Optional dependencies skipped: {len(failed_optional)}")
        print("These provide enhanced functionality but aren't required.")
    
    return len(failed_core) == 0

def setup_environment():
    """Set up environment configuration"""
    print("Setting up environment...")
    
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists() and env_template.exists():
        shutil.copy(env_template, env_file)
        print("✓ Created .env file from template")
        print("⚠️ Please edit .env file with your API keys")
    elif env_file.exists():
        print("✓ .env file already exists")
    else:
        # Create basic .env file
        env_content = """# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# WHO ICD API Configuration  
WHO_ICD_CLIENT_ID=your_who_client_id_here
WHO_ICD_CLIENT_SECRET=your_who_client_secret_here

# OpenFDA API Configuration (Optional)
OPENFDA_API_KEY=your_openfda_api_key_here

# Application Configuration
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_CONVERSATION_HISTORY=50
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✓ Created basic .env file")
        print("⚠️ Please edit .env file with your API keys")

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "logs",
        "data",
        "cache",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/ directory")

def test_imports():
    """Test critical imports"""
    print("Testing critical imports...")
    
    critical_imports = [
        ("streamlit", "Streamlit web framework"),
        ("requests", "HTTP requests library"),
        ("dotenv", "Environment variables"),
        ("pandas", "Data processing"),
    ]
    
    for module, description in critical_imports:
        try:
            __import__(module)
            print(f"✓ {description}")
        except ImportError:
            print(f"❌ {description} - import failed")

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your API keys:")
    print("   - Get Gemini API key: https://makersuite.google.com/app/apikey")
    print("   - Get WHO ICD credentials: https://icd.who.int/icdapi")
    print("   - Get OpenFDA key (optional): https://open.fda.gov/apis/authentication/")
    
    print("\n2. Test the setup:")
    print("   python -c \"from src.config import Config; print('Config loaded successfully')\"")
    
    print("\n3. Run the application:")
    print("   # CLI interface:")
    print("   python cli.py")
    print("   ")
    print("   # Web interface:")
    print("   streamlit run web_app.py")
    
    print("\n4. Try example queries:")
    print("   python cli.py --query \"What is diabetes?\"")
    print("   python cli.py --drug \"aspirin\"")
    
    print("\n📚 Documentation:")
    print("   - README.md - Complete setup and usage guide")
    print("   - DEMO.md - Demo scenarios and examples")

def main():
    """Main setup function"""
    print_header()
    
    try:
        check_python_version()
        install_dependencies()
        setup_environment()
        create_directories()
        test_imports()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()