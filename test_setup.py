#!/usr/bin/env python3
"""
Quick test script to verify the setup is working
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        from src.config import Config
        print("✅ Config")
        
        from src.utils import setup_logging, sanitize_input
        print("✅ Utils")
        
        from src.models import MedicalQuery, MedicalResponse
        print("✅ Models")
        
        from src.memory import ConversationMemory, CacheManager
        print("✅ Memory")
        
        from src.planner import QueryPlanner
        print("✅ Planner")
        
        from src.executor import ToolExecutor
        print("✅ Executor")
        
        from src.agent import HealthcareNavigatorAgent
        print("✅ Agent")
        
        import streamlit
        print("✅ Streamlit")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without API calls"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from src.agent import HealthcareNavigatorAgent
        from src.models import MedicalQuery
        
        # Test agent initialization
        agent = HealthcareNavigatorAgent("test_session")
        print("✅ Agent initialization")
        
        # Test query creation
        query = MedicalQuery(query_text="What is diabetes?")
        print("✅ Query creation")
        
        # Test input sanitization
        from src.utils import sanitize_input
        sanitized = sanitize_input("Test <script>alert('xss')</script> input")
        print("✅ Input sanitization")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🌍 Testing environment...")
    
    try:
        from src.config import Config
        
        # Check if .env file exists
        if os.path.exists('.env'):
            print("✅ .env file exists")
        else:
            print("⚠️ .env file not found (using defaults)")
        
        # Test config loading
        config_test = hasattr(Config, 'LOG_LEVEL')
        if config_test:
            print("✅ Config loading")
        else:
            print("❌ Config loading failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 Healthcare Navigator - Setup Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_environment():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 Next steps:")
        print("1. Add your API keys to .env file")
        print("2. Run: streamlit run web_app.py")
        print("3. Or try: python cli.py --query 'What is diabetes?'")
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())