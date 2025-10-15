#!/usr/bin/env python3
"""
Test script to verify the Chainlit app setup and configuration
"""

import os
import sys
from dotenv import load_dotenv

def test_environment_variables():
    """Test if environment variables are properly loaded"""
    print("🔍 Testing environment variable loading...")
    
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), '..', 'variables.env')
    load_dotenv(env_path)
    
    # Check required variables
    required_vars = ['LLM_URL', 'LLM_API_KEY', 'LLM_MODEL']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Mask sensitive values
            if 'KEY' in var:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
    
    if missing_vars:
        print(f"❌ Missing required variables: {', '.join(missing_vars)}")
        return False
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n📦 Testing package imports...")
    
    try:
        import chainlit as cl
        print("✅ chainlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import chainlit: {e}")
        return False
    
    try:
        import openai
        print("✅ openai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import openai: {e}")
        return False
    
    try:
        import httpx
        print("✅ httpx imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import httpx: {e}")
        return False
    
    return True

def test_chainlit_config():
    """Test if Chainlit configuration file exists"""
    print("\n⚙️  Testing Chainlit configuration...")
    
    config_path = os.path.join(os.path.dirname(__file__), '.chainlit', 'config.toml')
    if os.path.exists(config_path):
        print("✅ Chainlit config file found")
        return True
    else:
        print("❌ Chainlit config file not found")
        return False

def main():
    """Run all tests"""
    print("🏠 Zillow Home Finder - Setup Test\n")
    
    tests = [
        test_environment_variables,
        test_imports,
        test_chainlit_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Chainlit app is ready to run.")
        print("\nTo start the app, run:")
        print("  ./run.sh")
        print("  or")
        print("  chainlit run app.py --port 8022")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
