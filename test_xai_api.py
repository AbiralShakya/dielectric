#!/usr/bin/env python3
"""
Test xAI API Directly
Tests the xAI (Grok) API with a simple request matching the curl format.
"""

import os
import json
import requests
import sys

def test_xai_api():
    """Test xAI API with simple request."""
    
    # Get API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ XAI_API_KEY not set")
        print("Set it with: export XAI_API_KEY=your_key")
        sys.exit(1)
    
    endpoint = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    print("🧪 Testing xAI API Directly")
    print("=" * 50)
    print()
    
    # Test 1: Simple test (matching your format)
    print("1. Simple Test (grok-4-latest)...")
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a test assistant."
            },
            {
                "role": "user",
                "content": "Testing. Just say hi and hello world and nothing else."
            }
        ],
        "model": "grok-4-latest",
        "stream": False,
        "temperature": 0
    }
    
    try:
        response = requests.post(endpoint, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            print("✅ xAI API is working!")
            print(f"Response: {result['choices'][0]['message']['content']}")
            print(f"Model: {result.get('model', 'unknown')}")
            print(f"Usage: {result.get('usage', {})}")
        else:
            print("❌ Unexpected response format")
            print(json.dumps(result, indent=2))
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        sys.exit(1)
    
    print()
    print("2. Testing with grok-2-1212 (fallback model)...")
    data2 = {
        "messages": [
            {
                "role": "system",
                "content": "You are a PCB design expert."
            },
            {
                "role": "user",
                "content": "What is a PCB?"
            }
        ],
        "model": "grok-2-1212",
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response2 = requests.post(endpoint, json=data2, headers=headers, timeout=30)
        response2.raise_for_status()
        result2 = response2.json()
        
        if "choices" in result2 and len(result2["choices"]) > 0:
            print("✅ grok-2-1212 is working!")
            print(f"Response: {result2['choices'][0]['message']['content'][:100]}...")
        else:
            print("⚠️  grok-2-1212 test failed, but grok-4-latest works")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️  grok-2-1212 test failed: {e}")
    
    print()
    print("✅ xAI API tests complete!")
    print()
    print("Your API key is working. Dielectric can now use xAI for:")
    print("  - Design generation")
    print("  - Intent processing")
    print("  - Optimization reasoning")
    print("  - Post-optimization analysis")

if __name__ == "__main__":
    test_xai_api()

