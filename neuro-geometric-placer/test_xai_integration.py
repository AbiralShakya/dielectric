#!/usr/bin/env python3
"""
Test xAI Integration

Tests that xAI API integration works correctly.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ai.xai_client import XAIClient
from backend.agents.intent_agent import IntentAgent


async def test_xai_intent_agent():
    """Test Intent Agent with xAI."""
    print("🧪 Testing xAI Intent Agent Integration")
    print("=" * 50)

    try:
        agent = IntentAgent()

        test_intent = "Optimize for minimal trace length, but keep high-power components cool"
        context = {
            "num_components": 5,
            "board_area": 10000  # mm²
        }

        result = await agent.process(test_intent, context)

        if result["success"]:
            print("✅ Intent Agent: SUCCESS")
            print(f"   Intent: {test_intent}")
            print(f"   Weights: {result['weights']}")
            print(f"   Explanation: {result['explanation']}")
            return True
        else:
            print("❌ Intent Agent: FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Intent Agent: EXCEPTION - {str(e)}")
        return False


def test_xai_client():
    """Test xAI client directly."""
    print("\n🧪 Testing xAI Client Direct Integration")
    print("=" * 50)

    try:
        client = XAIClient()

        # Test basic connectivity (this will fail without valid key, but test structure)
        if hasattr(client, 'api_key') and client.api_key:
            print("✅ xAI Client: API key configured")
            return True
        else:
            print("❌ xAI Client: API key not configured")
            return False

    except Exception as e:
        print(f"❌ xAI Client: EXCEPTION - {str(e)}")
        return False


async def test_full_pipeline():
    """Test full pipeline with xAI."""
    print("\n🧪 Testing Full Pipeline with xAI")
    print("=" * 50)

    # Test xAI client
    client_ok = test_xai_client()

    # Test intent agent
    agent_ok = await test_xai_intent_agent()

    if client_ok and agent_ok:
        print("\n✅ All xAI tests PASSED!")
        print("   xAI integration is working correctly.")
        return True
    else:
        print("\n❌ Some xAI tests FAILED!")
        print("   Check your XAI_API_KEY in .env file.")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    sys.exit(0 if success else 1)
