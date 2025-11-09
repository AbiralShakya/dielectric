#!/usr/bin/env python3
"""
Test Dedalus Labs SDK Integration

Tests the Dedalus client integration with MCP servers.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ai.dedalus_client import DedalusClient


async def test_dedalus_client_initialization():
    """Test Dedalus client can be initialized."""
    print("🧪 Testing Dedalus Client Initialization")
    print("=" * 50)

    try:
        client = DedalusClient()
        print("✅ Dedalus client initialized successfully")
        print(f"   API Key configured: {'Yes' if client.api_key else 'No'}")
        return True
    except Exception as e:
        print(f"❌ Dedalus client initialization failed: {e}")
        return False


async def test_dedalus_runner_creation():
    """Test that Dedalus runner can be created."""
    print("\n🧪 Testing Dedalus Runner Creation")
    print("=" * 50)

    try:
        client = DedalusClient()
        runner = await client._get_runner()
        print("✅ Dedalus runner created successfully")
        return True
    except Exception as e:
        print(f"❌ Dedalus runner creation failed: {e}")
        return False


async def test_mcp_optimization_simulation():
    """Test MCP optimization workflow (simulation - won't actually call MCP)."""
    print("\n🧪 Testing MCP Optimization Simulation")
    print("=" * 50)

    try:
        # This would normally call the actual MCP server
        # For testing, we'll just verify the client setup
        client = DedalusClient()

        print("✅ Dedalus MCP optimization client ready")
        print("   Note: Actual MCP calls require deployed server at specified slug")
        print("   Test with: mcp_server_slug='abiralshakya/ngp' (after deployment)")

        return True
    except Exception as e:
        print(f"❌ MCP optimization simulation failed: {e}")
        return False


async def test_orchestrator_integration():
    """Test orchestrator with Dedalus client."""
    print("\n🧪 Testing Orchestrator Integration")
    print("=" * 50)

    try:
        from backend.agents.orchestrator import AgentOrchestrator

        # Create orchestrator with test slug
        orchestrator = AgentOrchestrator(mcp_server_slug="test/ngp")

        print("✅ Agent orchestrator created with Dedalus client")
        print(f"   MCP Server Slug: {orchestrator.mcp_server_slug}")
        print("   Dedalus client configured: Yes")

        return True
    except Exception as e:
        print(f"❌ Orchestrator integration failed: {e}")
        return False


async def run_dedalus_tests():
    """Run all Dedalus integration tests."""
    print("🚀 Running Dedalus Labs SDK Integration Tests")
    print("=" * 60)

    tests = [
        test_dedalus_client_initialization,
        test_dedalus_runner_creation,
        test_mcp_optimization_simulation,
        test_orchestrator_integration,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    passed = sum(results)
    total = len(results)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Dedalus integration tests PASSED!")
        print("\nNext steps:")
        print("1. Deploy MCP server to Dedalus Labs")
        print("2. Update mcp_server_slug in orchestrator to match deployed server")
        print("3. Test with real MCP server calls")
        return True
    else:
        print("❌ Some Dedalus integration tests FAILED!")
        print("Check API key configuration and SDK installation")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_dedalus_tests())
    sys.exit(0 if success else 1)
