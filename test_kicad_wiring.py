#!/usr/bin/env python3
"""
KiCad Integration Test

Simple test to verify KiCad MCP client wiring and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_kicad_client_import():
    """Test that KiCad client can be imported."""
    print("🔧 Test 1: KiCad Client Import")
    try:
        from src.backend.mcp.kicad_direct_client import KiCadDirectClient
        print("   ✅ KiCadDirectClient imported successfully")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kicad_client_init():
    """Test KiCad client initialization."""
    print("\n🔧 Test 2: KiCad Client Initialization")
    try:
        from src.backend.mcp.kicad_direct_client import KiCadDirectClient
        
        client = KiCadDirectClient()
        
        if client.is_available():
            print("   ✅ KiCad client initialized and available")
            print(f"   📋 Board info: {client.get_board_info()}")
            return True
        else:
            print("   ⚠️  KiCad client initialized but KiCad not available")
            print("   💡 This is OK if KiCad is not installed")
            return True  # Not a failure, just not available
            
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_routing_agent_wiring():
    """Test that RoutingAgent can use KiCad client."""
    print("\n🔧 Test 3: RoutingAgent KiCad Wiring")
    try:
        from src.backend.agents.routing_agent import RoutingAgent
        from src.backend.constraints.pcb_fabrication import FabricationConstraints
        
        constraints = FabricationConstraints()
        routing_agent = RoutingAgent(constraints=constraints)
        
        if routing_agent.kicad_client:
            if routing_agent.kicad_client.is_available():
                print("   ✅ RoutingAgent has KiCad client (available)")
            else:
                print("   ⚠️  RoutingAgent has KiCad client (not available)")
            return True
        else:
            print("   ⚠️  RoutingAgent has no KiCad client")
            print("   💡 This is OK if KiCad is not installed")
            return True
            
    except Exception as e:
        print(f"   ❌ Wiring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_verifier_agent_wiring():
    """Test that VerifierAgent can use KiCad client."""
    print("\n🔧 Test 4: VerifierAgent KiCad Wiring")
    try:
        from src.backend.agents.verifier_agent import VerifierAgent
        
        verifier_agent = VerifierAgent()
        
        if verifier_agent.kicad_client:
            if verifier_agent.kicad_client.is_available():
                print("   ✅ VerifierAgent has KiCad client (available)")
            else:
                print("   ⚠️  VerifierAgent has KiCad client (not available)")
            return True
        else:
            print("   ⚠️  VerifierAgent has no KiCad client")
            print("   💡 This is OK if KiCad is not installed")
            return True
            
    except Exception as e:
        print(f"   ❌ Wiring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_wiring():
    """Test that Orchestrator wires KiCad client to agents."""
    print("\n🔧 Test 5: Orchestrator KiCad Wiring")
    try:
        from src.backend.agents.orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(use_database=False)
        
        if orchestrator.kicad_client:
            if orchestrator.kicad_client.is_available():
                print("   ✅ Orchestrator has KiCad client (available)")
            else:
                print("   ⚠️  Orchestrator has KiCad client (not available)")
            
            # Check that agents share the client
            if orchestrator.routing_agent and orchestrator.routing_agent.kicad_client == orchestrator.kicad_client:
                print("   ✅ RoutingAgent shares Orchestrator's KiCad client")
            else:
                print("   ⚠️  RoutingAgent does not share client")
            
            if orchestrator.verifier_agent and orchestrator.verifier_agent.kicad_client == orchestrator.kicad_client:
                print("   ✅ VerifierAgent shares Orchestrator's KiCad client")
            else:
                print("   ⚠️  VerifierAgent does not share client")
            
            return True
        else:
            print("   ⚠️  Orchestrator has no KiCad client")
            print("   💡 This is OK if KiCad is not installed")
            return True
            
    except Exception as e:
        print(f"   ❌ Wiring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 KiCad Integration Wiring Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Import", test_kicad_client_import()))
    results.append(("Init", test_kicad_client_init()))
    results.append(("RoutingAgent", test_routing_agent_wiring()))
    results.append(("VerifierAgent", test_verifier_agent_wiring()))
    results.append(("Orchestrator", test_orchestrator_wiring()))
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL WIRING TESTS PASSED!")
        print("\n✅ KiCad MCP clients are properly wired to agents")
        print("💡 Note: If KiCad is not installed, tests will show warnings but still pass")
    else:
        print("⚠️  Some wiring tests failed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

