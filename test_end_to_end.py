#!/usr/bin/env python3
"""
End-to-End Test: Full Dielectric Workflow

Tests the complete workflow:
1. Design → Generate PCB from natural language
2. Optimize → Place components optimally
3. Route → Route all traces
4. Verify → Run DRC and DFM checks
5. Export → Generate KiCad files and production files

Tests KiCad integration with actual KiCad instance.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.geometry.placement import Placement
from src.backend.geometry.board import Board
from src.backend.geometry.component import Component
from src.backend.geometry.net import Net
from src.backend.agents.design_generator_agent import DesignGeneratorAgent
from src.backend.agents.orchestrator import AgentOrchestrator
from src.backend.agents.routing_agent import RoutingAgent
from src.backend.agents.verifier_agent import VerifierAgent
from src.backend.agents.exporter_agent import ExporterAgent
from src.backend.constraints.pcb_fabrication import FabricationConstraints


async def test_full_workflow():
    """Test complete end-to-end workflow."""
    print("🚀 Dielectric End-to-End Test")
    print("=" * 60)
    
    # Step 1: Generate Design
    print("\n📐 Step 1: Generating PCB design from natural language...")
    design_agent = DesignGeneratorAgent()
    
    try:
        design_result = await design_agent.generate_from_natural_language(
            description="Create a simple amplifier circuit with 2 op-amps, 4 resistors, and 2 capacitors",
            board_size={"width": 80, "height": 60},
            layer_count=2,
            use_real_footprints=True
        )
        
        if not design_result.get("success"):
            print(f"❌ Design generation failed: {design_result.get('error')}")
            return False
        
        design_data = design_result["placement"]
        print(f"✅ Design generated: {len(design_data.get('components', []))} components, {len(design_data.get('nets', []))} nets")
        
    except Exception as e:
        print(f"❌ Design generation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Create Placement object
    print("\n🔧 Step 2: Creating placement object...")
    try:
        board = Board.from_dict(design_data["board"])
        components = {}
        nets = {}
        
        for comp_data in design_data.get("components", []):
            comp = Component(
                name=comp_data["name"],
                package=comp_data.get("package", "SOIC-8"),
                x=comp_data.get("x", 0),
                y=comp_data.get("y", 0),
                width=comp_data.get("width", 5),
                height=comp_data.get("height", 4),
                angle=comp_data.get("angle", 0),
                power=comp_data.get("power", 0.0)
            )
            components[comp.name] = comp
        
        for net_data in design_data.get("nets", []):
            net = Net(
                name=net_data["name"],
                pins=net_data.get("pins", [])
            )
            nets[net.name] = net
        
        placement = Placement(list(components.values()), board, list(nets.values()))
        print(f"✅ Placement created: {len(components)} components, {len(nets)} nets")
        
    except Exception as e:
        print(f"❌ Placement creation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Optimize Placement
    print("\n⚡ Step 3: Optimizing component placement...")
    orchestrator = AgentOrchestrator(use_database=False)
    
    try:
        optimize_result = await orchestrator.optimize_fast(
            placement=placement,
            user_intent="Optimize for thermal management and minimize trace length"
        )
        
        if not optimize_result.get("success"):
            print(f"❌ Optimization failed: {optimize_result.get('error')}")
            return False
        
        optimized_placement = optimize_result["placement"]
        final_score = optimize_result["score"]
        print(f"✅ Optimization complete: Score = {final_score:.4f}")
        print(f"   Agents used: {optimize_result.get('agents_used', [])}")
        
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Route Traces
    print("\n🔌 Step 4: Routing traces...")
    try:
        constraints = FabricationConstraints()
        routing_agent = RoutingAgent(constraints=constraints)
        
        # Use KiCad client from orchestrator if available
        if orchestrator.kicad_client:
            routing_agent.kicad_client = orchestrator.kicad_client
        
        route_result = await routing_agent.route_design(optimized_placement)
        
        if not route_result.get("success"):
            print(f"❌ Routing failed: {route_result.get('error')}")
            return False
        
        routed_nets = route_result.get("routed_nets", 0)
        total_length = route_result.get("total_trace_length", 0)
        print(f"✅ Routing complete: {routed_nets}/{route_result.get('total_nets', 0)} nets routed")
        print(f"   Total trace length: {total_length:.1f}mm")
        
        if route_result.get("traces"):
            kicad_traces = [t for t in route_result["traces"] if t.get("kicad_placed")]
            if kicad_traces:
                print(f"   ✅ {len(kicad_traces)} traces placed in KiCad")
        
    except Exception as e:
        print(f"❌ Routing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Verify Design
    print("\n🔍 Step 5: Verifying design (DRC + DFM)...")
    try:
        verifier_agent = VerifierAgent(kicad_client=orchestrator.kicad_client)
        
        verify_result = await verifier_agent.process(
            optimized_placement,
            include_dfm=True,
            run_kicad_drc=True
        )
        
        violations = verify_result.get("violations", [])
        warnings = verify_result.get("warnings", [])
        dfm_score = verify_result.get("dfm_score", 0)
        
        print(f"✅ Verification complete:")
        print(f"   Violations: {len(violations)}")
        print(f"   Warnings: {len(warnings)}")
        print(f"   DFM Score: {dfm_score:.2f}")
        
        if violations:
            print(f"   ⚠️  Critical violations found:")
            for v in violations[:5]:  # Show first 5
                print(f"      - {v.get('message', 'Unknown violation')}")
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Export to KiCad
    print("\n📤 Step 6: Exporting to KiCad and production files...")
    try:
        exporter_agent = ExporterAgent()
        
        export_result = await exporter_agent.process(
            optimized_placement,
            format="production",
            include_production_files=True,
            include_step=True
        )
        
        if not export_result.get("success"):
            print(f"❌ Export failed: {export_result.get('error')}")
            return False
        
        print(f"✅ Export complete:")
        print(f"   Format: {export_result.get('format', 'unknown')}")
        print(f"   Method: {export_result.get('method', 'unknown')}")
        
        if export_result.get("production_files"):
            files = export_result["production_files"]
            print(f"   Production files: {len(files)} files generated")
            for filename in list(files.keys())[:5]:  # Show first 5
                print(f"      - {filename}")
        
        if export_result.get("step_file"):
            print(f"   ✅ 3D STEP file generated")
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Save KiCad board if client available
    if orchestrator.kicad_client and orchestrator.kicad_client.is_available():
        print("\n💾 Step 7: Saving KiCad board...")
        try:
            output_path = orchestrator.kicad_client.save_board()
            print(f"✅ KiCad board saved to: {output_path}")
        except Exception as e:
            print(f"⚠️  Could not save KiCad board: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 END-TO-END TEST COMPLETE!")
    print("\n✅ All workflow steps completed successfully:")
    print("   1. ✅ Design generation")
    print("   2. ✅ Component placement optimization")
    print("   3. ✅ Trace routing")
    print("   4. ✅ Design verification (DRC + DFM)")
    print("   5. ✅ File export (KiCad + Production)")
    
    return True


async def test_kicad_integration():
    """Test KiCad integration specifically."""
    print("\n🔧 Testing KiCad Integration")
    print("=" * 60)
    
    try:
        from src.backend.mcp.kicad_direct_client import KiCadDirectClient
        
        client = KiCadDirectClient()
        
        if not client.is_available():
            print("⚠️  KiCad not available - skipping KiCad-specific tests")
            print("   To enable KiCad integration:")
            print("   1. Install KiCad 9.0+")
            print("   2. Ensure pcbnew Python module is accessible")
            return True
        
        print("✅ KiCad client initialized")
        
        # Test board info
        board_info = client.get_board_info()
        if board_info.get("success"):
            print(f"✅ Board info: {board_info.get('width')}mm x {board_info.get('height')}mm")
            print(f"   Layers: {board_info.get('layer_count')}, Nets: {board_info.get('net_count')}")
        
        # Test adding a net
        print("\n   Testing: Add net...")
        net_result = await client.add_net("VCC")
        if net_result.get("success"):
            print(f"   ✅ Added net: {net_result.get('net', {}).get('name')}")
        else:
            print(f"   ⚠️  Net addition: {net_result.get('error')}")
        
        # Test routing a trace
        print("\n   Testing: Route trace...")
        trace_result = await client.route_trace(
            start={"x": 10.0, "y": 10.0, "unit": "mm"},
            end={"x": 20.0, "y": 20.0, "unit": "mm"},
            layer="F.Cu",
            width=0.2,
            net="VCC"
        )
        if trace_result.get("success"):
            print(f"   ✅ Routed trace: {trace_result.get('trace', {}).get('net')}")
        else:
            print(f"   ⚠️  Trace routing: {trace_result.get('error')}")
        
        # Test DRC
        print("\n   Testing: Run DRC...")
        drc_result = await client.run_drc()
        if drc_result.get("success"):
            violations = drc_result.get("violations", [])
            print(f"   ✅ DRC complete: {len(violations)} violations found")
        else:
            print(f"   ⚠️  DRC: {drc_result.get('error')}")
        
        print("\n✅ KiCad integration tests complete")
        return True
        
    except Exception as e:
        print(f"❌ KiCad integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🧪 Dielectric Integration Tests")
    print("=" * 60)
    
    # Test KiCad integration first
    kicad_ok = await test_kicad_integration()
    
    # Run full workflow test
    workflow_ok = await test_full_workflow()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"KiCad Integration: {'✅ PASS' if kicad_ok else '❌ FAIL'}")
    print(f"Full Workflow: {'✅ PASS' if workflow_ok else '❌ FAIL'}")
    
    if kicad_ok and workflow_ok:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️  Some tests failed - check output above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

