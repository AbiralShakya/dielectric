#!/usr/bin/env python3
"""
Quick test script for Dielectric system
Tests: Backend API, KiCAD Export, Frontend connectivity
"""

import requests
import json
import sys
import time

API_BASE = "http://localhost:8000"

def test_backend_health():
    """Test if backend is running"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend not running! Start it with: python deploy_simple.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_kicad_export():
    """Test KiCAD export with sample data"""
    print("\n🔍 Testing KiCAD export...")
    
    # Simple test placement
    test_placement = {
        "board": {
            "width": 100,
            "height": 100,
            "clearance": 0.5
        },
        "components": [
            {
                "name": "R1",
                "package": "0805",
                "x": 20,
                "y": 20,
                "angle": 0,
                "width": 2,
                "height": 1.25,
                "pins": [
                    {"name": "pin1", "x_offset": -1, "y_offset": 0, "net": "net1"},
                    {"name": "pin2", "x_offset": 1, "y_offset": 0, "net": "net2"}
                ]
            },
            {
                "name": "LED1",
                "package": "LED-5MM",
                "x": 50,
                "y": 50,
                "angle": 0,
                "width": 5,
                "height": 5,
                "pins": [
                    {"name": "anode", "x_offset": -2, "y_offset": 0, "net": "net2"},
                    {"name": "cathode", "x_offset": 2, "y_offset": 0, "net": "GND"}
                ]
            }
        ],
        "nets": [
            {
                "name": "net1",
                "pins": [["R1", "pin1"]]
            },
            {
                "name": "net2",
                "pins": [["R1", "pin2"], ["LED1", "anode"]]
            },
            {
                "name": "GND",
                "pins": [["LED1", "cathode"]]
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/export/kicad",
            json={"placement": test_placement},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ KiCAD export successful!")
            print(f"   File: {data['filename']}")
            print(f"   Size: {data['size_bytes']} bytes")
            print(f"   Format: {data['format']}")
            
            # Check if content looks valid
            content = data['content']
            if "(kicad_pcb" in content and "footprint" in content:
                print("✅ File content looks valid")
            else:
                print("⚠️  File content may be incomplete")
            
            return True
        else:
            print(f"❌ Export failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False

def test_optimize_endpoint():
    """Test optimization endpoint"""
    print("\n🔍 Testing optimization endpoint...")
    
    test_request = {
        "placement": {
            "board": {"width": 100, "height": 100, "clearance": 0.5},
            "components": [
                {
                    "name": "U1",
                    "package": "SOIC-8",
                    "x": 30,
                    "y": 30,
                    "angle": 0,
                    "width": 5,
                    "height": 6,
                    "power": 0.5
                }
            ],
            "nets": []
        },
        "user_intent": "minimize trace length"
    }
    
    try:
        print("   Sending request (this may take a few seconds)...")
        response = requests.post(
            f"{API_BASE}/optimize",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Optimization successful!")
            print(f"   Score: {data.get('score', 'N/A')}")
            print(f"   Agents used: {len(data.get('agents_used', []))}")
            return True
        else:
            print(f"❌ Optimization failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("🚀 Dielectric Quick Test")
    print("=" * 50)
    
    results = []
    
    # Test 1: Backend health
    results.append(("Backend Health", test_backend_health()))
    
    if not results[0][1]:
        print("\n⚠️  Backend not running. Start it first:")
        print("   python deploy_simple.py")
        sys.exit(1)
    
    # Test 2: KiCAD Export
    results.append(("KiCAD Export", test_kicad_export()))
    
    # Test 3: Optimization (optional - takes longer)
    if "--full" in sys.argv:
        results.append(("Optimization", test_optimize_endpoint()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\n💡 Next steps:")
        print("   1. Start frontend: streamlit run frontend/app_dielectric.py")
        print("   2. Open browser: http://localhost:8501")
        print("   3. Try generating and exporting a design")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

