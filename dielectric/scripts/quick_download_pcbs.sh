#!/bin/bash
# Quick script to download a few example PCBs for testing

echo "🚀 Quick PCB Download Script"
echo "============================"
echo ""

# Create examples directory
mkdir -p examples/real_pcbs

# Download KiCad examples (small, fast)
echo "📥 Downloading KiCad Examples..."
curl -L "https://github.com/KiCad/kicad-examples/archive/refs/heads/master.zip" -o examples/real_pcbs/kicad-examples.zip

if [ -f "examples/real_pcbs/kicad-examples.zip" ]; then
    echo "✅ Downloaded KiCad examples"
    echo "📦 Extracting..."
    cd examples/real_pcbs
    unzip -q kicad-examples.zip
    echo "✅ Extracted"
    
    # Find PCB files
    PCB_COUNT=$(find . -name "*.kicad_pcb" | wc -l)
    echo ""
    echo "📊 Found $PCB_COUNT KiCad PCB files"
    echo ""
    echo "💡 To upload these:"
    echo "   1. Go to frontend: http://localhost:8501"
    echo "   2. Select 'Optimize Design' → 'Folder/Zip'"
    echo "   3. Upload the kicad-examples-master folder or create a zip"
    echo ""
else
    echo "❌ Download failed"
    exit 1
fi

