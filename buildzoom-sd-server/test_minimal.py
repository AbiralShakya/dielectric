#!/usr/bin/env python3
"""
Minimal test server for BuildZoom AI
Tests the API pipeline without requiring large SD models
"""

import base64
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
import random

app = FastAPI(title="BuildZoom AI - Minimal Test Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_mock_remodeled_image(original_image: Image.Image, prompt: str) -> Image.Image:
    """Create a mock remodeled image by adding text overlay"""
    # Create a copy of the original image
    result = original_image.copy()

    # Add some visual changes (simple mock)
    draw = ImageDraw.Draw(result)

    # Add renovation text overlay
    text = f"Mock Renovation:\n{prompt[:50]}..."

    # Try to use default font, fallback to basic
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Add semi-transparent rectangle for text background
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5],
                  fill=(0, 0, 0, 128))

    # Add text
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    # Add some mock renovation elements (simple colored rectangles)
    width, height = result.size

    # Mock kitchen island
    if "island" in prompt.lower():
        draw.rectangle([width//4, height//2, width//2, height*3//4],
                      fill=(139, 69, 19, 180))  # Brown
        draw.text((width//4 + 10, height//2 + 10), "Kitchen Island",
                 fill=(255, 255, 255), font=font)

    # Mock cabinets
    if "cabinet" in prompt.lower():
        draw.rectangle([width//8, height//4, width//4, height//2],
                      fill=(210, 180, 140, 180))  # Beige
        draw.text((width//8 + 5, height//4 + 5), "New Cabinets",
                 fill=(0, 0, 0), font=font)

    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "minimal_test",
        "message": "Mock SD server - no real models loaded"
    }

@app.post("/generate")
async def generate_single_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    strength: float = Form(0.75),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(25)
):
    """Generate a mock remodeled image"""
    try:
        # Read and process image
        img_bytes = await image.read()
        original_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Resize for consistency
        original_image.thumbnail((1024, 768), Image.Resampling.LANCZOS)

        # Create mock remodeled image
        result_image = create_mock_remodeled_image(original_image, prompt)

        # Convert to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse({
            "success": True,
            "image": img_str,
            "prompt": prompt,
            "parameters": {
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "mode": "mock_generation"
            },
            "note": "This is a mock image for testing. Real SD generation requires model download."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock generation failed: {str(e)}")

@app.post("/generate-multi-angle")
async def generate_multi_angle_images(
    base_prompt: str = Form(...),
    image: UploadFile = File(...),
    num_angles: int = Form(3),  # Reduced for testing
    strength: float = Form(0.75),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(25)
):
    """Generate mock images from multiple angles"""
    try:
        # Read and process image
        img_bytes = await image.read()
        original_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_image.thumbnail((1024, 768), Image.Resampling.LANCZOS)

        # Define angle descriptions
        angle_descriptions = [
            "front view, eye level perspective",
            "view from left side, 45 degree angle",
            "view from right side, 45 degree angle",
            "wide angle view from doorway",
            "close up detail view"
        ]

        # Limit to requested number
        angles_to_generate = angle_descriptions[:num_angles]

        results = []

        for i, angle_desc in enumerate(angles_to_generate):
            print(f"Generating mock angle {i+1}/{len(angles_to_generate)}: {angle_desc}")

            # Create angle-specific prompt
            full_prompt = f"{base_prompt}, {angle_desc}, architectural photography"

            # Create mock image with angle-specific text
            result_image = original_image.copy()
            draw = ImageDraw.Draw(result_image)

            # Add angle indicator
            text = f"Mock Angle {i+1}:\n{angle_desc}"

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            # Add background
            bbox = draw.textbbox((20, 20), text, font=font)
            draw.rectangle([bbox[0]-10, bbox[1]-10, bbox[2]+10, bbox[3]+10],
                          fill=(0, 0, 0, 160))
            draw.text((20, 20), text, fill=(255, 255, 255), font=font)

            # Convert to base64
            buffered = io.BytesIO()
            result_image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            results.append({
                "angle": i + 1,
                "description": angle_desc,
                "image": img_str,
                "prompt": full_prompt
            })

        return JSONResponse({
            "success": True,
            "images": results,
            "total_angles": len(results),
            "base_prompt": base_prompt,
            "note": "These are mock images for testing. Real SD requires model download."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-angle mock generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting BuildZoom AI Minimal Test Server")
    print("📝 This server generates MOCK images for testing")
    print("🔗 Test endpoints:")
    print("   GET  /health")
    print("   POST /generate (with image + prompt)")
    print("   POST /generate-multi-angle")
    print("")

    uvicorn.run(
        "test_minimal:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
