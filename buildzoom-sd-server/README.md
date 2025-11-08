# BuildZoom AI - Stable Diffusion Server

Local GPU-powered image generation server for high-quality renovation visualizations.

## Features

- ✅ Stable Diffusion XL with ControlNet
- ✅ Image-to-image generation with Canny edge detection
- ✅ Multi-angle generation for AR experiences
- ✅ FastAPI with automatic CORS
- ✅ Optimized for RTX GPUs (also works on M1/M2 Mac)
- ✅ RESTful API for easy integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

```bash
python setup_models.py
```

This will download:
- SDXL Base model (~6.9GB)
- ControlNet Canny model (~5GB)

### 3. Run Server

```bash
python sd_server.py
```

Server will start on `http://localhost:8000`

### 4. Test

```bash
curl http://localhost:8000/health
```

## API Endpoints

### GET /health
Check server status and GPU availability

### POST /generate
Generate a single remodeled image

**Request:**
```json
{
  "prompt": "modern kitchen with white cabinets and quartz countertops",
  "image": "base64_encoded_image",
  "strength": 0.75,
  "guidance_scale": 7.5,
  "num_inference_steps": 25
}
```

**Response:**
```json
{
  "success": true,
  "image": "base64_encoded_result",
  "prompt": "modern kitchen...",
  "parameters": {...}
}
```

### POST /generate-multi-angle
Generate images from multiple angles for AR

**Request:**
```json
{
  "base_prompt": "modern kitchen renovation",
  "image": "base64_encoded_image",
  "num_angles": 5
}
```

**Response:**
```json
{
  "success": true,
  "images": [
    {
      "angle": 1,
      "description": "front view...",
      "image": "base64_encoded_image",
      "prompt": "full_prompt_used"
    }
  ]
}
```

## Hardware Requirements

### Minimum
- 8GB RAM
- 4GB VRAM (GTX 1060 or equivalent)
- Intel Core i5 or equivalent

### Recommended
- 16GB+ RAM
- 8GB+ VRAM (RTX 3060 or better)
- Intel Core i7/AMD Ryzen 7 or better

### MacBook Pro (M1/M2/M3)
- Works great with MPS (Metal Performance Shaders)
- 8GB unified memory minimum
- 16GB+ recommended for best performance

## Integration with Main Backend

Update your Node.js backend to use this server:

```javascript
// In your remodel.ts
async function generateRemodeledImage(prompt: string, imageBase64: string) {
  const formData = new FormData();
  formData.append('prompt', prompt);
  formData.append('image', base64ToBlob(imageBase64), 'room.jpg');

  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  return data.image;
}
```

## Performance Optimization

### For RTX GPUs:
- Uses FP16 for faster inference
- xFormers for memory efficiency
- ControlNet for better image conditioning

### Generation Speed:
- RTX 3060: ~15-20 seconds per image
- RTX 4070: ~8-12 seconds per image
- RTX 4090: ~5-8 seconds per image

### Multi-Angle Generation:
- 5 angles: ~1-2 minutes total
- 7 angles: ~2-3 minutes total

## Troubleshooting

### CUDA Issues (Windows/Linux)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### MPS Issues (Mac)
```bash
# Make sure you have latest PyTorch
pip install torch torchvision
```

### Memory Issues
- Reduce `num_inference_steps` to 20
- Use smaller images (768x512 instead of 1024x768)
- Close other GPU-intensive applications

### Model Download Issues
```bash
# Manual download if script fails
wget -O models/checkpoints/sd_xl_base_1.0.safetensors \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
```

## Architecture

```
Frontend (React)
    ↓
Backend (Node.js)
    ↓
Local SD Server (Python FastAPI)
    ↓
GPU (CUDA/MPS/CPU)
```

This setup gives you:
- ✅ Better quality than API
- ✅ Full control over generation
- ✅ No API costs or rate limits
- ✅ Privacy (images stay local)
- ✅ Fast iteration during development
