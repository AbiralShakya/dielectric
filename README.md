# BuildZoom AI - HackPrinceton 2025

AI-powered home renovation visualizer that generates photorealistic before/after images with cost estimates and feasibility analysis.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

1. **Clone and setup backend:**
```bash
cd buildzoom-ai-backend
npm install
npm run build
```

2. **Setup frontend:**
```bash
cd ../buildzoom-ai
npm install
npm run dev
```

3. **Start backend:**
```bash
cd ../buildzoom-ai-backend
npm run dev
```

### API Keys Setup

Create `.env` file in `buildzoom-ai-backend/` directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
XAI_API_KEY=your_xai_api_key_here
PORT=3001
NODE_ENV=development
```

**Get API Keys:**
- **Gemini API**: https://makersuite.google.com/app/apikey
- **xAI API**: https://docs.x.ai/docs#api-keys

## 🎯 Features

### MVP Core Features
- ✅ Photo upload with drag & drop
- ✅ Image compression and optimization
- ✅ Natural language renovation requests
- ✅ Before/after image comparison
- ✅ Cost estimates with breakdowns
- ✅ Feasibility scoring
- ✅ Materials list generation
- ✅ Structural concern warnings

### AI Pipeline
1. **Gemini Vision** - Analyzes room photos for dimensions, materials, condition
2. **xAI Grok** - Calculates costs, feasibility, and structural concerns
3. **Imagen 3** - Generates photorealistic remodeled images

## 🏗️ Architecture

```
Frontend (React + Tailwind)
    ↓
Backend (Node.js + Express)
    ↓
External APIs (Gemini + xAI)
```

## 🛠️ Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, Vite
- **Backend**: Node.js, Express, TypeScript
- **APIs**: Google Gemini 2.0, xAI Grok
- **Deployment**: Ready for Vercel/Netlify + serverless functions

## 📊 Demo Data

The app includes demo data for testing without API keys. To use real APIs:

1. Add your API keys to `.env`
2. Uncomment the real API calls in `src/routes/remodel.ts`
3. Test with actual room photos

## 🎨 User Flow

1. **Upload**: Drag & drop room photo (max 10MB)
2. **Describe**: Write renovation request in natural language
3. **Generate**: AI processes for ~15 seconds
4. **Results**: View before/after, costs, materials, warnings

## 🔧 Development

### Running Tests
```bash
# Backend tests
cd buildzoom-ai-backend
npm test

# Frontend tests
cd ../buildzoom-ai
npm test
```

### Building for Production
```bash
# Backend
cd buildzoom-ai-backend
npm run build
npm start

# Frontend
cd ../buildzoom-ai
npm run build
```

## 📝 API Endpoints

### POST `/api/generate-remodel`
Generates renovation plan from image and text description.

**Request:**
```json
{
  "imageBase64": "base64_encoded_image",
  "renovationRequest": "Add kitchen island, white cabinets..."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "beforeImage": "data:image/jpeg;base64,...",
    "afterImage": "generated_image_url",
    "costEstimate": {...},
    "feasibilityScore": 85,
    "materials": [...],
    "warnings": [...]
  }
}
```

## 🎯 HackPrinceton Prizes

This project is optimized for:
- **Best Use of xAI API** 🏆
- **Best Use of Gemini API** 🏆
- **Most Creative AI Application** 🏆

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for HackPrinceton 2025**
