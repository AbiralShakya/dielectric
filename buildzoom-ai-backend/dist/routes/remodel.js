"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.remodelRouter = void 0;
const express_1 = require("express");
const generative_ai_1 = require("@google/generative-ai");
const form_data_1 = __importDefault(require("form-data"));
const axios_1 = __importDefault(require("axios"));
const router = (0, express_1.Router)();
exports.remodelRouter = router;
// Initialize Gemini AI (only if we have a valid key)
const hasValidGeminiKey = !!(process.env.GEMINI_API_KEY &&
    process.env.GEMINI_API_KEY.length > 10 &&
    process.env.GEMINI_API_KEY !== 'your_gemini_api_key_here');
console.log('🔑 GEMINI_API_KEY configured:', hasValidGeminiKey);
console.log('🔑 XAI_API_KEY configured:', !!(process.env.XAI_API_KEY && process.env.XAI_API_KEY.length > 10));
const genAI = hasValidGeminiKey ? new generative_ai_1.GoogleGenerativeAI(process.env.GEMINI_API_KEY) : null;
// Room analysis with Gemini
async function analyzeRoom(imageBase64) {
    if (!hasValidGeminiKey) {
        // Return demo data
        return {
            roomType: "kitchen",
            dimensions: {
                estimated: "12ft x 14ft",
                squareFootage: 168
            },
            currentFeatures: {
                flooring: "laminate wood, medium oak color",
                walls: "painted drywall, cream color",
                cabinets: "raised panel, dark stain, approximately 15 years old",
                countertops: "laminate, beige pattern",
                appliances: ["refrigerator (stainless)", "electric stove", "over-range microwave"]
            },
            structuralElements: {
                windows: ["One 4ft window above sink"],
                doors: ["Entry from dining room", "pantry door"],
                loadBearingWalls: ["Cannot definitively identify without inspection"]
            },
            condition: "Dated but functional, cabinets showing wear",
            lighting: {
                natural: "Single window provides moderate natural light",
                artificial: "Recessed ceiling lights, under-cabinet lighting absent"
            },
            opportunities: [
                "Space available for kitchen island (6-8 ft clearance)",
                "Cabinet refacing or replacement would modernize",
                "Backsplash area available for upgrade"
            ]
        };
    }
    // Only execute this if we have a valid API key
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `Analyze this room image as a professional interior designer and contractor would. Provide:

1. Room type (kitchen, bathroom, bedroom, etc.)
2. Approximate dimensions (estimate based on visual cues)
3. Current features and materials:
   - Flooring type
   - Wall material/color
   - Existing fixtures/appliances
   - Cabinet/counter materials
4. Structural elements visible:
   - Windows and their placement
   - Doors and openings
   - Visible columns or beams
   - Load-bearing walls (if identifiable)
5. Current condition and age (modern, dated, needs repair)
6. Lighting situation (natural light sources, existing fixtures)
7. Spatial constraints or opportunities

Be specific and detailed. Format as structured JSON.`;
    const result = await model.generateContent([
        prompt,
        {
            inlineData: {
                mimeType: 'image/jpeg',
                data: imageBase64
            }
        }
    ]);
    const response = await result.response;
    const text = response.text();
    // Parse JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
    }
    return { rawAnalysis: text };
}
// Cost and feasibility analysis with xAI Grok
async function analyzeFeasibility(roomAnalysis, userRequest) {
    // Check if API key is configured (not placeholder)
    const hasValidXAIKey = !!(process.env.XAI_API_KEY &&
        process.env.XAI_API_KEY.length > 10);
    if (!hasValidXAIKey) {
        // Return demo data based on user request
        const isComplex = userRequest.toLowerCase().includes('remove wall') ||
            userRequest.toLowerCase().includes('structural') ||
            userRequest.toLowerCase().includes('load-bearing');
        return {
            feasibilityScore: isComplex ? 72 : 88,
            costEstimate: {
                low: isComplex ? 22000 : 15000,
                high: isComplex ? 32000 : 22000,
                currency: 'USD',
                breakdown: {
                    materials: isComplex ? 12000 : 8000,
                    labor: isComplex ? 14000 : 10000,
                    permits: isComplex ? 2500 : 1500,
                    contingency: isComplex ? 4000 : 3000
                }
            },
            timeline: {
                estimated: isComplex ? '4-6 weeks' : '2-3 weeks',
                phases: [
                    'Demolition (2-3 days)',
                    'Plumbing/electrical rough-in (3-5 days)',
                    'Cabinet installation (2-3 days)',
                    'Countertop templating and installation (5-7 days)',
                    'Finishing work (2-3 days)'
                ]
            },
            structuralConcerns: isComplex ? [
                'Wall between kitchen and dining room may be load-bearing - structural engineer consultation required ($500-1000)',
                'Additional electrical capacity may be needed for new outlets'
            ] : [],
            warnings: isComplex ? [
                '⚠️ Load-bearing wall removal requires structural engineer approval',
                '⚠️ Permit process may add 2-4 weeks to timeline',
                '⚠️ Additional electrical work may require panel upgrade'
            ] : [
                '⚠️ Verify local building codes for your area',
                '⚠️ Professional measurement recommended before purchasing materials'
            ],
            materialsList: [
                { item: 'White shaker cabinets (30 linear ft)', quantity: '1 set', estimatedCost: 4500 },
                { item: 'Quartz countertops (40 sq ft)', quantity: '40 sq ft', estimatedCost: 2800 },
                { item: 'Kitchen island base cabinet', quantity: '1', estimatedCost: 1200 },
                { item: 'Undermount sink', quantity: '1', estimatedCost: 350 },
                { item: 'Kitchen faucet (mid-range)', quantity: '1', estimatedCost: 250 }
            ],
            imageGenerationPrompt: `Preserve the exact same room layout, walls, perspective, and camera angle. Keep all existing structural elements unchanged. Only modify and update features based on: ${userRequest}. Maintain the same room dimensions, window positions, and overall structure. Professional architectural photography, high quality, realistic lighting, same perspective as original.`
        };
    }
    // Real xAI API call would go here
    return {
        feasibilityScore: 85,
        costEstimate: {
            low: 18000,
            high: 24000,
            currency: 'USD',
            breakdown: {
                materials: 8500,
                labor: 11000,
                permits: 1500,
                contingency: 3000
            }
        },
        timeline: {
            estimated: '3-4 weeks',
            phases: [
                'Demolition (2-3 days)',
                'Plumbing/electrical rough-in (3-5 days)',
                'Cabinet installation (2-3 days)',
                'Countertop templating and installation (5-7 days)',
                'Finishing work (2-3 days)'
            ]
        },
        structuralConcerns: [
            'Wall between kitchen and dining room may be load-bearing - structural engineer consultation required ($500-1000)'
        ],
        warnings: [
            '⚠️ Load-bearing wall removal requires structural engineer approval',
            '⚠️ Permit process may add 2-4 weeks to timeline'
        ],
        materialsList: [
            { item: 'White shaker cabinets (30 linear ft)', quantity: '1 set', estimatedCost: 4500 },
            { item: 'Quartz countertops (40 sq ft)', quantity: '40 sq ft', estimatedCost: 2800 },
            { item: 'Kitchen island base cabinet', quantity: '1', estimatedCost: 1200 }
        ],
        imageGenerationPrompt: `Preserve the exact same room layout, walls, perspective, and camera angle. Keep all existing structural elements unchanged. Only modify and update features based on the renovation request. Maintain the same room dimensions, window positions, and overall structure. Professional architectural photography, high quality, realistic lighting, same perspective as original.`
    };
}
// Generate remodeled image with local Stable Diffusion server
async function generateRemodeledImage(prompt, imageBase64) {
    console.log('🎨 generateRemodeledImage called');
    console.log('📝 Prompt:', prompt.substring(0, 100) + '...');
    console.log('🖼️ Image base64 length:', imageBase64.length);
    try {
        // Convert base64 to buffer - handle both with and without data URL prefix
        let base64Data = imageBase64;
        if (imageBase64.includes(',')) {
            base64Data = imageBase64.split(',')[1];
        }
        else if (imageBase64.startsWith('data:')) {
            base64Data = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');
        }
        if (!base64Data || base64Data.length < 100) {
            throw new Error('Invalid base64 image data');
        }
        const buffer = Buffer.from(base64Data, 'base64');
        console.log('📦 Buffer size:', buffer.length, 'bytes');
        // Create form data using form-data library (SDXL Turbo expects image first, then prompt as form)
        const form = new form_data_1.default();
        form.append('image', buffer, {
            filename: 'room.jpg',
            contentType: 'image/jpeg'
        });
        form.append('prompt', prompt);
        console.log('🌐 Calling SD server at http://localhost:8000/generate...');
        // Use axios instead of fetch - it handles form-data much better
        const response = await axios_1.default.post('http://localhost:8000/generate', form, {
            headers: {
                ...form.getHeaders(),
            },
            timeout: 120000, // 2 minute timeout
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
        });
        console.log('📡 SD server response status:', response.status);
        const data = response.data;
        console.log('✅ SD server response received, success:', data.success);
        if (data.success && data.image) {
            console.log('🖼️ Image generated, base64 length:', data.image.length);
            // Return data URL format for frontend (SDXL Turbo returns PNG)
            return {
                url: `data:image/png;base64,${data.image}`,
                alt: 'AI-generated remodeled room'
            };
        }
        else {
            throw new Error(`SD generation failed: ${JSON.stringify(data)}`);
        }
    }
    catch (error) {
        console.error('❌ CRITICAL ERROR generating with local SD:', error);
        console.error('Error name:', error.name);
        console.error('Error message:', error.message);
        console.error('Error code:', error.code);
        if (error.response) {
            console.error('Error response status:', error.response.status);
            console.error('Error response data:', error.response.data);
        }
        // Check if it's a connection error
        if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND' || error.message?.includes('connect')) {
            throw new Error(`Cannot connect to Stable Diffusion server at http://localhost:8000. Make sure the SD server is running. Error: ${error.message}`);
        }
        // Check if it's an axios error with response
        if (error.response) {
            const errorDetails = typeof error.response.data === 'string'
                ? error.response.data
                : JSON.stringify(error.response.data);
            throw new Error(`SD server error: ${error.response.status} ${error.response.statusText}. Details: ${errorDetails}`);
        }
        // Re-throw the error instead of silently falling back
        throw new Error(`Image generation failed: ${error.message}`);
    }
}
// Generate multi-angle images for AR experience
async function generateMultiAngleImages(basePrompt, imageBase64, numAngles = 5) {
    try {
        // Convert base64 to buffer
        const base64Data = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');
        const buffer = Buffer.from(base64Data, 'base64');
        // Create form data using form-data library (SDXL Turbo expects image first, then form fields)
        const form = new form_data_1.default();
        form.append('image', buffer, {
            filename: 'room.jpg',
            contentType: 'image/jpeg'
        });
        form.append('base_prompt', basePrompt);
        form.append('num_angles', numAngles.toString());
        const response = await axios_1.default.post('http://localhost:8000/generate-multi-angle', form, {
            headers: {
                ...form.getHeaders(),
            },
            timeout: 300000, // 5 minute timeout for multiple images
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
        });
        const data = response.data;
        if (data.success && data.images) {
            // Convert base64 images to data URLs (SDXL Turbo returns PNG)
            const images = data.images.map((img) => ({
                ...img,
                url: `data:image/png;base64,${img.image}`,
                image: undefined // Remove base64 string, keep URL
            }));
            return images;
        }
        else {
            throw new Error('Multi-angle generation failed');
        }
    }
    catch (error) {
        console.error('Error generating multi-angle images:', error);
        return []; // Return empty array as fallback
    }
}
// Main remodel endpoint
router.post('/generate-remodel', async (req, res) => {
    try {
        const { imageBase64, renovationRequest } = req.body;
        if (!imageBase64 || !renovationRequest) {
            return res.status(400).json({ error: 'Missing required fields: imageBase64 and renovationRequest' });
        }
        console.log('🎨 Starting AI pipeline...');
        // Step 1: Analyze room with Gemini
        console.log('🔍 Analyzing room...');
        const roomAnalysis = await analyzeRoom(imageBase64);
        // Step 2: Get cost/feasibility from Grok
        console.log('💰 Calculating costs...');
        const feasibilityAnalysis = await analyzeFeasibility(roomAnalysis, renovationRequest);
        // Step 3: Generate remodeled image
        console.log('🖼️ Generating image...');
        // Enhance prompt with user request and preservation instructions
        const enhancedPrompt = `Preserve the exact same room layout, walls, perspective, camera angle, and structural elements. Keep all existing room dimensions, window positions, door locations, and overall architecture unchanged. Only apply these specific renovations: ${renovationRequest}. Maintain the same lighting conditions and camera perspective as the original photo. Professional architectural photography, high quality, realistic.`;
        let remodeledImage;
        try {
            remodeledImage = await generateRemodeledImage(enhancedPrompt, imageBase64);
        }
        catch (error) {
            console.error('❌ Image generation failed:', error);
            return res.status(500).json({
                success: false,
                error: 'Image generation failed',
                details: error.message || 'Unknown error',
                hint: 'Make sure the Stable Diffusion server is running at http://localhost:8000'
            });
        }
        // Step 4: Compile results
        const results = {
            success: true,
            data: {
                roomAnalysis,
                beforeImage: `data:image/jpeg;base64,${imageBase64}`,
                afterImage: remodeledImage.url,
                costEstimate: feasibilityAnalysis.costEstimate,
                feasibilityScore: feasibilityAnalysis.feasibilityScore,
                timeline: feasibilityAnalysis.timeline,
                warnings: feasibilityAnalysis.warnings,
                materials: feasibilityAnalysis.materialsList,
                structuralConcerns: feasibilityAnalysis.structuralConcerns
            }
        };
        console.log('✅ Pipeline complete!');
        res.json(results);
    }
    catch (error) {
        console.error('❌ Pipeline error:', error);
        res.status(500).json({
            error: 'Processing failed',
            details: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
// Multi-angle generation endpoint for AR features
router.post('/generate-multi-angle', async (req, res) => {
    try {
        const { imageBase64, renovationRequest, numAngles = 5 } = req.body;
        if (!imageBase64 || !renovationRequest) {
            return res.status(400).json({ error: 'Missing required fields: imageBase64 and renovationRequest' });
        }
        console.log('🎨 Starting multi-angle generation...');
        // Generate multi-angle images
        console.log('🖼️ Generating multi-angle images...');
        const multiAngleImages = await generateMultiAngleImages(renovationRequest, imageBase64, numAngles);
        const results = {
            success: true,
            data: {
                images: multiAngleImages,
                totalAngles: multiAngleImages.length,
                basePrompt: renovationRequest
            }
        };
        console.log(`✅ Generated ${multiAngleImages.length} angles!`);
        res.json(results);
    }
    catch (error) {
        console.error('❌ Multi-angle generation error:', error);
        res.status(500).json({
            error: 'Multi-angle generation failed',
            details: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
