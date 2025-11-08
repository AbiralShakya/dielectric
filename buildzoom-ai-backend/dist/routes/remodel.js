"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.remodelRouter = void 0;
const express_1 = require("express");
const generative_ai_1 = require("@google/generative-ai");
const router = (0, express_1.Router)();
exports.remodelRouter = router;
// Initialize Gemini AI (only if we have a valid key)
const hasValidGeminiKey = false; // Force demo mode for development
console.log('GEMINI_API_KEY:', `"${process.env.GEMINI_API_KEY}"`);
console.log('hasValidGeminiKey:', hasValidGeminiKey);
console.log('Check 1:', !!process.env.GEMINI_API_KEY);
console.log('Check 2:', process.env.GEMINI_API_KEY !== 'your_gemini_api_key_here');
console.log('Check 3:', process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.length > 10);
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
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' });
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
    const hasValidXAIKey = false; // Force demo mode for development
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
            imageGenerationPrompt: `Modern kitchen interior with updated features based on user request: ${userRequest}. Professional architectural photography, high quality, realistic lighting.`
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
        imageGenerationPrompt: `Modern kitchen interior, white shaker style cabinets with brushed nickel hardware, quartz countertops in light gray, large kitchen island, subway tile backsplash, oak laminate flooring, professional architectural photography, 8K quality`
    };
}
// Generate remodeled image with Imagen 3
async function generateRemodeledImage(prompt) {
    if (!hasValidGeminiKey) {
        // Return demo image based on prompt content
        const hasIsland = prompt.toLowerCase().includes('island');
        const hasWhiteCabinets = prompt.toLowerCase().includes('white') && prompt.toLowerCase().includes('cabinet');
        const hasQuartz = prompt.toLowerCase().includes('quartz');
        let imageType = 'modern-kitchen';
        if (hasIsland && hasWhiteCabinets)
            imageType = 'modern-kitchen-island';
        if (hasQuartz)
            imageType += '-quartz';
        return {
            url: `https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&h=600&fit=crop&crop=center&q=80`,
            alt: 'AI-generated remodeled room - Demo Mode'
        };
    }
    // Real Imagen 3 API call would go here
    return {
        url: 'https://via.placeholder.com/800x600?text=Generated+Remodeled+Image',
        alt: 'AI-generated remodeled room'
    };
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
        const remodeledImage = await generateRemodeledImage(feasibilityAnalysis.imageGenerationPrompt);
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
