import { useState } from 'react';
import ImageUploader from './ImageUploader';

interface RoomAnalysis {
  roomType?: string;
  dimensions?: any;
  currentFeatures?: any;
  structuralElements?: any;
  condition?: string;
  lighting?: any;
  opportunities?: string[];
}

interface CostEstimate {
  low: number;
  high: number;
  currency: string;
  breakdown: {
    materials: number;
    labor: number;
    permits: number;
    contingency: number;
  };
}

interface MaterialItem {
  item: string;
  quantity: string;
  estimatedCost: number;
}

interface ResultsData {
  roomAnalysis: RoomAnalysis;
  beforeImage: string;
  afterImage: string;
  costEstimate: CostEstimate;
  feasibilityScore: number;
  timeline: {
    estimated: string;
    phases: string[];
  };
  warnings: string[];
  materials: MaterialItem[];
  structuralConcerns: string[];
}

function RemodelGenerator() {
  const [image, setImage] = useState<string | null>(null);
  const [request, setRequest] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ResultsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!image || !request) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:3002/api/generate-remodel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64: image,
          renovationRequest: request
        })
      });

      const data = await response.json();

      if (data.success) {
        setResults(data.data);
      } else {
        setError(data.error || 'Generation failed');
      }
    } catch (err) {
      setError('Network error. Make sure the backend is running.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">
          Visualize Your Dream Renovation
        </h2>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Upload a photo of your room, describe your renovation ideas, and get an AI-generated
          before/after visualization with realistic cost estimates and feasibility analysis.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
        {/* Input Section */}
        <div className="space-y-8">
          {/* Image Upload */}
          <div>
            <h3 className="text-xl font-semibold mb-4">Step 1: Upload Room Photo</h3>
            <ImageUploader onImageUpload={setImage} />
          </div>

          {/* Renovation Request */}
          <div>
            <h3 className="text-xl font-semibold mb-4">Step 2: Describe Your Renovation</h3>
            <textarea
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={6}
              placeholder="Example: Add a kitchen island with white quartz countertops, install white shaker cabinets, remove the wall to the dining room, add recessed lighting..."
              value={request}
              onChange={(e) => setRequest(e.target.value)}
            />
          </div>

          {/* Generate Button */}
          <div>
            <button
              onClick={handleGenerate}
              disabled={!image || !request || loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors duration-200"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Generating Your Renovation...
                </div>
              ) : (
                'Generate Renovation Plan'
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <div className="mt-2 text-sm text-red-700">{error}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Preview Section */}
        <div>
          <h3 className="text-xl font-semibold mb-4">Preview</h3>
          <div className="bg-gray-100 border-2 border-dashed border-gray-300 rounded-lg h-96 flex items-center justify-center">
            {image ? (
              <img
                src={`data:image/jpeg;base64,${image}`}
                alt="Room preview"
                className="max-w-full max-h-full object-contain rounded-lg"
              />
            ) : (
              <div className="text-center text-gray-500">
                <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                  <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <p className="mt-2">Upload an image to see preview</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="space-y-8">
          {/* Before/After Comparison */}
          <div>
            <h3 className="text-2xl font-bold mb-6 text-center">Your Renovation Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-lg font-semibold mb-3 text-center">Before</h4>
                <img
                  src={results.beforeImage}
                  alt="Before renovation"
                  className="w-full h-64 object-cover rounded-lg shadow-lg"
                />
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-3 text-center">After</h4>
                <img
                  src={results.afterImage}
                  alt="After renovation"
                  className="w-full h-64 object-cover rounded-lg shadow-lg"
                />
              </div>
            </div>
          </div>

          {/* Cost & Feasibility */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h4 className="text-xl font-bold mb-4">Project Analysis</h4>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  ${results.costEstimate.low.toLocaleString()} - ${results.costEstimate.high.toLocaleString()}
                </div>
                <div className="text-gray-600">Estimated Cost</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {results.feasibilityScore}/100
                </div>
                <div className="text-gray-600">Feasibility Score</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">
                  {results.timeline.estimated}
                </div>
                <div className="text-gray-600">Timeline</div>
              </div>
            </div>

            {/* Cost Breakdown */}
            <div className="mb-6">
              <h5 className="font-semibold mb-3">Cost Breakdown</h5>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(results.costEstimate.breakdown).map(([key, value]) => (
                  <div key={key} className="text-center">
                    <div className="font-semibold capitalize">{key}</div>
                    <div className="text-gray-600">${value.toLocaleString()}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Warnings */}
            {results.warnings.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
                <h5 className="font-semibold text-yellow-800 mb-2">⚠️ Important Considerations</h5>
                <ul className="list-disc pl-5 space-y-1">
                  {results.warnings.map((warning, i) => (
                    <li key={i} className="text-yellow-700 text-sm">{warning}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Structural Concerns */}
            {results.structuralConcerns.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <h5 className="font-semibold text-red-800 mb-2">🔧 Structural Concerns</h5>
                <ul className="list-disc pl-5 space-y-1">
                  {results.structuralConcerns.map((concern, i) => (
                    <li key={i} className="text-red-700 text-sm">{concern}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Materials List */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h4 className="text-xl font-bold mb-4">Materials & Supplies</h4>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 font-semibold">Item</th>
                    <th className="text-left py-2 font-semibold">Quantity</th>
                    <th className="text-right py-2 font-semibold">Est. Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {results.materials.map((item, i) => (
                    <tr key={i} className="border-b border-gray-100">
                      <td className="py-3">{item.item}</td>
                      <td className="py-3">{item.quantity}</td>
                      <td className="py-3 text-right">${item.estimatedCost.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default RemodelGenerator;
