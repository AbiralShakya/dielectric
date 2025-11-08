import { useState } from 'react';
import RemodelGenerator from './components/RemodelGenerator';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">BuildZoom AI</h1>
              <p className="text-gray-600 mt-1">AI-Powered Home Renovation Visualizer</p>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-500">
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">Gemini API</span>
              <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded">xAI Grok</span>
            </div>
          </div>
        </div>
      </header>

      <main>
        <RemodelGenerator />
      </main>

      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-500 text-sm">
            <p>Built for HackPrinceton 2025 • Powered by Gemini & xAI</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
