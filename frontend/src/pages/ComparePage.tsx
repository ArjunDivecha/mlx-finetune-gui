import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { 
  Send, 
  RefreshCw, 
  Copy, 
  Trash2,
  ArrowRight,
  Star,
  FolderOpen,
} from 'lucide-react';
import { RootState } from '../store/store';
import { LoadSessionModal } from '../components/LoadSessionModal';
import axios from 'axios';

interface Comparison {
  id: string;
  prompt: string;
  baseResponse: string;
  fineTunedResponse: string;
  rating?: 1 | 2 | 3 | 4 | 5;
  timestamp: Date;
}

export const ComparePage: React.FC = () => {
  const { config } = useSelector((state: RootState) => state.training);
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [comparisons, setComparisons] = useState<Comparison[]>([]);
  const [selectedComparison, setSelectedComparison] = useState<string | null>(null);
  const [isLoadSessionOpen, setIsLoadSessionOpen] = useState(false);
  const [loadedSession, setLoadedSession] = useState<any>(null);

  // Real model inference function
  const generateResponses = async (inputPrompt: string) => {
    setIsGenerating(true);
    
    try {
      // Get the current configuration (could be from current training or loaded session)
      const currentConfig = loadedSession || config;
      
      if (!currentConfig) {
        throw new Error('No model configuration available. Please complete a training session first.');
      }
      
      // Generate responses from both models in parallel
      const [baseResponse, fineTunedResponse] = await Promise.all([
        axios.post('http://localhost:8000/model/test-base', {
          prompt: inputPrompt,
          max_tokens: 1024,
          temperature: 0.7
        }),
        axios.post('http://localhost:8000/model/test', {
          prompt: inputPrompt,
          max_tokens: 1024,
          temperature: 0.7
        })
      ]);

      const baseData = baseResponse.data;
      const fineTunedData = fineTunedResponse.data;
      
      const newComparison: Comparison = {
        id: Math.random().toString(36).substr(2, 9),
        prompt: inputPrompt,
        baseResponse: baseData.response,
        fineTunedResponse: fineTunedData.response,
        timestamp: new Date()
      };
      
      setComparisons(prev => [newComparison, ...prev]);
      setSelectedComparison(newComparison.id);
    } catch (error: any) {
      console.error('Failed to generate responses:', error);
      
      let baseErrorMessage = 'Error: Failed to generate base model response';
      let fineTunedErrorMessage = 'Error: Failed to generate fine-tuned response';
      
      if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        baseErrorMessage = `Error: ${detail}`;
        fineTunedErrorMessage = `Error: ${detail}`;
      } else if (error.message) {
        baseErrorMessage = `Error: ${error.message}`;
        fineTunedErrorMessage = `Error: ${error.message}`;
      }
      
      // Show error in the comparison
      const errorComparison: Comparison = {
        id: Math.random().toString(36).substr(2, 9),
        prompt: inputPrompt,
        baseResponse: baseErrorMessage,
        fineTunedResponse: fineTunedErrorMessage,
        timestamp: new Date()
      };
      
      setComparisons(prev => [errorComparison, ...prev]);
      setSelectedComparison(errorComparison.id);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isGenerating) {
      generateResponses(prompt.trim());
      setPrompt('');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const rateComparison = (comparisonId: string, rating: 1 | 2 | 3 | 4 | 5) => {
    setComparisons(prev => 
      prev.map(comp => 
        comp.id === comparisonId 
          ? { ...comp, rating }
          : comp
      )
    );
  };

  const deleteComparison = (comparisonId: string) => {
    setComparisons(prev => prev.filter(comp => comp.id !== comparisonId));
    if (selectedComparison === comparisonId) {
      setSelectedComparison(null);
    }
  };

  const handleSessionLoaded = (session: any) => {
    setLoadedSession(session);
    // You might want to update the Redux store here as well
    console.log('Loaded session:', session);
  };

  const selectedComp = comparisons.find(comp => comp.id === selectedComparison);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Model Comparison
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Compare responses between your base model and fine-tuned model
        </p>
      </div>

      {/* Model Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card">
          <div className="card-body">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Base Model</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {config?.model_path.split('/').pop() || 'No model selected'}
            </p>
            <div className="mt-2">
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200">
                Original Model
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">Fine-tuned Model</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {loadedSession ? 
                    `${loadedSession.model_name} + ${loadedSession.adapter_name}` : 
                    (config?.adapter_name ? `${config.model_path.split('/').pop()} + ${config.adapter_name}` : 'No trained model available')
                  }
                </p>
              </div>
              <button
                onClick={() => setIsLoadSessionOpen(true)}
                className="btn-secondary flex items-center space-x-2 text-xs"
                title="Load a different trained model"
              >
                <FolderOpen className="h-3 w-3" />
                <span>Load Trained</span>
              </button>
            </div>
            <div className="mt-2 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                  (loadedSession || config?.adapter_name) ? 
                    'bg-success-100 dark:bg-success-900/30 text-success-800 dark:text-success-200' :
                    'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200'
                }`}>
                  {loadedSession ? 'Loaded Session' : 
                   config?.adapter_name ? 'Model Available' : 'No Model'}
                </span>
                {(loadedSession || config?.adapter_name) && (
                  <span className="text-xs text-success-600 dark:text-success-400 font-medium">
                    ✓ Ready for inference
                  </span>
                )}
              </div>
              {loadedSession && (
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {new Date(loadedSession.timestamp).toLocaleDateString()}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Prompt Input */}
      <div className="card">
        <div className="card-body">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Enter your prompt to compare model responses
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a prompt to test both models..."
                rows={4}
                className="input-field resize-none"
                disabled={isGenerating}
              />
            </div>
            
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={!prompt.trim() || isGenerating}
                className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    <span>Generate Comparison</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Comparison Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Comparison History */}
        <div className="lg:col-span-1">
          <div className="card h-fit max-h-96 overflow-hidden">
            <div className="card-header">
              <h3 className="text-lg font-semibold">Comparison History</h3>
            </div>
            <div className="overflow-y-auto">
              {comparisons.length === 0 ? (
                <div className="p-6 text-center text-gray-500 dark:text-gray-400">
                  <p>No comparisons yet</p>
                  <p className="text-sm">Generate your first comparison above</p>
                </div>
              ) : (
                <div className="space-y-1">
                  {comparisons.map((comparison) => (
                    <div
                      key={comparison.id}
                      onClick={() => setSelectedComparison(comparison.id)}
                      className={`p-3 cursor-pointer border-l-4 transition-colors ${
                        selectedComparison === comparison.id
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                          : 'border-transparent hover:bg-gray-50 dark:hover:bg-gray-800'
                      }`}
                    >
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100 line-clamp-2">
                        {comparison.prompt}
                      </div>
                      <div className="flex items-center justify-between mt-2">
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {comparison.timestamp.toLocaleDateString()}
                        </div>
                        {comparison.rating && (
                          <div className="flex items-center space-x-1">
                            {Array.from({ length: 5 }, (_, i) => (
                              <Star
                                key={i}
                                className={`h-3 w-3 ${
                                  i < comparison.rating!
                                    ? 'text-yellow-400 fill-current'
                                    : 'text-gray-300'
                                }`}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Comparison View */}
        <div className="lg:col-span-2">
          {selectedComp ? (
            <div className="space-y-6">
              {/* Prompt */}
              <div className="card">
                <div className="card-header">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Prompt</h3>
                    <button
                      onClick={() => deleteComparison(selectedComp.id)}
                      className="text-error-600 hover:text-error-700 p-1"
                      title="Delete comparison"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <div className="card-body">
                  <p className="text-gray-900 dark:text-gray-100">{selectedComp.prompt}</p>
                </div>
              </div>

              {/* Responses */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                {/* Base Model Response */}
                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">Base Model</h4>
                      <button
                        onClick={() => copyToClipboard(selectedComp.baseResponse)}
                        className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 p-1"
                        title="Copy response"
                      >
                        <Copy className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <div className="card-body">
                    <div className="prose dark:prose-invert max-w-none">
                      <p className="whitespace-pre-wrap text-sm">
                        {selectedComp.baseResponse}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Fine-tuned Model Response */}
                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">Fine-tuned Model</h4>
                      <button
                        onClick={() => copyToClipboard(selectedComp.fineTunedResponse)}
                        className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 p-1"
                        title="Copy response"
                      >
                        <Copy className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <div className="card-body">
                    <div className="prose dark:prose-invert max-w-none">
                      <p className="whitespace-pre-wrap text-sm">
                        {selectedComp.fineTunedResponse}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Rating */}
              <div className="card">
                <div className="card-header">
                  <h4 className="font-semibold">Rate Fine-tuned Response</h4>
                </div>
                <div className="card-body">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Quality:</span>
                    <div className="flex items-center space-x-1">
                      {Array.from({ length: 5 }, (_, i) => (
                        <button
                          key={i}
                          onClick={() => rateComparison(selectedComp.id, (i + 1) as 1 | 2 | 3 | 4 | 5)}
                          className="p-1 hover:scale-110 transition-transform"
                        >
                          <Star
                            className={`h-6 w-6 ${
                              selectedComp.rating && i < selectedComp.rating
                                ? 'text-yellow-400 fill-current'
                                : 'text-gray-300 hover:text-yellow-300'
                            }`}
                          />
                        </button>
                      ))}
                    </div>
                    {selectedComp.rating && (
                      <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">
                        ({selectedComp.rating}/5)
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-96">
              <div className="text-center text-gray-500 dark:text-gray-400">
                <ArrowRight className="h-12 w-12 mx-auto mb-4" />
                <p>Select a comparison from the history to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Load Session Modal */}
      <LoadSessionModal
        isOpen={isLoadSessionOpen}
        onClose={() => setIsLoadSessionOpen(false)}
        onSessionLoaded={handleSessionLoaded}
      />
    </div>
  );
};