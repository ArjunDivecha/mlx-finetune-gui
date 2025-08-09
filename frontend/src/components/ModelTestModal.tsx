import React, { useState } from 'react';
import { X, Send, Bot, User } from 'lucide-react';
import axios from 'axios';

interface ModelTestModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelInfo?: {
    base_model: string;
    adapter_name: string;
  };
}

interface TestResult {
  prompt: string;
  response: string;
  model_info: {
    base_model: string;
    adapter: string;
    max_tokens: number;
    temperature: number;
  };
}

const BACKEND_URL = 'http://localhost:8000';

export const ModelTestModal: React.FC<ModelTestModalProps> = ({
  isOpen,
  onClose,
  modelInfo
}) => {
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTest = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setIsLoading(true);
    setError(null);
    setTestResult(null);

    try {
      const response = await axios.post(`${BACKEND_URL}/model/test`, {
        prompt: prompt.trim(),
        max_tokens: maxTokens,
        temperature
      });

      setTestResult(response.data);
    } catch (err: any) {
      console.error('Model test error:', err);
      setError(err.response?.data?.detail || 'Failed to test model');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setPrompt('');
    setTestResult(null);
    setError(null);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Test Fine-tuned Model
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {modelInfo ? `${modelInfo.base_model} + ${modelInfo.adapter_name}` : 'Fine-tuned Model'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Max Tokens
              </label>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                min="1"
                max="1000"
                className="input-field"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Temperature
              </label>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                min="0"
                max="2"
                step="0.1"
                className="input-field"
              />
            </div>
          </div>

          {/* Prompt Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Test Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your test prompt here..."
              rows={4}
              className="input-field resize-none"
            />
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-3 mb-6">
            <button
              onClick={handleTest}
              disabled={isLoading || !prompt.trim()}
              className="btn-primary flex items-center space-x-2 disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
              <span>{isLoading ? 'Generating...' : 'Test Model'}</span>
            </button>
            <button
              onClick={handleClear}
              className="btn-secondary"
            >
              Clear
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-red-800 dark:text-red-200 text-sm">{error}</p>
            </div>
          )}

          {/* Results */}
          {testResult && (
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">Test Result</h3>
              
              {/* Conversation View */}
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 space-y-4">
                {/* User Prompt */}
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                      <User className="h-4 w-4 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
                      <p className="text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
                        {testResult.prompt}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Model Response */}
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                      <Bot className="h-4 w-4 text-white" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
                      <p className="text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
                        {testResult.response}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Model Info */}
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Generation Settings
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Model:</span>
                    <p className="font-medium">{testResult.model_info.base_model}</p>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Adapter:</span>
                    <p className="font-medium">{testResult.model_info.adapter}</p>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Max Tokens:</span>
                    <p className="font-medium">{testResult.model_info.max_tokens}</p>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Temperature:</span>
                    <p className="font-medium">{testResult.model_info.temperature}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};