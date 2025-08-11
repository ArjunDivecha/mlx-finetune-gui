import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { 
  BarChart3, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Download,
  Calendar
} from 'lucide-react';
import { RootState } from '../store/store';
import { ModelTestModal } from '../components/ModelTestModal';

export const ResultsPage: React.FC = () => {
  const { state: trainingState, metrics, config } = useSelector((state: RootState) => state.training);
  const [isModelTestOpen, setIsModelTestOpen] = useState(false);

  const formatDuration = (startTime: string) => {
    const start = new Date(startTime);
    const now = new Date();
    const diffMs = now.getTime() - start.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    return `${diffHours}h ${diffMinutes}m`;
  };

  const getStatusColor = (state: string) => {
    switch (state) {
      case 'completed':
        return 'text-success-600 bg-success-100 dark:bg-success-900/30';
      case 'running':
        return 'text-primary-600 bg-primary-100 dark:bg-primary-900/30';
      case 'error':
        return 'text-error-600 bg-error-100 dark:bg-error-900/30';
      case 'stopped':
        return 'text-warning-600 bg-warning-100 dark:bg-warning-900/30';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-800';
    }
  };

  const getStatusIcon = (state: string) => {
    switch (state) {
      case 'completed':
        return <CheckCircle className="h-5 w-5" />;
      case 'error':
      case 'stopped':
        return <AlertTriangle className="h-5 w-5" />;
      default:
        return <Clock className="h-5 w-5" />;
    }
  };

  if (trainingState === 'idle' || !metrics) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="h-8 w-8 text-gray-400" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            No Training Results
          </h2>
          <p className="text-gray-500 dark:text-gray-400">
            Complete a training session to view detailed results and analytics.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Training Results
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Detailed analysis and metrics from your fine-tuning session
        </p>
      </div>

      {/* Training Summary */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Training Summary</h2>
            <div className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(trainingState)}`}>
              {getStatusIcon(trainingState)}
              <span className="capitalize">{trainingState}</span>
            </div>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="text-2xl font-bold text-primary-600">
                {((metrics.current_step / metrics.total_steps) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Completion Rate
              </div>
            </div>

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="text-2xl font-bold text-success-600">
                {metrics.current_step.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Steps Completed
              </div>
            </div>

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="text-2xl font-bold text-error-600">
                {metrics.train_loss != null ? metrics.train_loss.toFixed(4) : '--'}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Final Train Loss
              </div>
            </div>

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="text-2xl font-bold text-warning-600">
                {metrics.val_loss ? metrics.val_loss.toFixed(4) : '--'}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Final Val Loss
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Configuration */}
      {config && (
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold">Configuration Used</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">Model Settings</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Model:</span>
                    <span className="font-medium">{config.model_path.split('/').pop()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Adapter Name:</span>
                    <span className="font-medium">{config.adapter_name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Max Sequence:</span>
                    <span className="font-medium">{config.max_seq_length}</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">Training Parameters</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Learning Rate:</span>
                    <span className="font-medium">{config.learning_rate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Batch Size:</span>
                    <span className="font-medium">{config.batch_size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Total Steps:</span>
                    <span className="font-medium">{config.iterations.toLocaleString()}</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">Optimization</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Early Stopping:</span>
                    <span className="font-medium">{config.early_stop ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Patience:</span>
                    <span className="font-medium">{config.patience} evaluations</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Save Every:</span>
                    <span className="font-medium">{config.save_every} steps</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Timing Information */}
      {metrics.start_time && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-2">
              <Calendar className="h-5 w-5 text-primary-600" />
              <h3 className="text-lg font-semibold">Timing Information</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Started</div>
                <div className="text-lg font-medium">
                  {new Date(metrics.start_time).toLocaleDateString()} at{' '}
                  {new Date(metrics.start_time).toLocaleTimeString()}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Duration</div>
                <div className="text-lg font-medium">
                  {formatDuration(metrics.start_time)}
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Avg. Time per Step</div>
                <div className="text-lg font-medium">
                  {metrics.current_step > 0 ? (
                    <>
                      {((new Date().getTime() - new Date(metrics.start_time).getTime()) / (metrics.current_step * 1000)).toFixed(2)}s
                    </>
                  ) : '--'}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end space-x-4">
        <button className="btn-secondary flex items-center space-x-2">
          <Download className="h-4 w-4" />
          <span>Export Report</span>
        </button>
        
        {trainingState === 'completed' && (
          <button 
            onClick={() => setIsModelTestOpen(true)}
            className="btn-primary"
          >
            Test Fine-tuned Model
          </button>
        )}
      </div>

      {/* Model Test Modal */}
      <ModelTestModal
        isOpen={isModelTestOpen}
        onClose={() => setIsModelTestOpen(false)}
        modelInfo={config ? {
          base_model: config.model_path.split('/').pop() || 'Unknown',
          adapter_name: config.adapter_name
        } : undefined}
      />
    </div>
  );
};