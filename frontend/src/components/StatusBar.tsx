import React from 'react';
import { useSelector } from 'react-redux';
import { Clock, Cpu, HardDrive, Activity } from 'lucide-react';
import { RootState } from '../store/store';

export const StatusBar: React.FC = () => {
  const { state: trainingState, metrics } = useSelector((state: RootState) => state.training);
  
  const formatTime = (seconds: number | null) => {
    if (!seconds) return '--';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hrs > 0) {
      return `${hrs}h ${mins}m ${secs}s`;
    } else if (mins > 0) {
      return `${mins}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const getProgress = () => {
    if (!metrics) return 0;
    return (metrics.current_step / metrics.total_steps) * 100;
  };

  return (
    <div className="h-8 bg-gray-100 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-4 flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
      {/* Left side - training status */}
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <div className={`h-2 w-2 rounded-full ${
            trainingState === 'running' 
              ? 'bg-success-500 animate-pulse' 
              : trainingState === 'completed'
              ? 'bg-success-500'
              : trainingState === 'error'
              ? 'bg-error-500'
              : 'bg-gray-400'
          }`} />
          <span className="capitalize">{trainingState}</span>
        </div>

        {metrics && trainingState === 'running' && (
          <>
            <div className="flex items-center space-x-1">
              <Activity className="h-3 w-3" />
              <span>
                Step {metrics.current_step.toLocaleString()} of {metrics.total_steps.toLocaleString()}
              </span>
            </div>

            <div className="flex items-center space-x-1">
              <Clock className="h-3 w-3" />
              <span>
                ETA: {formatTime(metrics.estimated_time_remaining)}
              </span>
            </div>
          </>
        )}
      </div>

      {/* Center - progress bar (when training) */}
      {metrics && trainingState === 'running' && (
        <div className="flex-1 max-w-xs mx-8">
          <div className="flex items-center space-x-2">
            <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary-500 transition-all duration-300 ease-out"
                style={{ width: `${getProgress()}%` }}
              />
            </div>
            <span className="text-xs font-medium min-w-10">
              {getProgress().toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Right side - system info */}
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-1">
          <Cpu className="h-3 w-3" />
          <span>Apple Silicon</span>
        </div>

        <div className="flex items-center space-x-1">
          <HardDrive className="h-3 w-3" />
          <span>MLX Ready</span>
        </div>

        {metrics && metrics.train_loss != null && (
          <div className="flex items-center space-x-1">
            <span>Loss: {metrics.train_loss.toFixed(4)}</span>
          </div>
        )}
      </div>
    </div>
  );
};