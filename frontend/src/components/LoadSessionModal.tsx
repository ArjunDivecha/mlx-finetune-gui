import React, { useState, useEffect } from 'react';
import { X, Calendar, CheckCircle, AlertTriangle, Clock, Loader2 } from 'lucide-react';
import axios from 'axios';

interface Session {
  session_id: string;
  timestamp: string;
  training_state: string;
  model_name: string;
  adapter_name: string;
  final_train_loss: number;
  final_val_loss: number;
  steps_completed: number;
  total_steps: number;
}

interface LoadSessionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSessionLoaded: (session: Session) => void;
}

const BACKEND_URL = 'http://localhost:8000';

export const LoadSessionModal: React.FC<LoadSessionModalProps> = ({
  isOpen,
  onClose,
  onSessionLoaded
}) => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchSessions();
    }
  }, [isOpen]);

  const fetchSessions = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${BACKEND_URL}/sessions`);
      setSessions(response.data.sessions);
    } catch (err: any) {
      console.error('Failed to fetch sessions:', err);
      setError(err.response?.data?.detail || 'Failed to load training sessions');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadSession = async (sessionId: string) => {
    setLoadingSessionId(sessionId);
    setError(null);

    try {
      const response = await axios.post(`${BACKEND_URL}/sessions/${sessionId}/load`);
      
      // Find the loaded session in our list
      const loadedSession = sessions.find(s => s.session_id === sessionId);
      if (loadedSession) {
        onSessionLoaded(loadedSession);
      }
      
      onClose();
    } catch (err: any) {
      console.error('Failed to load session:', err);
      setError(err.response?.data?.detail || 'Failed to load training session');
    } finally {
      setLoadingSessionId(null);
    }
  };

  const getStatusIcon = (state: string) => {
    switch (state) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-success-600" />;
      case 'error':
        return <AlertTriangle className="h-5 w-5 text-error-600" />;
      case 'stopped':
        return <AlertTriangle className="h-5 w-5 text-warning-600" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (state: string) => {
    switch (state) {
      case 'completed':
        return 'text-success-600 bg-success-100 dark:bg-success-900/30';
      case 'error':
        return 'text-error-600 bg-error-100 dark:bg-error-900/30';
      case 'stopped':
        return 'text-warning-600 bg-warning-100 dark:bg-warning-900/30';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-800';
    }
  };

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' at ' + date.toLocaleTimeString();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Load Trained Model
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Select a completed training session to load its model and configuration
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
          {/* Error */}
          {error && (
            <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-red-800 dark:text-red-200 text-sm">{error}</p>
            </div>
          )}

          {/* Loading */}
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-primary-600" />
              <span className="ml-2 text-gray-600 dark:text-gray-400">Loading training sessions...</span>
            </div>
          )}

          {/* Sessions List */}
          {!isLoading && sessions.length === 0 && (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              <Calendar className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No training sessions found</p>
              <p className="text-sm">Complete a training session to see it here</p>
            </div>
          )}

          {!isLoading && sessions.length > 0 && (
            <div className="space-y-4">
              {sessions.map((session) => (
                <div
                  key={session.session_id}
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      {/* Header */}
                      <div className="flex items-center space-x-3 mb-3">
                        {getStatusIcon(session.training_state)}
                        <div>
                          <h3 className="font-medium text-gray-900 dark:text-gray-100">
                            {session.model_name} + {session.adapter_name}
                          </h3>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {formatDate(session.timestamp)}
                          </p>
                        </div>
                        <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(session.training_state)}`}>
                          <span className="capitalize">{session.training_state}</span>
                        </div>
                      </div>

                      {/* Metrics */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">Progress:</span>
                          <p className="font-medium">
                            {session.steps_completed || 0}/{session.total_steps || 0} 
                            ({session.total_steps ? ((session.steps_completed / session.total_steps) * 100).toFixed(1) : '0.0'}%)
                          </p>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">Train Loss:</span>
                          <p className="font-medium">
                            {session.final_train_loss ? session.final_train_loss.toFixed(4) : '--'}
                          </p>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">Val Loss:</span>
                          <p className="font-medium">
                            {session.final_val_loss ? session.final_val_loss.toFixed(4) : '--'}
                          </p>
                        </div>
                        <div className="flex justify-end">
                          <button
                            onClick={() => handleLoadSession(session.session_id)}
                            disabled={loadingSessionId === session.session_id || session.training_state !== 'completed'}
                            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed text-sm px-3 py-1"
                          >
                            {loadingSessionId === session.session_id ? (
                              <>
                                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                                Loading...
                              </>
                            ) : session.training_state === 'completed' ? (
                              'Load Model'
                            ) : (
                              'Incomplete'
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};