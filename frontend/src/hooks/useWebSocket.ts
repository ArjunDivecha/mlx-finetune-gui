import { useCallback, useEffect, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store/store';
import {
  setConnectionStatus,
  setTrainingConfig,
  trainingStarted,
  trainingProgress,
  trainingCompleted,
  trainingStopped,
  trainingError,
  addLogLine,
  clearLogs,
} from '../store/slices/trainingSlice';
import { addNotification } from '../store/slices/uiSlice';

const WEBSOCKET_URL = 'ws://localhost:8000/ws';

export const useWebSocket = () => {
  const dispatch = useDispatch();
  const { isConnected } = useSelector((state: RootState) => state.training);
  const socketRef = useRef<WebSocket | null>(null);
  const lastLoggedStep = useRef<number>(-1);
  const lastRunStartTime = useRef<string | null>(null);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    
    const socket = new WebSocket(WEBSOCKET_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      dispatch(setConnectionStatus(true));
      dispatch(addNotification({
        type: 'success',
        title: 'Connected',
        message: 'WebSocket connection established'
      }));
    };

    socket.onclose = () => {
      dispatch(setConnectionStatus(false));
      dispatch(addNotification({
        type: 'warning',
        title: 'Disconnected',
        message: 'WebSocket connection lost'
      }));
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'training_started':
            dispatch(clearLogs()); // Clear previous logs when new training starts
            lastLoggedStep.current = -1; // Reset step tracking
            // Run boundary will be confirmed via polling using metrics.start_time
            dispatch(trainingStarted());
            break;
          case 'training_progress':
            // Ignore WebSocket training progress - using polling instead
            break;
          case 'training_completed':
            dispatch(trainingCompleted(data.data));
            break;
          case 'training_stopped':
            dispatch(trainingStopped());
            break;
          case 'training_error':
            dispatch(trainingError(data.data));
            break;
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      dispatch(addNotification({
        type: 'error',
        title: 'Connection Error',
        message: 'WebSocket connection failed'
      }));
    };

  }, [dispatch]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
      dispatch(setConnectionStatus(false));
    }
  }, [dispatch]);

  const send = useCallback((event: string, data: any) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({ type: event, payload: data });
      socketRef.current.send(message);
    } else {
    }
  }, []);

  // Removed fetchLogs - using polling for real-time updates only

  // Auto-connect on mount and start polling for training status
  useEffect(() => {
    connect();
    
    // Initial status and logs fetch
    const fetchInitialData = async () => {
      try {
        // Clear logs when page loads - force complete reset
        dispatch(clearLogs());
        lastLoggedStep.current = -1;
        
        const response = await fetch('http://localhost:8000/training/status');
        if (response.ok) {
          const status = await response.json();
          
          // Load the training config and state
          if (status.config) {
            dispatch(setTrainingConfig(status.config));
          }
          // Track current run start time if available
          if (status.metrics && status.metrics.start_time) {
            lastRunStartTime.current = status.metrics.start_time as string;
          }
          
          if (status.state === 'completed') {
            dispatch(trainingCompleted({ final_metrics: status.metrics }));
          } else if (status.state === 'error') {
            dispatch(trainingError({ error: 'Training failed' }));
          } else if (status.state === 'running') {
            // If training is running, start fresh
            lastLoggedStep.current = -1;
          }
        }
      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };
    
    fetchInitialData();
    
    // Poll training status every 2 seconds as fallback
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/training/status');
        if (response.ok) {
          const status = await response.json();
          
          // Update training state based on API status 
          if (status.state === 'running') {
            const metrics = status.metrics || {};
            const currentStep: number = metrics.current_step ?? 0;
            const runStartTime: string | null = metrics.start_time ?? null;

            // Detect a new run by start_time change (preferred) or step regression (fallback)
            const startTimeChanged = runStartTime && runStartTime !== lastRunStartTime.current;
            const stepRegressed = lastLoggedStep.current !== -1 && currentStep < lastLoggedStep.current;

            if (startTimeChanged || stepRegressed) {
              dispatch(clearLogs());
              lastLoggedStep.current = -1;
              if (runStartTime) {
                lastRunStartTime.current = runStartTime;
              }
              dispatch(trainingStarted());
            }

            // Only log when step changes - show iter, train loss, val loss
            let logLine = '';
            if (currentStep !== lastLoggedStep.current) {
              const trainLoss = metrics.train_loss != null ? Number(metrics.train_loss).toFixed(4) : 'N/A';
              const valLoss = metrics.val_loss != null ? Number(metrics.val_loss).toFixed(4) : 'N/A';
              logLine = `Iter ${currentStep}: Train loss ${trainLoss}, Val loss ${valLoss}`;
              lastLoggedStep.current = currentStep;
            }

            dispatch(trainingProgress({
              metrics: status.metrics,
              log_line: logLine
            }));
          } else if (status.state === 'completed') {
            // Ensure we set the training config if we don't have it
            if (status.config) {
              dispatch(setTrainingConfig(status.config));
            }
            dispatch(trainingCompleted({ final_metrics: status.metrics }));
          } else if (status.state === 'error') {
            dispatch(trainingError({ error: 'Training failed' }));
          }
        }
      } catch (error) {
        // Silent - don't spam console with polling errors
      }
    }, 2000);
    
    // Cleanup on unmount
    return () => {
      disconnect();
      clearInterval(pollInterval);
    };
  }, [connect, disconnect, dispatch]);

  return {
    connect,
    disconnect,
    send,
    isConnected,
  };
};