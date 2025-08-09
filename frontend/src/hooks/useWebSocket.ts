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
} from '../store/slices/trainingSlice';
import { addNotification } from '../store/slices/uiSlice';

const WEBSOCKET_URL = 'ws://localhost:8000/ws';

export const useWebSocket = () => {
  const dispatch = useDispatch();
  const { isConnected } = useSelector((state: RootState) => state.training);
  const socketRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    console.log('Connecting to WebSocket...');
    
    const socket = new WebSocket(WEBSOCKET_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      console.log('WebSocket connected');
      dispatch(setConnectionStatus(true));
      dispatch(addNotification({
        type: 'success',
        title: 'Connected',
        message: 'WebSocket connection established'
      }));
    };

    socket.onclose = () => {
      console.log('WebSocket disconnected');
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
        console.log('WebSocket message:', data);
        
        switch (data.type) {
          case 'training_started':
            console.log('Training started:', data);
            dispatch(trainingStarted());
            break;
          case 'training_progress':
            console.log('Training progress:', data);
            dispatch(trainingProgress(data.payload));
            break;
          case 'training_completed':
            console.log('Training completed:', data);
            dispatch(trainingCompleted(data.payload));
            break;
          case 'training_stopped':
            console.log('Training stopped:', data);
            dispatch(trainingStopped());
            break;
          case 'training_error':
            console.log('Training error:', data);
            dispatch(trainingError(data.payload));
            break;
          default:
            console.log('Unknown message type:', data.type);
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
      console.log('Disconnecting WebSocket...');
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
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  // Auto-connect on mount and start polling for training status
  useEffect(() => {
    connect();
    
    // Poll training status every 2 seconds as fallback
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/training/status');
        if (response.ok) {
          const status = await response.json();
          console.log('Polling status:', status.state, status.metrics);
          
          // Update training state based on API status
          if (status.state === 'running') {
            console.log('Dispatching training progress for running state');
            dispatch(trainingProgress({
              metrics: status.metrics,
              log_line: `Step ${status.metrics.current_step}/${status.metrics.total_steps} - Train Loss: ${status.metrics.train_loss}, Val Loss: ${status.metrics.val_loss}`
            }));
          } else if (status.state === 'completed') {
            console.log('Dispatching training completed for completed state');
            // Ensure we set the training config if we don't have it
            if (status.config) {
              dispatch(setTrainingConfig(status.config));
            }
            dispatch(trainingCompleted({ final_metrics: status.metrics }));
          } else if (status.state === 'error') {
            console.log('Dispatching training error for error state');
            dispatch(trainingError({ error: 'Training failed' }));
          } else {
            console.log('Unknown training state:', status.state);
          }
        }
      } catch (error) {
        console.error('Error polling training status:', error);
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