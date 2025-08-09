import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface TrainingMetrics {
  current_step: number;
  total_steps: number;
  train_loss: number;
  val_loss: number;
  learning_rate: number;
  start_time: string;
  estimated_time_remaining: number | null;
}

export interface TrainingConfig {
  model_path: string;
  train_data_path: string;
  val_data_path: string;
  learning_rate: number;
  batch_size: number;
  max_seq_length: number;
  iterations: number;
  steps_per_report: number;
  steps_per_eval: number;
  save_every: number;
  early_stop: boolean;
  patience: number;
  adapter_name: string;
}

export type TrainingState = 'idle' | 'running' | 'paused' | 'completed' | 'error' | 'stopped';

interface TrainingSliceState {
  state: TrainingState;
  config: TrainingConfig | null;
  metrics: TrainingMetrics | null;
  logs: string[];
  error: string | null;
  isConnected: boolean;
}

const initialState: TrainingSliceState = {
  state: 'idle',
  config: null,
  metrics: null,
  logs: [],
  error: null,
  isConnected: false,
};

export const trainingSlice = createSlice({
  name: 'training',
  initialState,
  reducers: {
    setTrainingState: (state, action: PayloadAction<TrainingState>) => {
      state.state = action.payload;
    },
    setTrainingConfig: (state, action: PayloadAction<TrainingConfig>) => {
      state.config = action.payload;
    },
    setTrainingMetrics: (state, action: PayloadAction<TrainingMetrics>) => {
      state.metrics = action.payload;
    },
    updateTrainingMetrics: (state, action: PayloadAction<Partial<TrainingMetrics>>) => {
      if (state.metrics) {
        state.metrics = { ...state.metrics, ...action.payload };
      }
    },
    addLogLine: (state, action: PayloadAction<string>) => {
      state.logs.push(action.payload);
      // Keep only last 1000 lines
      if (state.logs.length > 1000) {
        state.logs = state.logs.slice(-1000);
      }
    },
    clearLogs: (state) => {
      state.logs = [];
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    resetTraining: (state) => {
      state.state = 'idle';
      state.metrics = null;
      state.error = null;
      state.logs = [];
    },
    trainingStarted: (state) => {
      state.state = 'running';
      state.error = null;
    },
    trainingProgress: (state, action: PayloadAction<{ metrics: TrainingMetrics; log_line: string }>) => {
      state.metrics = action.payload.metrics;
      state.logs.push(action.payload.log_line);
      // Keep only last 1000 lines
      if (state.logs.length > 1000) {
        state.logs = state.logs.slice(-1000);
      }
    },
    trainingCompleted: (state, action: PayloadAction<{ final_metrics: TrainingMetrics }>) => {
      state.state = 'completed';
      state.metrics = action.payload.final_metrics;
    },
    trainingStopped: (state) => {
      state.state = 'stopped';
    },
    trainingError: (state, action: PayloadAction<{ error: string }>) => {
      state.state = 'error';
      state.error = action.payload.error;
    },
  },
});

export const {
  setTrainingState,
  setTrainingConfig,
  setTrainingMetrics,
  updateTrainingMetrics,
  addLogLine,
  clearLogs,
  setError,
  setConnectionStatus,
  resetTraining,
  trainingStarted,
  trainingProgress,
  trainingCompleted,
  trainingStopped,
  trainingError,
} = trainingSlice.actions;