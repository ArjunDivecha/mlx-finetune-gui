import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface ModelInfo {
  name: string;
  path: string;
  model_type: string;
  vocab_size: number;
}

interface ModelsSliceState {
  models: ModelInfo[];
  selectedModel: ModelInfo | null;
  isLoading: boolean;
  error: string | null;
}

const initialState: ModelsSliceState = {
  models: [],
  selectedModel: null,
  isLoading: false,
  error: null,
};

export const modelsSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    setModels: (state, action: PayloadAction<ModelInfo[]>) => {
      state.models = action.payload;
      state.isLoading = false;
      state.error = null;
    },
    setSelectedModel: (state, action: PayloadAction<ModelInfo | null>) => {
      state.selectedModel = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
      state.isLoading = false;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
});

export const {
  setModels,
  setSelectedModel,
  setLoading,
  setError,
  clearError,
} = modelsSlice.actions;