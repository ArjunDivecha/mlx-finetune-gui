import { configureStore } from '@reduxjs/toolkit';
import { trainingSlice } from './slices/trainingSlice';
import { modelsSlice } from './slices/modelsSlice';
import { uiSlice } from './slices/uiSlice';

export const store = configureStore({
  reducer: {
    training: trainingSlice.reducer,
    models: modelsSlice.reducer,
    ui: uiSlice.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['training/setWebSocketConnection'],
        ignoredPaths: ['training.wsConnection'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;