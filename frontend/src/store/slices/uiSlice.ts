import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ThemeMode = 'light' | 'dark' | 'system';
export type ActivePage = 'setup' | 'training' | 'results' | 'compare';

interface UISliceState {
  theme: ThemeMode;
  activeNavItem: ActivePage;
  sidebarCollapsed: boolean;
  showLogs: boolean;
  notifications: Notification[];
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: number;
  autoHide?: boolean;
}

const initialState: UISliceState = {
  theme: 'dark',
  activeNavItem: 'setup',
  sidebarCollapsed: false,
  showLogs: false,
  notifications: [],
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setTheme: (state, action: PayloadAction<ThemeMode>) => {
      state.theme = action.payload;
    },
    setActiveNavItem: (state, action: PayloadAction<ActivePage>) => {
      state.activeNavItem = action.payload;
    },
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
    },
    toggleLogs: (state) => {
      state.showLogs = !state.showLogs;
    },
    setShowLogs: (state, action: PayloadAction<boolean>) => {
      state.showLogs = action.payload;
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: Math.random().toString(36).substr(2, 9),
        timestamp: Date.now(),
      };
      state.notifications.push(notification);
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
  },
});

export const {
  setTheme,
  setActiveNavItem,
  toggleSidebar,
  setSidebarCollapsed,
  toggleLogs,
  setShowLogs,
  addNotification,
  removeNotification,
  clearNotifications,
} = uiSlice.actions;