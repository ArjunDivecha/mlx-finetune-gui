import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Moon, Sun, Monitor, Wifi, WifiOff } from 'lucide-react';
import { RootState } from '../store/store';
import { setTheme, type ThemeMode } from '../store/slices/uiSlice';

export const Header: React.FC = () => {
  const dispatch = useDispatch();
  const { theme } = useSelector((state: RootState) => state.ui);
  const { isConnected } = useSelector((state: RootState) => state.training);

  const toggleTheme = () => {
    const themes: ThemeMode[] = ['light', 'dark', 'system'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    dispatch(setTheme(nextTheme));
  };

  const getThemeIcon = () => {
    switch (theme) {
      case 'light':
        return <Sun className="h-4 w-4" />;
      case 'dark':
        return <Moon className="h-4 w-4" />;
      case 'system':
        return <Monitor className="h-4 w-4" />;
      default:
        return <Sun className="h-4 w-4" />;
    }
  };

  return (
    <header className="h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 flex items-center justify-between">
      {/* Left side - breadcrumb or page title could go here */}
      <div className="flex items-center space-x-4">
        <div className="text-sm text-gray-500 dark:text-gray-400">
          MLX Fine-Tuning GUI
        </div>
      </div>

      {/* Right side - controls */}
      <div className="flex items-center space-x-4">
        {/* Connection status */}
        <div className="flex items-center space-x-2">
          {isConnected ? (
            <>
              <Wifi className="h-4 w-4 text-success-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="h-4 w-4 text-error-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Disconnected</span>
            </>
          )}
        </div>

        {/* Divider */}
        <div className="h-6 w-px bg-gray-200 dark:bg-gray-700" />

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 transition-colors"
          title={`Current theme: ${theme}. Click to switch.`}
        >
          {getThemeIcon()}
        </button>
      </div>
    </header>
  );
};