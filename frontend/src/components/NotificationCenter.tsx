import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { X, CheckCircle, AlertTriangle, Info, XCircle } from 'lucide-react';
import { RootState } from '../store/store';
import { removeNotification } from '../store/slices/uiSlice';

export const NotificationCenter: React.FC = () => {
  const dispatch = useDispatch();
  const { notifications } = useSelector((state: RootState) => state.ui);

  useEffect(() => {
    // Auto-hide notifications after 5 seconds if autoHide is true
    notifications.forEach(notification => {
      if (notification.autoHide) {
        const timer = setTimeout(() => {
          dispatch(removeNotification(notification.id));
        }, 5000);

        return () => clearTimeout(timer);
      }
    });
  }, [notifications, dispatch]);

  if (notifications.length === 0) {
    return null;
  }

  const getIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-success-500" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-warning-500" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-error-500" />;
      default:
        return <Info className="h-5 w-5 text-primary-500" />;
    }
  };

  const getBorderColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'border-success-500';
      case 'warning':
        return 'border-warning-500';
      case 'error':
        return 'border-error-500';
      default:
        return 'border-primary-500';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 space-y-3 pointer-events-none">
      {notifications.map(notification => (
        <div
          key={notification.id}
          className={`pointer-events-auto max-w-sm bg-white dark:bg-gray-900 rounded-lg shadow-lg border-l-4 ${getBorderColor(notification.type)} animate-fade-in`}
        >
          <div className="p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                {getIcon(notification.type)}
              </div>
              
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {notification.title}
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  {notification.message}
                </p>
              </div>

              <button
                onClick={() => dispatch(removeNotification(notification.id))}
                className="flex-shrink-0 p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <X className="h-4 w-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};