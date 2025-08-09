import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Settings, 
  Play, 
  BarChart3, 
  GitCompare, 
  ChevronLeft,
  ChevronRight,
  Zap
} from 'lucide-react';
import { RootState } from '../store/store';
import { setActiveNavItem, toggleSidebar, type ActivePage } from '../store/slices/uiSlice';

interface NavItem {
  id: ActivePage;
  label: string;
  icon: React.ComponentType<any>;
  path: string;
  description: string;
}

const navItems: NavItem[] = [
  {
    id: 'setup',
    label: 'Setup',
    icon: Settings,
    path: '/setup',
    description: 'Configure models and datasets'
  },
  {
    id: 'training',
    label: 'Training',
    icon: Play,
    path: '/training',
    description: 'Monitor training progress'
  },
  {
    id: 'results',
    label: 'Results',
    icon: BarChart3,
    path: '/results',
    description: 'View training analytics'
  },
  {
    id: 'compare',
    label: 'Compare',
    icon: GitCompare,
    path: '/compare',
    description: 'Compare model outputs'
  }
];

export const Sidebar: React.FC = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const location = useLocation();
  const { sidebarCollapsed } = useSelector((state: RootState) => state.ui);
  const { state: trainingState } = useSelector((state: RootState) => state.training);

  const handleNavItemClick = (item: NavItem) => {
    dispatch(setActiveNavItem(item.id));
    navigate(item.path);
  };

  const toggleSidebarHandler = () => {
    dispatch(toggleSidebar());
  };

  return (
    <div className="h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between">
          {!sidebarCollapsed && (
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Zap className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  MLX Fine-Tuner
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Local LLM Training
                </p>
              </div>
            </div>
          )}
          <button
            onClick={toggleSidebarHandler}
            className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 transition-colors"
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          
          return (
            <button
              key={item.id}
              onClick={() => handleNavItemClick(item)}
              className={`w-full flex items-center space-x-3 px-3 py-3 rounded-lg transition-all duration-200 group ${
                isActive
                  ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
              title={sidebarCollapsed ? `${item.label}: ${item.description}` : undefined}
            >
              <div className="relative">
                <Icon 
                  className={`h-5 w-5 ${
                    isActive ? 'text-primary-600 dark:text-primary-400' : ''
                  }`} 
                />
                {/* Training indicator dot */}
                {item.id === 'training' && trainingState === 'running' && (
                  <div className="absolute -top-1 -right-1 h-3 w-3 bg-success-500 rounded-full border-2 border-white dark:border-gray-900 animate-pulse" />
                )}
              </div>
              
              {!sidebarCollapsed && (
                <div className="flex-1 text-left">
                  <div className="font-medium">{item.label}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300">
                    {item.description}
                  </div>
                </div>
              )}
            </button>
          );
        })}
      </nav>

      {/* Training status indicator */}
      {!sidebarCollapsed && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-800">
          <div className="flex items-center space-x-2">
            <div className={`h-3 w-3 rounded-full ${
              trainingState === 'running' 
                ? 'bg-success-500 animate-pulse' 
                : trainingState === 'completed'
                ? 'bg-success-500'
                : trainingState === 'error'
                ? 'bg-error-500'
                : 'bg-gray-300 dark:bg-gray-600'
            }`} />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Status: <span className="capitalize font-medium">{trainingState}</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
};