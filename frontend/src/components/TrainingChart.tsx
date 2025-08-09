import React, { useEffect, useRef } from 'react';
import { useSelector } from 'react-redux';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { RootState } from '../store/store';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface DataPoint {
  step: number;
  trainLoss: number;
  valLoss?: number;
  learningRate: number;
}

export const TrainingChart: React.FC = () => {
  const { metrics, state: trainingState } = useSelector((state: RootState) => state.training);
  const dataPointsRef = useRef<DataPoint[]>([]);

  useEffect(() => {
    if (metrics && trainingState === 'running') {
      // Add new data point
      const newPoint: DataPoint = {
        step: metrics.current_step,
        trainLoss: metrics.train_loss,
        valLoss: metrics.val_loss,
        learningRate: metrics.learning_rate
      };

      // Avoid duplicate points
      const lastPoint = dataPointsRef.current[dataPointsRef.current.length - 1];
      if (!lastPoint || lastPoint.step !== newPoint.step) {
        dataPointsRef.current.push(newPoint);
        
        // Keep only last 1000 points for performance
        if (dataPointsRef.current.length > 1000) {
          dataPointsRef.current = dataPointsRef.current.slice(-1000);
        }
      }
    }
  }, [metrics, trainingState]);

  // Reset data when training starts fresh
  useEffect(() => {
    if (trainingState === 'idle') {
      dataPointsRef.current = [];
    }
  }, [trainingState]);

  const data = {
    labels: dataPointsRef.current.map(point => point.step.toString()),
    datasets: [
      {
        label: 'Training Loss',
        data: dataPointsRef.current.map(point => point.trainLoss),
        borderColor: 'rgb(59, 130, 246)', // blue-500
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2,
      },
      ...(dataPointsRef.current.some(point => point.valLoss !== undefined) ? [{
        label: 'Validation Loss',
        data: dataPointsRef.current.map(point => point.valLoss),
        borderColor: 'rgb(16, 185, 129)', // green-500
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2,
      }] : []),
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      x: {
        type: 'category' as const,
        display: true,
        title: {
          display: true,
          text: 'Training Step',
          color: 'rgb(107, 114, 128)', // gray-500
        },
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
          maxTicksLimit: 10,
        },
      },
      y: {
        type: 'linear' as const,
        display: true,
        title: {
          display: true,
          text: 'Loss',
          color: 'rgb(107, 114, 128)',
        },
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: 'rgb(107, 114, 128)',
          usePointStyle: true,
          pointStyle: 'line',
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(107, 114, 128, 0.3)',
        borderWidth: 1,
        displayColors: true,
        callbacks: {
          title: function(tooltipItems: any[]) {
            return `Step: ${tooltipItems[0].label}`;
          },
          label: function(context: any) {
            const value = context.parsed.y;
            return `${context.dataset.label}: ${value.toFixed(6)}`;
          },
        },
      },
    },
    elements: {
      point: {
        hoverBackgroundColor: 'white',
        hoverBorderWidth: 2,
      },
    },
    animation: {
      duration: 0, // Disable animations for real-time updates
    },
  };

  if (dataPointsRef.current.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
        <div className="text-center">
          <div className="w-12 h-12 border-2 border-gray-300 dark:border-gray-600 border-dashed rounded-lg flex items-center justify-center mx-auto mb-2">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p>No training data available</p>
          <p className="text-sm">Start training to see real-time metrics</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-64">
      <Line data={data} options={options} />
    </div>
  );
};