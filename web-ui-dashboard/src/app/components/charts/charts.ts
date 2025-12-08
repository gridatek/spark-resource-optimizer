import { Component, OnInit, ChangeDetectionStrategy, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartType } from 'chart.js';
import { ApiService } from '../../services/api.service';
import { SparkJob } from '../../models/job.model';

@Component({
  selector: 'app-charts',
  imports: [CommonModule, BaseChartDirective],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="p-8 max-w-7xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Performance Analytics</h1>
        <p class="text-gray-600">Visualize job performance trends and resource utilization</p>
      </header>

      <!-- Loading State -->
      @if (loading()) {
        <div class="text-center py-16">
          <div class="w-12 h-12 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p class="text-gray-600">Loading analytics data...</p>
        </div>
      }

      <!-- Error State -->
      @if (error()) {
        <div class="text-center py-16">
          <p class="text-red-600 text-lg mb-4">{{ error() }}</p>
          <button
            (click)="loadChartData()"
            class="px-8 py-3 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
          >
            Retry
          </button>
        </div>
      }

      <!-- Charts Grid -->
      @if (!loading() && !error()) {
        <div class="space-y-8">
          <!-- Job Duration Trends -->
          <div class="bg-white rounded-lg p-6 shadow">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Job Duration Trends</h2>
            <div class="h-80">
              <canvas
                baseChart
                [type]="lineChartType"
                [data]="durationChartData()"
                [options]="lineChartOptions"
              ></canvas>
            </div>
          </div>

          <!-- Success Rate & Job Status Distribution -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Success Rate Over Time -->
            <div class="bg-white rounded-lg p-6 shadow">
              <h2 class="text-xl font-semibold text-gray-900 mb-4">Success Rate Trends</h2>
              <div class="h-64">
                <canvas
                  baseChart
                  [type]="lineChartType"
                  [data]="successRateChartData()"
                  [options]="percentageChartOptions"
                ></canvas>
              </div>
            </div>

            <!-- Job Status Distribution -->
            <div class="bg-white rounded-lg p-6 shadow">
              <h2 class="text-xl font-semibold text-gray-900 mb-4">Job Status Distribution</h2>
              <div class="h-64 flex items-center justify-center">
                <canvas
                  baseChart
                  [type]="doughnutChartType"
                  [data]="statusChartData()"
                  [options]="doughnutChartOptions"
                ></canvas>
              </div>
            </div>
          </div>

          <!-- Resource Utilization -->
          <div class="bg-white rounded-lg p-6 shadow">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Resource Utilization</h2>
            <div class="h-80">
              <canvas
                baseChart
                [type]="barChartType"
                [data]="resourceChartData()"
                [options]="barChartOptions"
              ></canvas>
            </div>
          </div>

          <!-- Cost Trends (Estimated) -->
          <div class="bg-white rounded-lg p-6 shadow">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Estimated Cost Trends</h2>
            <div class="text-sm text-gray-600 mb-4">
              * Cost estimates based on standard compute pricing
            </div>
            <div class="h-80">
              <canvas
                baseChart
                [type]="lineChartType"
                [data]="costChartData()"
                [options]="costChartOptions"
              ></canvas>
            </div>
          </div>

          <!-- Data Processing Volume -->
          <div class="bg-white rounded-lg p-6 shadow">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Data Processing Volume</h2>
            <div class="h-80">
              <canvas
                baseChart
                [type]="barChartType"
                [data]="dataVolumeChartData()"
                [options]="barChartOptions"
              ></canvas>
            </div>
          </div>
        </div>
      }
    </div>
  `
})
export class ChartsComponent implements OnInit {
  loading = signal(false);
  error = signal<string | null>(null);

  // Chart types
  lineChartType: ChartType = 'line';
  barChartType: ChartType = 'bar';
  doughnutChartType: ChartType = 'doughnut';

  // Chart data signals
  durationChartData = signal<ChartConfiguration['data']>({
    datasets: []
  });

  successRateChartData = signal<ChartConfiguration['data']>({
    datasets: []
  });

  statusChartData = signal<ChartConfiguration['data']>({
    datasets: []
  });

  resourceChartData = signal<ChartConfiguration['data']>({
    datasets: []
  });

  costChartData = signal<ChartConfiguration['data']>({
    datasets: []
  });

  dataVolumeChartData = signal<ChartConfiguration['data']>({
    datasets: []
  });

  // Chart options
  lineChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  percentageChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: (value) => value + '%'
        }
      }
    }
  };

  barChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    }
  };

  doughnutChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
    }
  };

  costChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: (value) => '$' + value
        }
      }
    }
  };

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadChartData();
  }

  loadChartData(): void {
    this.loading.set(true);
    this.error.set(null);

    // Fetch more jobs for better chart data
    this.apiService.getJobs({ limit: 100 }).subscribe({
      next: (response) => {
        this.processChartData(response.jobs);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set('Failed to load analytics data');
        console.error('Error loading charts:', err);
        this.loading.set(false);
      }
    });
  }

  private processChartData(jobs: SparkJob[]): void {
    if (jobs.length === 0) {
      this.error.set('No job data available for analytics');
      return;
    }

    // Sort jobs by start time
    const sortedJobs = [...jobs].sort((a, b) =>
      new Date(a.start_time).getTime() - new Date(b.start_time).getTime()
    );

    // Group jobs by date
    const jobsByDate = this.groupJobsByDate(sortedJobs);
    const dates = Object.keys(jobsByDate).sort();

    // Duration Trends
    this.durationChartData.set({
      labels: dates,
      datasets: [{
        label: 'Average Duration (minutes)',
        data: dates.map(date => {
          const dayJobs = jobsByDate[date];
          const avgDuration = dayJobs.reduce((sum, job) => sum + job.duration_ms, 0) / dayJobs.length;
          return Number((avgDuration / 60000).toFixed(2));
        }),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      }]
    });

    // Success Rate Trends
    this.successRateChartData.set({
      labels: dates,
      datasets: [{
        label: 'Success Rate (%)',
        data: dates.map(date => {
          const dayJobs = jobsByDate[date];
          const successCount = dayJobs.filter(j => j.status === 'completed').length;
          return Number(((successCount / dayJobs.length) * 100).toFixed(1));
        }),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: true
      }]
    });

    // Status Distribution
    const statusCounts = {
      completed: jobs.filter(j => j.status === 'completed').length,
      failed: jobs.filter(j => j.status === 'failed').length,
      running: jobs.filter(j => j.status === 'running').length
    };

    this.statusChartData.set({
      labels: ['Completed', 'Failed', 'Running'],
      datasets: [{
        data: [statusCounts.completed, statusCounts.failed, statusCounts.running],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(59, 130, 246, 0.8)'
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(239, 68, 68)',
          'rgb(59, 130, 246)'
        ],
        borderWidth: 1
      }]
    });

    // Resource Utilization (last 10 jobs)
    const recentJobs = sortedJobs.slice(-10);
    this.resourceChartData.set({
      labels: recentJobs.map(j => j.app_name.substring(0, 20)),
      datasets: [
        {
          label: 'Executors',
          data: recentJobs.map(j => j.configuration.num_executors),
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
        },
        {
          label: 'Executor Memory (GB)',
          data: recentJobs.map(j => Number((j.configuration.executor_memory_mb / 1024).toFixed(1))),
          backgroundColor: 'rgba(147, 51, 234, 0.8)',
        },
        {
          label: 'Executor Cores',
          data: recentJobs.map(j => j.configuration.executor_cores),
          backgroundColor: 'rgba(236, 72, 153, 0.8)',
        }
      ]
    });

    // Estimated Cost Trends
    this.costChartData.set({
      labels: dates,
      datasets: [{
        label: 'Estimated Daily Cost ($)',
        data: dates.map(date => {
          const dayJobs = jobsByDate[date];
          const totalCost = dayJobs.reduce((sum, job) => {
            return sum + this.estimateJobCost(job);
          }, 0);
          return Number(totalCost.toFixed(2));
        }),
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        fill: true
      }]
    });

    // Data Volume Processing
    this.dataVolumeChartData.set({
      labels: dates,
      datasets: [
        {
          label: 'Input Data (GB)',
          data: dates.map(date => {
            const dayJobs = jobsByDate[date];
            const totalInput = dayJobs.reduce((sum, job) => sum + (job.metrics.input_bytes || 0), 0);
            return Number((totalInput / (1024 ** 3)).toFixed(2));
          }),
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
        },
        {
          label: 'Output Data (GB)',
          data: dates.map(date => {
            const dayJobs = jobsByDate[date];
            const totalOutput = dayJobs.reduce((sum, job) => sum + (job.metrics.output_bytes || 0), 0);
            return Number((totalOutput / (1024 ** 3)).toFixed(2));
          }),
          backgroundColor: 'rgba(34, 197, 94, 0.8)',
        }
      ]
    });
  }

  private groupJobsByDate(jobs: SparkJob[]): Record<string, SparkJob[]> {
    return jobs.reduce((acc, job) => {
      const date = new Date(job.start_time).toISOString().split('T')[0];
      if (!acc[date]) {
        acc[date] = [];
      }
      acc[date].push(job);
      return acc;
    }, {} as Record<string, SparkJob[]>);
  }

  private estimateJobCost(job: SparkJob): number {
    // Rough cost estimation based on resource usage and duration
    // Assuming $0.10 per executor-hour
    const durationHours = job.duration_ms / (1000 * 60 * 60);
    const executorCost = 0.10;
    return job.configuration.num_executors * durationHours * executorCost;
  }
}
