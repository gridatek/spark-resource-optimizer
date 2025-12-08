import { Component, ChangeDetectionStrategy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartType } from 'chart.js';
import { ApiService } from '../../services/api.service';
import {
  RecommendationRequest,
  RecommendationResponse
} from '../../models/recommendation.model';

@Component({
  selector: 'app-recommendations',
  imports: [CommonModule, FormsModule, BaseChartDirective],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="p-8 max-w-7xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Get Resource Recommendations</h1>
        <p class="text-gray-600">Get optimal Spark configuration recommendations for your job</p>
      </header>

      <div class="grid lg:grid-cols-2 gap-8">
        <!-- Request Form -->
        <div class="bg-white rounded-lg p-8 shadow">
          <h2 class="text-2xl font-semibold text-gray-900 mb-6">Job Requirements</h2>

          <form (ngSubmit)="getRecommendation()" #recForm="ngForm">
            <div class="mb-6">
              <label for="inputSize" class="block text-sm font-medium text-gray-700 mb-2">
                Input Data Size (GB) *
              </label>
              <input
                type="number"
                id="inputSize"
                name="inputSize"
                [(ngModel)]="request.input_size_gb"
                min="0.1"
                step="0.1"
                required
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div class="mb-6">
              <label for="jobType" class="block text-sm font-medium text-gray-700 mb-2">
                Job Type
              </label>
              <select
                id="jobType"
                name="jobType"
                [(ngModel)]="request.job_type"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option [value]="undefined">-- Select Type --</option>
                @for (type of jobTypes; track type) {
                  <option [value]="type">{{ type | uppercase }}</option>
                }
              </select>
            </div>

            <div class="mb-6">
              <label for="appName" class="block text-sm font-medium text-gray-700 mb-2">
                Application Name (optional)
              </label>
              <input
                type="text"
                id="appName"
                name="appName"
                [(ngModel)]="request.app_name"
                placeholder="my-spark-job"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div class="mb-6">
              <label for="method" class="block text-sm font-medium text-gray-700 mb-2">
                Recommendation Method
              </label>
              <select
                id="method"
                name="method"
                [(ngModel)]="request.method"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option [value]="undefined">-- Auto Select --</option>
                @for (method of methods; track method) {
                  <option [value]="method">{{ method | titlecase }}</option>
                }
              </select>
            </div>

            <div class="flex gap-4 mt-8">
              <button
                type="submit"
                [disabled]="loading || !recForm.form.valid"
                class="px-6 py-3 bg-blue-500 text-white rounded-md font-medium hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {{ loading! ? 'Getting Recommendation...' : 'Get Recommendation' }}
              </button>
              <button
                type="button"
                (click)="reset()"
                class="px-6 py-3 bg-gray-200 text-gray-700 rounded-md font-medium hover:bg-gray-300 transition-colors"
              >
                Reset
              </button>
            </div>

            @if (error()) {
              <div class="mt-4 p-3 bg-red-100 text-red-800 rounded-md text-sm">
                {{ error() }}
              </div>
            }
          </form>
        </div>

        <!-- Recommendation Result -->
        @if (recommendation()) {
          <div class="bg-white rounded-lg p-8 shadow">
            <h2 class="text-2xl font-semibold text-gray-900 mb-6">Recommended Configuration</h2>

            <div [class]="'inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold mb-6 ' + getConfidenceColor(recommendation()?.confidence!)">
              <span>Confidence:</span>
              <span>{{ (recommendation()?.confidence! * 100).toFixed(0) }}%</span>
            </div>

            <div class="grid grid-cols-2 gap-4 mb-8">
              <div class="flex items-center gap-4 p-4 bg-gray-50 rounded-md">
                <div class="text-3xl">🔧</div>
                <div>
                  <h4 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
                    Executor Cores
                  </h4>
                  <p class="text-2xl font-bold text-gray-900">
                    {{ recommendation()?.configuration!.executor_cores! }}
                  </p>
                </div>
              </div>

              <div class="flex items-center gap-4 p-4 bg-gray-50 rounded-md">
                <div class="text-3xl">💾</div>
                <div>
                  <h4 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
                    Executor Memory
                  </h4>
                  <p class="text-2xl font-bold text-gray-900">
                    {{ formatMemory(recommendation()?.configuration!.executor_memory_mb!) }}
                  </p>
                </div>
              </div>

              <div class="flex items-center gap-4 p-4 bg-gray-50 rounded-md">
                <div class="text-3xl">⚡</div>
                <div>
                  <h4 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
                    Number of Executors
                  </h4>
                  <p class="text-2xl font-bold text-gray-900">
                    {{ recommendation()?.configuration?.num_executors! }}
                  </p>
                </div>
              </div>

              <div class="flex items-center gap-4 p-4 bg-gray-50 rounded-md">
                <div class="text-3xl">🚗</div>
                <div>
                  <h4 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
                    Driver Memory
                  </h4>
                  <p class="text-2xl font-bold text-gray-900">
                    {{ formatMemory(recommendation()?.configuration!.driver_memory_mb!) }}
                  </p>
                </div>
              </div>
            </div>

            <!-- Predicted Metrics -->
            @if (recommendation()?.predicted_metrics) {
              <div class="mb-8 p-4 bg-blue-50 rounded-md">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Predicted Performance</h3>
                <div class="grid grid-cols-2 gap-4">
                  <div class="flex justify-between">
                    <span class="text-blue-800 font-medium">Duration:</span>
                    <span class="text-gray-900 font-bold">
                      ~{{ recommendation()?.predicted_metrics?.duration_minutes! }} minutes
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-blue-800 font-medium">Estimated Cost:</span>
                    <span class="text-gray-900 font-bold">
                      \${{ recommendation()?.predicted_metrics?.cost_usd!.toFixed(2) }}
                    </span>
                  </div>
                </div>
              </div>
            }

            <!-- Metadata -->
            <div class="mb-8 p-4 bg-gray-50 rounded-md">
              <div class="text-gray-700 text-sm mb-2">
                <strong>Method:</strong> {{ recommendation()?.method | titlecase }}
              </div>
              @if (recommendation()?.metadata?.similar_jobs) {
                <div class="text-gray-700 text-sm">
                  <strong>Based on:</strong> {{ recommendation()?.metadata?.similar_jobs?.length }} similar jobs
                </div>
              }
            </div>

            <!-- Resource Allocation Visualization -->
            <div class="mb-8">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Resource Allocation</h3>
              <div class="bg-gray-50 rounded-md p-6">
                <div class="h-64">
                  <canvas
                    baseChart
                    [type]="barChartType"
                    [data]="resourceChartData()"
                    [options]="barChartOptions"
                  ></canvas>
                </div>
              </div>
            </div>

            <!-- Similar Jobs Visualization -->
            @if (hasSimilarJobs()) {
              <div class="mb-8">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Similar Jobs Analysis</h3>
                <div class="bg-gray-50 rounded-md p-4">
                  <p class="text-sm text-gray-600 mb-4">
                    This recommendation is based on {{ getSimilarJobsCount() }} similar historical jobs
                  </p>
                  <div class="space-y-2">
                    @for (similarJob of getSimilarJobs(); track similarJob.app_id) {
                      <div class="flex items-center gap-3 p-3 bg-white rounded border border-gray-200">
                        <div class="flex-1">
                          <div class="text-sm font-mono text-gray-900">{{ similarJob.app_id }}</div>
                        </div>
                        <div class="flex items-center gap-2">
                          <div class="text-sm text-gray-600">Similarity:</div>
                          <div class="flex items-center gap-2">
                            <div class="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                [style.width.%]="similarJob.similarity * 100"
                                class="h-full bg-gradient-to-r from-blue-500 to-green-500"
                              ></div>
                            </div>
                            <span class="text-sm font-semibold text-gray-900">
                              {{ (similarJob.similarity * 100).toFixed(0) }}%
                            </span>
                          </div>
                        </div>
                      </div>
                    }
                  </div>
                </div>
              </div>
            }

            <!-- Cost-Performance Trade-off -->
            @if (recommendation()?.predicted_metrics) {
              <div class="mb-8">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Cost-Performance Indicator</h3>
                <div class="bg-gray-50 rounded-md p-6">
                  <div class="flex items-center justify-between mb-4">
                    <div class="flex-1 text-center">
                      <div class="text-2xl mb-2">💰</div>
                      <div class="text-sm text-gray-600">Cost Efficient</div>
                    </div>
                    <div class="flex-1 px-8">
                      <div class="relative h-6 bg-gradient-to-r from-green-400 via-yellow-400 to-red-400 rounded-full">
                        <div
                          [style.left.%]="getCostPerformanceScore() * 100"
                          class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-blue-600 border-2 border-white rounded-full shadow-lg"
                        ></div>
                      </div>
                    </div>
                    <div class="flex-1 text-center">
                      <div class="text-2xl mb-2">⚡</div>
                      <div class="text-sm text-gray-600">High Performance</div>
                    </div>
                  </div>
                  <div class="text-center text-sm text-gray-600">
                    This configuration balances cost (\${{ recommendation()?.predicted_metrics?.cost_usd!.toFixed(2) }})
                    and performance (~{{ recommendation()?.predicted_metrics?.duration_minutes! }} min)
                  </div>
                </div>
              </div>
            }

            <!-- Spark Configuration -->
            <div>
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Spark Submit Command</h3>
              <div class="bg-gray-800 rounded-md p-4 overflow-x-auto">
                <pre class="text-gray-200 text-sm font-mono leading-relaxed"><code>spark-submit \
  --num-executors {{ recommendation()?.configuration?.num_executors! }} \
  --executor-cores {{ recommendation()?.configuration?.executor_cores! }} \
  --executor-memory {{ formatMemory(recommendation()?.configuration?.executor_memory_mb!) }} \
  --driver-memory {{ formatMemory(recommendation()?.configuration?.driver_memory_mb!) }} \
  your-spark-app.jar</code></pre>
              </div>
              <button
                (click)="copyToClipboard()"
                class="mt-3 px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition-colors text-sm"
              >
                {{ copied() ? '✓ Copied!' : '📋 Copy Command' }}
              </button>
            </div>
          </div>
        }

        <!-- Loading State -->
        @if (loading()) {
          <div class="bg-white rounded-lg p-12 shadow text-center">
            <div class="w-12 h-12 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
            <p class="text-gray-600">Analyzing historical data and generating recommendations...</p>
          </div>
        }
      </div>
    </div>
  `
})
export class Recommendations {
  // Form model (user-friendly with GB)
  formData = {
    input_size_gb: 10,
    job_type: 'etl' as 'etl' | 'ml' | 'sql' | 'streaming',
    app_name: '',
    priority: 'balanced' as 'performance' | 'cost' | 'balanced'
  };

  // For backward compatibility
  get request() { return this.formData; }
  set request(val: any) { Object.assign(this.formData, val); }

  // Use signals for reactive state
  recommendation = signal<RecommendationResponse | null>(null);
  loading = signal(false);
  error = signal<string | null>(null);
  copied = signal(false);

  jobTypes = ['etl', 'ml', 'sql', 'streaming'];
  methods = ['similarity', 'ml', 'rule_based', 'hybrid'];

  // Chart configuration
  barChartType: ChartType = 'bar';
  barChartOptions: ChartConfiguration['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        beginAtZero: true
      }
    }
  };

  resourceChartData = computed<ChartConfiguration['data']>(() => {
    const rec = this.recommendation();
    if (!rec) {
      return { datasets: [] };
    }

    const config = rec.configuration;
    return {
      labels: ['Executors', 'Executor Cores', 'Executor Memory (GB)', 'Driver Memory (GB)'],
      datasets: [{
        data: [
          config.num_executors,
          config.executor_cores,
          config.executor_memory_mb / 1024,
          config.driver_memory_mb / 1024
        ],
        backgroundColor: [
          'rgba(59, 130, 246, 0.8)',
          'rgba(147, 51, 234, 0.8)',
          'rgba(236, 72, 153, 0.8)',
          'rgba(245, 158, 11, 0.8)'
        ],
        borderColor: [
          'rgb(59, 130, 246)',
          'rgb(147, 51, 234)',
          'rgb(236, 72, 153)',
          'rgb(245, 158, 11)'
        ],
        borderWidth: 1
      }]
    };
  });

  constructor(private apiService: ApiService) {}

  getRecommendation(): void {
    if (!this.formData.input_size_gb || this.formData.input_size_gb <= 0) {
      this.error.set('Please enter a valid input size');
      return;
    }

    this.loading.set(true);
    this.error.set(null);
    this.recommendation.set(null);

    // Convert GB to bytes for API request
    const apiRequest: RecommendationRequest = {
      input_size_bytes: this.formData.input_size_gb * 1024 * 1024 * 1024,
      job_type: this.formData.job_type,
      app_name: this.formData.app_name || undefined,
      priority: this.formData.priority
    };

    this.apiService.getRecommendation(apiRequest).subscribe({
      next: (response) => {
        console.log('Received recommendation response:', response);
        this.recommendation.set(response);
        this.loading.set(false);
        console.log('Updated component state - loading:', this.loading(), 'recommendation:', this.recommendation());
      },
      error: (err) => {
        this.error.set('Failed to get recommendation. Please try again.');
        console.error('Error getting recommendation:', err);
        this.loading.set(false);
      }
    });
  }

  getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return 'bg-green-200 text-green-900';
    if (confidence >= 0.6) return 'bg-orange-200 text-orange-900';
    return 'bg-red-200 text-red-900';
  }

  formatMemory(mb: number): string {
    return `${(mb / 1024).toFixed(1)} GB`;
  }

  reset(): void {
    this.formData = {
      input_size_gb: 10,
      job_type: 'etl',
      app_name: '',
      priority: 'balanced'
    };
    this.recommendation.set(null);
    this.error.set(null);
    this.copied.set(false);
  }

  getCostPerformanceScore(): number {
    const rec = this.recommendation();
    if (!rec?.predicted_metrics) {
      return 0.5; // Default middle value
    }

    // Calculate a score between 0 (cost-efficient) and 1 (high-performance)
    // Lower cost and lower duration = more towards cost-efficient
    // Higher cost and lower duration = more towards high-performance
    const cost = rec.predicted_metrics.cost_usd;
    const duration = rec.predicted_metrics.duration_minutes;

    // Normalize: assuming reasonable ranges
    // Cost: $0-$10, Duration: 0-60 minutes
    const normalizedCost = Math.min(cost / 10, 1);
    const normalizedDuration = Math.min(duration / 60, 1);

    // Score: if cost is high but duration is low, it's performance-oriented
    // if cost is low but duration is higher, it's cost-oriented
    return (normalizedCost + (1 - normalizedDuration)) / 2;
  }

  copyToClipboard(): void {
    const rec = this.recommendation();
    if (!rec) return;

    const config = rec.configuration;
    const command = `spark-submit \\
  --num-executors ${config.num_executors} \\
  --executor-cores ${config.executor_cores} \\
  --executor-memory ${this.formatMemory(config.executor_memory_mb)} \\
  --driver-memory ${this.formatMemory(config.driver_memory_mb)} \\
  your-spark-app.jar`;

    navigator.clipboard.writeText(command).then(() => {
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 2000);
    }).catch(err => {
      console.error('Failed to copy:', err);
    });
  }

  hasSimilarJobs(): boolean {
    const rec = this.recommendation();
    return (rec?.metadata?.similar_jobs?.length ?? 0) > 0;
  }

  getSimilarJobsCount(): number {
    const rec = this.recommendation();
    return rec?.metadata?.similar_jobs?.length ?? 0;
  }

  getSimilarJobs(): Array<{ app_id: string; similarity: number }> {
    const rec = this.recommendation();
    return (rec?.metadata?.similar_jobs ?? []).slice(0, 5);
  }
}
