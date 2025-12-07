import { Component, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  RecommendationRequest,
  RecommendationResponse
} from '../../models/recommendation.model';

@Component({
  selector: 'app-recommendations',
  imports: [CommonModule, FormsModule],
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
                {{ loading ? 'Getting Recommendation...' : 'Get Recommendation' }}
              </button>
              <button
                type="button"
                (click)="reset()"
                class="px-6 py-3 bg-gray-200 text-gray-700 rounded-md font-medium hover:bg-gray-300 transition-colors"
              >
                Reset
              </button>
            </div>

            @if (error) {
              <div class="mt-4 p-3 bg-red-100 text-red-800 rounded-md text-sm">
                {{ error }}
              </div>
            }
          </form>
        </div>

        <!-- Recommendation Result -->
        @if (recommendation) {
          <div class="bg-white rounded-lg p-8 shadow">
            <h2 class="text-2xl font-semibold text-gray-900 mb-6">Recommended Configuration</h2>

            <div [class]="'inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold mb-6 ' + getConfidenceColor(recommendation.confidence)">
              <span>Confidence:</span>
              <span>{{ (recommendation.confidence * 100).toFixed(0) }}%</span>
            </div>

            <div class="grid grid-cols-2 gap-4 mb-8">
              <div class="flex items-center gap-4 p-4 bg-gray-50 rounded-md">
                <div class="text-3xl">🔧</div>
                <div>
                  <h4 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
                    Executor Cores
                  </h4>
                  <p class="text-2xl font-bold text-gray-900">
                    {{ recommendation.configuration.executor_cores }}
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
                    {{ formatMemory(recommendation.configuration.executor_memory_mb) }}
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
                    {{ recommendation.configuration.num_executors }}
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
                    {{ formatMemory(recommendation.configuration.driver_memory_mb) }}
                  </p>
                </div>
              </div>
            </div>

            <!-- Predicted Metrics -->
            @if (recommendation.predicted_metrics) {
              <div class="mb-8 p-4 bg-blue-50 rounded-md">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Predicted Performance</h3>
                <div class="grid grid-cols-2 gap-4">
                  <div class="flex justify-between">
                    <span class="text-blue-800 font-medium">Duration:</span>
                    <span class="text-gray-900 font-bold">
                      ~{{ recommendation.predicted_metrics.duration_minutes }} minutes
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-blue-800 font-medium">Estimated Cost:</span>
                    <span class="text-gray-900 font-bold">
                      \${{ recommendation.predicted_metrics.cost_usd.toFixed(2) }}
                    </span>
                  </div>
                </div>
              </div>
            }

            <!-- Metadata -->
            <div class="mb-8 p-4 bg-gray-50 rounded-md">
              <div class="text-gray-700 text-sm mb-2">
                <strong>Method:</strong> {{ recommendation.method | titlecase }}
              </div>
              @if (recommendation.metadata?.similar_jobs) {
                <div class="text-gray-700 text-sm">
                  <strong>Based on:</strong> {{ recommendation.metadata?.similar_jobs?.length }} similar jobs
                </div>
              }
            </div>

            <!-- Spark Configuration -->
            <div>
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Spark Configuration</h3>
              <div class="bg-gray-800 rounded-md p-4 overflow-x-auto">
                <pre class="text-gray-200 text-sm font-mono leading-relaxed"><code>spark-submit \
  --num-executors {{ recommendation.configuration.num_executors }} \
  --executor-cores {{ recommendation.configuration.executor_cores }} \
  --executor-memory {{ formatMemory(recommendation.configuration.executor_memory_mb) }} \
  --driver-memory {{ formatMemory(recommendation.configuration.driver_memory_mb) }} \
  your-spark-app.jar</code></pre>
              </div>
            </div>
          </div>
        }

        <!-- Loading State -->
        @if (loading) {
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
  request: RecommendationRequest = {
    input_size_gb: 10,
    job_type: 'etl'
  };

  recommendation: RecommendationResponse | null = null;
  loading = false;
  error: string | null = null;

  jobTypes = ['etl', 'ml', 'sql', 'streaming'];
  methods = ['similarity', 'ml', 'rule_based', 'hybrid'];

  constructor(private apiService: ApiService) {}

  getRecommendation(): void {
    if (!this.request.input_size_gb || this.request.input_size_gb <= 0) {
      this.error = 'Please enter a valid input size';
      return;
    }

    this.loading = true;
    this.error = null;
    this.recommendation = null;

    this.apiService.getRecommendation(this.request).subscribe({
      next: (response) => {
        this.recommendation = response;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to get recommendation. Please try again.';
        console.error('Error getting recommendation:', err);
        this.loading = false;
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
    this.request = {
      input_size_gb: 10,
      job_type: 'etl'
    };
    this.recommendation = null;
    this.error = null;
  }
}
