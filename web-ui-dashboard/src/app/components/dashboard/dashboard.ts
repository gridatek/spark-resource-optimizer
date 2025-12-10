import { Component, OnInit, OnDestroy, ChangeDetectionStrategy, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { SparkJob } from '../../models/job.model';
import { interval, Subscription } from 'rxjs';
import { switchMap } from 'rxjs/operators';

@Component({
  selector: 'app-dashboard',
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="p-8 max-w-7xl mx-auto">
      <header class="mb-8 flex justify-between items-center">
        <div>
          <h1 class="text-3xl font-bold text-gray-900 mb-2">Spark Resource Optimizer Dashboard</h1>
          <p class="text-gray-600">Monitor and optimize your Spark job configurations</p>
        </div>
        <div class="flex items-center gap-4">
          <button
            (click)="toggleAutoRefresh()"
            [class]="'px-4 py-2 rounded-md font-medium transition-colors ' + (autoRefreshEnabled() ? 'bg-green-500 text-white hover:bg-green-600' : 'bg-gray-200 text-gray-700 hover:bg-gray-300')"
          >
            {{ autoRefreshEnabled() ? '🔄 Auto-Refresh ON' : '⏸️ Auto-Refresh OFF' }}
          </button>
          @if (lastUpdated()) {
            <span class="text-sm text-gray-500">
              Last updated: {{ lastUpdated() | date:'short' }}
            </span>
          }
        </div>
      </header>

      <!-- Loading State -->
      @if (loading()) {
        <div class="text-center py-16">
          <div class="w-12 h-12 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p class="text-gray-600">Loading dashboard data...</p>
        </div>
      }

      <!-- Error State -->
      @if (error()) {
        <div class="text-center py-16">
          <p class="text-red-600 text-lg mb-4">{{ error() }}</p>
          <button
            (click)="loadDashboardData()"
            class="px-8 py-3 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
          >
            Retry
          </button>
        </div>
      }

      <!-- Dashboard Content -->
      @if (!loading() && !error()) {
        <div>
          <!-- Statistics Cards -->
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg p-6 shadow hover:shadow-md transition-shadow flex items-center gap-4">
              <div class="text-4xl">📊</div>
              <div>
                <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">
                  Total Jobs
                </h3>
                <p class="text-3xl font-bold text-gray-900">{{ stats().totalJobs }}</p>
              </div>
            </div>

            <div class="bg-white rounded-lg p-6 shadow hover:shadow-md transition-shadow flex items-center gap-4 border-l-4 border-green-500">
              <div class="text-4xl">✅</div>
              <div>
                <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">
                  Successful
                </h3>
                <p class="text-3xl font-bold text-gray-900">{{ stats().successfulJobs }}</p>
              </div>
            </div>

            <div class="bg-white rounded-lg p-6 shadow hover:shadow-md transition-shadow flex items-center gap-4 border-l-4 border-red-500">
              <div class="text-4xl">❌</div>
              <div>
                <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">
                  Failed
                </h3>
                <p class="text-3xl font-bold text-gray-900">{{ stats().failedJobs }}</p>
              </div>
            </div>

            <div class="bg-white rounded-lg p-6 shadow hover:shadow-md transition-shadow flex items-center gap-4">
              <div class="text-4xl">⏱️</div>
              <div>
                <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">
                  Avg Duration
                </h3>
                <p class="text-3xl font-bold text-gray-900">{{ formatDuration(stats().avgDuration) }}</p>
              </div>
            </div>
          </div>

          <!-- Recent Jobs Table -->
          <div class="bg-white rounded-lg p-6 shadow">
            <h2 class="text-2xl font-semibold text-gray-900 mb-6">Recent Jobs</h2>

            <div class="overflow-x-auto">
              <table class="w-full border-collapse">
                <thead class="bg-gray-50">
                  <tr>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      App ID
                    </th>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      App Name
                    </th>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      User
                    </th>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      Status
                    </th>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      Duration
                    </th>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      Input Data
                    </th>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      Executors
                    </th>
                    <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">
                      Start Time
                    </th>
                  </tr>
                </thead>
                <tbody>
                  @for (job of recentJobs(); track job.app_id) {
                    <tr class="hover:bg-gray-50 transition-colors">
                      <td class="px-4 py-4 border-b border-gray-200 font-mono text-xs text-gray-700">
                        {{ job.app_id }}
                      </td>
                      <td class="px-4 py-4 border-b border-gray-200 font-medium text-sm text-gray-900">
                        {{ job.app_name }}
                      </td>
                      <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">
                        {{ job.user }}
                      </td>
                      <td class="px-4 py-4 border-b border-gray-200">
                        <span [class]="'inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase ' + getStatusClass(job.status)">
                          {{ job.status }}
                        </span>
                      </td>
                      <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">
                        {{ formatDuration(job.duration_ms) }}
                      </td>
                      <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">
                        {{ formatBytes(job.metrics.input_bytes) }}
                      </td>
                      <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">
                        {{ job.configuration.num_executors }}
                      </td>
                      <td class="px-4 py-4 border-b border-gray-200 text-xs text-gray-600">
                        {{ job.start_time | date:'short' }}
                      </td>
                    </tr>
                  }
                </tbody>
              </table>

              @if (recentJobs().length === 0) {
                <div class="text-center py-12 text-gray-600">
                  <p>No jobs found. Start by collecting Spark job data.</p>
                </div>
              }
            </div>
          </div>
        </div>
      }
    </div>
  `
})
export class Dashboard implements OnInit, OnDestroy {
  private readonly apiService = inject(ApiService);
  private refreshSubscription?: Subscription;
  private readonly REFRESH_INTERVAL = 30000; // 30 seconds

  // Reactive state using signals
  protected readonly recentJobs = signal<SparkJob[]>([]);
  protected readonly totalJobs = signal(0);
  protected readonly loading = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly autoRefreshEnabled = signal(true);
  protected readonly lastUpdated = signal<Date | null>(null);

  // Computed statistics
  protected readonly stats = computed(() => {
    const jobs = this.recentJobs();
    const total = this.totalJobs();
    const successfulJobs = jobs.filter(j => j.status === 'completed').length;
    const failedJobs = jobs.filter(j => j.status === 'failed').length;
    const durations = jobs.filter(j => j.duration_ms > 0).map(j => j.duration_ms);
    const avgDuration = durations.length > 0
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : 0;

    return { totalJobs: total, successfulJobs, failedJobs, avgDuration };
  });

  ngOnInit(): void {
    this.loadDashboardData();
    this.startAutoRefresh();
  }

  ngOnDestroy(): void {
    this.stopAutoRefresh();
  }

  loadDashboardData(): void {
    this.loading.set(true);
    this.error.set(null);

    this.apiService.getJobs({ limit: 10 }).subscribe({
      next: (response) => {
        this.recentJobs.set(response.jobs);
        this.totalJobs.set(response.total);
        this.lastUpdated.set(new Date());
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Failed to load dashboard data');
        this.loading.set(false);
      }
    });
  }

  toggleAutoRefresh(): void {
    this.autoRefreshEnabled.update(enabled => !enabled);
    if (this.autoRefreshEnabled()) {
      this.startAutoRefresh();
    } else {
      this.stopAutoRefresh();
    }
  }

  private startAutoRefresh(): void {
    if (this.refreshSubscription) {
      return; // Already running
    }

    this.refreshSubscription = interval(this.REFRESH_INTERVAL)
      .pipe(
        switchMap(() => this.apiService.getJobs({ limit: 10 }))
      )
      .subscribe({
        next: (response) => {
          if (this.autoRefreshEnabled()) {
            this.recentJobs.set(response.jobs);
            this.totalJobs.set(response.total);
            this.lastUpdated.set(new Date());
          }
        },
        error: () => {
          // Silently handle auto-refresh errors
        }
      });
  }

  private stopAutoRefresh(): void {
    if (this.refreshSubscription) {
      this.refreshSubscription.unsubscribe();
      this.refreshSubscription = undefined;
    }
  }

  getStatusClass(status: string | undefined): string {
    if (status === 'completed') return 'bg-green-200 text-green-900';
    if (status === 'failed') return 'bg-red-200 text-red-900';
    if (status === 'running') return 'bg-blue-200 text-blue-900';
    return 'bg-gray-200 text-gray-900';
  }

  formatDuration(ms: number): string {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }

  formatBytes(bytes: number): string {
    const gb = bytes / (1024 ** 3);
    return `${gb.toFixed(2)} GB`;
  }
}
