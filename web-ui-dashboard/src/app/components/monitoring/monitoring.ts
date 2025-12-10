import { Component, OnInit, OnDestroy, ChangeDetectionStrategy, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { interval, Subscription } from 'rxjs';

interface Application {
  app_id: string;
  app_name: string;
  status: string;
  progress: number;
  active_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  current_memory_mb: number;
  current_cpu_percent: number;
  executors: number;
  duration_seconds: number;
  last_updated: string;
}

interface Alert {
  id: string;
  app_id: string;
  severity: string;
  title: string;
  message: string;
  created_at: string;
  acknowledged: boolean;
}

@Component({
  selector: 'app-monitoring',
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="p-8 max-w-7xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Real-Time Monitoring</h1>
        <p class="text-gray-600">Monitor your Spark applications in real-time</p>
      </header>

      <!-- Connection Status -->
      <div class="mb-6 flex items-center gap-4">
        <div class="flex items-center gap-2">
          <span [class]="'w-3 h-3 rounded-full ' + (connected() ? 'bg-green-500' : 'bg-red-500')"></span>
          <span class="text-sm text-gray-600">{{ connected() ? 'Connected' : 'Disconnected' }}</span>
        </div>
        <button
          (click)="toggleConnection()"
          [class]="'px-4 py-2 rounded-md font-medium transition-colors ' + (connected() ? 'bg-red-500 text-white hover:bg-red-600' : 'bg-green-500 text-white hover:bg-green-600')"
        >
          {{ connected() ? 'Disconnect' : 'Connect' }}
        </button>
      </div>

      <!-- Active Alerts -->
      @if (activeAlerts().length > 0) {
        <div class="mb-8">
          <h2 class="text-xl font-semibold text-gray-900 mb-4">Active Alerts</h2>
          <div class="space-y-3">
            @for (alert of activeAlerts(); track alert.id) {
              <div [class]="'p-4 rounded-lg border-l-4 ' + getAlertClass(alert.severity)">
                <div class="flex justify-between items-start">
                  <div>
                    <h3 class="font-semibold">{{ alert.title }}</h3>
                    <p class="text-sm text-gray-600 mt-1">{{ alert.message }}</p>
                    <p class="text-xs text-gray-500 mt-2">App: {{ alert.app_id }} | {{ alert.created_at | date:'short' }}</p>
                  </div>
                  <button
                    (click)="acknowledgeAlert(alert.id)"
                    class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300 transition-colors"
                  >
                    Acknowledge
                  </button>
                </div>
              </div>
            }
          </div>
        </div>
      }

      <!-- Statistics Cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4">
          <div class="text-4xl">📱</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Running Apps</h3>
            <p class="text-3xl font-bold text-gray-900">{{ runningApps() }}</p>
          </div>
        </div>

        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4">
          <div class="text-4xl">⚡</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Active Tasks</h3>
            <p class="text-3xl font-bold text-gray-900">{{ totalActiveTasks() }}</p>
          </div>
        </div>

        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4 border-l-4 border-yellow-500">
          <div class="text-4xl">⚠️</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Active Alerts</h3>
            <p class="text-3xl font-bold text-gray-900">{{ activeAlerts().length }}</p>
          </div>
        </div>

        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4">
          <div class="text-4xl">🖥️</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Total Executors</h3>
            <p class="text-3xl font-bold text-gray-900">{{ totalExecutors() }}</p>
          </div>
        </div>
      </div>

      <!-- Applications Table -->
      <div class="bg-white rounded-lg p-6 shadow">
        <h2 class="text-2xl font-semibold text-gray-900 mb-6">Monitored Applications</h2>

        <div class="overflow-x-auto">
          <table class="w-full border-collapse">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">App ID</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Name</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Status</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Progress</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Tasks</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">CPU</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Memory</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Executors</th>
              </tr>
            </thead>
            <tbody>
              @for (app of applications(); track app.app_id) {
                <tr class="hover:bg-gray-50 transition-colors">
                  <td class="px-4 py-4 border-b border-gray-200 font-mono text-xs text-gray-700">{{ app.app_id }}</td>
                  <td class="px-4 py-4 border-b border-gray-200 font-medium text-sm text-gray-900">{{ app.app_name }}</td>
                  <td class="px-4 py-4 border-b border-gray-200">
                    <span [class]="'inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase ' + getStatusClass(app.status)">
                      {{ app.status }}
                    </span>
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200">
                    <div class="flex items-center gap-2">
                      <div class="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          class="h-full bg-blue-500 transition-all duration-300"
                          [style.width.%]="app.progress * 100"
                        ></div>
                      </div>
                      <span class="text-xs text-gray-600">{{ (app.progress * 100).toFixed(0) }}%</span>
                    </div>
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">
                    <span class="text-blue-600">{{ app.active_tasks }}</span> /
                    <span class="text-green-600">{{ app.completed_tasks }}</span> /
                    <span class="text-red-600">{{ app.failed_tasks }}</span>
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">{{ app.current_cpu_percent.toFixed(1) }}%</td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">{{ formatMemory(app.current_memory_mb) }}</td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">{{ app.executors }}</td>
                </tr>
              }
            </tbody>
          </table>

          @if (applications().length === 0) {
            <div class="text-center py-12 text-gray-600">
              <p>No applications being monitored. Connect to start monitoring.</p>
            </div>
          }
        </div>
      </div>
    </div>
  `
})
export class Monitoring implements OnInit, OnDestroy {
  private refreshSubscription?: Subscription;
  private readonly REFRESH_INTERVAL = 5000;

  protected readonly connected = signal(false);
  protected readonly applications = signal<Application[]>([]);
  protected readonly activeAlerts = signal<Alert[]>([]);

  protected readonly runningApps = computed(() =>
    this.applications().filter(a => a.status === 'running').length
  );

  protected readonly totalActiveTasks = computed(() =>
    this.applications().reduce((sum, a) => sum + a.active_tasks, 0)
  );

  protected readonly totalExecutors = computed(() =>
    this.applications().reduce((sum, a) => sum + a.executors, 0)
  );

  ngOnInit(): void {
    // Load sample data for demo
    this.loadSampleData();
  }

  ngOnDestroy(): void {
    this.stopRefresh();
  }

  toggleConnection(): void {
    this.connected.update(c => !c);
    if (this.connected()) {
      this.startRefresh();
    } else {
      this.stopRefresh();
    }
  }

  acknowledgeAlert(alertId: string): void {
    this.activeAlerts.update(alerts =>
      alerts.filter(a => a.id !== alertId)
    );
  }

  private startRefresh(): void {
    this.refreshSubscription = interval(this.REFRESH_INTERVAL).subscribe(() => {
      this.updateMetrics();
    });
  }

  private stopRefresh(): void {
    this.refreshSubscription?.unsubscribe();
  }

  private loadSampleData(): void {
    this.applications.set([
      {
        app_id: 'app-20241210-001',
        app_name: 'ETL Pipeline - Sales Data',
        status: 'running',
        progress: 0.65,
        active_tasks: 24,
        completed_tasks: 156,
        failed_tasks: 2,
        current_memory_mb: 12288,
        current_cpu_percent: 78.5,
        executors: 8,
        duration_seconds: 1234,
        last_updated: new Date().toISOString()
      },
      {
        app_id: 'app-20241210-002',
        app_name: 'ML Training Job',
        status: 'running',
        progress: 0.32,
        active_tasks: 48,
        completed_tasks: 89,
        failed_tasks: 0,
        current_memory_mb: 24576,
        current_cpu_percent: 92.3,
        executors: 16,
        duration_seconds: 567,
        last_updated: new Date().toISOString()
      }
    ]);

    this.activeAlerts.set([
      {
        id: 'alert-001',
        app_id: 'app-20241210-001',
        severity: 'warning',
        title: 'High GC Time',
        message: 'GC time is 12.5%, exceeding 10% threshold',
        created_at: new Date().toISOString(),
        acknowledged: false
      }
    ]);
  }

  private updateMetrics(): void {
    this.applications.update(apps =>
      apps.map(app => ({
        ...app,
        progress: Math.min(1, app.progress + Math.random() * 0.05),
        active_tasks: Math.max(0, app.active_tasks + Math.floor(Math.random() * 10) - 5),
        completed_tasks: app.completed_tasks + Math.floor(Math.random() * 5),
        current_cpu_percent: Math.min(100, Math.max(0, app.current_cpu_percent + (Math.random() - 0.5) * 10)),
        current_memory_mb: Math.max(0, app.current_memory_mb + (Math.random() - 0.5) * 1024),
        last_updated: new Date().toISOString()
      }))
    );
  }

  getStatusClass(status: string): string {
    if (status === 'running') return 'bg-blue-200 text-blue-900';
    if (status === 'completed') return 'bg-green-200 text-green-900';
    if (status === 'failed') return 'bg-red-200 text-red-900';
    return 'bg-gray-200 text-gray-900';
  }

  getAlertClass(severity: string): string {
    if (severity === 'critical') return 'bg-red-50 border-red-500';
    if (severity === 'error') return 'bg-orange-50 border-orange-500';
    if (severity === 'warning') return 'bg-yellow-50 border-yellow-500';
    return 'bg-blue-50 border-blue-500';
  }

  formatMemory(mb: number): string {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  }
}
