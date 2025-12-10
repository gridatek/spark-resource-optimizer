import { Component, OnInit, ChangeDetectionStrategy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface TuningSession {
  session_id: string;
  app_id: string;
  app_name: string;
  strategy: string;
  target_metric: string;
  status: string;
  iterations: number;
  started_at: string;
  best_metric_value: number | null;
  adjustments_count: number;
}

interface Adjustment {
  parameter: string;
  old_value: any;
  new_value: any;
  reason: string;
  applied: boolean;
  timestamp: string;
}

@Component({
  selector: 'app-tuning',
  imports: [CommonModule, FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="p-8 max-w-7xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Auto-Tuning</h1>
        <p class="text-gray-600">Automatically optimize Spark configurations based on real-time metrics</p>
      </header>

      <!-- New Session Form -->
      <div class="bg-white rounded-lg p-6 shadow mb-8">
        <h2 class="text-xl font-semibold text-gray-900 mb-4">Start New Tuning Session</h2>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">App ID</label>
            <input
              type="text"
              [(ngModel)]="newSession.appId"
              placeholder="app-20241210-001"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Strategy</label>
            <select
              [(ngModel)]="newSession.strategy"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="conservative">Conservative</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Target Metric</label>
            <select
              [(ngModel)]="newSession.targetMetric"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="duration">Duration</option>
              <option value="cost">Cost</option>
              <option value="throughput">Throughput</option>
            </select>
          </div>
          <div class="flex items-end">
            <button
              (click)="startSession()"
              class="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors font-medium"
            >
              Start Session
            </button>
          </div>
        </div>
      </div>

      <!-- Statistics Cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4">
          <div class="text-4xl">🔧</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Active Sessions</h3>
            <p class="text-3xl font-bold text-gray-900">{{ activeSessions() }}</p>
          </div>
        </div>

        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4 border-l-4 border-green-500">
          <div class="text-4xl">✅</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Completed</h3>
            <p class="text-3xl font-bold text-gray-900">{{ completedSessions() }}</p>
          </div>
        </div>

        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4">
          <div class="text-4xl">📊</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Total Adjustments</h3>
            <p class="text-3xl font-bold text-gray-900">{{ totalAdjustments() }}</p>
          </div>
        </div>

        <div class="bg-white rounded-lg p-6 shadow flex items-center gap-4">
          <div class="text-4xl">📈</div>
          <div>
            <h3 class="text-xs font-medium text-gray-600 uppercase tracking-wide mb-2">Avg Improvement</h3>
            <p class="text-3xl font-bold text-gray-900">{{ avgImprovement() }}%</p>
          </div>
        </div>
      </div>

      <!-- Sessions Table -->
      <div class="bg-white rounded-lg p-6 shadow mb-8">
        <h2 class="text-2xl font-semibold text-gray-900 mb-6">Tuning Sessions</h2>

        <div class="overflow-x-auto">
          <table class="w-full border-collapse">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Session ID</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">App</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Strategy</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Target</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Status</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Iterations</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Best Value</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Actions</th>
              </tr>
            </thead>
            <tbody>
              @for (session of sessions(); track session.session_id) {
                <tr class="hover:bg-gray-50 transition-colors">
                  <td class="px-4 py-4 border-b border-gray-200 font-mono text-xs text-gray-700">{{ session.session_id }}</td>
                  <td class="px-4 py-4 border-b border-gray-200">
                    <div class="font-medium text-sm text-gray-900">{{ session.app_name }}</div>
                    <div class="text-xs text-gray-500">{{ session.app_id }}</div>
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200">
                    <span [class]="'inline-block px-2 py-1 rounded text-xs font-medium ' + getStrategyClass(session.strategy)">
                      {{ session.strategy }}
                    </span>
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700 capitalize">{{ session.target_metric }}</td>
                  <td class="px-4 py-4 border-b border-gray-200">
                    <span [class]="'inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase ' + getStatusClass(session.status)">
                      {{ session.status }}
                    </span>
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">{{ session.iterations }}</td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">
                    {{ session.best_metric_value !== null ? session.best_metric_value.toFixed(2) : '-' }}
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200">
                    <div class="flex gap-2">
                      @if (session.status === 'active') {
                        <button
                          (click)="pauseSession(session.session_id)"
                          class="px-2 py-1 text-xs bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200 transition-colors"
                        >
                          Pause
                        </button>
                        <button
                          (click)="endSession(session.session_id)"
                          class="px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors"
                        >
                          End
                        </button>
                      }
                      @if (session.status === 'paused') {
                        <button
                          (click)="resumeSession(session.session_id)"
                          class="px-2 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
                        >
                          Resume
                        </button>
                      }
                      <button
                        (click)="selectSession(session.session_id)"
                        class="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                      >
                        Details
                      </button>
                    </div>
                  </td>
                </tr>
              }
            </tbody>
          </table>

          @if (sessions().length === 0) {
            <div class="text-center py-12 text-gray-600">
              <p>No tuning sessions. Start a new session above.</p>
            </div>
          }
        </div>
      </div>

      <!-- Selected Session Details -->
      @if (selectedSession()) {
        <div class="bg-white rounded-lg p-6 shadow">
          <h2 class="text-xl font-semibold text-gray-900 mb-4">Session Details: {{ selectedSession()?.session_id }}</h2>

          <h3 class="text-lg font-medium text-gray-800 mb-3">Adjustments</h3>
          <div class="space-y-3">
            @for (adj of selectedAdjustments(); track adj.timestamp) {
              <div class="p-4 bg-gray-50 rounded-lg border">
                <div class="flex justify-between items-start">
                  <div>
                    <div class="font-medium text-gray-900">{{ adj.parameter }}</div>
                    <div class="text-sm text-gray-600 mt-1">
                      {{ adj.old_value }} → {{ adj.new_value }}
                    </div>
                    <div class="text-xs text-gray-500 mt-1">{{ adj.reason }}</div>
                  </div>
                  <span [class]="'px-2 py-1 rounded text-xs font-medium ' + (adj.applied ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600')">
                    {{ adj.applied ? 'Applied' : 'Pending' }}
                  </span>
                </div>
              </div>
            }

            @if (selectedAdjustments().length === 0) {
              <p class="text-gray-500 text-center py-4">No adjustments yet</p>
            }
          </div>
        </div>
      }
    </div>
  `
})
export class Tuning implements OnInit {
  protected readonly sessions = signal<TuningSession[]>([]);
  protected readonly selectedSession = signal<TuningSession | null>(null);
  protected readonly selectedAdjustments = signal<Adjustment[]>([]);

  protected newSession = {
    appId: '',
    strategy: 'moderate',
    targetMetric: 'duration'
  };

  protected readonly activeSessions = computed(() =>
    this.sessions().filter(s => s.status === 'active').length
  );

  protected readonly completedSessions = computed(() =>
    this.sessions().filter(s => s.status === 'completed').length
  );

  protected readonly totalAdjustments = computed(() =>
    this.sessions().reduce((sum, s) => sum + s.adjustments_count, 0)
  );

  protected readonly avgImprovement = computed(() => {
    const completed = this.sessions().filter(s => s.status === 'completed' && s.best_metric_value !== null);
    if (completed.length === 0) return 0;
    return 15; // Demo value
  });

  ngOnInit(): void {
    this.loadSampleData();
  }

  startSession(): void {
    if (!this.newSession.appId) return;

    const session: TuningSession = {
      session_id: `tune-${Date.now()}`,
      app_id: this.newSession.appId,
      app_name: `App ${this.newSession.appId}`,
      strategy: this.newSession.strategy,
      target_metric: this.newSession.targetMetric,
      status: 'active',
      iterations: 0,
      started_at: new Date().toISOString(),
      best_metric_value: null,
      adjustments_count: 0
    };

    this.sessions.update(sessions => [session, ...sessions]);
    this.newSession.appId = '';
  }

  pauseSession(sessionId: string): void {
    this.sessions.update(sessions =>
      sessions.map(s => s.session_id === sessionId ? { ...s, status: 'paused' } : s)
    );
  }

  resumeSession(sessionId: string): void {
    this.sessions.update(sessions =>
      sessions.map(s => s.session_id === sessionId ? { ...s, status: 'active' } : s)
    );
  }

  endSession(sessionId: string): void {
    this.sessions.update(sessions =>
      sessions.map(s => s.session_id === sessionId ? { ...s, status: 'completed' } : s)
    );
  }

  selectSession(sessionId: string): void {
    const session = this.sessions().find(s => s.session_id === sessionId);
    this.selectedSession.set(session || null);

    if (session) {
      this.selectedAdjustments.set([
        {
          parameter: 'spark.executor.memory',
          old_value: '4096MB',
          new_value: '6144MB',
          reason: 'Memory spilling detected (ratio: 0.15)',
          applied: true,
          timestamp: new Date().toISOString()
        },
        {
          parameter: 'spark.sql.shuffle.partitions',
          old_value: 200,
          new_value: 300,
          reason: 'High shuffle spill ratio (0.25)',
          applied: true,
          timestamp: new Date().toISOString()
        }
      ]);
    }
  }

  private loadSampleData(): void {
    this.sessions.set([
      {
        session_id: 'tune-1702234567890',
        app_id: 'app-20241210-001',
        app_name: 'ETL Pipeline - Sales Data',
        strategy: 'moderate',
        target_metric: 'duration',
        status: 'active',
        iterations: 5,
        started_at: new Date(Date.now() - 3600000).toISOString(),
        best_metric_value: 85.2,
        adjustments_count: 3
      },
      {
        session_id: 'tune-1702234123456',
        app_id: 'app-20241209-005',
        app_name: 'ML Training Pipeline',
        strategy: 'aggressive',
        target_metric: 'cost',
        status: 'completed',
        iterations: 12,
        started_at: new Date(Date.now() - 86400000).toISOString(),
        best_metric_value: 45.8,
        adjustments_count: 8
      }
    ]);
  }

  getStatusClass(status: string): string {
    if (status === 'active') return 'bg-blue-200 text-blue-900';
    if (status === 'completed') return 'bg-green-200 text-green-900';
    if (status === 'paused') return 'bg-yellow-200 text-yellow-900';
    if (status === 'failed') return 'bg-red-200 text-red-900';
    return 'bg-gray-200 text-gray-900';
  }

  getStrategyClass(strategy: string): string {
    if (strategy === 'conservative') return 'bg-green-100 text-green-700';
    if (strategy === 'moderate') return 'bg-blue-100 text-blue-700';
    if (strategy === 'aggressive') return 'bg-orange-100 text-orange-700';
    return 'bg-gray-100 text-gray-700';
  }
}
