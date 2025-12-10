import { Component, OnInit, ChangeDetectionStrategy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface CostEstimate {
  job_id: string;
  total_cost: number;
  breakdown: {
    resource_type: string;
    total_cost: number;
    quantity: number;
    unit: string;
  }[];
  instance_type: string;
  cloud_provider: string;
  estimated_duration_hours: number;
  spot_savings: number;
  reserved_savings: number;
  recommendations: string[];
}

interface ProviderComparison {
  provider: string;
  instance_type: string;
  hourly_price: number;
  total_cost: number;
  vcpus: number;
  memory_gb: number;
}

@Component({
  selector: 'app-cost',
  imports: [CommonModule, FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="p-8 max-w-7xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">Cost Optimization</h1>
        <p class="text-gray-600">Estimate and optimize costs for your Spark jobs across cloud providers</p>
      </header>

      <!-- Cost Estimator Form -->
      <div class="bg-white rounded-lg p-6 shadow mb-8">
        <h2 class="text-xl font-semibold text-gray-900 mb-4">Cost Estimator</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Executors</label>
            <input
              type="number"
              [(ngModel)]="estimatorConfig.executors"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Cores/Executor</label>
            <input
              type="number"
              [(ngModel)]="estimatorConfig.cores"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Memory (MB)</label>
            <input
              type="number"
              [(ngModel)]="estimatorConfig.memoryMb"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Duration (hours)</label>
            <input
              type="number"
              [(ngModel)]="estimatorConfig.durationHours"
              step="0.5"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Cloud Provider</label>
            <select
              [(ngModel)]="estimatorConfig.provider"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            >
              <option value="aws">AWS</option>
              <option value="gcp">GCP</option>
              <option value="azure">Azure</option>
            </select>
          </div>
          <div class="flex items-end">
            <button
              (click)="calculateEstimate()"
              class="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors font-medium"
            >
              Estimate Cost
            </button>
          </div>
        </div>
      </div>

      <!-- Cost Estimate Result -->
      @if (currentEstimate()) {
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <!-- Cost Summary -->
          <div class="bg-white rounded-lg p-6 shadow">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Cost Summary</h2>

            <div class="text-center mb-6">
              <p class="text-sm text-gray-600 uppercase tracking-wide">Estimated Total Cost</p>
              <p class="text-5xl font-bold text-gray-900 mt-2">\${{ currentEstimate()!.total_cost.toFixed(2) }}</p>
              <p class="text-sm text-gray-500 mt-1">{{ currentEstimate()!.cloud_provider | uppercase }} | {{ currentEstimate()!.estimated_duration_hours }}h</p>
            </div>

            <div class="grid grid-cols-2 gap-4 mb-6">
              <div class="bg-green-50 p-4 rounded-lg text-center">
                <p class="text-xs text-green-700 uppercase font-medium">Spot Savings</p>
                <p class="text-2xl font-bold text-green-600">\${{ currentEstimate()!.spot_savings.toFixed(2) }}</p>
              </div>
              <div class="bg-blue-50 p-4 rounded-lg text-center">
                <p class="text-xs text-blue-700 uppercase font-medium">Reserved Savings</p>
                <p class="text-2xl font-bold text-blue-600">\${{ currentEstimate()!.reserved_savings.toFixed(2) }}</p>
              </div>
            </div>

            <h3 class="text-sm font-semibold text-gray-700 uppercase mb-3">Cost Breakdown</h3>
            <div class="space-y-2">
              @for (item of currentEstimate()!.breakdown; track item.resource_type) {
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                  <div>
                    <span class="font-medium text-gray-800 capitalize">{{ item.resource_type }}</span>
                    <span class="text-xs text-gray-500 ml-2">{{ item.quantity.toFixed(1) }} {{ item.unit }}</span>
                  </div>
                  <span class="font-semibold text-gray-900">\${{ item.total_cost.toFixed(4) }}</span>
                </div>
              }
            </div>
          </div>

          <!-- Recommendations -->
          <div class="bg-white rounded-lg p-6 shadow">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Recommendations</h2>

            <div class="space-y-3">
              @for (rec of currentEstimate()!.recommendations; track rec) {
                <div class="flex items-start gap-3 p-3 bg-blue-50 rounded-lg">
                  <span class="text-blue-500 text-xl">💡</span>
                  <p class="text-sm text-gray-700">{{ rec }}</p>
                </div>
              }

              @if (currentEstimate()!.recommendations.length === 0) {
                <p class="text-gray-500 text-center py-4">No specific recommendations</p>
              }
            </div>
          </div>
        </div>
      }

      <!-- Provider Comparison -->
      <div class="bg-white rounded-lg p-6 shadow mb-8">
        <h2 class="text-xl font-semibold text-gray-900 mb-4">Cloud Provider Comparison</h2>
        <p class="text-sm text-gray-600 mb-4">Compare costs across AWS, GCP, and Azure for your configuration</p>

        <div class="overflow-x-auto">
          <table class="w-full border-collapse">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Provider</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Instance Type</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">vCPUs</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Memory</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Hourly Price</th>
                <th class="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wide border-b-2 border-gray-200">Total Cost</th>
              </tr>
            </thead>
            <tbody>
              @for (comp of providerComparison(); track comp.provider) {
                <tr class="hover:bg-gray-50 transition-colors" [class.bg-green-50]="comp === cheapestProvider()">
                  <td class="px-4 py-4 border-b border-gray-200">
                    <div class="flex items-center gap-2">
                      <span class="text-xl">{{ getProviderIcon(comp.provider) }}</span>
                      <span class="font-medium text-gray-900 uppercase">{{ comp.provider }}</span>
                      @if (comp === cheapestProvider()) {
                        <span class="px-2 py-0.5 bg-green-200 text-green-800 text-xs rounded-full font-medium">Cheapest</span>
                      }
                    </div>
                  </td>
                  <td class="px-4 py-4 border-b border-gray-200 font-mono text-sm text-gray-700">{{ comp.instance_type }}</td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">{{ comp.vcpus }}</td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">{{ comp.memory_gb }} GB</td>
                  <td class="px-4 py-4 border-b border-gray-200 text-sm text-gray-700">\${{ comp.hourly_price.toFixed(4) }}/hr</td>
                  <td class="px-4 py-4 border-b border-gray-200 font-semibold text-gray-900">\${{ comp.total_cost.toFixed(2) }}</td>
                </tr>
              }
            </tbody>
          </table>
        </div>
      </div>

      <!-- Monthly Cost Projection -->
      <div class="bg-white rounded-lg p-6 shadow">
        <h2 class="text-xl font-semibold text-gray-900 mb-4">Monthly Cost Projection</h2>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="p-6 bg-gray-50 rounded-lg text-center">
            <h3 class="text-sm font-medium text-gray-600 uppercase mb-2">On-Demand (24/7)</h3>
            <p class="text-3xl font-bold text-gray-900">\${{ monthlyOnDemand().toFixed(0) }}</p>
            <p class="text-xs text-gray-500 mt-1">720 hours/month</p>
          </div>

          <div class="p-6 bg-green-50 rounded-lg text-center">
            <h3 class="text-sm font-medium text-green-700 uppercase mb-2">With Spot Instances</h3>
            <p class="text-3xl font-bold text-green-600">\${{ monthlySpot().toFixed(0) }}</p>
            <p class="text-xs text-green-600 mt-1">Save {{ spotSavingsPercent() }}%</p>
          </div>

          <div class="p-6 bg-blue-50 rounded-lg text-center">
            <h3 class="text-sm font-medium text-blue-700 uppercase mb-2">With Reserved (1yr)</h3>
            <p class="text-3xl font-bold text-blue-600">\${{ monthlyReserved().toFixed(0) }}</p>
            <p class="text-xs text-blue-600 mt-1">Save {{ reservedSavingsPercent() }}%</p>
          </div>
        </div>
      </div>
    </div>
  `
})
export class Cost implements OnInit {
  protected readonly currentEstimate = signal<CostEstimate | null>(null);
  protected readonly providerComparison = signal<ProviderComparison[]>([]);

  protected estimatorConfig = {
    executors: 10,
    cores: 4,
    memoryMb: 8192,
    durationHours: 2,
    provider: 'aws'
  };

  protected readonly cheapestProvider = computed(() => {
    const comparison = this.providerComparison();
    if (comparison.length === 0) return null;
    return comparison.reduce((min, p) => p.total_cost < min.total_cost ? p : min);
  });

  protected readonly monthlyOnDemand = computed(() => {
    const estimate = this.currentEstimate();
    if (!estimate) return 0;
    const hourlyRate = estimate.total_cost / estimate.estimated_duration_hours;
    return hourlyRate * 720;
  });

  protected readonly monthlySpot = computed(() => {
    return this.monthlyOnDemand() * 0.3;
  });

  protected readonly monthlyReserved = computed(() => {
    return this.monthlyOnDemand() * 0.6;
  });

  protected readonly spotSavingsPercent = computed(() => {
    return 70;
  });

  protected readonly reservedSavingsPercent = computed(() => {
    return 40;
  });

  ngOnInit(): void {
    this.calculateEstimate();
    this.loadProviderComparison();
  }

  calculateEstimate(): void {
    const { executors, cores, memoryMb, durationHours, provider } = this.estimatorConfig;

    // Simulated cost calculation
    const computeCost = executors * cores * durationHours * 0.05;
    const memoryCost = (executors * memoryMb / 1024) * durationHours * 0.005;
    const totalCost = computeCost + memoryCost;

    this.currentEstimate.set({
      job_id: `estimate-${Date.now()}`,
      total_cost: totalCost,
      breakdown: [
        { resource_type: 'compute', total_cost: computeCost, quantity: executors * cores, unit: 'vCPU-hours' },
        { resource_type: 'memory', total_cost: memoryCost, quantity: executors * memoryMb / 1024, unit: 'GB-hours' }
      ],
      instance_type: 'on_demand',
      cloud_provider: provider,
      estimated_duration_hours: durationHours,
      spot_savings: totalCost * 0.7,
      reserved_savings: totalCost * 0.4,
      recommendations: [
        `Consider using spot instances to save \$${(totalCost * 0.7).toFixed(2)} (70%)`,
        'For recurring jobs, reserved instances could provide significant savings',
        executors > 10 ? 'Consider reducing executor count if CPU utilization is low' : ''
      ].filter(r => r)
    });

    this.loadProviderComparison();
  }

  loadProviderComparison(): void {
    const { executors, durationHours } = this.estimatorConfig;

    this.providerComparison.set([
      {
        provider: 'aws',
        instance_type: 'm5.xlarge',
        vcpus: 4,
        memory_gb: 16,
        hourly_price: 0.192,
        total_cost: 0.192 * executors * durationHours
      },
      {
        provider: 'gcp',
        instance_type: 'n1-standard-4',
        vcpus: 4,
        memory_gb: 15,
        hourly_price: 0.190,
        total_cost: 0.190 * executors * durationHours
      },
      {
        provider: 'azure',
        instance_type: 'Standard_D4s_v3',
        vcpus: 4,
        memory_gb: 16,
        hourly_price: 0.192,
        total_cost: 0.192 * executors * durationHours
      }
    ]);
  }

  getProviderIcon(provider: string): string {
    if (provider === 'aws') return '☁️';
    if (provider === 'gcp') return '🌐';
    if (provider === 'azure') return '💠';
    return '☁️';
  }
}
