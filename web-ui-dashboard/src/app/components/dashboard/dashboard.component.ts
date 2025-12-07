import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { SparkJob } from '../../models/job.model';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class Dashboard implements OnInit {
  recentJobs: SparkJob[] = [];
  totalJobs = 0;
  loading = false;
  error: string | null = null;

  // Statistics
  stats = {
    totalJobs: 0,
    successfulJobs: 0,
    failedJobs: 0,
    avgDuration: 0
  };

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadDashboardData();
  }

  loadDashboardData(): void {
    this.loading = true;
    this.error = null;

    this.apiService.getJobs({ limit: 10 }).subscribe({
      next: (response) => {
        this.recentJobs = response.jobs;
        this.totalJobs = response.total;
        this.calculateStats();
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load dashboard data';
        console.error('Error loading dashboard:', err);
        this.loading = false;
      }
    });
  }

  calculateStats(): void {
    this.stats.totalJobs = this.totalJobs;
    this.stats.successfulJobs = this.recentJobs.filter(j => j.status === 'completed').length;
    this.stats.failedJobs = this.recentJobs.filter(j => j.status === 'failed').length;

    const durations = this.recentJobs
      .filter(j => j.duration_ms > 0)
      .map(j => j.duration_ms);

    this.stats.avgDuration = durations.length > 0
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : 0;
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
