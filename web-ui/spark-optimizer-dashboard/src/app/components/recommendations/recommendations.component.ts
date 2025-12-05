import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  RecommendationRequest,
  RecommendationResponse
} from '../../models/recommendation.model';

@Component({
  selector: 'app-recommendations',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './recommendations.component.html',
  styleUrls: ['./recommendations.component.css']
})
export class RecommendationsComponent {
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
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
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
