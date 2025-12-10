import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SparkJob, JobListResponse } from '../models/job.model';
import {
  RecommendationRequest,
  RecommendationResponse,
  JobAnalysis
} from '../models/recommendation.model';
import { ConfigService } from './config.service';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly configService = inject(ConfigService);

  private getApiUrl(): string {
    return this.configService.apiUrl;
  }

  // Health Check
  healthCheck(): Observable<{ status: string; service: string }> {
    return this.http.get<{ status: string; service: string }>(`${this.getApiUrl()}/health`);
  }

  // Jobs
  getJobs(params?: {
    limit?: number;
    offset?: number;
    app_name?: string;
    user?: string;
    date_from?: string;
    date_to?: string;
    status?: string;
  }): Observable<JobListResponse> {
    let httpParams = new HttpParams();
    if (params) {
      Object.keys(params).forEach(key => {
        const value = params[key as keyof typeof params];
        if (value !== undefined && value !== null) {
          httpParams = httpParams.set(key, value.toString());
        }
      });
    }
    return this.http.get<JobListResponse>(`${this.getApiUrl()}/api/v1/jobs`, { params: httpParams });
  }

  getJobDetails(appId: string): Observable<SparkJob> {
    return this.http.get<SparkJob>(`${this.getApiUrl()}/api/v1/jobs/${appId}`);
  }

  analyzeJob(appId: string): Observable<JobAnalysis> {
    return this.http.get<JobAnalysis>(`${this.getApiUrl()}/api/v1/jobs/${appId}/analyze`);
  }

  // Recommendations
  getRecommendation(request: RecommendationRequest): Observable<RecommendationResponse> {
    return this.http.post<RecommendationResponse>(`${this.getApiUrl()}/api/v1/recommend`, request);
  }

  // Feedback
  submitFeedback(data: {
    recommendation_id: number;
    actual_performance?: any;
    satisfaction_score?: number;
    comments?: string;
  }): Observable<{ status: string; message: string }> {
    return this.http.post<{ status: string; message: string }>(`${this.getApiUrl()}/api/v1/feedback`, data);
  }

  // Collection
  collectJobs(data: {
    source_type: string;
    source_path: string;
    config?: any;
  }): Observable<{
    status: string;
    jobs_collected: number;
    jobs_stored: number;
    errors: number;
  }> {
    return this.http.post<any>(`${this.getApiUrl()}/api/v1/collect`, data);
  }
}
