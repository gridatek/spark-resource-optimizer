import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SparkJob, JobListResponse } from '../models/job.model';
import {
  RecommendationRequest,
  RecommendationResponse,
  JobAnalysis
} from '../models/recommendation.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  // Health Check
  healthCheck(): Observable<{ status: string; service: string }> {
    return this.http.get<{ status: string; service: string }>(`${this.apiUrl}/health`);
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
    return this.http.get<JobListResponse>(`${this.apiUrl}/jobs`, { params: httpParams });
  }

  getJobDetails(appId: string): Observable<SparkJob> {
    return this.http.get<SparkJob>(`${this.apiUrl}/jobs/${appId}`);
  }

  analyzeJob(appId: string): Observable<JobAnalysis> {
    return this.http.get<JobAnalysis>(`${this.apiUrl}/analyze/${appId}`);
  }

  // Recommendations
  getRecommendation(request: RecommendationRequest): Observable<RecommendationResponse> {
    return this.http.post<RecommendationResponse>(`${this.apiUrl}/recommend`, request);
  }

  // Feedback
  submitFeedback(data: {
    recommendation_id: number;
    actual_performance?: any;
    satisfaction_score?: number;
    comments?: string;
  }): Observable<{ status: string; message: string }> {
    return this.http.post<{ status: string; message: string }>(`${this.apiUrl}/feedback`, data);
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
    return this.http.post<any>(`${this.apiUrl}/collect`, data);
  }
}
