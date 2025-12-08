import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, switchMap, shareReplay, map, catchError, of } from 'rxjs';
import { SparkJob, JobListResponse } from '../models/job.model';
import {
  RecommendationRequest,
  RecommendationResponse,
  JobAnalysis
} from '../models/recommendation.model';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private config$: Observable<{ apiUrl: string }>;

  constructor(private http: HttpClient) {
    this.config$ = this.http.get<{ apiUrl: string }>('/config.json').pipe(
      catchError(error => {
        console.error('Failed to load config.json, using default', error);
        return of({ apiUrl: 'http://localhost:8080' });
      }),
      shareReplay(1)
    );
  }

  private getApiUrl(): Observable<string> {
    return this.config$.pipe(
      map(config => config.apiUrl)
    );
  }

  // Health Check
  healthCheck(): Observable<{ status: string; service: string }> {
    return this.getApiUrl().pipe(
      switchMap(apiUrl => this.http.get<{ status: string; service: string }>(`${apiUrl}/health`))
    );
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
    return this.getApiUrl().pipe(
      switchMap(apiUrl => this.http.get<JobListResponse>(`${apiUrl}/api/v1/jobs`, { params: httpParams }))
    );
  }

  getJobDetails(appId: string): Observable<SparkJob> {
    return this.getApiUrl().pipe(
      switchMap(apiUrl => this.http.get<SparkJob>(`${apiUrl}/api/v1/jobs/${appId}`))
    );
  }

  analyzeJob(appId: string): Observable<JobAnalysis> {
    return this.getApiUrl().pipe(
      switchMap(apiUrl => this.http.get<JobAnalysis>(`${apiUrl}/api/v1/jobs/${appId}/analyze`))
    );
  }

  // Recommendations
  getRecommendation(request: RecommendationRequest): Observable<RecommendationResponse> {
    return this.getApiUrl().pipe(
      switchMap(apiUrl => {
        console.log('Making recommendation request to:', `${apiUrl}/api/v1/recommend`);
        console.log('Request payload:', request);
        return this.http.post<RecommendationResponse>(`${apiUrl}/api/v1/recommend`, request);
      })
    );
  }

  // Feedback
  submitFeedback(data: {
    recommendation_id: number;
    actual_performance?: any;
    satisfaction_score?: number;
    comments?: string;
  }): Observable<{ status: string; message: string }> {
    return this.getApiUrl().pipe(
      switchMap(apiUrl => this.http.post<{ status: string; message: string }>(`${apiUrl}/api/v1/feedback`, data))
    );
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
    return this.getApiUrl().pipe(
      switchMap(apiUrl => this.http.post<any>(`${apiUrl}/api/v1/collect`, data))
    );
  }
}
