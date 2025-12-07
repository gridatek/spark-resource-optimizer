export interface RecommendationRequest {
  input_size_gb: number;
  job_type?: 'etl' | 'ml' | 'sql' | 'streaming';
  app_name?: string;
  method?: 'similarity' | 'ml' | 'rule_based' | 'hybrid';
  additional_params?: Record<string, any>;
}

export interface RecommendationResponse {
  configuration: {
    executor_cores: number;
    executor_memory_mb: number;
    num_executors: number;
    driver_memory_mb: number;
  };
  predicted_metrics?: {
    duration_minutes: number;
    cost_usd: number;
  };
  confidence: number;
  method: string;
  metadata?: {
    similar_jobs?: Array<{
      app_id: string;
      similarity: number;
    }>;
    feature_importance?: Record<string, number>;
  };
}

export interface JobAnalysis {
  app_id: string;
  app_name: string;
  analysis: {
    resource_efficiency: {
      cpu_efficiency: number;
      memory_efficiency: number;
      io_efficiency: number;
    };
    bottlenecks: Array<{
      type: string;
      severity: 'low' | 'medium' | 'high';
      description: string;
      affected_stages?: number[];
    }>;
    issues: Array<{
      type: string;
      severity: 'low' | 'medium' | 'high';
      description: string;
      recommendation?: string;
    }>;
  };
  suggestions: Array<{
    category: string;
    suggestion: string;
    expected_improvement: string;
  }>;
}
