export interface SparkJob {
  app_id: string;
  app_name: string;
  user: string;
  submit_time?: string;
  start_time: string;
  end_time?: string;
  duration_ms: number;
  status?: 'completed' | 'failed' | 'running';
  spark_version?: string;
  configuration: JobConfiguration;
  metrics: JobMetrics;
  estimated_cost?: number;
}

export interface JobConfiguration {
  executor_cores: number;
  executor_memory_mb: number;
  num_executors: number;
  driver_memory_mb: number;
}

export interface JobMetrics {
  total_tasks: number;
  failed_tasks: number;
  total_stages: number;
  failed_stages: number;
  input_bytes: number;
  output_bytes: number;
  shuffle_read_bytes: number;
  shuffle_write_bytes: number;
  cpu_time_ms?: number;
  memory_spilled_bytes?: number;
  disk_spilled_bytes?: number;
}

export interface JobListResponse {
  jobs: SparkJob[];
  total: number;
  limit: number;
  offset: number;
}
