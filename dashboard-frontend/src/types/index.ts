// Common data types for the LDTS Reranker Dashboard

export interface Tool {
  id: string;
  name: string;
  description: string;
  source: string;
  category?: string;
  tags?: string[];
}

export interface SearchQuery {
  query: string;
  filters?: {
    category?: string;
    source?: string;
    tags?: string[];
  };
  limit?: number;
}

export interface SearchResult {
  tool: Tool;
  score: number;
  rank: number;
  reasoning?: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  metadata: {
    total_found: number;
    search_time: number;
    reranker_used?: string;
    original_results?: SearchResult[];
  };
}

export interface RerankerConfig {
  enabled: boolean;
  model: string;
  provider: string;
  parameters: {
    temperature?: number;
    max_tokens?: number;
    [key: string]: any;
  };
}

export interface ConfigurationPreset {
  id: string;
  name: string;
  description: string;
  config: RerankerConfig;
  isDefault?: boolean;
}

export interface EvaluationRating {
  result_id: string;
  query: string;
  relevance_score: number; // 1-5
  usefulness_score: number; // 1-5
  notes?: string;
  timestamp: Date;
}

export interface Analytics {
  total_searches: number;
  average_response_time: number;
  top_queries: { query: string; count: number }[];
  reranker_performance: {
    improvement_rate: number;
    accuracy_gain: number;
  };
  usage_stats: {
    daily_searches: { date: string; count: number }[];
    model_usage: { model: string; count: number }[];
  };
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}