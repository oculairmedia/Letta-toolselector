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

export interface EmbeddingConfig {
  embedding_model: string;
  embedding_provider: string;
  dimensions?: number;
  batch_size?: number;
}

export interface ExtendedTool extends Tool {
  mcp_server_name?: string;
  last_updated?: string;
  registered_in_letta?: boolean;
  embedding_id?: string;
}

export interface ToolBrowseResponse {
  tools: ExtendedTool[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

export interface ToolDetailResponse extends ExtendedTool {
  json_schema?: any;
  parameters?: any;
  metadata?: Record<string, any>;
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
  search_count: number;
  total_evaluations: number;
  avg_rating: number;
  date_range: {
    start: string | null;
    end: string | null;
  };
  recent_searches: Array<{
    query: string;
    timestamp: string;
    result_count?: number;
  }>;
  top_tools: Array<{
    tool_name: string;
    usage_count: number;
  }>;
  tool_usage: Record<string, number>;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}