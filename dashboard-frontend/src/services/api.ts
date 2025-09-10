import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  SearchQuery,
  SearchResponse,
  RerankerConfig,
  ConfigurationPreset,
  EvaluationRating,
  Analytics,
  Tool,
  ApiResponse,
} from '../types';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        console.error('API Response Error:', error);
        if (error.response?.status === 401) {
          // Handle authentication errors
          console.error('Authentication failed');
        }
        return Promise.reject(error);
      }
    );
  }

  // Search endpoints
  async searchTools(query: SearchQuery): Promise<SearchResponse> {
    const response = await this.client.post<ApiResponse<SearchResponse>>('/tools/search', query);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Search failed');
    }
    return response.data.data!;
  }

  async searchWithReranking(query: SearchQuery, config: RerankerConfig): Promise<SearchResponse> {
    const response = await this.client.post<ApiResponse<SearchResponse>>('/tools/search/rerank', {
      query,
      reranker_config: config,
    });
    if (!response.data.success) {
      throw new Error(response.data.error || 'Reranked search failed');
    }
    return response.data.data!;
  }

  // Configuration endpoints
  async getRerankerConfig(): Promise<RerankerConfig> {
    const response = await this.client.get<ApiResponse<RerankerConfig>>('/config/reranker');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get reranker config');
    }
    return response.data.data!;
  }

  async updateRerankerConfig(config: RerankerConfig): Promise<void> {
    const response = await this.client.put<ApiResponse<void>>('/config/reranker', config);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to update reranker config');
    }
  }

  async testRerankerConnection(config: RerankerConfig): Promise<boolean> {
    const response = await this.client.post<ApiResponse<{ connected: boolean }>>('/config/reranker/test', config);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to test reranker connection');
    }
    return response.data.data!.connected;
  }

  async getOllamaModels(): Promise<{ models: Array<{ name: string; size: number; modified_at: string; details: any }>, base_url: string, total: number }> {
    const response = await this.client.get<ApiResponse<{ models: Array<{ name: string; size: number; modified_at: string; details: any }>, base_url: string, total: number }>>('/ollama/models');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get Ollama models');
    }
    return response.data.data!;
  }

  // Configuration presets
  async getConfigurationPresets(): Promise<ConfigurationPreset[]> {
    const response = await this.client.get<ApiResponse<ConfigurationPreset[]>>('/config/presets');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get configuration presets');
    }
    return response.data.data!;
  }

  async createConfigurationPreset(preset: Omit<ConfigurationPreset, 'id'>): Promise<ConfigurationPreset> {
    const response = await this.client.post<ApiResponse<ConfigurationPreset>>('/config/presets', preset);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to create configuration preset');
    }
    return response.data.data!;
  }

  async updateConfigurationPreset(id: string, preset: Partial<ConfigurationPreset>): Promise<ConfigurationPreset> {
    const response = await this.client.put<ApiResponse<ConfigurationPreset>>(`/config/presets/${id}`, preset);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to update configuration preset');
    }
    return response.data.data!;
  }

  async deleteConfigurationPreset(id: string): Promise<void> {
    const response = await this.client.delete<ApiResponse<void>>(`/config/presets/${id}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to delete configuration preset');
    }
  }

  // Evaluation endpoints
  async submitEvaluation(evaluation: Omit<EvaluationRating, 'timestamp'>): Promise<EvaluationRating> {
    const response = await this.client.post<ApiResponse<EvaluationRating>>('/evaluations', evaluation);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to submit evaluation');
    }
    return response.data.data!;
  }

  async getEvaluations(query?: string, limit?: number): Promise<EvaluationRating[]> {
    const params = new URLSearchParams();
    if (query) params.append('query', query);
    if (limit) params.append('limit', limit.toString());
    
    const response = await this.client.get<ApiResponse<EvaluationRating[]>>(`/evaluations?${params}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get evaluations');
    }
    return response.data.data!;
  }

  // Analytics endpoints
  async getAnalytics(dateRange?: { start: string; end: string }): Promise<Analytics> {
    const params = new URLSearchParams();
    if (dateRange) {
      params.append('start_date', dateRange.start);
      params.append('end_date', dateRange.end);
    }
    
    const response = await this.client.get<ApiResponse<Analytics>>(`/analytics?${params}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get analytics');
    }
    return response.data.data!;
  }

  // Tool management endpoints
  async getAllTools(): Promise<Tool[]> {
    const response = await this.client.get<ApiResponse<Tool[]>>('/tools');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get tools');
    }
    return response.data.data!;
  }

  async refreshToolIndex(): Promise<void> {
    const response = await this.client.post<ApiResponse<void>>('/tools/refresh');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to refresh tool index');
    }
  }

  // Advanced search endpoints
  async searchWithOverrides(query: string, options: {
    limit?: number;
    alpha?: number;
    distance_metric?: string;
    reranker_enabled?: boolean;
    reranker_model?: string;
  }): Promise<SearchResponse & { metadata: any }> {
    const params = new URLSearchParams();
    params.append('query', query);
    if (options.limit) params.append('limit', options.limit.toString());
    if (options.alpha !== undefined) params.append('alpha', options.alpha.toString());
    if (options.distance_metric) params.append('distance_metric', options.distance_metric);
    if (options.reranker_enabled !== undefined) params.append('reranker_enabled', options.reranker_enabled.toString());
    if (options.reranker_model) params.append('reranker_model', options.reranker_model);

    const response = await this.client.get<ApiResponse<SearchResponse & { metadata: any }>>(`/search/test?${params}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Search with overrides failed');
    }
    return response.data.data!;
  }

  async compareRankerConfigurations(query: string, configA: RerankerConfig, configB: RerankerConfig, limit?: number): Promise<{
    query: string;
    results_a: any[];
    results_b: any[];
    comparison_metrics: {
      total_results_a: number;
      total_results_b: number;
      avg_score_a: number;
      avg_score_b: number;
      top_5_overlap: number;
      rank_correlation: number;
    };
    config_a_name: string;
    config_b_name: string;
    timestamp: number;
  }> {
    const response = await this.client.post<ApiResponse<any>>('/rerank/compare', {
      query,
      config_a: configA,
      config_b: configB,
      limit: limit || 10
    });
    if (!response.data.success) {
      throw new Error(response.data.error || 'Reranker comparison failed');
    }
    return response.data.data!;
  }

  // Model discovery endpoints
  async getEmbeddingModels(): Promise<{
    models: Array<{
      id: string;
      name: string;
      provider: string;
      dimensions: number | string;
      max_tokens: number | string;
      cost_per_1k: number;
      recommended: boolean;
      size?: number;
      modified_at?: string;
    }>;
    total: number;
    providers: string[];
  }> {
    const response = await this.client.get<ApiResponse<any>>('/models/embedding');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get embedding models');
    }
    return response.data.data!;
  }

  async getRerankerModels(): Promise<{
    models: Array<{
      id: string;
      name: string;
      provider: string;
      type: string;
      cost_per_1k: number;
      recommended: boolean;
      size?: number;
      modified_at?: string;
    }>;
    total: number;
    providers: string[];
    types: string[];
  }> {
    const response = await this.client.get<ApiResponse<any>>('/models/reranker');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get reranker models');
    }
    return response.data.data!;
  }

  // Reranker Model Registry endpoints
  async getRerankerModelRegistry(): Promise<{
    models: Array<{
      id: string;
      name: string;
      provider: string;
      type: string;
      cost_per_1k: number;
      recommended: boolean;
      registered: boolean;
      validated: boolean;
      last_tested: string | null;
      test_status: string;
      registry_notes: string;
      configuration?: any;
      registered_at?: string;
      last_updated?: string;
      test_results?: any;
    }>;
    total: number;
    last_updated: string | null;
    providers: string[];
    types: string[];
    validated_count: number;
    pending_count: number;
  }> {
    const response = await this.client.get<ApiResponse<any>>('/reranker/models/registry');
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to get reranker model registry');
    }
    return response.data.data!;
  }

  async registerRerankerModel(model: {
    id: string;
    name: string;
    provider: string;
    type: string;
    cost_per_1k?: number;
    recommended?: boolean;
    notes?: string;
    configuration?: any;
    size?: number;
    modified_at?: string;
    dimensions?: number;
    max_tokens?: number;
    description?: string;
  }): Promise<any> {
    const response = await this.client.post<ApiResponse<any>>('/reranker/models/registry', model);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to register reranker model');
    }
    return response.data.data!;
  }

  async updateRerankerModel(modelId: string, updates: {
    name?: string;
    recommended?: boolean;
    registry_notes?: string;
    configuration?: any;
    validated?: boolean;
    test_status?: string;
    description?: string;
  }): Promise<any> {
    const response = await this.client.put<ApiResponse<any>>(`/reranker/models/registry/${modelId}`, updates);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to update reranker model');
    }
    return response.data.data!;
  }

  async testRerankerModel(modelId: string): Promise<{
    model_id: string;
    test_timestamp: string;
    connectivity: boolean;
    functionality: boolean;
    latency_ms: number | null;
    error: string | null;
    details: any;
  }> {
    const response = await this.client.post<ApiResponse<any>>(`/reranker/models/registry/${modelId}/test`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to test reranker model');
    }
    return response.data.data!;
  }

  async unregisterRerankerModel(modelId: string): Promise<void> {
    const response = await this.client.delete<ApiResponse<void>>(`/reranker/models/registry/${modelId}`);
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to unregister reranker model');
    }
  }

  // Health check
  async healthCheck(): Promise<{ status: string; version?: string }> {
    const response = await this.client.get<ApiResponse<{ status: string; version?: string }>>('/health');
    return response.data.data || { status: 'unknown' };
  }
}

export const apiService = new ApiService();
export default apiService;