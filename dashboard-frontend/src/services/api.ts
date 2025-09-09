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

  // Health check
  async healthCheck(): Promise<{ status: string; version?: string }> {
    const response = await this.client.get<ApiResponse<{ status: string; version?: string }>>('/health');
    return response.data.data || { status: 'unknown' };
  }
}

export const apiService = new ApiService();
export default apiService;