import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../services/api';
import {
  SearchQuery,
  SearchResponse,
  RerankerConfig,
  ConfigurationPreset,
  EvaluationRating,
  Analytics,
  Tool,
  ToolBrowseResponse,
  ToolDetailResponse,
} from '../types';

// Query keys for React Query
export const queryKeys = {
  tools: ['tools'] as const,
  search: (query: SearchQuery) => ['search', query] as const,
  searchWithOverrides: (query: string, options: any) => ['searchWithOverrides', query, options] as const,
  rerankerConfig: ['rerankerConfig'] as const,
  ollamaModels: ['ollamaModels'] as const,
  embeddingModels: ['embeddingModels'] as const,
  rerankerModels: ['rerankerModels'] as const,
  embeddingConfig: ['embeddingConfig'] as const,
  reembeddingProgress: ['reembeddingProgress'] as const,
  toolSelectorConfig: ['toolSelectorConfig'] as const,
  configPresets: ['configPresets'] as const,
  evaluations: (query?: string, limit?: number) => ['evaluations', query, limit] as const,
  analytics: (dateRange?: { start: string; end: string }) => ['analytics', dateRange] as const,
  health: ['health'] as const,
};

// Tools
export const useTools = () => {
  return useQuery({
    queryKey: queryKeys.tools,
    queryFn: () => apiService.getAllTools(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useRefreshTools = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: () => apiService.refreshToolIndex(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.tools });
      queryClient.invalidateQueries({ queryKey: ['browseTools'] });
    },
  });
};

// Tool Browser hooks
export const useBrowseTools = (params?: {
  page?: number;
  limit?: number;
  search?: string;
  category?: string;
  source?: string;
  mcp_server?: string;
  sort?: 'name' | 'category' | 'updated' | 'relevance';
  order?: 'asc' | 'desc';
}) => {
  console.log('useBrowseTools hook called with params:', params);
  
  const result = useQuery<ToolBrowseResponse>({
    queryKey: ['browseTools', params],
    queryFn: async () => {
      console.log('useBrowseTools queryFn executing with params:', params);
      try {
        const data = await apiService.browseTools(params);
        console.log('useBrowseTools queryFn success:', {
          toolsCount: data.tools?.length,
          total: data.total,
          page: data.page,
          limit: data.limit,
          has_more: data.has_more
        });
        return data;
      } catch (error) {
        console.error('useBrowseTools queryFn error:', error);
        throw error;
      }
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
    retry: (failureCount, error) => {
      console.log('useBrowseTools retry:', { failureCount, error: error.message });
      return failureCount < 3;
    }
  });

  // React Query v5 removed onError/onSuccess callbacks, so we log manually
  if (result.error) {
    console.error('useBrowseTools error state:', result.error);
  }
  if (result.data) {
    console.log('useBrowseTools data available:', {
      toolsCount: result.data.tools?.length,
      total: result.data.total
    });
  }

  console.log('useBrowseTools hook result:', {
    isLoading: result.isLoading,
    isFetching: result.isFetching,
    hasData: !!result.data,
    error: result.error?.message,
    status: result.status
  });

  return result;
};

export const useToolDetail = (toolId: string, enabled: boolean = true) => {
  return useQuery<ToolDetailResponse>({
    queryKey: ['toolDetail', toolId],
    queryFn: () => apiService.getToolDetail(toolId),
    enabled: enabled && !!toolId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useToolCategories = () => {
  return useQuery({
    queryKey: ['toolCategories'],
    queryFn: () => apiService.getToolCategories(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useToolSources = () => {
  return useQuery({
    queryKey: ['toolSources'],
    queryFn: () => apiService.getToolSources(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useExportTools = () => {
  return useMutation({
    mutationFn: (format: 'json' | 'csv' = 'json') => apiService.exportTools(format),
  });
};

// Search
export const useSearch = (query: SearchQuery, enabled: boolean = true) => {
  return useQuery({
    queryKey: queryKeys.search(query),
    queryFn: () => apiService.searchTools(query),
    enabled: enabled && !!query.query.trim(),
    staleTime: 30 * 1000, // 30 seconds
  });
};

export const useSearchWithReranking = () => {
  return useMutation({
    mutationFn: ({ query, config }: { query: SearchQuery; config: RerankerConfig }) =>
      apiService.searchWithReranking(query, config),
  });
};

// Configuration
export const useRerankerConfig = () => {
  return useQuery({
    queryKey: queryKeys.rerankerConfig,
    queryFn: () => apiService.getRerankerConfig(),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useUpdateRerankerConfig = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (config: RerankerConfig) => apiService.updateRerankerConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.rerankerConfig });
    },
  });
};

export const useTestRerankerConnection = () => {
  return useMutation({
    mutationFn: (config: RerankerConfig) => apiService.testRerankerConnection(config),
  });
};

export const useOllamaModels = () => {
  return useQuery({
    queryKey: queryKeys.ollamaModels,
    queryFn: () => apiService.getOllamaModels(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Configuration presets
export const useConfigurationPresets = () => {
  return useQuery({
    queryKey: queryKeys.configPresets,
    queryFn: () => apiService.getConfigurationPresets(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useCreateConfigurationPreset = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (preset: Omit<ConfigurationPreset, 'id'>) =>
      apiService.createConfigurationPreset(preset),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.configPresets });
    },
  });
};

export const useUpdateConfigurationPreset = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, preset }: { id: string; preset: Partial<ConfigurationPreset> }) =>
      apiService.updateConfigurationPreset(id, preset),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.configPresets });
    },
  });
};

export const useDeleteConfigurationPreset = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => apiService.deleteConfigurationPreset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.configPresets });
    },
  });
};

// Evaluations
export const useEvaluations = (query?: string, limit?: number) => {
  return useQuery({
    queryKey: queryKeys.evaluations(query, limit),
    queryFn: () => apiService.getEvaluations(query, limit),
    staleTime: 1 * 60 * 1000, // 1 minute
  });
};

export const useSubmitEvaluation = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (evaluation: Omit<EvaluationRating, 'timestamp'>) =>
      apiService.submitEvaluation(evaluation),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['evaluations'] });
      queryClient.invalidateQueries({ queryKey: queryKeys.analytics() });
    },
  });
};

// Analytics
export const useAnalytics = (dateRange?: { start: string; end: string }) => {
  return useQuery({
    queryKey: queryKeys.analytics(dateRange),
    queryFn: () => apiService.getAnalytics(dateRange),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

// Advanced search hooks
export const useSearchWithOverrides = (
  query: string, 
  options: {
    limit?: number;
    alpha?: number;
    distance_metric?: string;
    reranker_enabled?: boolean;
    reranker_model?: string;
  },
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: queryKeys.searchWithOverrides(query, options),
    queryFn: () => apiService.searchWithOverrides(query, options),
    enabled: enabled && !!query.trim(),
    staleTime: 30 * 1000, // 30 seconds
  });
};

export const useCompareRankerConfigurations = () => {
  return useMutation({
    mutationFn: ({ 
      query, 
      configA, 
      configB, 
      limit 
    }: { 
      query: string; 
      configA: RerankerConfig; 
      configB: RerankerConfig; 
      limit?: number;
    }) => apiService.compareRankerConfigurations(query, configA, configB, limit),
  });
};

// Model discovery hooks
export const useEmbeddingModels = () => {
  return useQuery({
    queryKey: queryKeys.embeddingModels,
    queryFn: () => apiService.getEmbeddingModels(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useRerankerModels = () => {
  return useQuery({
    queryKey: queryKeys.rerankerModels,
    queryFn: () => apiService.getRerankerModels(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Reranker Model Registry hooks
export const useRerankerModelRegistry = () => {
  return useQuery({
    queryKey: ['rerankerModelRegistry'],
    queryFn: () => apiService.getRerankerModelRegistry(),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useRegisterRerankerModel = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (model: any) => apiService.registerRerankerModel(model),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rerankerModelRegistry'] });
      queryClient.invalidateQueries({ queryKey: queryKeys.rerankerModels });
    },
  });
};

export const useUpdateRerankerModel = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ modelId, updates }: { modelId: string; updates: any }) =>
      apiService.updateRerankerModel(modelId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rerankerModelRegistry'] });
    },
  });
};

export const useTestRerankerModel = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (modelId: string) => apiService.testRerankerModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rerankerModelRegistry'] });
    },
  });
};

export const useUnregisterRerankerModel = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (modelId: string) => apiService.unregisterRerankerModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['rerankerModelRegistry'] });
      queryClient.invalidateQueries({ queryKey: queryKeys.rerankerModels });
    },
  });
};

// Embedding Configuration hooks
export const useEmbeddingConfig = () => {
  return useQuery({
    queryKey: queryKeys.embeddingConfig,
    queryFn: () => apiService.getEmbeddingConfig(),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useUpdateEmbeddingConfig = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (config: any) => apiService.updateEmbeddingConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.embeddingConfig });
      queryClient.invalidateQueries({ queryKey: queryKeys.embeddingModels });
    },
  });
};

export const useReembeddingProgress = () => {
  return useQuery({
    queryKey: queryKeys.reembeddingProgress,
    queryFn: () => apiService.getReembeddingProgress(),
    refetchInterval: 5000, // Every 5 seconds to respect rate limits
    staleTime: 0,
  });
};

export const useStartReembedding = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (config: { embedding_model: string; batch_size?: number }) => 
      apiService.startReembedding(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.reembeddingProgress });
    },
  });
};

export const useCancelReembedding = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: () => apiService.cancelReembedding(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.reembeddingProgress });
    },
  });
};

// Tool Selector Configuration hooks
export const useToolSelectorConfig = () => {
  return useQuery({
    queryKey: queryKeys.toolSelectorConfig,
    queryFn: () => apiService.getToolSelectorConfig(),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useUpdateToolSelectorConfig = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (config: any) => apiService.updateToolSelectorConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.toolSelectorConfig });
    },
  });
};

// Health check
export const useHealthCheck = () => {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => apiService.healthCheck(),
    refetchInterval: 30 * 1000, // Every 30 seconds
    staleTime: 0,
  });
};