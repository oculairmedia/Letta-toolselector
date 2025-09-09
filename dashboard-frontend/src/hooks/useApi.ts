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
} from '../types';

// Query keys for React Query
export const queryKeys = {
  tools: ['tools'] as const,
  search: (query: SearchQuery) => ['search', query] as const,
  rerankerConfig: ['rerankerConfig'] as const,
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
    },
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

// Health check
export const useHealthCheck = () => {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => apiService.healthCheck(),
    refetchInterval: 30 * 1000, // Every 30 seconds
    staleTime: 0,
  });
};