import React from 'react';
import {
  Box,
  Chip,
  Tooltip,
  Typography,
  Paper,
  Stack,
  CircularProgress,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Memory as EmbeddingIcon,
  TrendingUp as RerankerIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

import { useRerankerConfig, useEmbeddingModels, useRerankerModels } from '../../hooks/useApi';

interface ModelIndicatorProps {
  compact?: boolean;
  showDetails?: boolean;
}

const ModelIndicator: React.FC<ModelIndicatorProps> = ({ 
  compact = false, 
  showDetails = false 
}) => {
  const { data: rerankerConfig, isLoading: rerankerConfigLoading } = useRerankerConfig();
  const { data: embeddingModels, isLoading: embeddingModelsLoading } = useEmbeddingModels();
  const { data: rerankerModels, isLoading: rerankerModelsLoading } = useRerankerModels();

  const isLoading = rerankerConfigLoading || embeddingModelsLoading || rerankerModelsLoading;

  // Get current embedding model (assuming first recommended or first available)
  const currentEmbeddingModel = embeddingModels?.models.find(m => m.recommended) || embeddingModels?.models[0];
  
  // Get current reranker model info
  const currentRerankerModel = rerankerModels?.models.find(m => m.id === rerankerConfig?.model);

  // Format model name for display
  const formatModelName = (modelName: string | undefined) => {
    if (!modelName) return 'Unknown';
    
    // Handle Qwen models specially
    if (modelName.includes('Qwen3')) {
      if (modelName.includes('Reranker')) return 'Qwen3 Reranker';
      if (modelName.includes('Embedding')) return 'Qwen3 Embedding';
    }
    
    // Handle OpenAI models
    if (modelName.includes('text-embedding')) {
      if (modelName.includes('3-small')) return 'OpenAI Embedding 3 Small';
      if (modelName.includes('3-large')) return 'OpenAI Embedding 3 Large';
      if (modelName.includes('ada-002')) return 'OpenAI Ada-002';
    }
    
    // Handle other common models
    if (modelName.includes('mistral')) return 'Mistral 7B';
    if (modelName.includes('llama2')) return modelName.replace('llama2:', 'Llama2 ');
    
    // Fallback to original name, truncated if too long
    return modelName.length > 20 ? modelName.substring(0, 17) + '...' : modelName;
  };

  const renderModelChip = (
    label: string, 
    model: string | undefined, 
    icon: React.ReactElement, 
    color: 'primary' | 'secondary' | 'default' = 'default'
  ) => {
    const modelName = formatModelName(model);
    
    if (compact) {
      return (
        <Tooltip title={`${label}: ${model || 'Unknown'}`}>
          <Chip
            label={modelName}
            size="small"
            icon={icon}
            variant="outlined"
            color={color}
          />
        </Tooltip>
      );
    }

    return (
      <Tooltip title={`Full model ID: ${model || 'Unknown'}`}>
        <Chip
          label={`${label}: ${modelName}`}
          size="small"
          icon={icon}
          variant="outlined"
          color={color}
        />
      </Tooltip>
    );
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="caption" color="text.secondary">
          Loading models...
        </Typography>
      </Box>
    );
  }

  if (compact) {
    return (
      <Stack direction="row" spacing={1} alignItems="center">
        {rerankerConfig?.enabled && (
          renderModelChip(
            'Reranker', 
            rerankerConfig.model, 
            <RerankerIcon fontSize="small" />, 
            'primary'
          )
        )}
        {currentEmbeddingModel && (
          renderModelChip(
            'Embedding', 
            currentEmbeddingModel.id, 
            <EmbeddingIcon fontSize="small" />, 
            'secondary'
          )
        )}
      </Stack>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SettingsIcon fontSize="small" />
        Current Models
      </Typography>
      
      <Stack direction="row" spacing={1} flexWrap="wrap">
        {currentEmbeddingModel && (
          renderModelChip(
            'Embedding', 
            currentEmbeddingModel.id, 
            <EmbeddingIcon fontSize="small" />, 
            'secondary'
          )
        )}
        
        {rerankerConfig?.enabled ? (
          renderModelChip(
            'Reranker', 
            rerankerConfig.model, 
            <RerankerIcon fontSize="small" />, 
            'primary'
          )
        ) : (
          <Chip
            label="Reranker: Disabled"
            size="small"
            icon={<InfoIcon fontSize="small" />}
            variant="outlined"
            color="default"
          />
        )}
      </Stack>

      {showDetails && (
        <Paper sx={{ p: 2, mt: 1, bgcolor: 'background.default' }}>
          <Typography variant="caption" color="text.secondary">
            <strong>Details:</strong><br />
            • Embedding: {currentEmbeddingModel?.name || 'Not available'}<br />
            • Provider: {currentEmbeddingModel?.provider || 'Unknown'}<br />
            • Dimensions: {currentEmbeddingModel?.dimensions || 'Unknown'}<br />
            {rerankerConfig?.enabled && (
              <>
                • Reranker: {currentRerankerModel?.name || rerankerConfig.model}<br />
                • Reranker Provider: {rerankerConfig.provider}<br />
              </>
            )}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default ModelIndicator;