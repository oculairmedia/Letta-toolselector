import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Grid,
  Autocomplete,
  Alert,
  CircularProgress,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Save as SaveIcon,
  Science as TestIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  ExpandMore as ExpandMoreIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';

import { RerankerConfig as RerankerConfigType, EmbeddingConfig } from '../../types';
import { 
  useUpdateRerankerConfig, 
  useTestRerankerConnection, 
  useOllamaModels,
  useRerankerModels,
  useEmbeddingModels,
  useRerankerModelRegistry,
  useTestRerankerModel,
  useRegisterRerankerModel,
  useEmbeddingConfig,
  useUpdateEmbeddingConfig,
  useStartReembedding,
  useReembeddingProgress
} from '../../hooks/useApi';
import ReembeddingProgress from '../ReembeddingProgress';

interface RerankerConfigProps {
  config?: RerankerConfigType;
  isLoading?: boolean;
}

const PROVIDER_OPTIONS = [
  { label: 'Ollama', value: 'ollama' },
  { label: 'OpenAI', value: 'openai' },
  { label: 'Anthropic', value: 'anthropic' },
  { label: 'Cohere', value: 'cohere' },
];

const MODEL_OPTIONS: Record<string, string[]> = {
  ollama: ['llama2:7b', 'llama2:13b', 'codellama:7b', 'mistral:7b', 'neural-chat:7b'],
  openai: ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'],
  anthropic: ['claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'],
  cohere: ['command', 'command-light', 'command-nightly'],
};

const RerankerConfig: React.FC<RerankerConfigProps> = ({ config, isLoading }) => {
  const [formData, setFormData] = useState<RerankerConfigType & { 
    embedding_model?: string; 
    embedding_provider?: string;
  }>({
    enabled: true,
    model: 'mistral:7b',
    provider: 'ollama',
    embedding_model: 'text-embedding-3-small',
    embedding_provider: 'openai',
    parameters: {
      temperature: 0.1,
      max_tokens: 150,
    },
  });

  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0); // 0: Reranker, 1: Embedding

  const updateConfigMutation = useUpdateRerankerConfig();
  const testConnectionMutation = useTestRerankerConnection();
  const { data: ollamaModelsData, isLoading: ollamaModelsLoading } = useOllamaModels();
  const { data: rerankerModelsData, isLoading: rerankerModelsLoading } = useRerankerModels();
  const { data: embeddingModelsData, isLoading: embeddingModelsLoading } = useEmbeddingModels();
  const { data: rerankerRegistryData, isLoading: registryLoading } = useRerankerModelRegistry();
  const testRerankerModelMutation = useTestRerankerModel();
  const registerRerankerModelMutation = useRegisterRerankerModel();
  
  // Embedding configuration hooks
  const { data: embeddingConfig, isLoading: embeddingConfigLoading } = useEmbeddingConfig();
  const updateEmbeddingConfigMutation = useUpdateEmbeddingConfig();
  const startReembeddingMutation = useStartReembedding();
  const { data: reembeddingProgress } = useReembeddingProgress();

  // Update form data when config loads
  useEffect(() => {
    if (config) {
      setFormData(prev => ({
        ...prev,
        ...config,
      }));
      setHasUnsavedChanges(false);
    }
  }, [config]);

  // Update form data with embedding config
  useEffect(() => {
    if (embeddingConfig && typeof embeddingConfig === 'object' && 'model' in embeddingConfig) {
      setFormData(prev => ({
        ...prev,
        embedding_model: embeddingConfig.model,
        embedding_provider: embeddingConfig.provider,
      }));
    }
  }, [embeddingConfig]);

  const handleChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
    setHasUnsavedChanges(true);
  };

  const handleParameterChange = (parameter: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [parameter]: value,
      },
    }));
    setHasUnsavedChanges(true);
  };

  const handleProviderChange = (provider: string) => {
    let defaultModel = '';
    if (provider === 'ollama' && ollamaModelsData?.models && ollamaModelsData.models.length > 0) {
      defaultModel = ollamaModelsData.models[0].name;
    } else {
      defaultModel = MODEL_OPTIONS[provider]?.[0] || '';
    }
    
    setFormData(prev => ({
      ...prev,
      provider,
      model: defaultModel,
    }));
    setHasUnsavedChanges(true);
  };

  const handleSave = async () => {
    try {
      await updateConfigMutation.mutateAsync(formData);
      setHasUnsavedChanges(false);
    } catch (error) {
      console.error('Failed to save configuration:', error);
    }
  };

  const handleEmbeddingModelChange = async (modelId: string, provider: string) => {
    // Update form data
    setFormData(prev => ({
      ...prev,
      embedding_model: modelId,
      embedding_provider: provider,
    }));
    setHasUnsavedChanges(true);

    try {
      // Update embedding configuration
      await updateEmbeddingConfigMutation.mutateAsync({
        model: modelId,
        provider: provider,
      });

      // Start re-embedding process
      await startReembeddingMutation.mutateAsync({
        embedding_model: modelId,
        batch_size: 100, // Default batch size
      });
    } catch (error) {
      console.error('Failed to update embedding model:', error);
      setTestResult({
        success: false,
        message: `Failed to update embedding model: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  };

  const handleTest = async () => {
    try {
      const connected = await testConnectionMutation.mutateAsync(formData);
      setTestResult({
        success: connected,
        message: connected ? 'Connection successful!' : 'Connection failed',
      });
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : 'Connection test failed',
      });
    }
  };

  // Get available reranker models from registry and discovery
  const getAvailableRerankerModels = () => {
    const models: Array<{ id: string; name: string; provider: string; recommended: boolean; validated: boolean }> = [];
    
    // Add registry models
    if (rerankerRegistryData?.models) {
      models.push(...rerankerRegistryData.models.map(model => ({
        id: model.id,
        name: model.name,
        provider: model.provider,
        recommended: model.recommended,
        validated: model.validated,
      })));
    }
    
    // Add discovered models (if not already in registry)
    if (rerankerModelsData?.models) {
      const registryIds = new Set(rerankerRegistryData?.models?.map(m => m.id) || []);
      rerankerModelsData.models.forEach(model => {
        if (!registryIds.has(model.id)) {
          models.push({
            id: model.id,
            name: model.name,
            provider: model.provider,
            recommended: model.recommended,
            validated: false,
          });
        }
      });
    }
    
    return models;
  };

  // Get available embedding models
  const getAvailableEmbeddingModels = () => {
    return embeddingModelsData?.models?.map(model => ({
      id: model.id,
      name: model.name,
      provider: model.provider,
      recommended: model.recommended,
      cost_per_1k: model.cost_per_1k,
      dimensions: model.dimensions,
    })) || [];
  };
  
  const availableRerankerModels = getAvailableRerankerModels();
  const availableEmbeddingModels = getAvailableEmbeddingModels();

  // Test specific reranker model
  const handleTestRerankerModel = async (modelId: string) => {
    try {
      const result = await testRerankerModelMutation.mutateAsync(modelId);
      setTestResult({
        success: result.connectivity && result.functionality,
        message: result.error || `Connectivity: ${result.connectivity ? '✓' : '✗'}, Functionality: ${result.functionality ? '✓' : '✗'} (${result.latency_ms}ms)`,
      });
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : 'Model test failed',
      });
    }
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TuneIcon />
        Model Configuration
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure reranker and embedding models to optimize search performance. Changing the embedding model will trigger automatic re-embedding of the search index.
      </Typography>

      {/* Save/Test Actions */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={!hasUnsavedChanges || updateConfigMutation.isPending}
          startIcon={updateConfigMutation.isPending ? <CircularProgress size={16} /> : <SaveIcon />}
        >
          {updateConfigMutation.isPending ? 'Saving...' : 'Save Configuration'}
        </Button>
        
        <Button
          variant="outlined"
          onClick={handleTest}
          disabled={testConnectionMutation.isPending}
          startIcon={testConnectionMutation.isPending ? <CircularProgress size={16} /> : <TestIcon />}
        >
          {testConnectionMutation.isPending ? 'Testing...' : 'Test Connection'}
        </Button>
      </Box>

      {/* Success/Error Messages */}
      {updateConfigMutation.isSuccess && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => updateConfigMutation.reset()}>
          Configuration saved successfully!
        </Alert>
      )}
      
      {updateConfigMutation.error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => updateConfigMutation.reset()}>
          Failed to save configuration: {updateConfigMutation.error.message}
        </Alert>
      )}

      {testResult && (
        <Alert 
          severity={testResult.success ? 'success' : 'error'} 
          sx={{ mb: 2 }}
          icon={testResult.success ? <CheckCircleIcon /> : <ErrorIcon />}
          onClose={() => setTestResult(null)}
        >
          {testResult.message}
        </Alert>
      )}

      {/* Re-embedding Progress */}
      <ReembeddingProgress
        onComplete={() => {
          setTestResult({
            success: true,
            message: 'Re-embedding completed successfully!',
          });
        }}
        onError={(error) => {
          setTestResult({
            success: false,
            message: `Re-embedding failed: ${error}`,
          });
        }}
      />

      {/* Configuration Form */}
      <Grid container spacing={3}>
        {/* Basic Settings */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Basic Settings
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={formData.enabled}
                        onChange={(e) => handleChange('enabled', e.target.checked)}
                      />
                    }
                    label="Enable Reranker"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Autocomplete
                    options={PROVIDER_OPTIONS}
                    value={PROVIDER_OPTIONS.find(opt => opt.value === formData.provider) || null}
                    onChange={(_, value) => handleProviderChange(value?.value || 'openai')}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Provider"
                        required
                        disabled={!formData.enabled}
                      />
                    )}
                    isOptionEqualToValue={(option, value) => option.value === value.value}
                    getOptionLabel={(option) => option.label}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Tabs 
                    value={selectedTab} 
                    onChange={(_, value) => setSelectedTab(value)}
                    variant="fullWidth"
                    sx={{ mb: 2 }}
                  >
                    <Tab label="Reranker Model" />
                    <Tab label="Embedding Model" />
                  </Tabs>

                  {selectedTab === 0 && (
                    // Reranker Model Selection
                    <Box>
                      <Autocomplete
                        options={availableRerankerModels}
                        value={availableRerankerModels.find(model => model.id === formData.model) || null}
                        onChange={(_, value) => {
                          if (value) {
                            handleChange('model', value.id);
                            handleChange('provider', value.provider);
                          }
                        }}
                        loading={rerankerModelsLoading || registryLoading}
                        renderInput={(params) => (
                          <TextField
                            {...params}
                            label="Reranker Model"
                            required
                            disabled={!formData.enabled}
                            helperText={`${availableRerankerModels.length} models available`}
                            InputProps={{
                              ...params.InputProps,
                              endAdornment: (
                                <>
                                  {(rerankerModelsLoading || registryLoading) && (
                                    <CircularProgress color="inherit" size={20} />
                                  )}
                                  {params.InputProps.endAdornment}
                                </>
                              ),
                            }}
                          />
                        )}
                        renderOption={(props, option) => (
                          <Box component="li" {...props}>
                            <Box sx={{ flexGrow: 1 }}>
                              <Typography variant="body1">
                                {option.name}
                                {option.recommended && (
                                  <Chip label="Recommended" size="small" color="primary" sx={{ ml: 1 }} />
                                )}
                                {option.validated && (
                                  <Chip label="Validated" size="small" color="success" sx={{ ml: 1 }} />
                                )}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {option.provider} • {option.id}
                              </Typography>
                            </Box>
                            <Button
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleTestRerankerModel(option.id);
                              }}
                              disabled={testRerankerModelMutation.isPending}
                            >
                              Test
                            </Button>
                          </Box>
                        )}
                        getOptionLabel={(option) => option.name}
                        isOptionEqualToValue={(option, value) => option.id === value.id}
                      />
                    </Box>
                  )}

                  {selectedTab === 1 && (
                    // Embedding Model Selection
                    <Box>
                      <Autocomplete
                        options={availableEmbeddingModels}
                        value={availableEmbeddingModels.find(model => model.id === formData.embedding_model) || null}
                        onChange={(_, value) => {
                          if (value) {
                            handleEmbeddingModelChange(value.id, value.provider);
                          }
                        }}
                        loading={embeddingModelsLoading}
                        renderInput={(params) => (
                          <TextField
                            {...params}
                            label="Embedding Model"
                            helperText={`${availableEmbeddingModels.length} embedding models available`}
                            InputProps={{
                              ...params.InputProps,
                              endAdornment: (
                                <>
                                  {embeddingModelsLoading && (
                                    <CircularProgress color="inherit" size={20} />
                                  )}
                                  {params.InputProps.endAdornment}
                                </>
                              ),
                            }}
                          />
                        )}
                        renderOption={(props, option) => (
                          <Box component="li" {...props}>
                            <Box sx={{ flexGrow: 1 }}>
                              <Typography variant="body1">
                                {option.name}
                                {option.recommended && (
                                  <Chip label="Recommended" size="small" color="primary" sx={{ ml: 1 }} />
                                )}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {option.provider} • ${option.cost_per_1k}/1K tokens • {option.dimensions} dims
                              </Typography>
                            </Box>
                          </Box>
                        )}
                        getOptionLabel={(option) => option.name}
                        isOptionEqualToValue={(option, value) => option.id === value.id}
                      />
                    </Box>
                  )}
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Advanced Parameters */}
        <Grid item xs={12}>
          <Accordion defaultExpanded={false}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Advanced Parameters</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Temperature: {formData.parameters.temperature}
                  </Typography>
                  <Slider
                    value={formData.parameters.temperature || 0.1}
                    onChange={(_, value) => handleParameterChange('temperature', value)}
                    min={0}
                    max={2}
                    step={0.1}
                    marks={[
                      { value: 0, label: '0 (Deterministic)' },
                      { value: 1, label: '1 (Balanced)' },
                      { value: 2, label: '2 (Creative)' },
                    ]}
                    disabled={!formData.enabled}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Max Tokens"
                    type="number"
                    value={formData.parameters.max_tokens || 150}
                    onChange={(e) => handleParameterChange('max_tokens', parseInt(e.target.value) || 150)}
                    inputProps={{
                      min: 1,
                      max: 4000,
                    }}
                    disabled={!formData.enabled}
                    helperText="Maximum tokens for reranking explanation"
                  />
                </Grid>

                {/* Additional provider-specific parameters can be added here */}
                {formData.provider === 'ollama' && (
                  <Grid item xs={12}>
                    <Alert severity="info">
                      <Typography variant="subtitle2" gutterBottom>
                        Ollama Provider Settings
                      </Typography>
                      <Typography variant="body2">
                        Make sure Ollama is running and the model is installed locally.
                        Use `ollama pull {formData.model}` to install the model.
                      </Typography>
                    </Alert>
                  </Grid>
                )}

                {formData.provider === 'openai' && (
                  <Grid item xs={12}>
                    <Alert severity="info">
                      <Typography variant="subtitle2" gutterBottom>
                        OpenAI Provider Settings
                      </Typography>
                      <Typography variant="body2">
                        Ensure your OpenAI API key is configured in the system environment.
                      </Typography>
                    </Alert>
                  </Grid>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Configuration Preview */}
        <Grid item xs={12}>
          <Card sx={{ bgcolor: 'grey.50' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Configuration Preview
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                <Chip 
                  label={formData.enabled ? 'Enabled' : 'Disabled'}
                  color={formData.enabled ? 'success' : 'default'}
                  size="small"
                />
                <Chip label={`Reranker: ${formData.provider}`} size="small" />
                <Chip label={`Model: ${formData.model}`} size="small" />
                {formData.embedding_model && (
                  <Chip label={`Embedding: ${formData.embedding_model}`} size="small" color="secondary" />
                )}
                <Chip label={`Temperature: ${formData.parameters.temperature}`} size="small" />
                <Chip label={`Max Tokens: ${formData.parameters.max_tokens}`} size="small" />
              </Box>

              {/* Model Status Summary */}
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Reranker Models Available
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {availableRerankerModels.slice(0, 3).map(model => (
                      <Chip
                        key={model.id}
                        label={model.name}
                        size="small"
                        variant={model.id === formData.model ? 'filled' : 'outlined'}
                        color={model.validated ? 'success' : model.recommended ? 'primary' : 'default'}
                      />
                    ))}
                    {availableRerankerModels.length > 3 && (
                      <Chip label={`+${availableRerankerModels.length - 3} more`} size="small" variant="outlined" />
                    )}
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Embedding Models Available
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {availableEmbeddingModels.slice(0, 3).map(model => (
                      <Chip
                        key={model.id}
                        label={model.name}
                        size="small"
                        variant={model.id === formData.embedding_model ? 'filled' : 'outlined'}
                        color={model.recommended ? 'primary' : 'default'}
                      />
                    ))}
                    {availableEmbeddingModels.length > 3 && (
                      <Chip label={`+${availableEmbeddingModels.length - 3} more`} size="small" variant="outlined" />
                    )}
                  </Box>
                </Grid>
              </Grid>

              {hasUnsavedChanges && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  You have unsaved changes. Don't forget to save your configuration.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RerankerConfig;