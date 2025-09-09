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
} from '@mui/material';
import {
  Save as SaveIcon,
  Science as TestIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  ExpandMore as ExpandMoreIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';

import { RerankerConfig as RerankerConfigType } from '../../types';
import { useUpdateRerankerConfig, useTestRerankerConnection, useOllamaModels } from '../../hooks/useApi';

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
  const [formData, setFormData] = useState<RerankerConfigType>({
    enabled: true,
    model: 'gpt-3.5-turbo',
    provider: 'openai',
    parameters: {
      temperature: 0.1,
      max_tokens: 150,
    },
  });

  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const updateConfigMutation = useUpdateRerankerConfig();
  const testConnectionMutation = useTestRerankerConnection();
  const { data: ollamaModelsData, isLoading: ollamaModelsLoading } = useOllamaModels();

  // Update form data when config loads
  useEffect(() => {
    if (config) {
      setFormData(config);
      setHasUnsavedChanges(false);
    }
  }, [config]);

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

  // Get available models based on provider
  const getAvailableModels = () => {
    if (formData.provider === 'ollama' && ollamaModelsData?.models) {
      return ollamaModelsData.models.map(model => model.name);
    }
    return MODEL_OPTIONS[formData.provider] || [];
  };
  
  const availableModels = getAvailableModels();

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
        Reranker Configuration
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure the reranker model settings to improve search result relevance and ranking accuracy.
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

                <Grid item xs={12} md={6}>
                  <Autocomplete
                    options={availableModels}
                    value={formData.model}
                    onChange={(_, value) => handleChange('model', value || '')}
                    loading={formData.provider === 'ollama' && ollamaModelsLoading}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Model"
                        required
                        disabled={!formData.enabled}
                        helperText={
                          formData.provider === 'ollama' && ollamaModelsLoading
                            ? 'Loading models from Ollama...'
                            : formData.provider === 'ollama'
                            ? `${availableModels.length} models available`
                            : undefined
                        }
                        InputProps={{
                          ...params.InputProps,
                          endAdornment: (
                            <>
                              {formData.provider === 'ollama' && ollamaModelsLoading && (
                                <CircularProgress color="inherit" size={20} />
                              )}
                              {params.InputProps.endAdornment}
                            </>
                          ),
                        }}
                      />
                    )}
                    freeSolo
                  />
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
                <Chip label={`Provider: ${formData.provider}`} size="small" />
                <Chip label={`Model: ${formData.model}`} size="small" />
                <Chip label={`Temperature: ${formData.parameters.temperature}`} size="small" />
                <Chip label={`Max Tokens: ${formData.parameters.max_tokens}`} size="small" />
              </Box>

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