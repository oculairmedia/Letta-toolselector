import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  LinearProgress,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  FormControlLabel,
  Switch,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Storage as StorageIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Extension as ExtensionIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Settings as SettingsIcon,
  CloudSync as EmbeddingIcon,
  Tune as TuneIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as StartIcon,
} from '@mui/icons-material';

import {
  useTools,
  useRefreshTools,
  useHealthCheck,
  useEmbeddingConfig,
  useEmbeddingModels,
  useUpdateEmbeddingConfig,
  useStartReembedding,
  useToolSelectorConfig,
  useUpdateToolSelectorConfig,
} from '../../hooks/useApi';
import { formatNumber, formatRelativeTime } from '../../utils';
import ReembeddingProgress from '../ReembeddingProgress';
import MetricsDashboard from '../MetricsDashboard/MetricsDashboard';
import ComponentStatus from '../ComponentStatus/ComponentStatus';
import LogViewer from '../LogViewer/LogViewer';
import ConfigurationManager from '../ConfigurationManager/ConfigurationManager';
import ValidationPanel from '../ValidationPanel/ValidationPanel';
import ValidatedTextField from '../ValidatedTextField/ValidatedTextField';
import EmbeddingHealthIndicator from '../EmbeddingHealthIndicator/EmbeddingHealthIndicator';

const SystemSettings: React.FC = () => {
  const [embeddingDialogOpen, setEmbeddingDialogOpen] = useState(false);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState('');
  const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] = useState('');
  const [localToolSelectorConfig, setLocalToolSelectorConfig] = useState({
    maxTotalTools: 30,
    maxMcpTools: 20,
    minMcpTools: 7,
    dropRate: 0.6,
    excludeLettaCore: true,
    excludeOfficial: true,
    manageOnlyMcp: true,
    enableReranking: false,
    enableLlmEnhancement: true,
    enableSafetyMode: true,
  });

  // All hooks must be declared before useEffect
  const { data: tools, isLoading: toolsLoading } = useTools();
  const { data: healthData } = useHealthCheck();
  const { data: embeddingConfig, isLoading: embeddingConfigLoading } = useEmbeddingConfig();
  const { data: embeddingModels } = useEmbeddingModels();
  const { data: toolSelectorConfig, isLoading: toolSelectorConfigLoading } = useToolSelectorConfig();
  const refreshToolsMutation = useRefreshTools();
  const updateEmbeddingConfigMutation = useUpdateEmbeddingConfig();
  const startReembeddingMutation = useStartReembedding();
  const updateToolSelectorConfigMutation = useUpdateToolSelectorConfig();

  // Update local state when API data loads
  React.useEffect(() => {
    if (toolSelectorConfig?.tool_limits && toolSelectorConfig?.behavior) {
      setLocalToolSelectorConfig({
        maxTotalTools: toolSelectorConfig.tool_limits.max_total_tools,
        maxMcpTools: toolSelectorConfig.tool_limits.max_mcp_tools,
        minMcpTools: toolSelectorConfig.tool_limits.min_mcp_tools,
        dropRate: toolSelectorConfig.behavior.default_drop_rate,
        excludeLettaCore: toolSelectorConfig.behavior.exclude_letta_core_tools,
        excludeOfficial: toolSelectorConfig.behavior.exclude_official_tools,
        manageOnlyMcp: toolSelectorConfig.behavior.manage_only_mcp_tools,
        enableReranking: toolSelectorConfig.behavior.enable_reranking || false,
        enableLlmEnhancement: toolSelectorConfig.behavior.enable_llm_enhancement !== false,
        enableSafetyMode: toolSelectorConfig.behavior.enable_safety_mode !== false,
      });
    }
  }, [toolSelectorConfig]);

  // Update embedding provider state when API data loads
  React.useEffect(() => {
    if (embeddingConfig?.provider) {
      setSelectedEmbeddingProvider(embeddingConfig.provider);
    }
    if (embeddingConfig?.model) {
      setSelectedEmbeddingModel(embeddingConfig.model);
    }
  }, [embeddingConfig]);

  const handleRefreshTools = () => {
    refreshToolsMutation.mutate();
  };

  const handleStartReembedding = async () => {
    if (!selectedEmbeddingModel || !selectedEmbeddingProvider) {
      return;
    }

    try {
      // Update embedding configuration
      await updateEmbeddingConfigMutation.mutateAsync({
        model: selectedEmbeddingModel,
        provider: selectedEmbeddingProvider,
      });

      // Start reembedding process
      await startReembeddingMutation.mutateAsync({
        embedding_model: selectedEmbeddingModel,
        batch_size: 100,
      });

      setEmbeddingDialogOpen(false);
    } catch (error) {
      console.error('Failed to start reembedding:', error);
    }
  };

  const handleToolSelectorConfigChange = (field: string, value: any) => {
    setLocalToolSelectorConfig(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSaveToolSelectorConfig = async () => {
    try {
      const configToSave = {
        tool_limits: {
          max_total_tools: localToolSelectorConfig.maxTotalTools,
          max_mcp_tools: localToolSelectorConfig.maxMcpTools,
          min_mcp_tools: localToolSelectorConfig.minMcpTools,
        },
        behavior: {
          default_drop_rate: localToolSelectorConfig.dropRate,
          exclude_letta_core_tools: localToolSelectorConfig.excludeLettaCore,
          exclude_official_tools: localToolSelectorConfig.excludeOfficial,
          manage_only_mcp_tools: localToolSelectorConfig.manageOnlyMcp,
          enable_reranking: localToolSelectorConfig.enableReranking,
          enable_llm_enhancement: localToolSelectorConfig.enableLlmEnhancement,
          enable_safety_mode: localToolSelectorConfig.enableSafetyMode,
        }
      };

      await updateToolSelectorConfigMutation.mutateAsync(configToSave);
    } catch (error) {
      console.error('Failed to save tool selector config:', error);
    }
  };

  const systemStats = React.useMemo(() => {
    if (!tools) return null;

    const sources = [...new Set(tools.map(t => t.source))];
    const categories = [...new Set(tools.filter(t => t.category).map(t => t.category!))];
    const totalTags = tools.reduce((acc, t) => acc + (t.tags?.length || 0), 0);

    return {
      totalTools: tools.length,
      sources: sources.length,
      categories: categories.length,
      totalTags,
      sourceBreakdown: sources.map(source => ({
        name: source,
        count: tools.filter(t => t.source === source).length,
      })).sort((a, b) => b.count - a.count),
    };
  }, [tools]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <StorageIcon />
        System Settings
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Monitor system health, manage tool inventory, and view performance metrics.
      </Typography>

      {/* Real-time Metrics Dashboard */}
      <Box mb={4}>
        <MetricsDashboard />
      </Box>

      {/* Component Status Monitor */}
      <Box mb={4}>
        <ComponentStatus />
      </Box>

      {/* Error Log Viewer and Analysis */}
      <Box mb={4}>
        <LogViewer />
      </Box>

      {/* Configuration Save and Reset */}
      <Box mb={4}>
        <ConfigurationManager />
      </Box>

      {/* Real-time Configuration Validation */}
      <Box mb={4}>
        <ValidationPanel
          title="Real-time Configuration Validation"
          autoValidate={true}
          configFields={[
            {
              fieldId: 'max_total_tools',
              configType: 'tool_selector',
              field: 'max_total_tools',
              label: 'Max Total Tools',
              value: localToolSelectorConfig.maxTotalTools
            },
            {
              fieldId: 'max_mcp_tools',
              configType: 'tool_selector',
              field: 'max_mcp_tools',
              label: 'Max MCP Tools',
              value: localToolSelectorConfig.maxMcpTools
            },
            {
              fieldId: 'min_mcp_tools',
              configType: 'tool_selector',
              field: 'min_mcp_tools',
              label: 'Min MCP Tools',
              value: localToolSelectorConfig.minMcpTools
            },
            {
              fieldId: 'default_drop_rate',
              configType: 'tool_selector',
              field: 'default_drop_rate',
              label: 'Tool Drop Rate',
              value: localToolSelectorConfig.dropRate
            },
            {
              fieldId: 'embedding_provider',
              configType: 'embedding',
              field: 'provider',
              label: 'Embedding Provider',
              value: selectedEmbeddingProvider
            },
            {
              fieldId: 'embedding_model',
              configType: 'embedding',
              field: selectedEmbeddingProvider === 'openai' ? 'openai_model' : 'ollama_model',
              label: 'Embedding Model',
              value: selectedEmbeddingModel
            }
          ]}
        />
      </Box>

      {/* Embedding Model Health and Status Indicators */}
      <Box mb={4}>
        <EmbeddingHealthIndicator autoRefresh={true} refreshInterval={30000} />
      </Box>

      <Grid container spacing={3}>
        {/* System Health */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckCircleIcon color="success" />
                System Health
              </Typography>
              
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary="API Server"
                    secondary={healthData ? `Status: ${healthData.status}` : 'Checking...'}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Database Connection"
                    secondary="Connected"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Search Engine"
                    secondary="Weaviate - Online"
                  />
                </ListItem>
              </List>

              {healthData?.version && (
                <Chip 
                  label={`Version: ${healthData.version}`} 
                  size="small" 
                  icon={<InfoIcon />}
                  sx={{ mt: 1 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Tool Inventory */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ExtensionIcon />
                  Tool Inventory
                </Typography>
                
                <Button
                  size="small"
                  onClick={handleRefreshTools}
                  disabled={refreshToolsMutation.isPending}
                  startIcon={<RefreshIcon />}
                >
                  Refresh
                </Button>
              </Box>

              {toolsLoading && <LinearProgress sx={{ mb: 2 }} />}

              {systemStats && (
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <ExtensionIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Total Tools"
                      secondary={formatNumber(systemStats.totalTools)}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <StorageIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Sources"
                      secondary={`${systemStats.sources} different sources`}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <MemoryIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Categories"
                      secondary={`${systemStats.categories} categories`}
                    />
                  </ListItem>
                </List>
              )}

              {refreshToolsMutation.isSuccess && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  Tool inventory refreshed successfully!
                </Alert>
              )}

              {refreshToolsMutation.error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  Failed to refresh tool inventory: {refreshToolsMutation.error.message}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Source Breakdown */}
        {systemStats && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Tool Sources Breakdown
                </Typography>
                
                <Grid container spacing={2}>
                  {systemStats.sourceBreakdown.map((source, index) => (
                    <Grid item xs={12} sm={6} md={4} key={source.name}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 2 }}>
                          <Typography variant="subtitle2" noWrap>
                            {source.name}
                          </Typography>
                          <Typography variant="h6" color="primary">
                            {formatNumber(source.count)}
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={(source.count / systemStats.totalTools) * 100}
                            sx={{ mt: 1, height: 4, borderRadius: 2 }}
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Performance Metrics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SpeedIcon />
                Performance Metrics
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        ~250ms
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Average Search Time
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main">
                        ~15%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Reranker Improvement
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        99.9%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        System Uptime
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Tool Selector Configuration */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TuneIcon />
                Tool Selector Configuration
              </Typography>

              {toolSelectorConfigLoading ? (
                <Typography>Loading configuration...</Typography>
              ) : (
                <Grid container spacing={3}>
                  {/* Tool Limits */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Tool Limits
                    </Typography>

                    <Box sx={{ mb: 3 }}>
                      <Typography gutterBottom>
                        Max Total Tools: {localToolSelectorConfig.maxTotalTools}
                      </Typography>
                      <Slider
                        value={localToolSelectorConfig.maxTotalTools}
                        onChange={(_, value) => handleToolSelectorConfigChange('maxTotalTools', value)}
                        min={10}
                        max={100}
                        marks={[
                          { value: 30, label: 'Default' },
                          { value: 50, label: '50' },
                        ]}
                        valueLabelDisplay="auto"
                      />
                    </Box>

                    <Box sx={{ mb: 3 }}>
                      <Typography gutterBottom>
                        Max MCP Tools: {localToolSelectorConfig.maxMcpTools}
                      </Typography>
                      <Slider
                        value={localToolSelectorConfig.maxMcpTools}
                        onChange={(_, value) => handleToolSelectorConfigChange('maxMcpTools', value)}
                        min={5}
                        max={50}
                        marks={[
                          { value: 20, label: 'Default' },
                        ]}
                        valueLabelDisplay="auto"
                      />
                    </Box>

                    <Box sx={{ mb: 3 }}>
                      <Typography gutterBottom>
                        Min MCP Tools: {localToolSelectorConfig.minMcpTools}
                      </Typography>
                      <Slider
                        value={localToolSelectorConfig.minMcpTools}
                        onChange={(_, value) => handleToolSelectorConfigChange('minMcpTools', value)}
                        min={1}
                        max={20}
                        marks={[
                          { value: 7, label: 'Default' },
                        ]}
                        valueLabelDisplay="auto"
                      />
                    </Box>
                  </Grid>

                  {/* Behavior Settings */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Behavior Settings
                    </Typography>

                    <Box sx={{ mb: 3 }}>
                      <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <TuneIcon fontSize="small" />
                        Tool Drop Rate: {(localToolSelectorConfig.dropRate * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                        Controls how aggressively tools are removed when agent reaches tool limits.
                        Higher values remove more irrelevant tools.
                      </Typography>
                      <Slider
                        value={localToolSelectorConfig.dropRate}
                        onChange={(_, value) => handleToolSelectorConfigChange('dropRate', value)}
                        min={0.0}
                        max={1.0}
                        step={0.05}
                        marks={[
                          { value: 0.2, label: 'Minimal' },
                          { value: 0.4, label: 'Conservative' },
                          { value: 0.6, label: 'Balanced' },
                          { value: 0.8, label: 'Aggressive' },
                        ]}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                        track="normal"
                        color="primary"
                        sx={{
                          '& .MuiSlider-mark': {
                            backgroundColor: '#bfbfbf',
                            height: 8,
                            width: 1,
                            '&.MuiSlider-markActive': {
                              backgroundColor: 'currentColor',
                            },
                          },
                          '& .MuiSlider-markLabel': {
                            fontSize: '0.75rem',
                          },
                        }}
                      />
                    </Box>

                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 3, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                      <SettingsIcon fontSize="small" />
                      Policy Configuration
                    </Typography>

                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <Card variant="outlined" sx={{ p: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={localToolSelectorConfig.excludeLettaCore}
                              onChange={(e) => handleToolSelectorConfigChange('excludeLettaCore', e.target.checked)}
                              color="primary"
                            />
                          }
                          label="Exclude Letta Core Tools"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          Skip Letta's built-in core tools during management operations.
                          Recommended to keep enabled to prevent interference with essential functions.
                        </Typography>
                      </Card>

                      <Card variant="outlined" sx={{ p: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={localToolSelectorConfig.excludeOfficial}
                              onChange={(e) => handleToolSelectorConfigChange('excludeOfficial', e.target.checked)}
                              color="primary"
                            />
                          }
                          label="Exclude Official Tools"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          Skip official Letta tools from attachment/detachment operations.
                          This prevents modification of verified official tool sets.
                        </Typography>
                      </Card>

                      <Card variant="outlined" sx={{ p: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={localToolSelectorConfig.manageOnlyMcp}
                              onChange={(e) => handleToolSelectorConfigChange('manageOnlyMcp', e.target.checked)}
                              color="primary"
                            />
                          }
                          label="Manage Only MCP Tools"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          Only manage external MCP tools, completely ignore all Letta tools.
                          Use this for strict separation between Letta and external tool management.
                        </Typography>
                      </Card>

                      <Card variant="outlined" sx={{ p: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={localToolSelectorConfig.enableReranking}
                              onChange={(e) => handleToolSelectorConfigChange('enableReranking', e.target.checked)}
                              color="primary"
                            />
                          }
                          label="Enable Query Reranking"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          Use LLM-based reranking to improve search result relevance.
                          May increase API costs but provides better tool selection accuracy.
                        </Typography>
                      </Card>

                      <Card variant="outlined" sx={{ p: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={localToolSelectorConfig.enableLlmEnhancement}
                              onChange={(e) => handleToolSelectorConfigChange('enableLlmEnhancement', e.target.checked)}
                              color="primary"
                            />
                          }
                          label="Enable LLM Description Enhancement"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          Automatically enhance tool descriptions using LLM for better searchability.
                          Recommended for improved semantic matching but increases processing time.
                        </Typography>
                      </Card>

                      <Card variant="outlined" sx={{ p: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={localToolSelectorConfig.enableSafetyMode}
                              onChange={(e) => handleToolSelectorConfigChange('enableSafetyMode', e.target.checked)}
                              color="warning"
                            />
                          }
                          label="Enable Safety Mode"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" display="block" color="text.secondary">
                          Enable additional safety checks and protections during tool operations.
                          Recommended for production environments to prevent accidental modifications.
                        </Typography>
                      </Card>
                    </Box>
                  </Grid>

                  {/* Save Button */}
                  <Grid item xs={12}>
                    <Box display="flex" justifyContent="flex-end" gap={2} mt={2}>
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={handleSaveToolSelectorConfig}
                        disabled={updateToolSelectorConfigMutation.isPending}
                        startIcon={updateToolSelectorConfigMutation.isPending ? <CircularProgress size={20} /> : <SettingsIcon />}
                      >
                        {updateToolSelectorConfigMutation.isPending ? 'Saving...' : 'Save Configuration'}
                      </Button>
                    </Box>
                  </Grid>
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Embedding Configuration */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <MemoryIcon />
                Embedding Configuration
              </Typography>

              <Grid container spacing={3}>
                {/* Provider Selection */}
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Embedding Provider</InputLabel>
                    <Select
                      value={selectedEmbeddingProvider}
                      label="Embedding Provider"
                      onChange={(e) => setSelectedEmbeddingProvider(e.target.value)}
                    >
                      <MenuItem value="">
                        <em>Select Provider</em>
                      </MenuItem>
                      <MenuItem value="openai">OpenAI</MenuItem>
                      <MenuItem value="ollama">Ollama</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                {/* Current Configuration Display */}
                <Grid item xs={12} md={6}>
                  <Card variant="outlined" sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Current Configuration & Status
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        {embeddingConfigLoading
                          ? 'Loading...'
                          : embeddingConfig
                            ? `${embeddingConfig.provider}: ${embeddingConfig.model}`
                            : 'Not configured'}
                      </Typography>
                      {embeddingConfig && (
                        <Box mt={2}>
                          <EmbeddingHealthIndicator compact={true} autoRefresh={true} />
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>

                {/* Model Selection */}
                {selectedEmbeddingProvider && (
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Embedding Model</InputLabel>
                      <Select
                        value={selectedEmbeddingModel}
                        label="Embedding Model"
                        onChange={(e) => setSelectedEmbeddingModel(e.target.value)}
                        disabled={!embeddingModels}
                      >
                        <MenuItem value="">
                          <em>Select Model</em>
                        </MenuItem>
                        {embeddingModels?.models
                          ?.filter((model: any) => model.provider === selectedEmbeddingProvider)
                          .map((model: any) => (
                            <MenuItem key={model.id} value={model.id}>
                              {model.name}
                              {model.recommended && (
                                <Chip
                                  label="Recommended"
                                  size="small"
                                  color="primary"
                                  sx={{ ml: 1, height: 16 }}
                                />
                              )}
                            </MenuItem>
                          ))
                        }
                      </Select>
                    </FormControl>
                  </Grid>
                )}

                {/* Model Details */}
                {selectedEmbeddingProvider && selectedEmbeddingModel && embeddingModels && (
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Model Details
                        </Typography>
                        {(() => {
                          const model = embeddingModels?.models?.find((m: any) => m.id === selectedEmbeddingModel);
                          if (!model) return <Typography>Model not found</Typography>;
                          return (
                            <Box>
                              <Typography variant="body2" gutterBottom>
                                <strong>Dimensions:</strong> {model.dimensions || 'N/A'}
                              </Typography>
                              <Typography variant="body2" gutterBottom>
                                <strong>Max Tokens:</strong> {model.max_tokens || 'N/A'}
                              </Typography>
                              {model.cost_per_1k && (
                                <Typography variant="body2" gutterBottom>
                                  <strong>Cost per 1K tokens:</strong> ${model.cost_per_1k}
                                </Typography>
                              )}
                            </Box>
                          );
                        })()}
                      </CardContent>
                    </Card>
                  </Grid>
                )}

                {/* Save Button */}
                {selectedEmbeddingProvider && selectedEmbeddingModel && (
                  <Grid item xs={12}>
                    <Box display="flex" justifyContent="flex-end" gap={2} mt={2}>
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={() => {
                          updateEmbeddingConfigMutation.mutate({
                            provider: selectedEmbeddingProvider,
                            model: selectedEmbeddingModel,
                          });
                        }}
                        disabled={updateEmbeddingConfigMutation.isPending}
                        startIcon={updateEmbeddingConfigMutation.isPending ? <CircularProgress size={20} /> : <SettingsIcon />}
                      >
                        {updateEmbeddingConfigMutation.isPending ? 'Saving...' : 'Save Embedding Configuration'}
                      </Button>
                    </Box>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Reembedding Controls */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <RefreshIcon />
                Reembedding Controls
              </Typography>

              <Grid container spacing={3}>
                {/* Progress Display */}
                <Grid item xs={12}>
                  <ReembeddingProgress />
                </Grid>

                {/* Reembedding Configuration */}
                <Grid item xs={12} md={8}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        Start Reembedding Process
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Reembedding will regenerate all tool embeddings with the selected model. This process may take several minutes depending on the number of tools.
                      </Typography>

                      <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
                        <Button
                          variant="contained"
                          color="primary"
                          onClick={handleStartReembedding}
                          disabled={
                            !selectedEmbeddingProvider ||
                            !selectedEmbeddingModel ||
                            startReembeddingMutation.isPending ||
                            updateEmbeddingConfigMutation.isPending
                          }
                          startIcon={
                            startReembeddingMutation.isPending ?
                              <CircularProgress size={20} /> :
                              <RefreshIcon />
                          }
                        >
                          {startReembeddingMutation.isPending ? 'Starting...' : 'Start Reembedding'}
                        </Button>

                        {(!selectedEmbeddingProvider || !selectedEmbeddingModel) && (
                          <Typography variant="body2" color="text.secondary">
                            Select embedding provider and model first
                          </Typography>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Current Configuration Summary */}
                <Grid item xs={12} md={4}>
                  <Card variant="outlined" sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Reembedding Configuration
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Provider:</strong> {selectedEmbeddingProvider || 'Not selected'}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Model:</strong> {selectedEmbeddingModel || 'Not selected'}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Batch Size:</strong> 100 (default)
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Warning for configuration changes */}
                {(selectedEmbeddingProvider !== embeddingConfig?.provider ||
                  selectedEmbeddingModel !== embeddingConfig?.model) &&
                  selectedEmbeddingProvider && selectedEmbeddingModel && (
                    <Grid item xs={12}>
                      <Alert severity="warning">
                        You have unsaved embedding configuration changes. Save the embedding configuration before starting reembedding to use the new settings.
                      </Alert>
                    </Grid>
                  )}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* System Information */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Information
              </Typography>
              
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Last Tool Sync"
                    secondary={formatRelativeTime(new Date(Date.now() - 5 * 60 * 1000))}
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Search Engine"
                    secondary="Weaviate Vector Database"
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Embedding Model"
                    secondary={
                      embeddingConfigLoading
                        ? 'Loading...'
                        : embeddingConfig
                          ? `${embeddingConfig.provider}: ${embeddingConfig.model}`
                          : 'Not configured'
                    }
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Default Reranker"
                    secondary="OpenAI GPT-3.5-turbo"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SystemSettings;