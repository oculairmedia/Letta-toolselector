import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  CircularProgress,
  Tooltip,
  Grid,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Science as TestIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';

import { 
  useRerankerModelRegistry, 
  useRegisterRerankerModel,
  useUnregisterRerankerModel,
  useTestRerankerModel,
  useUpdateRerankerModel
} from '../../hooks/useApi';

interface ModelRegistryManagerProps {}

const ModelRegistryManager: React.FC<ModelRegistryManagerProps> = () => {
  const [registerDialogOpen, setRegisterDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [newModelData, setNewModelData] = useState({
    id: '',
    name: '',
    provider: 'ollama',
    type: 'cross-encoder',
    description: '',
    notes: '',
  });

  const { data: registryData, isLoading, refetch } = useRerankerModelRegistry();
  const registerModelMutation = useRegisterRerankerModel();
  const unregisterModelMutation = useUnregisterRerankerModel();
  const testModelMutation = useTestRerankerModel();
  const updateModelMutation = useUpdateRerankerModel();

  const handleRegisterModel = async () => {
    try {
      await registerModelMutation.mutateAsync(newModelData);
      setRegisterDialogOpen(false);
      setNewModelData({
        id: '',
        name: '',
        provider: 'ollama',
        type: 'cross-encoder',
        description: '',
        notes: '',
      });
    } catch (error) {
      console.error('Failed to register model:', error);
    }
  };

  const handleTestModel = async (modelId: string) => {
    try {
      await testModelMutation.mutateAsync(modelId);
    } catch (error) {
      console.error('Model test failed:', error);
    }
  };

  const handleUnregisterModel = async (modelId: string) => {
    if (window.confirm(`Are you sure you want to unregister model "${modelId}"?`)) {
      try {
        await unregisterModelMutation.mutateAsync(modelId);
      } catch (error) {
        console.error('Failed to unregister model:', error);
      }
    }
  };

  const getStatusColor = (status: string, validated: boolean) => {
    if (validated) return 'success';
    if (status === 'pending') return 'warning';
    if (status === 'failed') return 'error';
    return 'default';
  };

  const getStatusIcon = (status: string, validated: boolean) => {
    if (validated) return <CheckCircleIcon fontSize="small" />;
    if (status === 'failed') return <ErrorIcon fontSize="small" />;
    return undefined;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <StorageIcon />
        Model Registry Manager
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Manage registered reranker models, test their functionality, and monitor their status.
      </Typography>

      {/* Registry Summary */}
      {registryData && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="primary">
                  {registryData.total}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Models
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="success.main">
                  {registryData.validated_count}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Validated
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="warning.main">
                  {registryData.pending_count}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Pending
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h4">
                  {registryData.providers.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Providers
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Actions */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setRegisterDialogOpen(true)}
        >
          Register Model
        </Button>
        <Button
          variant="outlined"
          startIcon={isLoading ? <CircularProgress size={16} /> : <RefreshIcon />}
          onClick={() => refetch()}
          disabled={isLoading}
        >
          Refresh Registry
        </Button>
      </Box>

      {/* Models Table */}
      <Card>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell>Provider</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Last Tested</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={6} sx={{ textAlign: 'center', py: 4 }}>
                    <CircularProgress />
                  </TableCell>
                </TableRow>
              ) : registryData?.models?.length ? (
                registryData.models.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell>
                      <Box>
                        <Typography variant="body2" fontWeight="medium">
                          {model.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {model.id}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip label={model.provider} size="small" />
                    </TableCell>
                    <TableCell>
                      <Chip label={model.type} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={model.validated ? 'Validated' : model.test_status}
                        size="small"
                        color={getStatusColor(model.test_status, model.validated)}
                        icon={getStatusIcon(model.test_status, model.validated)}
                      />
                      {model.recommended && (
                        <Chip label="Recommended" size="small" color="primary" sx={{ ml: 1 }} />
                      )}
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">
                        {model.last_tested 
                          ? new Date(model.last_tested).toLocaleDateString()
                          : 'Never'
                        }
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Test Model">
                          <IconButton
                            size="small"
                            onClick={() => handleTestModel(model.id)}
                            disabled={testModelMutation.isPending}
                          >
                            <TestIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Model Details">
                          <IconButton
                            size="small"
                            onClick={() => setSelectedModel(model)}
                          >
                            <InfoIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Unregister Model">
                          <IconButton
                            size="small"
                            onClick={() => handleUnregisterModel(model.id)}
                            disabled={unregisterModelMutation.isPending}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={6} sx={{ textAlign: 'center', py: 4 }}>
                    <Typography color="text.secondary">
                      No models registered
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {/* Register Model Dialog */}
      <Dialog open={registerDialogOpen} onClose={() => setRegisterDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Register New Model</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Model ID"
                value={newModelData.id}
                onChange={(e) => setNewModelData({ ...newModelData, id: e.target.value })}
                helperText="Unique identifier for the model (e.g., llama2:7b)"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Model Name"
                value={newModelData.name}
                onChange={(e) => setNewModelData({ ...newModelData, name: e.target.value })}
                helperText="Human-readable name for the model"
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                select
                label="Provider"
                value={newModelData.provider}
                onChange={(e) => setNewModelData({ ...newModelData, provider: e.target.value })}
                SelectProps={{ native: true }}
              >
                <option value="ollama">Ollama</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="cohere">Cohere</option>
                <option value="custom">Custom</option>
              </TextField>
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                select
                label="Type"
                value={newModelData.type}
                onChange={(e) => setNewModelData({ ...newModelData, type: e.target.value })}
                SelectProps={{ native: true }}
              >
                <option value="cross-encoder">Cross-Encoder</option>
                <option value="bi-encoder">Bi-Encoder</option>
                <option value="generative">Generative</option>
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Description"
                value={newModelData.description}
                onChange={(e) => setNewModelData({ ...newModelData, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Registry Notes"
                value={newModelData.notes}
                onChange={(e) => setNewModelData({ ...newModelData, notes: e.target.value })}
              />
            </Grid>
          </Grid>
          
          {registerModelMutation.error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {registerModelMutation.error.message}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRegisterDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleRegisterModel}
            variant="contained"
            disabled={!newModelData.id || !newModelData.name || registerModelMutation.isPending}
          >
            {registerModelMutation.isPending ? 'Registering...' : 'Register'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Model Details Dialog */}
      <Dialog open={!!selectedModel} onClose={() => setSelectedModel(null)} maxWidth="md" fullWidth>
        <DialogTitle>Model Details</DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>Model ID</Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>{selectedModel.id}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>Name</Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>{selectedModel.name}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>Provider</Typography>
                <Chip label={selectedModel.provider} size="small" />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>Type</Typography>
                <Chip label={selectedModel.type} size="small" variant="outlined" />
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>Status</Typography>
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Chip
                    label={selectedModel.validated ? 'Validated' : selectedModel.test_status}
                    size="small"
                    color={getStatusColor(selectedModel.test_status, selectedModel.validated)}
                    icon={getStatusIcon(selectedModel.test_status, selectedModel.validated)}
                  />
                  {selectedModel.recommended && (
                    <Chip label="Recommended" size="small" color="primary" />
                  )}
                </Box>
              </Grid>
              {selectedModel.registry_notes && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>Registry Notes</Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>{selectedModel.registry_notes}</Typography>
                </Grid>
              )}
              {selectedModel.test_results && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>Last Test Results</Typography>
                  <Alert 
                    severity={selectedModel.test_results.connectivity && selectedModel.test_results.functionality ? 'success' : 'error'}
                  >
                    <Typography variant="body2">
                      Connectivity: {selectedModel.test_results.connectivity ? '✓' : '✗'}<br />
                      Functionality: {selectedModel.test_results.functionality ? '✓' : '✗'}<br />
                      {selectedModel.test_results.latency_ms && `Latency: ${selectedModel.test_results.latency_ms}ms`}<br />
                      {selectedModel.test_results.error && `Error: ${selectedModel.test_results.error}`}
                    </Typography>
                  </Alert>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedModel(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ModelRegistryManager;