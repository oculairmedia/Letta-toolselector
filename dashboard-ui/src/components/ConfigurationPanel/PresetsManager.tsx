import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  Bookmark as BookmarkIcon,
} from '@mui/icons-material';

import {
  useConfigurationPresets,
  useCreateConfigurationPreset,
  useUpdateConfigurationPreset,
  useDeleteConfigurationPreset,
} from '../../hooks/useApi';
import { ConfigurationPreset, RerankerConfig } from '../../types';

const PresetsManager: React.FC = () => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingPreset, setEditingPreset] = useState<ConfigurationPreset | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState<string | null>(null);
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    config: {
      enabled: true,
      model: 'gpt-3.5-turbo',
      provider: 'openai',
      parameters: {
        temperature: 0.1,
        max_tokens: 150,
      },
    } as RerankerConfig,
  });

  const { data: presets, isLoading } = useConfigurationPresets();
  const createPresetMutation = useCreateConfigurationPreset();
  const updatePresetMutation = useUpdateConfigurationPreset();
  const deletePresetMutation = useDeleteConfigurationPreset();

  const handleCreatePreset = () => {
    setEditingPreset(null);
    setFormData({
      name: '',
      description: '',
      config: {
        enabled: true,
        model: 'gpt-3.5-turbo',
        provider: 'openai',
        parameters: {
          temperature: 0.1,
          max_tokens: 150,
        },
      },
    });
    setDialogOpen(true);
  };

  const handleEditPreset = (preset: ConfigurationPreset) => {
    setEditingPreset(preset);
    setFormData({
      name: preset.name,
      description: preset.description,
      config: preset.config,
    });
    setDialogOpen(true);
  };

  const handleSavePreset = async () => {
    try {
      if (editingPreset) {
        await updatePresetMutation.mutateAsync({
          id: editingPreset.id,
          preset: formData,
        });
      } else {
        await createPresetMutation.mutateAsync(formData);
      }
      setDialogOpen(false);
    } catch (error) {
      console.error('Failed to save preset:', error);
    }
  };

  const handleDeletePreset = async (id: string) => {
    try {
      await deletePresetMutation.mutateAsync(id);
      setDeleteConfirmOpen(null);
    } catch (error) {
      console.error('Failed to delete preset:', error);
    }
  };

  const handleSetDefault = async (preset: ConfigurationPreset) => {
    try {
      await updatePresetMutation.mutateAsync({
        id: preset.id,
        preset: { isDefault: true },
      });
    } catch (error) {
      console.error('Failed to set default preset:', error);
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
        <BookmarkIcon />
        Configuration Presets
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Save and manage different reranker configurations for different use cases and scenarios.
      </Typography>

      {/* Add Preset Button */}
      <Box sx={{ mb: 3 }}>
        <Button
          variant="contained"
          onClick={handleCreatePreset}
          startIcon={<AddIcon />}
        >
          Create New Preset
        </Button>
      </Box>

      {/* Presets List */}
      {presets && presets.length > 0 ? (
        <Grid container spacing={2}>
          {presets.map((preset) => (
            <Grid item xs={12} md={6} lg={4} key={preset.id}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {preset.name}
                      {preset.isDefault && <StarIcon color="primary" fontSize="small" />}
                    </Typography>
                    
                    <Box>
                      <IconButton
                        size="small"
                        onClick={() => handleSetDefault(preset)}
                        disabled={preset.isDefault}
                      >
                        {preset.isDefault ? <StarIcon color="primary" /> : <StarBorderIcon />}
                      </IconButton>
                      <IconButton size="small" onClick={() => handleEditPreset(preset)}>
                        <EditIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => setDeleteConfirmOpen(preset.id)}
                        disabled={preset.isDefault}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </Box>

                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {preset.description}
                  </Typography>

                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    <Chip
                      label={preset.config.enabled ? 'Enabled' : 'Disabled'}
                      color={preset.config.enabled ? 'success' : 'default'}
                      size="small"
                    />
                    <Chip
                      label={`${preset.config.provider}/${preset.config.model}`}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={`T: ${preset.config.parameters.temperature}`}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 4 }}>
            <BookmarkIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No presets found
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Create your first configuration preset to get started
            </Typography>
            <Button variant="outlined" onClick={handleCreatePreset} startIcon={<AddIcon />}>
              Create Preset
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Create/Edit Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingPreset ? 'Edit Preset' : 'Create New Preset'}
        </DialogTitle>
        
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              fullWidth
              label="Preset Name"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              required
            />
            
            <TextField
              fullWidth
              label="Description"
              multiline
              rows={2}
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
            />

            {/* Configuration Preview */}
            <Card sx={{ bgcolor: 'action.hover' }}>
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Configuration Summary
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  <Chip
                    label={formData.config.enabled ? 'Enabled' : 'Disabled'}
                    color={formData.config.enabled ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip label={`Provider: ${formData.config.provider}`} size="small" />
                  <Chip label={`Model: ${formData.config.model}`} size="small" />
                  <Chip label={`Temperature: ${formData.config.parameters.temperature}`} size="small" />
                  <Chip label={`Max Tokens: ${formData.config.parameters.max_tokens}`} size="small" />
                </Box>
              </CardContent>
            </Card>

            <Alert severity="info">
              This preset will save the current reranker configuration. Make sure to configure 
              the reranker settings first in the "Reranker Settings" tab.
            </Alert>
          </Box>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSavePreset}
            variant="contained"
            disabled={!formData.name.trim() || createPresetMutation.isPending || updatePresetMutation.isPending}
          >
            {(createPresetMutation.isPending || updatePresetMutation.isPending) ? 'Saving...' : 'Save Preset'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={!!deleteConfirmOpen} onClose={() => setDeleteConfirmOpen(null)}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this preset? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(null)}>
            Cancel
          </Button>
          <Button
            onClick={() => deleteConfirmOpen && handleDeletePreset(deleteConfirmOpen)}
            color="error"
            disabled={deletePresetMutation.isPending}
          >
            {deletePresetMutation.isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Success/Error Messages */}
      {(createPresetMutation.isSuccess || updatePresetMutation.isSuccess) && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Preset saved successfully!
        </Alert>
      )}

      {deletePresetMutation.isSuccess && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Preset deleted successfully!
        </Alert>
      )}
    </Box>
  );
};

export default PresetsManager;