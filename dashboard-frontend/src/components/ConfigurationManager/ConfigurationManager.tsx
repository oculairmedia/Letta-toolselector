import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Divider,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Checkbox,
  FormGroup,
} from '@mui/material';
import {
  Save as SaveIcon,
  RestartAlt as ResetIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Backup as BackupIcon,
  RestoreFromTrash as RestoreIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Timeline as TimelineIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';

interface SavedConfiguration {
  name: string;
  description: string;
  saved_at: string;
  configurations_count: number;
  include_secrets: boolean;
  version: string;
  file_size_bytes: number;
}

interface ResetAction {
  action: string;
  section: string;
  key?: string;
  new_value?: string;
  previous_value?: string;
  config_name?: string;
  config?: any;
}

const ConfigurationManager: React.FC = () => {
  const [savedConfigs, setSavedConfigs] = useState<SavedConfiguration[]>([]);
  const [loading, setLoading] = useState(true);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  // Save dialog state
  const [saveName, setSaveName] = useState('');
  const [saveDescription, setSaveDescription] = useState('');
  const [includeSecrets, setIncludeSecrets] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);

  // Reset dialog state
  const [resetType, setResetType] = useState<'defaults' | 'saved'>('defaults');
  const [selectedConfigName, setSelectedConfigName] = useState('');
  const [resetSections, setResetSections] = useState<string[]>([]);
  const [resetDryRun, setResetDryRun] = useState(true);
  const [resetLoading, setResetLoading] = useState(false);
  const [resetPreview, setResetPreview] = useState<ResetAction[] | null>(null);

  // Delete dialog state
  const [configToDelete, setConfigToDelete] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  const availableSections = [
    { id: 'tool_selector', label: 'Tool Selector Configuration' },
    { id: 'embedding', label: 'Embedding Configuration' },
    { id: 'weaviate', label: 'Weaviate Configuration' },
    { id: 'letta_api', label: 'Letta API Configuration' },
  ];

  const fetchSavedConfigurations = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/config/saves');
      const result = await response.json();

      if (result.success) {
        setSavedConfigs(result.data.saves || []);
      }
    } catch (error) {
      console.error('Failed to fetch saved configurations:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSavedConfigurations();
  }, []);

  const handleSaveConfiguration = async () => {
    try {
      setSaveLoading(true);
      const response = await fetch('/api/v1/config/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: saveName || undefined,
          description: saveDescription || undefined,
          include_secrets: includeSecrets
        })
      });

      const result = await response.json();
      if (result.success) {
        setSaveDialogOpen(false);
        setSaveName('');
        setSaveDescription('');
        setIncludeSecrets(false);
        fetchSavedConfigurations(); // Refresh list
      } else {
        console.error('Failed to save configuration:', result.error);
      }
    } catch (error) {
      console.error('Failed to save configuration:', error);
    } finally {
      setSaveLoading(false);
    }
  };

  const handlePreviewReset = async () => {
    try {
      setResetLoading(true);
      const response = await fetch('/api/v1/config/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: resetType,
          config_name: resetType === 'saved' ? selectedConfigName : undefined,
          sections: resetSections.length > 0 ? resetSections : undefined,
          dry_run: true
        })
      });

      const result = await response.json();
      if (result.success) {
        setResetPreview(result.data.actions);
      } else {
        console.error('Failed to preview reset:', result.error);
        setResetPreview([]);
      }
    } catch (error) {
      console.error('Failed to preview reset:', error);
      setResetPreview([]);
    } finally {
      setResetLoading(false);
    }
  };

  const handleExecuteReset = async () => {
    try {
      setResetLoading(true);
      const response = await fetch('/api/v1/config/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: resetType,
          config_name: resetType === 'saved' ? selectedConfigName : undefined,
          sections: resetSections.length > 0 ? resetSections : undefined,
          dry_run: false
        })
      });

      const result = await response.json();
      if (result.success) {
        setResetDialogOpen(false);
        setResetPreview(null);
        setResetSections([]);
        setSelectedConfigName('');
      } else {
        console.error('Failed to execute reset:', result.error);
      }
    } catch (error) {
      console.error('Failed to execute reset:', error);
    } finally {
      setResetLoading(false);
    }
  };

  const handleDeleteConfiguration = async () => {
    if (!configToDelete) return;

    try {
      setDeleteLoading(true);
      const response = await fetch(`/api/v1/config/saves/${configToDelete}`, {
        method: 'DELETE'
      });

      const result = await response.json();
      if (result.success) {
        setDeleteDialogOpen(false);
        setConfigToDelete(null);
        fetchSavedConfigurations(); // Refresh list
      } else {
        console.error('Failed to delete configuration:', result.error);
      }
    } catch (error) {
      console.error('Failed to delete configuration:', error);
    } finally {
      setDeleteLoading(false);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <StorageIcon />
          Configuration Save & Reset
        </Typography>

        <Box display="flex" alignItems="center" gap={1}>
          <Button
            startIcon={<SaveIcon />}
            onClick={() => setSaveDialogOpen(true)}
            variant="contained"
            size="small"
          >
            Save Config
          </Button>
          <Button
            startIcon={<ResetIcon />}
            onClick={() => setResetDialogOpen(true)}
            variant="outlined"
            size="small"
            color="warning"
          >
            Reset Config
          </Button>
          <IconButton onClick={fetchSavedConfigurations} size="small">
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Overview Cards */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="primary.main">
                {savedConfigs.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Saved Configurations
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="success.main">
                {savedConfigs.reduce((total, config) => total + config.file_size_bytes, 0) > 0
                  ? formatFileSize(savedConfigs.reduce((total, config) => total + config.file_size_bytes, 0))
                  : '0 B'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Storage Used
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="info.main">
                {savedConfigs.filter(c => c.include_secrets).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                With Secrets
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Saved Configurations List */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <BackupIcon />
            Saved Configurations
          </Typography>

          {loading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : savedConfigs.length === 0 ? (
            <Alert severity="info">
              No saved configurations found. Create your first configuration save above.
            </Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Description</TableCell>
                    <TableCell>Saved At</TableCell>
                    <TableCell>Configs</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Secrets</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {savedConfigs.map((config, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {config.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {config.description}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatTimestamp(config.saved_at)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={config.configurations_count}
                          size="small"
                          color="primary"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatFileSize(config.file_size_bytes)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {config.include_secrets ? (
                          <Chip label="Yes" size="small" color="warning" />
                        ) : (
                          <Chip label="No" size="small" color="default" />
                        )}
                      </TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => {
                            setConfigToDelete(config.name);
                            setDeleteDialogOpen(true);
                          }}
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Save Configuration Dialog */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SaveIcon />
          Save Current Configuration
        </DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            This will save all current system configuration settings. You can restore to this state later.
          </Alert>

          <TextField
            fullWidth
            label="Configuration Name"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="Leave blank for auto-generated name"
            sx={{ mb: 2 }}
          />

          <TextField
            fullWidth
            label="Description"
            value={saveDescription}
            onChange={(e) => setSaveDescription(e.target.value)}
            placeholder="Optional description"
            multiline
            rows={2}
            sx={{ mb: 2 }}
          />

          <FormControlLabel
            control={
              <Switch
                checked={includeSecrets}
                onChange={(e) => setIncludeSecrets(e.target.checked)}
              />
            }
            label="Include API keys and secrets (stored securely)"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleSaveConfiguration}
            variant="contained"
            disabled={saveLoading}
            startIcon={saveLoading ? <CircularProgress size={20} /> : <SaveIcon />}
          >
            {saveLoading ? 'Saving...' : 'Save Configuration'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Reset Configuration Dialog */}
      <Dialog open={resetDialogOpen} onClose={() => setResetDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ResetIcon />
          Reset Configuration
        </DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 3 }}>
            This will reset system configuration. This action cannot be undone without a backup.
          </Alert>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Reset Type</InputLabel>
                <Select
                  value={resetType}
                  label="Reset Type"
                  onChange={(e) => setResetType(e.target.value as any)}
                >
                  <MenuItem value="defaults">Reset to Defaults</MenuItem>
                  <MenuItem value="saved">Reset to Saved Configuration</MenuItem>
                </Select>
              </FormControl>

              {resetType === 'saved' && (
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Saved Configuration</InputLabel>
                  <Select
                    value={selectedConfigName}
                    label="Saved Configuration"
                    onChange={(e) => setSelectedConfigName(e.target.value)}
                  >
                    {savedConfigs.map((config) => (
                      <MenuItem key={config.name} value={config.name}>
                        {config.name} ({formatTimestamp(config.saved_at)})
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              <Typography variant="subtitle2" gutterBottom>
                Configuration Sections (leave empty for all)
              </Typography>
              <FormGroup sx={{ mb: 2 }}>
                {availableSections.map((section) => (
                  <FormControlLabel
                    key={section.id}
                    control={
                      <Checkbox
                        checked={resetSections.includes(section.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setResetSections([...resetSections, section.id]);
                          } else {
                            setResetSections(resetSections.filter(s => s !== section.id));
                          }
                        }}
                      />
                    }
                    label={section.label}
                  />
                ))}
              </FormGroup>

              <Button
                fullWidth
                onClick={handlePreviewReset}
                disabled={resetLoading}
                startIcon={resetLoading ? <CircularProgress size={20} /> : <InfoIcon />}
              >
                {resetLoading ? 'Loading Preview...' : 'Preview Changes'}
              </Button>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Reset Preview
              </Typography>
              {resetPreview ? (
                <Box sx={{ maxHeight: 300, overflowY: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
                  {resetPreview.length === 0 ? (
                    <Alert severity="info">No changes to preview</Alert>
                  ) : (
                    <List dense>
                      {resetPreview.map((action, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <CheckCircleIcon color="primary" fontSize="small" />
                          </ListItemIcon>
                          <ListItemText
                            primary={`${action.section}: ${action.key || 'Configuration'}`}
                            secondary={action.new_value || action.config_name || 'Reset action'}
                          />
                        </ListItem>
                      ))}
                    </List>
                  )}
                </Box>
              ) : (
                <Alert severity="info">
                  Click "Preview Changes" to see what will be reset
                </Alert>
              )}
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleExecuteReset}
            variant="contained"
            color="warning"
            disabled={!resetPreview || resetPreview.length === 0 || resetLoading}
            startIcon={resetLoading ? <CircularProgress size={20} /> : <ResetIcon />}
          >
            {resetLoading ? 'Resetting...' : 'Execute Reset'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Configuration Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DeleteIcon />
          Delete Configuration
        </DialogTitle>
        <DialogContent>
          <Alert severity="warning">
            Are you sure you want to delete the saved configuration "{configToDelete}"?
            This action cannot be undone.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleDeleteConfiguration}
            variant="contained"
            color="error"
            disabled={deleteLoading}
            startIcon={deleteLoading ? <CircularProgress size={20} /> : <DeleteIcon />}
          >
            {deleteLoading ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ConfigurationManager;