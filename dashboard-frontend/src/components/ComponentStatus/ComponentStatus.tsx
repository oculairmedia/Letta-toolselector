import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
  IconButton,
  Tooltip,
  LinearProgress,
  Collapse,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Timeline as TimelineIcon,
  Storage as StorageIcon,
  Computer as ComputerIcon,
  Cloud as CloudIcon,
  Memory as MemoryIcon,
  RestartAlt as RestartIcon,
  Build as BuildIcon,
} from '@mui/icons-material';
import { apiService } from '../../services/api';

interface ComponentDetails {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  type: 'service' | 'database' | 'external' | 'internal';
  version?: string;
  uptime?: number;
  memory_usage?: number;
  cpu_usage?: number;
  connections?: number;
  response_time?: number;
  last_check?: string;
  error_message?: string;
  details?: any;
  actions?: string[];
}

const ComponentStatus: React.FC = () => {
  const [components, setComponents] = useState<ComponentDetails[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedComponents, setExpandedComponents] = useState<Set<string>>(new Set());
  const [actionDialogOpen, setActionDialogOpen] = useState(false);
  const [selectedComponent, setSelectedComponent] = useState<ComponentDetails | null>(null);
  const [selectedAction, setSelectedAction] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  const fetchComponentStatus = async () => {
    try {
      setLoading(true);
      const maintenanceStatus = await apiService.getMaintenanceStatus();

      // Transform maintenance status into component details
      const componentDetails: ComponentDetails[] = [
        {
          name: 'API Server',
          type: 'service',
          status: 'healthy', // Assuming healthy since we got a response
          uptime: maintenanceStatus.system.uptime_seconds,
          memory_usage: maintenanceStatus.system.memory_usage.percent,
          cpu_usage: maintenanceStatus.system.cpu_info.cpu_percent,
          last_check: new Date().toISOString(),
          actions: ['restart', 'optimize'],
          details: {
            memory_rss: maintenanceStatus.system.memory_usage.rss_mb,
            memory_vms: maintenanceStatus.system.memory_usage.vms_mb,
            cpu_cores: maintenanceStatus.system.cpu_info.cpu_count,
            last_restart: maintenanceStatus.system.last_restart
          }
        },
        {
          name: 'Weaviate Database',
          type: 'database',
          status: maintenanceStatus.services.weaviate.available ? 'healthy' : 'error',
          version: maintenanceStatus.services.weaviate.version,
          connections: maintenanceStatus.services.weaviate.class_count,
          last_check: new Date().toISOString(),
          error_message: maintenanceStatus.services.weaviate.error,
          actions: ['restart', 'optimize', 'reindex'],
          details: {
            classes: maintenanceStatus.services.weaviate.class_count,
            url: maintenanceStatus.services.weaviate.url
          }
        },
        {
          name: 'Ollama Service',
          type: 'external',
          status: maintenanceStatus.services.ollama.available ? 'healthy' :
                   (maintenanceStatus.services.ollama.available === false ? 'warning' : 'unknown'),
          version: maintenanceStatus.services.ollama.version,
          connections: maintenanceStatus.services.ollama.model_count,
          last_check: new Date().toISOString(),
          error_message: maintenanceStatus.services.ollama.error,
          actions: ['test_connection', 'restart'],
          details: {
            models: maintenanceStatus.services.ollama.model_count,
            host: maintenanceStatus.services.ollama.host,
            port: maintenanceStatus.services.ollama.port
          }
        },
        {
          name: 'Letta API',
          type: 'external',
          status: maintenanceStatus.services.letta_api.available ? 'healthy' : 'error',
          last_check: new Date().toISOString(),
          error_message: maintenanceStatus.services.letta_api.error,
          actions: ['test_connection'],
          details: {
            status: maintenanceStatus.services.letta_api.status
          }
        },
        {
          name: 'Tool Cache',
          type: 'internal',
          status: maintenanceStatus.database.tool_count > 0 ? 'healthy' : 'warning',
          last_check: maintenanceStatus.database.last_sync,
          actions: ['refresh', 'clear'],
          details: {
            tool_count: maintenanceStatus.database.tool_count,
            cache_size: maintenanceStatus.database.cache_size,
            index_status: maintenanceStatus.database.index_status
          }
        },
        {
          name: 'Log System',
          type: 'internal',
          status: maintenanceStatus.logs.error_count === 0 ? 'healthy' :
                   (maintenanceStatus.logs.error_count < 10 ? 'warning' : 'error'),
          actions: ['view_logs', 'clear_logs', 'rotate'],
          details: {
            log_level: maintenanceStatus.logs.log_level,
            log_size: maintenanceStatus.logs.log_size,
            error_count: maintenanceStatus.logs.error_count,
            warning_count: maintenanceStatus.logs.warning_count
          }
        }
      ];

      setComponents(componentDetails);
    } catch (error) {
      console.error('Failed to fetch component status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComponentStatus();

    // Auto-refresh every minute
    const interval = setInterval(fetchComponentStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleIcon color="success" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'error': return <ErrorIcon color="error" />;
      default: return <InfoIcon color="disabled" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'service': return <ComputerIcon />;
      case 'database': return <StorageIcon />;
      case 'external': return <CloudIcon />;
      case 'internal': return <MemoryIcon />;
      default: return <SettingsIcon />;
    }
  };

  const toggleExpanded = (componentName: string) => {
    const newExpanded = new Set(expandedComponents);
    if (newExpanded.has(componentName)) {
      newExpanded.delete(componentName);
    } else {
      newExpanded.add(componentName);
    }
    setExpandedComponents(newExpanded);
  };

  const handleActionClick = (component: ComponentDetails, action: string) => {
    setSelectedComponent(component);
    setSelectedAction(action);
    setActionDialogOpen(true);
  };

  const executeAction = async () => {
    if (!selectedComponent || !selectedAction) return;

    setActionLoading(true);
    try {
      switch (selectedAction) {
        case 'restart':
          await apiService.restartSystemComponents([selectedComponent.name.toLowerCase().replace(' ', '_')]);
          break;
        case 'optimize':
          await apiService.optimizeSystem([selectedComponent.name.toLowerCase().replace(' ', '_')]);
          break;
        case 'refresh':
          if (selectedComponent.name === 'Tool Cache') {
            // Implement tool cache refresh
            console.log('Refreshing tool cache...');
          }
          break;
        case 'test_connection':
          // Test connection logic would go here
          console.log('Testing connection...');
          break;
        default:
          console.log(`Executing action: ${selectedAction} on ${selectedComponent.name}`);
      }

      // Refresh component status after action
      await fetchComponentStatus();
    } catch (error) {
      console.error('Failed to execute action:', error);
    } finally {
      setActionLoading(false);
      setActionDialogOpen(false);
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>Component Status</Typography>
        <LinearProgress />
      </Box>
    );
  }

  const healthyCount = components.filter(c => c.status === 'healthy').length;
  const warningCount = components.filter(c => c.status === 'warning').length;
  const errorCount = components.filter(c => c.status === 'error').length;

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TimelineIcon />
          Component Status Monitor
        </Typography>

        <Box display="flex" alignItems="center" gap={2}>
          <Chip label={`${healthyCount} Healthy`} color="success" size="small" />
          {warningCount > 0 && <Chip label={`${warningCount} Warning`} color="warning" size="small" />}
          {errorCount > 0 && <Chip label={`${errorCount} Error`} color="error" size="small" />}

          <IconButton onClick={fetchComponentStatus} size="small">
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {errorCount > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {errorCount} component{errorCount > 1 ? 's' : ''} require{errorCount === 1 ? 's' : ''} attention
        </Alert>
      )}

      <Grid container spacing={2}>
        {components.map((component) => {
          const isExpanded = expandedComponents.has(component.name);

          return (
            <Grid item xs={12} md={6} key={component.name}>
              <Card variant="outlined" sx={{ height: 'fit-content' }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Avatar sx={{ width: 32, height: 32, bgcolor: 'grey.100' }}>
                        {getTypeIcon(component.type)}
                      </Avatar>
                      <Box>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                          {component.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {component.type}
                        </Typography>
                      </Box>
                    </Box>

                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip
                        icon={getStatusIcon(component.status)}
                        label={component.status.toUpperCase()}
                        color={getStatusColor(component.status)}
                        size="small"
                      />
                      <IconButton
                        size="small"
                        onClick={() => toggleExpanded(component.name)}
                      >
                        {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                      </IconButton>
                    </Box>
                  </Box>

                  {/* Quick Stats */}
                  <Box display="flex" gap={2} mb={1}>
                    {component.version && (
                      <Typography variant="caption" color="text.secondary">
                        v{component.version}
                      </Typography>
                    )}
                    {component.uptime && (
                      <Typography variant="caption" color="text.secondary">
                        Up {formatUptime(component.uptime)}
                      </Typography>
                    )}
                    {component.response_time && (
                      <Typography variant="caption" color="text.secondary">
                        {component.response_time}ms
                      </Typography>
                    )}
                  </Box>

                  {component.error_message && (
                    <Alert severity="error" sx={{ mb: 1 }}>
                      {component.error_message}
                    </Alert>
                  )}

                  {/* Resource Usage */}
                  {(component.memory_usage || component.cpu_usage) && (
                    <Box mb={1}>
                      {component.memory_usage && (
                        <Box mb={0.5}>
                          <Typography variant="caption" color="text.secondary">
                            Memory: {component.memory_usage}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={component.memory_usage}
                            color={component.memory_usage > 80 ? 'warning' : 'primary'}
                            sx={{ height: 6 }}
                          />
                        </Box>
                      )}
                      {component.cpu_usage && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            CPU: {component.cpu_usage}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={component.cpu_usage}
                            color={component.cpu_usage > 80 ? 'warning' : 'primary'}
                            sx={{ height: 6 }}
                          />
                        </Box>
                      )}
                    </Box>
                  )}

                  {/* Expanded Details */}
                  <Collapse in={isExpanded}>
                    <Box pt={1} borderTop={1} borderColor="grey.200">
                      <Typography variant="subtitle2" gutterBottom>Details</Typography>

                      {component.details && (
                        <Grid container spacing={1} sx={{ mb: 1 }}>
                          {Object.entries(component.details).map(([key, value]) => (
                            <Grid item xs={6} key={key}>
                              <Typography variant="caption" color="text.secondary" display="block">
                                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </Typography>
                              <Typography variant="body2">
                                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                              </Typography>
                            </Grid>
                          ))}
                        </Grid>
                      )}

                      {/* Actions */}
                      {component.actions && component.actions.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>Actions</Typography>
                          <Box display="flex" gap={1} flexWrap="wrap">
                            {component.actions.map((action) => (
                              <Button
                                key={action}
                                size="small"
                                variant="outlined"
                                onClick={() => handleActionClick(component, action)}
                                startIcon={action === 'restart' ? <RestartIcon /> : <BuildIcon />}
                              >
                                {action.replace(/_/g, ' ')}
                              </Button>
                            ))}
                          </Box>
                        </Box>
                      )}

                      {component.last_check && (
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          Last checked: {new Date(component.last_check).toLocaleString()}
                        </Typography>
                      )}
                    </Box>
                  </Collapse>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Action Confirmation Dialog */}
      <Dialog open={actionDialogOpen} onClose={() => setActionDialogOpen(false)}>
        <DialogTitle>Confirm Action</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to {selectedAction?.replace(/_/g, ' ')} {selectedComponent?.name}?
          </Typography>
          {selectedAction === 'restart' && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              This action may temporarily interrupt service availability.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setActionDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={executeAction}
            variant="contained"
            disabled={actionLoading}
            startIcon={actionLoading ? <CircularProgress size={16} /> : undefined}
          >
            {actionLoading ? 'Executing...' : 'Execute'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ComponentStatus;