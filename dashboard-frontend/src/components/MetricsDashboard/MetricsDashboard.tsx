import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  LinearProgress,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Timeline as TimelineIcon,
  Computer as ComputerIcon,
  Database as DatabaseIcon,
} from '@mui/icons-material';
import { apiService } from '../../services/api';

interface SystemMetrics {
  system: {
    uptime_seconds: number;
    memory_usage: {
      rss_mb: number;
      vms_mb: number;
      percent: number;
    };
    disk_usage: {
      total_gb: number;
      used_gb: number;
      free_gb: number;
      percent: number;
    };
    cpu_info: {
      cpu_percent: number;
      cpu_count: number;
      load_average?: number[];
    };
    last_restart: string;
  };
  services: {
    weaviate: {
      available: boolean;
      version?: string;
      class_count?: number;
      error?: string;
    };
    ollama: {
      available: boolean;
      version?: string;
      model_count?: number;
      error?: string;
    };
    letta_api: {
      available: boolean;
      status?: string;
      error?: string;
    };
  };
  database: {
    tool_count: number;
    cache_size: number;
    last_sync: string;
    index_status: {
      status: string;
      class_count?: number;
      version?: string;
      error?: string;
    };
  };
  logs: {
    log_level: string;
    log_size: any;
    error_count: number;
    warning_count: number;
  };
  health: {
    status: string;
    issues: string[];
    last_check: string;
  };
}

const MetricsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchMetrics = async () => {
    try {
      const response = await apiService.getMaintenanceStatus();
      setMetrics(response);
      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchMetrics, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'degraded': return 'warning';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const getServiceStatusIcon = (available: boolean) => {
    return available ?
      <CheckCircleIcon color="success" fontSize="small" /> :
      <ErrorIcon color="error" fontSize="small" />;
  };

  if (loading && !metrics) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
        <CircularProgress />
      </Box>
    );
  }

  if (error && !metrics) {
    return (
      <Alert severity="error" action={
        <IconButton onClick={fetchMetrics} size="small">
          <RefreshIcon />
        </IconButton>
      }>
        {error}
      </Alert>
    );
  }

  if (!metrics) return null;

  return (
    <Box>
      {/* Header with refresh controls */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TimelineIcon />
          Real-time Metrics Dashboard
        </Typography>

        <Box display="flex" alignItems="center" gap={2}>
          {lastUpdate && (
            <Typography variant="body2" color="text.secondary">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </Typography>
          )}

          <Tooltip title="Refresh metrics">
            <IconButton onClick={fetchMetrics} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Health Status Overview */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">System Health</Typography>
                <Chip
                  label={metrics.health.status.toUpperCase()}
                  color={getStatusColor(metrics.health.status)}
                  variant="filled"
                />
              </Box>

              {metrics.health.issues.length > 0 && (
                <Alert severity={metrics.health.status === 'critical' ? 'error' : 'warning'} sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Issues detected:</Typography>
                  <List dense>
                    {metrics.health.issues.map((issue, index) => (
                      <ListItem key={index} sx={{ py: 0 }}>
                        <ListItemIcon sx={{ minWidth: 20 }}>
                          <WarningIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={issue} />
                      </ListItem>
                    ))}
                  </List>
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* System Resources */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ComputerIcon />
                System Resources
              </Typography>

              {/* Uptime */}
              <Box mb={2}>
                <Typography variant="body2" color="text.secondary">Uptime</Typography>
                <Typography variant="h6">{formatUptime(metrics.system.uptime_seconds)}</Typography>
              </Box>

              {/* Memory Usage */}
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">Memory Usage</Typography>
                  <Typography variant="body2">{metrics.system.memory_usage.percent}%</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={metrics.system.memory_usage.percent}
                  color={metrics.system.memory_usage.percent > 80 ? 'warning' : 'primary'}
                />
                <Typography variant="caption" color="text.secondary">
                  {metrics.system.memory_usage.rss_mb} MB RSS / {metrics.system.memory_usage.vms_mb} MB VMS
                </Typography>
              </Box>

              {/* Disk Usage */}
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">Disk Usage</Typography>
                  <Typography variant="body2">{metrics.system.disk_usage.percent}%</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={metrics.system.disk_usage.percent}
                  color={metrics.system.disk_usage.percent > 85 ? 'warning' : 'primary'}
                />
                <Typography variant="caption" color="text.secondary">
                  {metrics.system.disk_usage.used_gb} GB / {metrics.system.disk_usage.total_gb} GB
                </Typography>
              </Box>

              {/* CPU Usage */}
              <Box>
                <Typography variant="body2" color="text.secondary">CPU Usage</Typography>
                <Typography variant="h6">{metrics.system.cpu_info.cpu_percent}%</Typography>
                <Typography variant="caption" color="text.secondary">
                  {metrics.system.cpu_info.cpu_count} cores
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Services Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SpeedIcon />
                Services Status
              </Typography>

              <List dense>
                <ListItem>
                  <ListItemIcon>
                    {getServiceStatusIcon(metrics.services.weaviate.available)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Weaviate Database"
                    secondary={
                      metrics.services.weaviate.available
                        ? `v${metrics.services.weaviate.version} - ${metrics.services.weaviate.class_count} classes`
                        : metrics.services.weaviate.error
                    }
                  />
                </ListItem>

                <ListItem>
                  <ListItemIcon>
                    {getServiceStatusIcon(metrics.services.ollama.available)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Ollama Service"
                    secondary={
                      metrics.services.ollama.available
                        ? `v${metrics.services.ollama.version} - ${metrics.services.ollama.model_count} models`
                        : metrics.services.ollama.error || 'Service unavailable'
                    }
                  />
                </ListItem>

                <ListItem>
                  <ListItemIcon>
                    {getServiceStatusIcon(metrics.services.letta_api.available)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Letta API"
                    secondary={
                      metrics.services.letta_api.available
                        ? metrics.services.letta_api.status
                        : metrics.services.letta_api.error
                    }
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Database Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <DatabaseIcon />
                Database & Cache
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Tool Count</Typography>
                  <Typography variant="h5">{metrics.database.tool_count.toLocaleString()}</Typography>
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Cache Size</Typography>
                  <Typography variant="h5">{metrics.database.cache_size} MB</Typography>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">Index Status</Typography>
                  <Chip
                    label={metrics.database.index_status.status}
                    color={metrics.database.index_status.status === 'healthy' ? 'success' : 'warning'}
                    size="small"
                  />
                </Grid>

                {metrics.database.last_sync && (
                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary">Last Sync</Typography>
                    <Typography variant="body2">
                      {new Date(metrics.database.last_sync).toLocaleString()}
                    </Typography>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Logs & Monitoring */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <MemoryIcon />
                Logs & Monitoring
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Log Level</Typography>
                  <Chip label={metrics.logs.log_level} size="small" />
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Log Size</Typography>
                  <Typography variant="body2">
                    {typeof metrics.logs.log_size === 'object'
                      ? `${metrics.logs.log_size.size_mb || 0} MB`
                      : 'N/A'
                    }
                  </Typography>
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Recent Errors</Typography>
                  <Typography variant="h6" color={metrics.logs.error_count > 0 ? 'error.main' : 'text.primary'}>
                    {metrics.logs.error_count}
                  </Typography>
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Recent Warnings</Typography>
                  <Typography variant="h6" color={metrics.logs.warning_count > 0 ? 'warning.main' : 'text.primary'}>
                    {metrics.logs.warning_count}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MetricsDashboard;