import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  CheckCircle as HealthyIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Refresh as RefreshIcon,
  Speed as PerformanceIcon,
  Memory as MemoryIcon,
  CloudSync as SyncIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

interface EmbeddingHealth {
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  provider: string;
  model: string;
  availability: boolean;
  response_time_ms: number;
  last_checked: string;
  error_message?: string;
  performance_metrics?: {
    avg_response_time: number;
    success_rate: number;
    total_requests: number;
    failed_requests: number;
  };
  model_info?: {
    dimensions: number;
    max_tokens: number;
    cost_per_1k?: number;
  };
}

interface EmbeddingHealthIndicatorProps {
  compact?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const EmbeddingHealthIndicator: React.FC<EmbeddingHealthIndicatorProps> = ({
  compact = false,
  autoRefresh = true,
  refreshInterval = 30000, // 30 seconds
}) => {
  const [healthData, setHealthData] = useState<EmbeddingHealth | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchEmbeddingHealth = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/embedding/health');
      if (response.ok) {
        const data = await response.json();
        setHealthData(data.success ? data.data : null);
        setLastRefresh(new Date());
      } else {
        console.error('Failed to fetch embedding health');
        setHealthData(null);
      }
    } catch (error) {
      console.error('Error fetching embedding health:', error);
      setHealthData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEmbeddingHealth();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchEmbeddingHealth, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <HealthyIcon color="success" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatResponseTime = (ms: number) => {
    if (ms < 1000) {
      return `${ms}ms`;
    }
    return `${(ms / 1000).toFixed(1)}s`;
  };

  if (compact && healthData) {
    return (
      <Box display="flex" alignItems="center" gap={1}>
        {getStatusIcon(healthData.status)}
        <Typography variant="body2">
          {healthData.provider}: {healthData.model}
        </Typography>
        <Chip
          size="small"
          label={healthData.status}
          color={getStatusColor(healthData.status) as any}
        />
        <Typography variant="caption" color="text.secondary">
          {formatResponseTime(healthData.response_time_ms)}
        </Typography>
        {loading && <CircularProgress size={16} />}
      </Box>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <MemoryIcon />
            Embedding Model Health
          </Typography>
          <Box display="flex" alignItems="center" gap={1}>
            {lastRefresh && (
              <Typography variant="caption" color="text.secondary">
                Last checked: {lastRefresh.toLocaleTimeString()}
              </Typography>
            )}
            <Tooltip title="Refresh health status">
              <IconButton size="small" onClick={fetchEmbeddingHealth} disabled={loading}>
                {loading ? <CircularProgress size={20} /> : <RefreshIcon />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {loading && !healthData && <LinearProgress sx={{ mb: 2 }} />}

        {healthData ? (
          <Box>
            {/* Status Overview */}
            <Alert
              severity={getStatusColor(healthData.status) as any}
              sx={{ mb: 2 }}
              icon={getStatusIcon(healthData.status)}
            >
              <Typography variant="body2">
                <strong>{healthData.provider}: {healthData.model}</strong> is {healthData.status}
                {healthData.error_message && ` - ${healthData.error_message}`}
              </Typography>
            </Alert>

            {/* Detailed Metrics */}
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <SyncIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Availability"
                  secondary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip
                        size="small"
                        label={healthData.availability ? 'Online' : 'Offline'}
                        color={healthData.availability ? 'success' : 'error'}
                      />
                    </Box>
                  }
                />
              </ListItem>

              <ListItem>
                <ListItemIcon>
                  <PerformanceIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Response Time"
                  secondary={formatResponseTime(healthData.response_time_ms)}
                />
              </ListItem>

              {healthData.model_info && (
                <>
                  <Divider />
                  <ListItem>
                    <ListItemIcon>
                      <InfoIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Model Dimensions"
                      secondary={healthData.model_info.dimensions?.toLocaleString()}
                    />
                  </ListItem>

                  <ListItem>
                    <ListItemIcon>
                      <MemoryIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Max Tokens"
                      secondary={healthData.model_info.max_tokens?.toLocaleString()}
                    />
                  </ListItem>

                  {healthData.model_info.cost_per_1k && (
                    <ListItem>
                      <ListItemText
                        primary="Cost per 1K tokens"
                        secondary={`$${healthData.model_info.cost_per_1k}`}
                      />
                    </ListItem>
                  )}
                </>
              )}

              {healthData.performance_metrics && (
                <>
                  <Divider />
                  <ListItem>
                    <ListItemText
                      primary="Performance Metrics"
                      secondary={
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" display="block">
                            Success Rate: {(healthData.performance_metrics.success_rate * 100).toFixed(1)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={healthData.performance_metrics.success_rate * 100}
                            sx={{ mb: 1, height: 4 }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {healthData.performance_metrics.total_requests} total requests,
                            {healthData.performance_metrics.failed_requests} failed
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                </>
              )}
            </List>
          </Box>
        ) : (
          <Alert severity="info">
            No embedding model health data available. Configure an embedding model to see health status.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default EmbeddingHealthIndicator;