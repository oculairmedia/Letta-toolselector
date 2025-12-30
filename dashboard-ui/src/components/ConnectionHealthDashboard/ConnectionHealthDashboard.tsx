import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  MonitorHeart as HealthIcon,
  CheckCircle as HealthyIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Timeline as MetricsIcon,
  Storage as DatabaseIcon,
  CloudQueue as ApiIcon,
  Memory as ModelIcon,
  Router as NetworkIcon,
  Speed as PerformanceIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  Notifications as AlertIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

// Types for health monitoring
interface ServiceHealth {
  service: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  responseTime: number;
  lastChecked: Date;
  uptime: number;
  errorCount: number;
  details?: {
    version?: string;
    endpoint?: string;
    authentication?: boolean;
    rateLimit?: {
      remaining: number;
      reset: Date;
    };
    queue?: {
      pending: number;
      processing: number;
    };
  };
  metrics?: Array<{
    timestamp: Date;
    responseTime: number;
    success: boolean;
  }>;
}

interface SystemHealth {
  overall: 'healthy' | 'warning' | 'error';
  services: ServiceHealth[];
  lastUpdate: Date;
  totalChecks: number;
  successRate: number;
}

const ConnectionHealthDashboard: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    overall: 'warning',
    services: [],
    lastUpdate: new Date(),
    totalChecks: 0,
    successRate: 0,
  });

  const [isLoading, setIsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30000); // 30 seconds
  const [selectedService, setSelectedService] = useState<ServiceHealth | null>(null);
  const [showMetricsDialog, setShowMetricsDialog] = useState(false);
  const [showLogsDialog, setShowLogsDialog] = useState(false);

  // Mock data for demonstration - in real implementation, this would come from API
  const mockServices: ServiceHealth[] = [
    {
      service: 'Weaviate Database',
      status: 'healthy',
      responseTime: 45,
      lastChecked: new Date(),
      uptime: 99.8,
      errorCount: 0,
      details: {
        version: '1.25.0',
        endpoint: 'http://localhost:8080',
        authentication: true,
      },
      metrics: generateMockMetrics(24),
    },
    {
      service: 'Letta API',
      status: 'healthy',
      responseTime: 120,
      lastChecked: new Date(),
      uptime: 99.5,
      errorCount: 2,
      details: {
        version: '0.4.0',
        endpoint: 'https://letta2.oculair.ca/v1',
        authentication: true,
        rateLimit: {
          remaining: 8500,
          reset: new Date(Date.now() + 3600000),
        },
      },
      metrics: generateMockMetrics(24),
    },
    {
      service: 'Ollama Embedding Service',
      status: 'warning',
      responseTime: 2500,
      lastChecked: new Date(),
      uptime: 97.2,
      errorCount: 5,
      details: {
        version: '0.3.0',
        endpoint: 'http://192.168.50.80:11434',
        queue: {
          pending: 3,
          processing: 1,
        },
      },
      metrics: generateMockMetrics(24),
    },
    {
      service: 'MCP Server',
      status: 'healthy',
      responseTime: 25,
      lastChecked: new Date(),
      uptime: 99.9,
      errorCount: 0,
      details: {
        version: '1.0.0',
        endpoint: 'http://localhost:3020',
        authentication: false,
      },
      metrics: generateMockMetrics(24),
    },
    {
      service: 'API Server',
      status: 'healthy',
      responseTime: 85,
      lastChecked: new Date(),
      uptime: 99.7,
      errorCount: 1,
      details: {
        version: '1.0.0',
        endpoint: 'http://localhost:8020',
        queue: {
          pending: 0,
          processing: 2,
        },
      },
      metrics: generateMockMetrics(24),
    },
    {
      service: 'OpenAI API',
      status: 'healthy',
      responseTime: 450,
      lastChecked: new Date(),
      uptime: 99.9,
      errorCount: 0,
      details: {
        version: 'v1',
        endpoint: 'https://api.openai.com/v1',
        authentication: true,
        rateLimit: {
          remaining: 4500,
          reset: new Date(Date.now() + 3600000),
        },
      },
      metrics: generateMockMetrics(24),
    },
    {
      service: 'Reranking Service',
      status: 'error',
      responseTime: 0,
      lastChecked: new Date(),
      uptime: 85.2,
      errorCount: 15,
      details: {
        version: '1.0.0',
        endpoint: 'http://localhost:8091',
        queue: {
          pending: 10,
          processing: 0,
        },
      },
      metrics: generateMockMetrics(24),
    },
  ];

  function generateMockMetrics(hours: number) {
    const metrics = [];
    const now = new Date();

    for (let i = hours; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
      metrics.push({
        timestamp,
        responseTime: Math.random() * 500 + 50,
        success: Math.random() > 0.05, // 95% success rate
      });
    }

    return metrics;
  }

  const fetchSystemHealth = useCallback(async () => {
    setIsLoading(true);

    try {
      // In a real implementation, this would call the health check API
      // const response = await fetch('/api/v1/health/system');
      // const data = await response.json();

      // Mock implementation
      await new Promise(resolve => setTimeout(resolve, 1000));

      const services = mockServices;
      const healthyServices = services.filter(s => s.status === 'healthy').length;
      const totalServices = services.length;

      let overall: 'healthy' | 'warning' | 'error' = 'healthy';
      if (services.some(s => s.status === 'error')) {
        overall = 'error';
      } else if (services.some(s => s.status === 'warning')) {
        overall = 'warning';
      }

      setSystemHealth({
        overall,
        services,
        lastUpdate: new Date(),
        totalChecks: totalServices,
        successRate: (healthyServices / totalServices) * 100,
      });
    } catch (error) {
      console.error('Failed to fetch system health:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const testServiceConnection = async (service: string) => {
    setIsLoading(true);
    try {
      // In real implementation: await fetch(`/api/v1/health/test/${service}`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      await fetchSystemHealth();
    } catch (error) {
      console.error(`Failed to test ${service}:`, error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemHealth();

    if (autoRefresh) {
      const interval = setInterval(fetchSystemHealth, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, fetchSystemHealth]);

  const getStatusIcon = (status: ServiceHealth['status']) => {
    switch (status) {
      case 'healthy': return <HealthyIcon color="success" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'error': return <ErrorIcon color="error" />;
      default: return <NetworkIcon color="disabled" />;
    }
  };

  const getStatusColor = (status: ServiceHealth['status']) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getOverallHealthCard = () => {
    const { overall, services, successRate } = systemHealth;
    const healthyCount = services.filter(s => s.status === 'healthy').length;
    const warningCount = services.filter(s => s.status === 'warning').length;
    const errorCount = services.filter(s => s.status === 'error').length;

    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Box>
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getStatusIcon(overall)}
                System Health Overview
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Last updated: {systemHealth.lastUpdate.toLocaleTimeString()}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                    size="small"
                  />
                }
                label="Auto-refresh"
                sx={{ mr: 2 }}
              />

              <Button
                variant="outlined"
                onClick={fetchSystemHealth}
                disabled={isLoading}
                startIcon={<RefreshIcon />}
                size="small"
              >
                Refresh
              </Button>
            </Box>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="success.main">
                  {healthyCount}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Healthy Services
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="warning.main">
                  {warningCount}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Warning Services
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="error.main">
                  {errorCount}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Error Services
                </Typography>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color={successRate > 90 ? 'success.main' : successRate > 70 ? 'warning.main' : 'error.main'}>
                  {successRate.toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Success Rate
                </Typography>
              </Box>
            </Grid>
          </Grid>

          {isLoading && <LinearProgress sx={{ mt: 2 }} />}
        </CardContent>
      </Card>
    );
  };

  return (
    <Box>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <HealthIcon />
          Connection Health Monitoring
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Real-time monitoring of all critical system services and external integrations.
        </Typography>
      </Paper>

      {/* Overall Health Card */}
      {getOverallHealthCard()}

      {/* Service Health Cards */}
      <Grid container spacing={3}>
        {systemHealth.services.map((service) => (
          <Grid item xs={12} sm={6} lg={4} key={service.service}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getStatusIcon(service.status)}
                      {service.service}
                    </Typography>
                    <Chip
                      label={service.status.toUpperCase()}
                      color={getStatusColor(service.status) as any}
                      size="small"
                    />
                  </Box>

                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <Tooltip title="View metrics">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSelectedService(service);
                          setShowMetricsDialog(true);
                        }}
                      >
                        <MetricsIcon />
                      </IconButton>
                    </Tooltip>

                    <Tooltip title="Test connection">
                      <IconButton
                        size="small"
                        onClick={() => testServiceConnection(service.service)}
                        disabled={isLoading}
                      >
                        <RefreshIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Response Time
                  </Typography>
                  <Typography variant="h6" color={service.responseTime > 1000 ? 'warning.main' : 'text.primary'}>
                    {service.responseTime}ms
                  </Typography>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Uptime
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={service.uptime}
                      sx={{ flexGrow: 1, height: 8 }}
                      color={service.uptime > 99 ? 'success' : service.uptime > 95 ? 'warning' : 'error'}
                    />
                    <Typography variant="body2" sx={{ minWidth: 50 }}>
                      {service.uptime.toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>

                {service.details && (
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="body2">Service Details</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {service.details.version && (
                          <ListItem>
                            <ListItemText
                              primary="Version"
                              secondary={service.details.version}
                            />
                          </ListItem>
                        )}

                        {service.details.endpoint && (
                          <ListItem>
                            <ListItemText
                              primary="Endpoint"
                              secondary={service.details.endpoint}
                            />
                          </ListItem>
                        )}

                        {service.details.rateLimit && (
                          <ListItem>
                            <ListItemText
                              primary="Rate Limit"
                              secondary={`${service.details.rateLimit.remaining} remaining`}
                            />
                          </ListItem>
                        )}

                        {service.details.queue && (
                          <ListItem>
                            <ListItemText
                              primary="Queue Status"
                              secondary={`${service.details.queue.pending} pending, ${service.details.queue.processing} processing`}
                            />
                          </ListItem>
                        )}

                        <ListItem>
                          <ListItemText
                            primary="Error Count"
                            secondary={service.errorCount}
                          />
                        </ListItem>
                      </List>
                    </AccordionDetails>
                  </Accordion>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Service Metrics Dialog */}
      <Dialog
        open={showMetricsDialog}
        onClose={() => setShowMetricsDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          {selectedService?.service} - Performance Metrics
        </DialogTitle>
        <DialogContent>
          {selectedService?.metrics && (
            <Box sx={{ height: 400, width: '100%' }}>
              <ResponsiveContainer>
                <LineChart data={selectedService.metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis />
                  <ChartTooltip
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: any) => [`${value.toFixed(0)}ms`, 'Response Time']}
                  />
                  <Line
                    type="monotone"
                    dataKey="responseTime"
                    stroke="#8884d8"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowMetricsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ConnectionHealthDashboard;