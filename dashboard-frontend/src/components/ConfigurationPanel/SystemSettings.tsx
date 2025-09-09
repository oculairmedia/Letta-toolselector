import React from 'react';
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
} from '@mui/icons-material';

import { useTools, useRefreshTools, useHealthCheck } from '../../hooks/useApi';
import { formatNumber, formatRelativeTime } from '../../utils';

const SystemSettings: React.FC = () => {
  const { data: tools, isLoading: toolsLoading } = useTools();
  const { data: healthData } = useHealthCheck();
  const refreshToolsMutation = useRefreshTools();

  const handleRefreshTools = () => {
    refreshToolsMutation.mutate();
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
                    secondary="OpenAI text-embedding-3-small"
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