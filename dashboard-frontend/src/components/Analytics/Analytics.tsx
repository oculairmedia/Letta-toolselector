import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Tabs,
  Tab,
  TextField,
  useTheme,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp as TrendingUpIcon,
  Search as SearchIcon,
  Speed as SpeedIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

import { useAnalytics } from '../../hooks/useApi';
import { formatNumber, formatDuration } from '../../utils';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const Analytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [dateRange, setDateRange] = useState<{ start: string; end: string } | undefined>();
  
  const theme = useTheme();
  const { data: analytics } = useAnalytics(dateRange);

  // Mock data for demonstration
  const mockAnalytics = {
    total_searches: 1250,
    average_response_time: 0.245,
    top_queries: [
      { query: 'file operations', count: 125 },
      { query: 'data processing', count: 98 },
      { query: 'API integration', count: 87 },
      { query: 'authentication', count: 76 },
      { query: 'database queries', count: 65 },
    ],
    reranker_performance: {
      improvement_rate: 0.15,
      accuracy_gain: 0.12,
    },
    usage_stats: {
      daily_searches: [
        { date: '2024-01-01', count: 45 },
        { date: '2024-01-02', count: 52 },
        { date: '2024-01-03', count: 38 },
        { date: '2024-01-04', count: 61 },
        { date: '2024-01-05', count: 49 },
        { date: '2024-01-06', count: 55 },
        { date: '2024-01-07', count: 42 },
      ],
      model_usage: [
        { model: 'gpt-3.5-turbo', count: 780 },
        { model: 'claude-3-haiku', count: 320 },
        { model: 'llama2:7b', count: 150 },
      ],
    },
  };

  const data = analytics || mockAnalytics;

  const COLORS = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
  ];

  return (
    <Box>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DashboardIcon />
          Analytics Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Monitor search performance, reranker effectiveness, and usage patterns over time.
        </Typography>
      </Paper>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary" gutterBottom>
                {formatNumber(data.total_searches)}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                <SearchIcon fontSize="small" />
                Total Searches
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main" gutterBottom>
                {formatDuration(data.average_response_time * 1000)}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                <SpeedIcon fontSize="small" />
                Avg Response Time
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info.main" gutterBottom>
                +{(data.reranker_performance.improvement_rate * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                <TrendingUpIcon fontSize="small" />
                Improvement Rate
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="secondary.main" gutterBottom>
                +{(data.reranker_performance.accuracy_gain * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                <AssessmentIcon fontSize="small" />
                Accuracy Gain
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Analytics Tabs */}
      <Paper elevation={1}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
            <Tab label="Usage Trends" />
            <Tab label="Top Queries" />
            <Tab label="Model Performance" />
            <Tab label="Response Times" />
          </Tabs>
        </Box>

        <TabPanel value={activeTab} index={0}>
          {/* Usage Trends */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Daily Search Volume
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.usage_stats.daily_searches}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="count" 
                      stroke={theme.palette.primary.main}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          {/* Top Queries */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Most Popular Search Queries
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.top_queries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="query" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill={theme.palette.primary.main} />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          {/* Model Performance */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Model Usage Distribution
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={data.usage_stats.model_usage}
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="count"
                          label={(entry: any) => `${entry.model}: ${entry.count}`}
                        >
                          {data.usage_stats.model_usage.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Model Performance Comparison
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.usage_stats.model_usage}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="model" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill={theme.palette.secondary.main} />
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          {/* Response Times */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Average Response Times by Day
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Monitor system performance and identify potential bottlenecks
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.usage_stats.daily_searches.map(item => ({
                    ...item,
                    response_time: Math.random() * 0.5 + 0.2, // Mock response time data
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={(value) => `${(value * 1000).toFixed(0)}ms`} />
                    <Tooltip formatter={(value: number) => [`${(value * 1000).toFixed(0)}ms`, 'Response Time']} />
                    <Line 
                      type="monotone" 
                      dataKey="response_time" 
                      stroke={theme.palette.success.main}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default Analytics;