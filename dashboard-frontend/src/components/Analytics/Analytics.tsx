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

  // Mock data for demonstration when API data is empty
  const mockData = {
    search_count: 1250,
    total_evaluations: 45,
    avg_rating: 4.2,
    date_range: { start: null, end: null },
    recent_searches: [
      { query: 'file operations', timestamp: '2024-01-01T10:00:00Z', result_count: 15 },
      { query: 'data processing', timestamp: '2024-01-01T11:00:00Z', result_count: 12 },
      { query: 'API integration', timestamp: '2024-01-01T12:00:00Z', result_count: 18 },
      { query: 'authentication', timestamp: '2024-01-01T13:00:00Z', result_count: 8 },
      { query: 'database queries', timestamp: '2024-01-01T14:00:00Z', result_count: 22 },
    ],
    top_tools: [
      { tool_name: 'search_files', usage_count: 125 },
      { tool_name: 'read_file', usage_count: 98 },
      { tool_name: 'api_call', usage_count: 87 },
      { tool_name: 'auth_check', usage_count: 76 },
      { tool_name: 'db_query', usage_count: 65 },
    ],
    tool_usage: {
      'search_files': 125,
      'read_file': 98,
      'api_call': 87,
      'auth_check': 76,
      'db_query': 65,
    }
  };

  // Use API data if available, otherwise fall back to mock data
  const data = analytics || mockData;

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
                {formatNumber(data.search_count)}
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
                {formatNumber(data.total_evaluations)}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                <AssessmentIcon fontSize="small" />
                Total Evaluations
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info.main" gutterBottom>
                {data.avg_rating.toFixed(1)}⭐
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                <TrendingUpIcon fontSize="small" />
                Average Rating
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="secondary.main" gutterBottom>
                {formatNumber(data.top_tools.length)}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                <SpeedIcon fontSize="small" />
                Active Tools
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Analytics Tabs */}
      <Paper elevation={1}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
            <Tab label="Recent Searches" />
            <Tab label="Top Tools" />
            <Tab label="Tool Usage" />
            <Tab label="Search Analytics" />
          </Tabs>
        </Box>

        <TabPanel value={activeTab} index={0}>
          {/* Recent Searches */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Search Activity
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.recent_searches.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="query" 
                      tick={{ fontSize: 12 }}
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="result_count" fill={theme.palette.primary.main} />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          {/* Top Tools */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Most Popular Tools
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.top_tools}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="tool_name" 
                      tick={{ fontSize: 12 }}
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="usage_count" fill={theme.palette.primary.main} />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          {/* Tool Usage */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Tool Usage Distribution
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={Object.entries(data.tool_usage).map(([tool_name, count]) => ({ tool_name, count }))}
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="count"
                          label={(entry: any) => `${entry.tool_name}: ${entry.count}`}
                        >
                          {Object.entries(data.tool_usage).map((_, index) => (
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
                    Tool Usage Comparison
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.top_tools.slice(0, 5)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="tool_name" 
                          tick={{ fontSize: 12 }}
                          angle={-45}
                          textAnchor="end"
                          height={100}
                        />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="usage_count" fill={theme.palette.secondary.main} />
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          {/* Search Analytics */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Search Query Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Analyze recent search patterns and result effectiveness
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.recent_searches.slice(0, 10).map((item, index) => ({
                    ...item,
                    index: index + 1,
                    query_short: item.query.length > 15 ? item.query.substring(0, 15) + '...' : item.query
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="query_short" tick={{ fontSize: 10 }} />
                    <YAxis />
                    <Tooltip formatter={(value: number) => [value, 'Result Count']} />
                    <Line 
                      type="monotone" 
                      dataKey="result_count" 
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