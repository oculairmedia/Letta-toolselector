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
  Avatar,
  IconButton,
  Tooltip,
  LinearProgress,
  Collapse,
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Divider,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Debug as DebugIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  Clear as ClearIcon,
  Analytics as AnalyticsIcon,
  Timeline as TimelineIcon,
  Computer as ComputerIcon,
  Search as SearchIcon,
  Visibility as VisibilityIcon,
} from '@mui/icons-material';
import { apiService } from '../../services/api';

interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  line_number?: number;
  thread?: string;
  exception?: string;
  stack_trace?: string;
}

interface LogAnalysis {
  timeframe: string;
  total_entries: number;
  summary: {
    error_count: number;
    warning_count: number;
    info_count: number;
    debug_count: number;
    error_rate: number;
  };
  top_error_patterns: Array<{ pattern: string; count: number }>;
  logger_activity: Record<string, { total: number; errors: number; warnings: number }>;
  recommendations: string[];
  recent_errors?: LogEntry[];
}

const LogViewer: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [analysis, setAnalysis] = useState<LogAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [expandedLogs, setExpandedLogs] = useState<Set<number>>(new Set());
  const [filterDialogOpen, setFilterDialogOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);

  // Filter state
  const [filters, setFilters] = useState({
    level: 'all',
    lines: 100,
    search: '',
    timeframe: '24h',
  });

  // Export state
  const [exportFormat, setExportFormat] = useState<'json' | 'csv' | 'text'>('json');
  const [exportLoading, setExportLoading] = useState(false);

  // Clear logs state
  const [clearBackup, setClearBackup] = useState(true);
  const [clearOlderThan, setClearOlderThan] = useState(0);
  const [clearLoading, setClearLoading] = useState(false);

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filters.level !== 'all') params.append('level', filters.level);
      if (filters.lines) params.append('lines', filters.lines.toString());
      if (filters.search) params.append('search', filters.search);

      const response = await fetch(`/api/v1/logs?${params.toString()}`);
      const result = await response.json();

      if (result.success) {
        setLogs(result.data.logs || []);
      }
    } catch (error) {
      console.error('Failed to fetch logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalysis = async () => {
    try {
      setAnalysisLoading(true);
      const params = new URLSearchParams();
      params.append('timeframe', filters.timeframe);
      params.append('include_details', 'true');

      const response = await fetch(`/api/v1/logs/analysis?${params.toString()}`);
      const result = await response.json();

      if (result.success) {
        setAnalysis(result.data);
      }
    } catch (error) {
      console.error('Failed to fetch log analysis:', error);
    } finally {
      setAnalysisLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    fetchAnalysis();
  }, [filters]);

  const handleRefresh = () => {
    fetchLogs();
    fetchAnalysis();
  };

  const toggleLogExpanded = (index: number) => {
    const newExpanded = new Set(expandedLogs);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedLogs(newExpanded);
  };

  const getLogLevelIcon = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return <ErrorIcon color="error" />;
      case 'WARNING':
        return <WarningIcon color="warning" />;
      case 'INFO':
        return <InfoIcon color="info" />;
      case 'DEBUG':
        return <DebugIcon color="disabled" />;
      default:
        return <InfoIcon />;
    }
  };

  const getLogLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return 'error';
      case 'WARNING':
        return 'warning';
      case 'INFO':
        return 'info';
      case 'DEBUG':
        return 'default';
      default:
        return 'default';
    }
  };

  const handleExport = async () => {
    try {
      setExportLoading(true);
      const response = await fetch('/api/v1/logs/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          format: exportFormat,
          filters: filters
        })
      });

      const result = await response.json();
      if (result.success) {
        // Create download link
        const blob = new Blob([result.data.data], { type: result.data.content_type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = result.data.filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        setExportDialogOpen(false);
      }
    } catch (error) {
      console.error('Failed to export logs:', error);
    } finally {
      setExportLoading(false);
    }
  };

  const handleClearLogs = async () => {
    try {
      setClearLoading(true);
      const response = await fetch('/api/v1/logs/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backup: clearBackup,
          older_than_days: clearOlderThan
        })
      });

      const result = await response.json();
      if (result.success) {
        setClearDialogOpen(false);
        handleRefresh(); // Refresh logs after clearing
      }
    } catch (error) {
      console.error('Failed to clear logs:', error);
    } finally {
      setClearLoading(false);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TimelineIcon />
          Error Log Viewer & Analysis
        </Typography>

        <Box display="flex" alignItems="center" gap={1}>
          <Button
            startIcon={<FilterIcon />}
            onClick={() => setFilterDialogOpen(true)}
            variant="outlined"
            size="small"
          >
            Filters
          </Button>
          <Button
            startIcon={<DownloadIcon />}
            onClick={() => setExportDialogOpen(true)}
            variant="outlined"
            size="small"
          >
            Export
          </Button>
          <Button
            startIcon={<ClearIcon />}
            onClick={() => setClearDialogOpen(true)}
            variant="outlined"
            size="small"
            color="warning"
          >
            Clear
          </Button>
          <IconButton onClick={handleRefresh} size="small">
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Log Analysis Summary */}
      {analysis && (
        <Grid container spacing={3} mb={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <AnalyticsIcon />
                  Log Analysis Summary ({analysis.timeframe})
                </Typography>

                <Grid container spacing={2} mb={2}>
                  <Grid item xs={6} md={3}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                        <Typography variant="h4" color="error.main">
                          {analysis.summary.error_count}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Errors ({analysis.summary.error_rate}%)
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                        <Typography variant="h4" color="warning.main">
                          {analysis.summary.warning_count}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Warnings
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                        <Typography variant="h4" color="info.main">
                          {analysis.summary.info_count}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Info
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 1 }}>
                        <Typography variant="h4" color="text.secondary">
                          {analysis.total_entries}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Total Entries
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                {/* Top Error Patterns */}
                {analysis.top_error_patterns.length > 0 && (
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Top Error Patterns</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {analysis.top_error_patterns.slice(0, 5).map((pattern, index) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <ErrorIcon color="error" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={pattern.pattern}
                              secondary={`${pattern.count} occurrences`}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>
                )}

                {/* Recommendations */}
                {analysis.recommendations.length > 0 && (
                  <Box mt={2}>
                    <Typography variant="subtitle2" gutterBottom>Recommendations</Typography>
                    {analysis.recommendations.map((rec, index) => (
                      <Alert key={index} severity="info" sx={{ mt: 1 }}>
                        {rec}
                      </Alert>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Log Entries */}
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              Log Entries ({logs.length})
            </Typography>
            <Box>
              <Chip
                label={`Level: ${filters.level}`}
                size="small"
                variant="outlined"
                sx={{ mr: 1 }}
              />
              <Chip
                label={`Lines: ${filters.lines}`}
                size="small"
                variant="outlined"
              />
            </Box>
          </Box>

          {loading && <LinearProgress sx={{ mb: 2 }} />}

          <List>
            {logs.map((log, index) => {
              const isExpanded = expandedLogs.has(index);
              return (
                <React.Fragment key={index}>
                  <ListItem>
                    <ListItemIcon>
                      <Avatar sx={{ width: 32, height: 32 }}>
                        {getLogLevelIcon(log.level)}
                      </Avatar>
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Chip
                            label={log.level}
                            size="small"
                            color={getLogLevelColor(log.level) as any}
                          />
                          <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                            {log.message}
                          </Typography>
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            {formatTimestamp(log.timestamp)} | {log.logger}
                            {log.thread && ` | ${log.thread}`}
                            {log.line_number && ` | Line ${log.line_number}`}
                          </Typography>
                        </Box>
                      }
                    />
                    {(log.exception || log.stack_trace) && (
                      <IconButton
                        size="small"
                        onClick={() => toggleLogExpanded(index)}
                      >
                        {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                      </IconButton>
                    )}
                  </ListItem>

                  {/* Expanded details for errors with stack traces */}
                  {(log.exception || log.stack_trace) && (
                    <Collapse in={isExpanded}>
                      <Box sx={{ pl: 8, pr: 2, pb: 2 }}>
                        <Card variant="outlined" sx={{ bgcolor: 'error.light', color: 'error.contrastText' }}>
                          <CardContent sx={{ py: 1 }}>
                            {log.exception && (
                              <Box mb={1}>
                                <Typography variant="subtitle2" gutterBottom>Exception:</Typography>
                                <Typography variant="body2" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                                  {log.exception}
                                </Typography>
                              </Box>
                            )}
                            {log.stack_trace && (
                              <Box>
                                <Typography variant="subtitle2" gutterBottom>Stack Trace:</Typography>
                                <Typography variant="body2" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                                  {log.stack_trace}
                                </Typography>
                              </Box>
                            )}
                          </CardContent>
                        </Card>
                      </Box>
                    </Collapse>
                  )}

                  {index < logs.length - 1 && <Divider />}
                </React.Fragment>
              );
            })}
          </List>

          {logs.length === 0 && !loading && (
            <Alert severity="info">
              No log entries found with the current filters.
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Filter Dialog */}
      <Dialog open={filterDialogOpen} onClose={() => setFilterDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Log Filters</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Log Level</InputLabel>
                <Select
                  value={filters.level}
                  label="Log Level"
                  onChange={(e) => setFilters({ ...filters, level: e.target.value })}
                >
                  <MenuItem value="all">All Levels</MenuItem>
                  <MenuItem value="error">Error</MenuItem>
                  <MenuItem value="warning">Warning</MenuItem>
                  <MenuItem value="info">Info</MenuItem>
                  <MenuItem value="debug">Debug</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Max Lines"
                value={filters.lines}
                onChange={(e) => setFilters({ ...filters, lines: parseInt(e.target.value) || 100 })}
                inputProps={{ min: 10, max: 10000 }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Search in Messages"
                value={filters.search}
                onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                placeholder="Enter search term..."
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Analysis Timeframe</InputLabel>
                <Select
                  value={filters.timeframe}
                  label="Analysis Timeframe"
                  onChange={(e) => setFilters({ ...filters, timeframe: e.target.value })}
                >
                  <MenuItem value="1h">Last Hour</MenuItem>
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFilterDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={() => {
              setFilterDialogOpen(false);
              // Triggers useEffect to refresh data
            }}
            variant="contained"
          >
            Apply Filters
          </Button>
        </DialogActions>
      </Dialog>

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)}>
        <DialogTitle>Export Logs</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Export Format</InputLabel>
            <Select
              value={exportFormat}
              label="Export Format"
              onChange={(e) => setExportFormat(e.target.value as any)}
            >
              <MenuItem value="json">JSON</MenuItem>
              <MenuItem value="csv">CSV</MenuItem>
              <MenuItem value="text">Text</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleExport}
            variant="contained"
            disabled={exportLoading}
            startIcon={exportLoading ? <CircularProgress size={20} /> : <DownloadIcon />}
          >
            {exportLoading ? 'Exporting...' : 'Export'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Clear Logs Dialog */}
      <Dialog open={clearDialogOpen} onClose={() => setClearDialogOpen(false)}>
        <DialogTitle>Clear Log Files</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            This action will clear log files. This operation cannot be undone.
          </Alert>
          <FormControlLabel
            control={
              <Switch
                checked={clearBackup}
                onChange={(e) => setClearBackup(e.target.checked)}
              />
            }
            label="Create backup before clearing"
          />
          <TextField
            fullWidth
            type="number"
            label="Clear files older than (days)"
            value={clearOlderThan}
            onChange={(e) => setClearOlderThan(parseInt(e.target.value) || 0)}
            helperText="0 = clear all files"
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleClearLogs}
            variant="contained"
            color="warning"
            disabled={clearLoading}
            startIcon={clearLoading ? <CircularProgress size={20} /> : <ClearIcon />}
          >
            {clearLoading ? 'Clearing...' : 'Clear Logs'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default LogViewer;