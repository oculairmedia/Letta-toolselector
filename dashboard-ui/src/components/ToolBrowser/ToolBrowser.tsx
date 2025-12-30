import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Chip,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Paper,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Autocomplete,
  Stack,
  Divider,
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  FilterList as FilterIcon,
  Clear as ClearIcon,
  Storage as StorageIcon,
  CheckCircle as RegisteredIcon,
  RadioButtonUnchecked as UnregisteredIcon,
} from '@mui/icons-material';

import {
  useBrowseTools,
  useToolDetail,
  useToolCategories,
  useToolSources,
  useRefreshTools,
  useExportTools,
} from '../../hooks/useApi';
import { ExtendedTool, ToolDetailResponse, ToolBrowseResponse } from '../../types';

const ToolBrowser: React.FC = () => {
  // State for filters and pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [selectedMcpServer, setSelectedMcpServer] = useState<string>('');
  const [sortBy, setSortBy] = useState<'name' | 'category' | 'updated' | 'relevance'>('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  
  // State for tool detail modal
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);

  // Build query parameters for API
  const queryParams = useMemo(() => ({
    page: page, // API expects 0-based pagination
    limit: rowsPerPage,
    search: searchQuery || undefined,
    category: selectedCategory || undefined,
    source: selectedSource || undefined,
    mcp_server: selectedMcpServer || undefined,
    sort: sortBy,
    order: sortOrder,
  }), [page, rowsPerPage, searchQuery, selectedCategory, selectedSource, selectedMcpServer, sortBy, sortOrder]);

  // API hooks
  const { data: browseData, isLoading, error, refetch } = useBrowseTools(queryParams) as {
    data: ToolBrowseResponse | undefined;
    isLoading: boolean;
    error: any;
    refetch: () => void;
  };
  const { data: categories } = useToolCategories();
  const { data: sources } = useToolSources();
  const { data: toolDetail, isLoading: isLoadingDetail } = useToolDetail(selectedToolId || '', !!selectedToolId);
  const refreshToolsMutation = useRefreshTools();
  const exportToolsMutation = useExportTools();

  // Debug logging
  console.log('ToolBrowser Debug:', {
    queryParams,
    isLoading,
    hasData: !!browseData,
    dataKeys: browseData ? Object.keys(browseData) : null,
    toolsCount: browseData?.tools?.length,
    error: error ? {
      message: error.message,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      stack: error.stack
    } : null
  });

  // Event handlers
  const handleSearchChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
    setPage(0); // Reset to first page when searching
  }, []);

  const handleClearFilters = () => {
    setSearchQuery('');
    setSelectedCategory('');
    setSelectedSource('');
    setSelectedMcpServer('');
    setPage(0);
  };

  const handleSort = (column: 'name' | 'category' | 'updated') => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('asc');
    }
    setPage(0);
  };

  const handleViewTool = (toolId: string) => {
    setSelectedToolId(toolId);
    setIsDetailModalOpen(true);
  };

  const handleCloseDetail = () => {
    setIsDetailModalOpen(false);
    setSelectedToolId(null);
  };

  const handleRefresh = async () => {
    try {
      await refreshToolsMutation.mutateAsync();
      refetch();
    } catch (error) {
      console.error('Failed to refresh tools:', error);
    }
  };

  const handleExport = async (format: 'json' | 'csv') => {
    try {
      const blob = await exportToolsMutation.mutateAsync(format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `tools-export.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export tools:', error);
    }
  };

  // Get unique MCP servers from current data
  const mcpServers = useMemo(() => {
    if (!browseData?.tools) return [];
    const servers = new Set(
      browseData.tools
        .map(tool => tool.mcp_server_name)
        .filter((server): server is string => Boolean(server))
    );
    return Array.from(servers).sort();
  }, [browseData?.tools]);

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Unknown';
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return 'Invalid date';
    }
  };

  const getRegistrationStatus = (tool: ExtendedTool) => {
    if (tool.registered_in_letta === true) {
      return <Chip icon={<RegisteredIcon />} label="Registered" color="success" size="small" />;
    }
    return <Chip icon={<UnregisteredIcon />} label="Unregistered" color="default" size="small" />;
  };

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load tools: {error.message}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <StorageIcon />
        Tool Browser
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Browse and manage all tools indexed in the Weaviate vector database.
      </Typography>

      {/* Toolbar */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            {/* Search */}
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                placeholder="Search tools..."
                value={searchQuery}
                onChange={handleSearchChange}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
              />
            </Grid>

            {/* Category Filter */}
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Category</InputLabel>
                <Select
                  value={selectedCategory}
                  label="Category"
                  onChange={(e) => {
                    setSelectedCategory(e.target.value);
                    setPage(0);
                  }}
                >
                  <MenuItem value="">All Categories</MenuItem>
                  {categories?.map((category) => (
                    <MenuItem key={category} value={category}>
                      {category}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Source Filter */}
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Source</InputLabel>
                <Select
                  value={selectedSource}
                  label="Source"
                  onChange={(e) => {
                    setSelectedSource(e.target.value);
                    setPage(0);
                  }}
                >
                  <MenuItem value="">All Sources</MenuItem>
                  {sources?.map((source) => (
                    <MenuItem key={source} value={source}>
                      {source}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* MCP Server Filter */}
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>MCP Server</InputLabel>
                <Select
                  value={selectedMcpServer}
                  label="MCP Server"
                  onChange={(e) => {
                    setSelectedMcpServer(e.target.value);
                    setPage(0);
                  }}
                >
                  <MenuItem value="">All Servers</MenuItem>
                  {mcpServers.map((server) => (
                    <MenuItem key={server} value={server}>
                      {server}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Actions */}
            <Grid item xs={12} md={2}>
              <Stack direction="row" spacing={1}>
                <Tooltip title="Clear Filters">
                  <IconButton onClick={handleClearFilters} size="small">
                    <ClearIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh Index">
                  <IconButton 
                    onClick={handleRefresh} 
                    size="small"
                    disabled={refreshToolsMutation.isPending}
                  >
                    {refreshToolsMutation.isPending ? <CircularProgress size={20} /> : <RefreshIcon />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="Export Tools">
                  <IconButton 
                    onClick={() => handleExport('json')} 
                    size="small"
                    disabled={exportToolsMutation.isPending}
                  >
                    <DownloadIcon />
                  </IconButton>
                </Tooltip>
              </Stack>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Results Summary */}
      {browseData && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Showing {browseData.tools.length} of {browseData.total} tools
          {searchQuery && ` matching "${searchQuery}"`}
        </Typography>
      )}

      {/* Tools Table */}
      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>
                  <TableSortLabel
                    active={sortBy === 'name'}
                    direction={sortBy === 'name' ? sortOrder : 'asc'}
                    onClick={() => handleSort('name')}
                  >
                    Name
                  </TableSortLabel>
                </TableCell>
                <TableCell>Description</TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortBy === 'category'}
                    direction={sortBy === 'category' ? sortOrder : 'asc'}
                    onClick={() => handleSort('category')}
                  >
                    Category
                  </TableSortLabel>
                </TableCell>
                <TableCell>MCP Server</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortBy === 'updated'}
                    direction={sortBy === 'updated' ? sortOrder : 'asc'}
                    onClick={() => handleSort('updated')}
                  >
                    Updated
                  </TableSortLabel>
                </TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                    <CircularProgress />
                  </TableCell>
                </TableRow>
              ) : browseData?.tools.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                    <Typography color="text.secondary">
                      No tools found {searchQuery && `matching "${searchQuery}"`}
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                browseData?.tools.map((tool) => (
                  <TableRow key={tool.id} hover>
                    <TableCell>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'medium' }}>
                        {tool.name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography 
                        variant="body2" 
                        color="text.secondary"
                        sx={{
                          maxWidth: 300,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {tool.description}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {tool.category && (
                        <Chip label={tool.category} size="small" variant="outlined" />
                      )}
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {tool.mcp_server_name || 'Unknown'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {getRegistrationStatus(tool)}
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(tool.last_updated)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleViewTool(tool.id)}
                        >
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Pagination */}
        {browseData && browseData.total > 0 && (
          <TablePagination
            component="div"
            count={browseData.total}
            page={page}
            onPageChange={(_, newPage) => setPage(newPage)}
            rowsPerPage={rowsPerPage}
            onRowsPerPageChange={(event) => {
              setRowsPerPage(parseInt(event.target.value, 10));
              setPage(0);
            }}
            rowsPerPageOptions={[10, 25, 50, 100]}
          />
        )}
      </Paper>

      {/* Tool Detail Modal */}
      <Dialog
        open={isDetailModalOpen}
        onClose={handleCloseDetail}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Tool Details
          {toolDetail && (
            <Typography variant="subtitle2" color="text.secondary">
              {toolDetail.name}
            </Typography>
          )}
        </DialogTitle>
        <DialogContent dividers>
          {isLoadingDetail ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : toolDetail ? (
            <Stack spacing={3}>
              {/* Basic Information */}
              <Box>
                <Typography variant="h6" gutterBottom>Basic Information</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">Name</Typography>
                    <Typography variant="body1">{toolDetail.name}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">ID</Typography>
                    <Typography variant="body1" sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                      {toolDetail.id}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" color="text.secondary">Description</Typography>
                    <Typography variant="body1">{toolDetail.description}</Typography>
                  </Grid>
                </Grid>
              </Box>

              <Divider />

              {/* Metadata */}
              <Box>
                <Typography variant="h6" gutterBottom>Metadata</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">Category</Typography>
                    <Typography variant="body1">{toolDetail.category || 'Uncategorized'}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">Source</Typography>
                    <Typography variant="body1">{toolDetail.source || 'Unknown'}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">MCP Server</Typography>
                    <Typography variant="body1">{toolDetail.mcp_server_name || 'Unknown'}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">Registration Status</Typography>
                    {getRegistrationStatus(toolDetail)}
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">Last Updated</Typography>
                    <Typography variant="body1">{formatDate(toolDetail.last_updated)}</Typography>
                  </Grid>
                  {toolDetail.embedding_id && (
                    <Grid item xs={12} sm={6}>
                      <Typography variant="subtitle2" color="text.secondary">Embedding ID</Typography>
                      <Typography variant="body1" sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                        {toolDetail.embedding_id}
                      </Typography>
                    </Grid>
                  )}
                </Grid>
              </Box>

              {/* Tags */}
              {toolDetail.tags && toolDetail.tags.length > 0 && (
                <>
                  <Divider />
                  <Box>
                    <Typography variant="h6" gutterBottom>Tags</Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
                      {toolDetail.tags.map((tag, index) => (
                        <Chip key={index} label={tag} size="small" />
                      ))}
                    </Stack>
                  </Box>
                </>
              )}

              {/* JSON Schema */}
              {toolDetail.json_schema && (
                <>
                  <Divider />
                  <Box>
                    <Typography variant="h6" gutterBottom>JSON Schema</Typography>
                    <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                      <pre style={{ margin: 0, fontSize: '0.75rem', overflow: 'auto' }}>
                        {JSON.stringify(toolDetail.json_schema, null, 2)}
                      </pre>
                    </Paper>
                  </Box>
                </>
              )}
            </Stack>
          ) : (
            <Typography color="error">Failed to load tool details</Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDetail}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ToolBrowser;