import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  InputAdornment,
  IconButton,
  Button,
  Typography,
  Grid,
  Chip,
  Divider,
  CircularProgress,
  Alert,
  Autocomplete,
  FormControlLabel,
  Switch,
  Tooltip,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Badge,
} from '@mui/material';
import {
  Search as SearchIcon,
  Clear as ClearIcon,
  FilterList as FilterIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Speed as SpeedIcon,
  Extension as ExtensionIcon,
} from '@mui/icons-material';

import { useSearch, useSearchWithReranking, useRerankerConfig } from '../../hooks/useApi';
import { SearchQuery, SearchResult, RerankerConfig } from '../../types';
import { debounce, formatDuration, getScoreColor, truncateText } from '../../utils';
import SearchResultCard from './SearchResultCard';
import SearchFilters from './SearchFilters';
import ModelIndicator from '../ModelIndicator';

interface SearchInterfaceProps {}

const SearchInterface: React.FC<SearchInterfaceProps> = () => {
  // State management
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [filters, setFilters] = useState({
    category: '',
    source: '',
    tags: [] as string[],
    limit: 10,
  });
  const [showFilters, setShowFilters] = useState(false);
  const [useReranker, setUseReranker] = useState(true);
  const [recentQueries, setRecentQueries] = useState<string[]>(() => {
    const saved = localStorage.getItem('recentSearchQueries');
    return saved ? JSON.parse(saved) : [];
  });

  // API hooks
  const { data: rerankerConfig } = useRerankerConfig();
  const searchWithRerankingMutation = useSearchWithReranking();

  // Build search query object
  const searchQueryObject: SearchQuery = {
    query: debouncedQuery,
    filters: filters.category || filters.source || filters.tags.length > 0 ? {
      ...(filters.category && { category: filters.category }),
      ...(filters.source && { source: filters.source }),
      ...(filters.tags.length > 0 && { tags: filters.tags }),
    } : undefined,
    limit: filters.limit,
  };

  // Regular search (fallback when reranker is disabled)
  const {
    data: regularSearchData,
    isLoading: isRegularSearchLoading,
    error: regularSearchError,
  } = useSearch(searchQueryObject, !useReranker && !!debouncedQuery);

  // Determine which data to use
  const searchData = useReranker && searchWithRerankingMutation.data 
    ? searchWithRerankingMutation.data 
    : regularSearchData;
  const isLoading = useReranker 
    ? searchWithRerankingMutation.isPending 
    : isRegularSearchLoading;
  const error = useReranker 
    ? searchWithRerankingMutation.error 
    : regularSearchError;

  // Debounce search query
  const debouncedSetQuery = useCallback(
    debounce((query: string) => setDebouncedQuery(query), 300),
    []
  );

  useEffect(() => {
    debouncedSetQuery(searchQuery);
  }, [searchQuery, debouncedSetQuery]);

  // Perform search with reranking when query changes
  useEffect(() => {
    if (useReranker && debouncedQuery && rerankerConfig) {
      searchWithRerankingMutation.mutate({
        query: searchQueryObject,
        config: rerankerConfig,
      });
    }
  }, [debouncedQuery, JSON.stringify(filters), useReranker, rerankerConfig]);

  // Handle search submit
  const handleSearchSubmit = () => {
    if (!searchQuery.trim()) return;

    // Add to recent queries
    const newRecentQueries = [
      searchQuery,
      ...recentQueries.filter(q => q !== searchQuery),
    ].slice(0, 10);
    setRecentQueries(newRecentQueries);
    localStorage.setItem('recentSearchQueries', JSON.stringify(newRecentQueries));

    // Force search
    if (useReranker && rerankerConfig) {
      searchWithRerankingMutation.mutate({
        query: searchQueryObject,
        config: rerankerConfig,
      });
    }
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setDebouncedQuery('');
  };

  const handleRecentQuerySelect = (query: string) => {
    setSearchQuery(query);
  };

  const handleFilterChange = (newFilters: typeof filters) => {
    setFilters(newFilters);
  };

  return (
    <Box>
      {/* Search Header */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SearchIcon />
          Tool Search Interface
          {useReranker && (
            <Chip 
              label="Reranker Enabled" 
              color="primary" 
              size="small"
              icon={<SpeedIcon />}
            />
          )}
        </Typography>
        
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Search through the tool inventory using semantic search with optional reranking for improved relevance.
        </Typography>

        {/* Main Search Bar */}
        <Box sx={{ mt: 2, mb: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <TextField
                fullWidth
                placeholder="Search for tools... (e.g., 'file operations', 'data processing', 'API integration')"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearchSubmit()}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon color="action" />
                    </InputAdornment>
                  ),
                  endAdornment: searchQuery && (
                    <InputAdornment position="end">
                      <IconButton onClick={handleClearSearch} size="small">
                        <ClearIcon />
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="contained"
                  onClick={handleSearchSubmit}
                  disabled={!searchQuery.trim() || isLoading}
                  startIcon={isLoading ? <CircularProgress size={16} /> : <SearchIcon />}
                >
                  Search
                </Button>
                
                <Tooltip title="Toggle Filters">
                  <IconButton onClick={() => setShowFilters(!showFilters)}>
                    <Badge badgeContent={
                      (filters.category ? 1 : 0) + 
                      (filters.source ? 1 : 0) + 
                      filters.tags.length
                    } color="primary">
                      <FilterIcon />
                    </Badge>
                  </IconButton>
                </Tooltip>
              </Box>
            </Grid>
          </Grid>
        </Box>

        {/* Search Options */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <FormControlLabel
            control={
              <Switch
                checked={useReranker}
                onChange={(e) => setUseReranker(e.target.checked)}
                disabled={!rerankerConfig?.enabled}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SpeedIcon fontSize="small" />
                Use Reranker
              </Box>
            }
          />
          
          <ModelIndicator compact={true} />
        </Box>

        {/* Recent Queries */}
        {recentQueries.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Recent searches:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
              {recentQueries.slice(0, 5).map((query, index) => (
                <Chip
                  key={index}
                  label={truncateText(query, 30)}
                  size="small"
                  variant="outlined"
                  clickable
                  onClick={() => handleRecentQuerySelect(query)}
                />
              ))}
            </Box>
          </Box>
        )}
      </Paper>

      {/* Search Filters */}
      {showFilters && (
        <SearchFilters
          filters={filters}
          onChange={handleFilterChange}
          onClose={() => setShowFilters(false)}
        />
      )}

      {/* Search Results */}
      <Box>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Search failed: {error instanceof Error ? error.message : 'Unknown error'}
          </Alert>
        )}

        {isLoading && (
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <CircularProgress />
              <Typography variant="body2" sx={{ mt: 2 }}>
                Searching tools...
              </Typography>
            </CardContent>
          </Card>
        )}

        {searchData && !isLoading && (
          <>
            {/* Search Statistics */}
            <Paper elevation={0} sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                <Typography variant="body2">
                  Found <strong>{searchData.results.length}</strong> results
                  {searchData.metadata.total_found > searchData.results.length && 
                    ` of ${searchData.metadata.total_found} total`}
                </Typography>
                
                <Chip
                  label={`${formatDuration(searchData.metadata.search_time * 1000)}`}
                  size="small"
                  icon={<SpeedIcon />}
                />
                
                {searchData.metadata.reranker_used && (
                  <Chip
                    label={`Reranked by ${searchData.metadata.reranker_used}`}
                    size="small"
                    color="primary"
                    icon={<ExtensionIcon />}
                  />
                )}
              </Box>
            </Paper>

            {/* Results List */}
            {searchData.results.length > 0 ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {searchData.results.map((result, index) => (
                  <SearchResultCard
                    key={`${result.tool.id}-${index}`}
                    result={result}
                    query={searchQuery}
                    showRank={true}
                    showScore={true}
                  />
                ))}
              </Box>
            ) : (
              <Card>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  <SearchIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary">
                    No tools found
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Try adjusting your search query or filters
                  </Typography>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {!searchData && !isLoading && !error && (
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <SearchIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                Ready to search
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Enter a search query to find relevant tools
              </Typography>
            </CardContent>
          </Card>
        )}
      </Box>
    </Box>
  );
};

export default SearchInterface;