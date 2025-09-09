import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Alert,
  Divider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Compare as CompareIcon,
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  SwapHoriz as SwapIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';

import { useSearch, useSearchWithReranking, useRerankerConfig } from '../../hooks/useApi';
import { SearchQuery, SearchResult } from '../../types';
import { formatScore, getScoreColor, debounce } from '../../utils';
import SearchResultCard from '../SearchInterface/SearchResultCard';
import ComparisonChart from './ComparisonChart';
import ComparisonMetrics from './ComparisonMetrics';

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
      id={`comparison-tabpanel-${index}`}
      aria-labelledby={`comparison-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const ResultsComparison: React.FC = () => {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  const [showMetrics, setShowMetrics] = useState(true);

  const { data: rerankerConfig } = useRerankerConfig();

  // Create search query object
  const searchQuery: SearchQuery = {
    query: debouncedQuery,
    limit: 10,
  };

  // Get both original and reranked results
  const { data: originalResults, isLoading: originalLoading } = useSearch(
    searchQuery,
    !!debouncedQuery
  );

  const { data: rerankedResults, mutate: searchWithReranking, isPending: rerankedLoading } = useSearchWithReranking();

  // Debounce search
  const debouncedSetQuery = useMemo(
    () => debounce((newQuery: string) => setDebouncedQuery(newQuery), 500),
    []
  );

  React.useEffect(() => {
    debouncedSetQuery(query);
  }, [query, debouncedSetQuery]);

  // Trigger reranked search when query changes
  React.useEffect(() => {
    if (debouncedQuery && rerankerConfig) {
      searchWithReranking({
        query: searchQuery,
        config: rerankerConfig,
      });
    }
  }, [debouncedQuery, rerankerConfig]);

  // Comparison analysis
  const comparisonData = useMemo(() => {
    if (!originalResults || !rerankedResults) return null;

    const originalRanks = new Map(originalResults.results.map((r, i) => [r.tool.id, i + 1]));
    const rerankedRanks = new Map(rerankedResults.results.map((r, i) => [r.tool.id, i + 1]));

    const improvements = [];
    const declines = [];
    const unchanged = [];

    // Analyze ranking changes
    for (const result of rerankedResults.results) {
      const toolId = result.tool.id;
      const originalRank = originalRanks.get(toolId);
      const rerankedRank = rerankedRanks.get(toolId);

      if (originalRank && rerankedRank) {
        const change = originalRank - rerankedRank; // Positive = improvement (lower rank number)
        
        if (change > 0) {
          improvements.push({
            tool: result.tool,
            originalRank,
            rerankedRank,
            change,
            originalScore: originalResults.results.find(r => r.tool.id === toolId)?.score || 0,
            rerankedScore: result.score,
          });
        } else if (change < 0) {
          declines.push({
            tool: result.tool,
            originalRank,
            rerankedRank,
            change: Math.abs(change),
            originalScore: originalResults.results.find(r => r.tool.id === toolId)?.score || 0,
            rerankedScore: result.score,
          });
        } else {
          unchanged.push({
            tool: result.tool,
            rank: originalRank,
            originalScore: originalResults.results.find(r => r.tool.id === toolId)?.score || 0,
            rerankedScore: result.score,
          });
        }
      }
    }

    return {
      improvements: improvements.sort((a, b) => b.change - a.change),
      declines: declines.sort((a, b) => b.change - a.change),
      unchanged,
      totalChanges: improvements.length + declines.length,
      avgScoreChange: rerankedResults.results.reduce((sum, result, i) => {
        const originalScore = originalResults.results[i]?.score || 0;
        return sum + (result.score - originalScore);
      }, 0) / Math.min(rerankedResults.results.length, originalResults.results.length),
    };
  }, [originalResults, rerankedResults]);

  const handleSearch = () => {
    if (!query.trim()) return;
    setDebouncedQuery(query);
  };

  const isLoading = originalLoading || rerankedLoading;
  const hasResults = originalResults || rerankedResults;

  return (
    <Box>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CompareIcon />
          Results Comparison
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Compare search results before and after reranking to analyze performance improvements.
        </Typography>
      </Paper>

      {/* Search Interface */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              placeholder="Enter search query to compare results..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Button
                variant="contained"
                onClick={handleSearch}
                disabled={!query.trim() || isLoading}
                startIcon={<SearchIcon />}
              >
                Compare
              </Button>
              <FormControlLabel
                control={
                  <Switch
                    checked={showMetrics}
                    onChange={(e) => setShowMetrics(e.target.checked)}
                  />
                }
                label="Show Metrics"
              />
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Results */}
      {hasResults && !isLoading && (
        <Box>
          {/* Comparison Metrics */}
          {showMetrics && comparisonData && (
            <ComparisonMetrics
              data={comparisonData}
              originalResults={originalResults!}
              rerankedResults={rerankedResults!}
            />
          )}

          {/* Comparison Tabs */}
          <Paper elevation={1}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                <Tab 
                  label="Side by Side" 
                  icon={<SwapIcon />} 
                  iconPosition="start" 
                />
                <Tab 
                  label="Ranking Changes" 
                  icon={<TimelineIcon />} 
                  iconPosition="start" 
                />
                <Tab 
                  label="Performance Chart" 
                  icon={<TrendingUpIcon />} 
                  iconPosition="start" 
                />
              </Tabs>
            </Box>

            <TabPanel value={activeTab} index={0}>
              {/* Side by Side Comparison */}
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    Original Results
                    <Chip label={`${originalResults?.results.length || 0} results`} size="small" />
                  </Typography>
                  <Box sx={{ maxHeight: '600px', overflowY: 'auto' }}>
                    {originalResults?.results.map((result, index) => (
                      <SearchResultCard
                        key={`original-${result.tool.id}`}
                        result={result}
                        query={query}
                        compact={true}
                        showReasoning={false}
                      />
                    ))}
                  </Box>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    Reranked Results
                    <Chip label={`${rerankedResults?.results.length || 0} results`} size="small" color="primary" />
                  </Typography>
                  <Box sx={{ maxHeight: '600px', overflowY: 'auto' }}>
                    {rerankedResults?.results.map((result, index) => (
                      <SearchResultCard
                        key={`reranked-${result.tool.id}`}
                        result={result}
                        query={query}
                        compact={true}
                        showReasoning={true}
                      />
                    ))}
                  </Box>
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={activeTab} index={1}>
              {/* Ranking Changes */}
              {comparisonData && (
                <Box>
                  {/* Improvements */}
                  {comparisonData.improvements.length > 0 && (
                    <Card sx={{ mb: 3 }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <TrendingUpIcon color="success" />
                          Improved Rankings ({comparisonData.improvements.length})
                        </Typography>
                        <TableContainer>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Tool</TableCell>
                                <TableCell align="center">Original Rank</TableCell>
                                <TableCell align="center">New Rank</TableCell>
                                <TableCell align="center">Improvement</TableCell>
                                <TableCell align="center">Score Change</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {comparisonData.improvements.map((item, index) => (
                                <TableRow key={index}>
                                  <TableCell>
                                    <Typography variant="body2" fontWeight="medium">
                                      {item.tool.name}
                                    </Typography>
                                  </TableCell>
                                  <TableCell align="center">#{item.originalRank}</TableCell>
                                  <TableCell align="center">#{item.rerankedRank}</TableCell>
                                  <TableCell align="center">
                                    <Chip 
                                      label={`+${item.change}`} 
                                      size="small" 
                                      color="success"
                                      icon={<TrendingUpIcon />}
                                    />
                                  </TableCell>
                                  <TableCell align="center">
                                    <Typography 
                                      variant="body2" 
                                      color={item.rerankedScore > item.originalScore ? 'success.main' : 'error.main'}
                                    >
                                      {formatScore(item.rerankedScore - item.originalScore)}
                                    </Typography>
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </CardContent>
                    </Card>
                  )}

                  {/* Declines */}
                  {comparisonData.declines.length > 0 && (
                    <Card sx={{ mb: 3 }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <TrendingDownIcon color="error" />
                          Declined Rankings ({comparisonData.declines.length})
                        </Typography>
                        <TableContainer>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Tool</TableCell>
                                <TableCell align="center">Original Rank</TableCell>
                                <TableCell align="center">New Rank</TableCell>
                                <TableCell align="center">Decline</TableCell>
                                <TableCell align="center">Score Change</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {comparisonData.declines.map((item, index) => (
                                <TableRow key={index}>
                                  <TableCell>
                                    <Typography variant="body2" fontWeight="medium">
                                      {item.tool.name}
                                    </Typography>
                                  </TableCell>
                                  <TableCell align="center">#{item.originalRank}</TableCell>
                                  <TableCell align="center">#{item.rerankedRank}</TableCell>
                                  <TableCell align="center">
                                    <Chip 
                                      label={`-${item.change}`} 
                                      size="small" 
                                      color="error"
                                      icon={<TrendingDownIcon />}
                                    />
                                  </TableCell>
                                  <TableCell align="center">
                                    <Typography 
                                      variant="body2" 
                                      color={item.rerankedScore > item.originalScore ? 'success.main' : 'error.main'}
                                    >
                                      {formatScore(item.rerankedScore - item.originalScore)}
                                    </Typography>
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </CardContent>
                    </Card>
                  )}
                </Box>
              )}
            </TabPanel>

            <TabPanel value={activeTab} index={2}>
              {/* Performance Chart */}
              {originalResults && rerankedResults && (
                <ComparisonChart
                  originalResults={originalResults.results}
                  rerankedResults={rerankedResults.results}
                />
              )}
            </TabPanel>
          </Paper>
        </Box>
      )}

      {/* No Results State */}
      {!hasResults && !isLoading && (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 8 }}>
            <CompareIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Ready to Compare
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Enter a search query above to compare original vs reranked results
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Loading State */}
      {isLoading && (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 4 }}>
            <Typography>Comparing search results...</Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default ResultsComparison;