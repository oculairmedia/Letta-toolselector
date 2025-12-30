import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';

import { SearchResponse } from '../../types';
import { formatScore, formatDuration, getScoreColor } from '../../utils';

interface ComparisonMetricsProps {
  data: {
    improvements: any[];
    declines: any[];
    unchanged: any[];
    totalChanges: number;
    avgScoreChange: number;
  };
  originalResults: SearchResponse;
  rerankedResults: SearchResponse;
}

const ComparisonMetrics: React.FC<ComparisonMetricsProps> = ({
  data,
  originalResults,
  rerankedResults,
}) => {
  const improvementRate = (data.improvements.length / (data.improvements.length + data.declines.length)) * 100;
  const timeComparison = rerankedResults.metadata.search_time - originalResults.metadata.search_time;
  
  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Comparison Metrics
        </Typography>
        
        <Grid container spacing={3}>
          {/* Improvement Rate */}
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary" gutterBottom>
                {improvementRate.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Improvement Rate
              </Typography>
              <Box sx={{ mt: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={improvementRate}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              </Box>
            </Box>
          </Grid>

          {/* Position Changes */}
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main" gutterBottom>
                {data.improvements.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Improved Positions
              </Typography>
              <Chip
                icon={<TrendingUpIcon />}
                label={`vs ${data.declines.length} declined`}
                size="small"
                color="success"
                sx={{ mt: 1 }}
              />
            </Box>
          </Grid>

          {/* Average Score Change */}
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography 
                variant="h4" 
                color={data.avgScoreChange >= 0 ? 'success.main' : 'error.main'}
                gutterBottom
              >
                {data.avgScoreChange >= 0 ? '+' : ''}{formatScore(data.avgScoreChange)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Score Change
              </Typography>
              <Chip
                icon={data.avgScoreChange >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                label={data.avgScoreChange >= 0 ? 'Improved' : 'Declined'}
                size="small"
                color={data.avgScoreChange >= 0 ? 'success' : 'error'}
                sx={{ mt: 1 }}
              />
            </Box>
          </Grid>

          {/* Time Comparison */}
          <Grid item xs={12} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography 
                variant="h4" 
                color={timeComparison <= 0 ? 'success.main' : 'warning.main'}
                gutterBottom
              >
                {timeComparison >= 0 ? '+' : ''}{formatDuration(timeComparison * 1000)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Time Difference
              </Typography>
              <Chip
                icon={<SpeedIcon />}
                label={timeComparison <= 0 ? 'Faster' : 'Slower'}
                size="small"
                color={timeComparison <= 0 ? 'success' : 'warning'}
                sx={{ mt: 1 }}
              />
            </Box>
          </Grid>
        </Grid>

        {/* Additional Metrics Row */}
        <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Typography variant="body2" color="text.secondary">
                Original Search Time: {formatDuration(originalResults.metadata.search_time * 1000)}
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="body2" color="text.secondary">
                Reranked Search Time: {formatDuration(rerankedResults.metadata.search_time * 1000)}
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="body2" color="text.secondary">
                Reranker Used: {rerankedResults.metadata.reranker_used || 'Unknown'}
              </Typography>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ComparisonMetrics;