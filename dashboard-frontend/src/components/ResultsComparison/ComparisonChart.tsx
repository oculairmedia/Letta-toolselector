import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  useTheme,
} from '@mui/material';
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
  ScatterChart,
  Scatter,
} from 'recharts';

import { SearchResult } from '../../types';
import { formatScore } from '../../utils';

interface ComparisonChartProps {
  originalResults: SearchResult[];
  rerankedResults: SearchResult[];
}

const ComparisonChart: React.FC<ComparisonChartProps> = ({
  originalResults,
  rerankedResults,
}) => {
  const theme = useTheme();

  // Prepare data for comparison
  const comparisonData = React.useMemo(() => {
    const data = [];
    const maxLength = Math.max(originalResults.length, rerankedResults.length);

    for (let i = 0; i < maxLength; i++) {
      const original = originalResults[i];
      const reranked = rerankedResults[i];

      data.push({
        rank: i + 1,
        originalScore: original?.score || 0,
        rerankedScore: reranked?.score || 0,
        originalTool: original?.tool.name || '',
        rerankedTool: reranked?.tool.name || '',
      });
    }

    return data;
  }, [originalResults, rerankedResults]);

  // Prepare scatter plot data for score correlation
  const scatterData = React.useMemo(() => {
    const toolScores = new Map();
    
    // Map original scores
    originalResults.forEach(result => {
      toolScores.set(result.tool.id, { original: result.score });
    });

    // Add reranked scores
    rerankedResults.forEach(result => {
      if (toolScores.has(result.tool.id)) {
        toolScores.get(result.tool.id).reranked = result.score;
        toolScores.get(result.tool.id).name = result.tool.name;
      }
    });

    return Array.from(toolScores.values()).filter(item => 
      item.original !== undefined && item.reranked !== undefined
    );
  }, [originalResults, rerankedResults]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ 
          bgcolor: 'background.paper', 
          p: 2, 
          border: 1, 
          borderColor: 'divider', 
          borderRadius: 1,
          boxShadow: 2,
        }}>
          <Typography variant="subtitle2">Rank #{label}</Typography>
          {payload.map((entry: any, index: number) => (
            <Typography key={index} variant="body2" sx={{ color: entry.color }}>
              {entry.dataKey === 'originalScore' ? 'Original' : 'Reranked'}: {formatScore(entry.value)}
            </Typography>
          ))}
        </Box>
      );
    }
    return null;
  };

  const ScatterTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Box sx={{ 
          bgcolor: 'background.paper', 
          p: 2, 
          border: 1, 
          borderColor: 'divider', 
          borderRadius: 1,
          boxShadow: 2,
        }}>
          <Typography variant="subtitle2">{data.name}</Typography>
          <Typography variant="body2">Original: {formatScore(data.original)}</Typography>
          <Typography variant="body2">Reranked: {formatScore(data.reranked)}</Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Box>
      {/* Score Comparison Line Chart */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Score Comparison by Rank
          </Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="rank" 
                  label={{ value: 'Rank Position', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
                  tickFormatter={formatScore}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="originalScore" 
                  stroke={theme.palette.secondary.main}
                  strokeWidth={2}
                  name="Original Results"
                  dot={{ r: 4 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="rerankedScore" 
                  stroke={theme.palette.primary.main}
                  strokeWidth={2}
                  name="Reranked Results"
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Score Correlation Scatter Plot */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Score Correlation Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Each point represents a tool. Points above the diagonal line indicate score improvements.
          </Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={scatterData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="original" 
                  type="number"
                  domain={[0, 1]}
                  label={{ value: 'Original Score', position: 'insideBottom', offset: -5 }}
                  tickFormatter={formatScore}
                />
                <YAxis 
                  dataKey="reranked" 
                  type="number"
                  domain={[0, 1]}
                  label={{ value: 'Reranked Score', angle: -90, position: 'insideLeft' }}
                  tickFormatter={formatScore}
                />
                <Tooltip content={<ScatterTooltip />} />
                
                {/* Diagonal reference line (y = x) */}
                <Line
                  type="linear"
                  dataKey="original"
                  stroke={theme.palette.grey[400]}
                  strokeDasharray="5 5"
                  dot={false}
                  activeDot={false}
                />
                
                <Scatter 
                  dataKey="reranked" 
                  fill={theme.palette.primary.main}
                  fillOpacity={0.7}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Score Distribution Bar Chart */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Score Distribution Comparison
          </Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="rank" 
                  label={{ value: 'Rank Position', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
                  tickFormatter={formatScore}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar 
                  dataKey="originalScore" 
                  fill={theme.palette.secondary.main}
                  name="Original Results"
                  opacity={0.7}
                />
                <Bar 
                  dataKey="rerankedScore" 
                  fill={theme.palette.primary.main}
                  name="Reranked Results"
                  opacity={0.7}
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ComparisonChart;