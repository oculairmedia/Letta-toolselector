import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Chip,
  Box,
  IconButton,
  Tooltip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Extension as ExtensionIcon,
  Source as SourceIcon,
  ExpandMore as ExpandMoreIcon,
  ContentCopy as CopyIcon,
  OpenInNew as OpenInNewIcon,
  Psychology as ReasoningIcon,
} from '@mui/icons-material';

import { SearchResult } from '../../types';
import { formatScore, getScoreColor, truncateText, highlightSearchTerms } from '../../utils';

interface SearchResultCardProps {
  result: SearchResult;
  query: string;
  showRank?: boolean;
  showScore?: boolean;
  showReasoning?: boolean;
  compact?: boolean;
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({
  result,
  query,
  showRank = true,
  showScore = true,
  showReasoning = true,
  compact = false,
}) => {
  const { tool, score, rank, reasoning } = result;

  const handleCopyId = () => {
    navigator.clipboard.writeText(tool.id);
  };

  const handleCopyName = () => {
    navigator.clipboard.writeText(tool.name);
  };

  const ScoreDisplay = () => (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Typography variant="caption" color="text.secondary">
        Score:
      </Typography>
      <Box sx={{ minWidth: 60 }}>
        <LinearProgress
          variant="determinate"
          value={score * 100}
          sx={{
            height: 6,
            borderRadius: 3,
            bgcolor: 'rgba(0,0,0,0.1)',
            '& .MuiLinearProgress-bar': {
              bgcolor: getScoreColor(score),
            },
          }}
        />
      </Box>
      <Typography
        variant="caption"
        sx={{ 
          color: getScoreColor(score),
          fontWeight: 'bold',
          minWidth: 40,
        }}
      >
        {formatScore(score)}
      </Typography>
    </Box>
  );

  const RankDisplay = () => (
    <Chip
      label={`#${rank}`}
      size="small"
      color="primary"
      variant="outlined"
    />
  );

  if (compact) {
    return (
      <Card sx={{ mb: 1 }}>
        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {showRank && <RankDisplay />}
            
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="subtitle2" noWrap>
                {tool.name}
              </Typography>
            </Box>

            {showScore && (
              <Box sx={{ minWidth: 120 }}>
                <ScoreDisplay />
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ mb: 2, transition: 'all 0.2s ease-in-out' }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2, mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ExtensionIcon color="primary" />
            {showRank && <RankDisplay />}
          </Box>

          <Box sx={{ flexGrow: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="h6" component="h3">
                <span dangerouslySetInnerHTML={{ 
                  __html: highlightSearchTerms(tool.name, query) 
                }} />
              </Typography>
              
              <Tooltip title="Copy tool name">
                <IconButton size="small" onClick={handleCopyName}>
                  <CopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>

            <Typography 
              variant="body2" 
              color="text.secondary"
              sx={{ 
                mb: 1,
                display: '-webkit-box',
                WebkitLineClamp: 3,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
              }}
            >
              <span dangerouslySetInnerHTML={{ 
                __html: highlightSearchTerms(tool.description, query) 
              }} />
            </Typography>

            {/* Tool Metadata */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              <Chip
                label={tool.source}
                size="small"
                icon={<SourceIcon />}
                variant="outlined"
              />
              
              {tool.category && (
                <Chip
                  label={tool.category}
                  size="small"
                  color="secondary"
                  variant="outlined"
                />
              )}
              
              {tool.tags && tool.tags.length > 0 && (
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  {tool.tags.slice(0, 3).map((tag, index) => (
                    <Chip
                      key={index}
                      label={tag}
                      size="small"
                      variant="filled"
                      sx={{ bgcolor: 'action.hover' }}
                    />
                  ))}
                  {tool.tags.length > 3 && (
                    <Chip
                      label={`+${tool.tags.length - 3}`}
                      size="small"
                      variant="outlined"
                    />
                  )}
                </Box>
              )}
            </Box>
          </Box>

          {/* Score Display */}
          {showScore && (
            <Box sx={{ minWidth: 140, textAlign: 'right' }}>
              <ScoreDisplay />
            </Box>
          )}
        </Box>

        {/* Tool ID */}
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1, 
          p: 1, 
          bgcolor: 'action.hover',
          borderRadius: 1,
          mb: reasoning ? 1 : 0,
        }}>
          <Typography variant="caption" color="text.secondary">
            ID:
          </Typography>
          <Typography variant="caption" fontFamily="monospace" sx={{ flexGrow: 1 }}>
            {tool.id}
          </Typography>
          <Tooltip title="Copy tool ID">
            <IconButton size="small" onClick={handleCopyId}>
              <CopyIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Reasoning (if available) */}
        {showReasoning && reasoning && (
          <Accordion sx={{ mt: 1, boxShadow: 'none', '&:before': { display: 'none' } }}>
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              sx={{ 
                px: 0, 
                minHeight: 'auto',
                '& .MuiAccordionSummary-content': { margin: '8px 0' },
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ReasoningIcon fontSize="small" color="action" />
                <Typography variant="caption" color="text.secondary">
                  Reranker Reasoning
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0, pt: 0 }}>
              <Typography
                variant="body2"
                sx={{
                  bgcolor: 'action.hover',
                  p: 2,
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'divider',
                  fontStyle: 'italic',
                }}
              >
                {reasoning}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}
      </CardContent>
    </Card>
  );
};

export default SearchResultCard;