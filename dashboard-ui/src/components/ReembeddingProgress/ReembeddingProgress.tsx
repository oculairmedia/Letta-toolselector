import React from 'react';
import {
  Box,
  Card,
  CardContent,
  LinearProgress,
  Typography,
  Button,
  Alert,
  Chip,
  Stack,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  Cancel as CancelIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Schedule as ScheduleIcon,
  CheckCircle as CompleteIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

import { useReembeddingProgress, useCancelReembedding } from '../../hooks/useApi';

interface ReembeddingProgressProps {
  onComplete?: () => void;
  onError?: (error: string) => void;
}

const ReembeddingProgress: React.FC<ReembeddingProgressProps> = ({
  onComplete,
  onError,
}) => {
  const [expanded, setExpanded] = React.useState(true);
  const { data: progress, isLoading } = useReembeddingProgress();
  const cancelMutation = useCancelReembedding();

  const handleCancel = async () => {
    try {
      await cancelMutation.mutateAsync();
    } catch (error) {
      console.error('Failed to cancel re-embedding:', error);
    }
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  };

  const formatTimeRemaining = (etaSeconds?: number) => {
    if (!etaSeconds) return 'Calculating...';
    return `~${formatDuration(etaSeconds)} remaining`;
  };

  React.useEffect(() => {
    if (progress?.status === 'completed' && onComplete) {
      onComplete();
    } else if (progress?.status === 'error' && onError && progress.error) {
      onError(progress.error);
    }
  }, [progress?.status, onComplete, onError, progress?.error]);

  // Don't render if no active re-embedding process
  if (isLoading || !progress || progress.status === 'idle') {
    return null;
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'cancelled': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <ScheduleIcon />;
      case 'completed': return <CompleteIcon />;
      case 'error': return <ErrorIcon />;
      case 'cancelled': return <CancelIcon />;
      default: return <ScheduleIcon />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return 'Re-embedding in Progress';
      case 'completed': return 'Re-embedding Completed';
      case 'error': return 'Re-embedding Failed';
      case 'cancelled': return 'Re-embedding Cancelled';
      default: return status;
    }
  };

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {getStatusIcon(progress.status)}
            <Typography variant="h6">
              {getStatusText(progress.status)}
            </Typography>
            <Chip
              label={progress.status.toUpperCase()}
              size="small"
              color={getStatusColor(progress.status)}
            />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {progress.status === 'running' && (
              <Button
                size="small"
                color="error"
                variant="outlined"
                startIcon={<CancelIcon />}
                onClick={handleCancel}
                disabled={cancelMutation.isPending}
              >
                Cancel
              </Button>
            )}
            <IconButton
              size="small"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? <CollapseIcon /> : <ExpandIcon />}
            </IconButton>
          </Box>
        </Box>

        <Collapse in={expanded}>
          {progress.status === 'running' && progress.progress && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Progress: {progress.progress.current} / {progress.progress.total}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {progress.progress.percentage.toFixed(1)}%
                </Typography>
              </Box>
              
              <LinearProgress
                variant="determinate"
                value={progress.progress.percentage}
                sx={{ mb: 2, height: 6, borderRadius: 3 }}
              />

              <Stack spacing={1}>
                {progress.progress.current_item && (
                  <Typography variant="caption" color="text.secondary">
                    Currently processing: {progress.progress.current_item}
                  </Typography>
                )}
                
                {progress.progress.eta_seconds && (
                  <Typography variant="caption" color="text.secondary">
                    {formatTimeRemaining(progress.progress.eta_seconds)}
                  </Typography>
                )}
              </Stack>
            </Box>
          )}

          {progress.status === 'error' && progress.error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Error:</strong> {progress.error}
              </Typography>
            </Alert>
          )}

          {progress.status === 'completed' && (
            <Alert severity="success" sx={{ mb: 2 }}>
              <Typography variant="body2">
                Re-embedding completed successfully! The search index has been updated with the new embedding model.
              </Typography>
            </Alert>
          )}

          {progress.status === 'cancelled' && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <Typography variant="body2">
                Re-embedding was cancelled. The search index remains unchanged.
              </Typography>
            </Alert>
          )}

          {/* Timing information */}
          {(progress.started_at || progress.completed_at) && (
            <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="caption" color="text.secondary">
                {progress.started_at && `Started: ${new Date(progress.started_at).toLocaleString()}`}
                {progress.completed_at && ` • Completed: ${new Date(progress.completed_at).toLocaleString()}`}
              </Typography>
            </Box>
          )}
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default ReembeddingProgress;