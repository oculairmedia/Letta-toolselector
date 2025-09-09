import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Rating,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  Chip,
  Divider,
  Alert,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  Star as StarIcon,
  Save as SaveIcon,
  History as HistoryIcon,
} from '@mui/icons-material';

import { useSubmitEvaluation, useEvaluations } from '../../hooks/useApi';
import { EvaluationRating } from '../../types';
import { formatRelativeTime } from '../../utils';

const EvaluationInterface: React.FC = () => {
  const [selectedResult, setSelectedResult] = useState<any>(null);
  const [relevanceRating, setRelevanceRating] = useState<number>(0);
  const [usefulnessRating, setUsefulnessRating] = useState<number>(0);
  const [notes, setNotes] = useState('');

  const { data: evaluations } = useEvaluations();
  const submitEvaluationMutation = useSubmitEvaluation();

  const handleSubmitEvaluation = async () => {
    if (!selectedResult || !relevanceRating || !usefulnessRating) return;

    try {
      await submitEvaluationMutation.mutateAsync({
        result_id: selectedResult.id,
        query: 'sample query', // This would come from the search context
        relevance_score: relevanceRating,
        usefulness_score: usefulnessRating,
        notes: notes.trim() || undefined,
      });

      // Reset form
      setRelevanceRating(0);
      setUsefulnessRating(0);
      setNotes('');
      setSelectedResult(null);
    } catch (error) {
      console.error('Failed to submit evaluation:', error);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AssessmentIcon />
          Manual Evaluation Interface
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manually evaluate search results to improve reranker performance and build evaluation datasets.
        </Typography>
      </Paper>

      <Grid container spacing={3}>
        {/* Evaluation Form */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Evaluate Search Result
              </Typography>

              {!selectedResult ? (
                <Alert severity="info">
                  Select a search result from a recent query to begin evaluation.
                </Alert>
              ) : (
                <Box>
                  {/* Result Summary */}
                  <Card variant="outlined" sx={{ mb: 3 }}>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        {selectedResult.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {selectedResult.description}
                      </Typography>
                    </CardContent>
                  </Card>

                  {/* Rating Form */}
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Typography component="legend" gutterBottom>
                        Relevance Rating
                      </Typography>
                      <Rating
                        name="relevance-rating"
                        value={relevanceRating}
                        onChange={(_, newValue) => setRelevanceRating(newValue || 0)}
                        size="large"
                        icon={<StarIcon fontSize="inherit" />}
                      />
                      <Typography variant="caption" display="block" color="text.secondary">
                        How relevant is this result to the search query?
                      </Typography>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Typography component="legend" gutterBottom>
                        Usefulness Rating
                      </Typography>
                      <Rating
                        name="usefulness-rating"
                        value={usefulnessRating}
                        onChange={(_, newValue) => setUsefulnessRating(newValue || 0)}
                        size="large"
                        icon={<StarIcon fontSize="inherit" />}
                      />
                      <Typography variant="caption" display="block" color="text.secondary">
                        How useful is this tool for the intended task?
                      </Typography>
                    </Grid>

                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        multiline
                        rows={3}
                        label="Notes (Optional)"
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        placeholder="Add any additional comments about this evaluation..."
                      />
                    </Grid>

                    <Grid item xs={12}>
                      <Button
                        variant="contained"
                        onClick={handleSubmitEvaluation}
                        disabled={!relevanceRating || !usefulnessRating || submitEvaluationMutation.isPending}
                        startIcon={<SaveIcon />}
                      >
                        {submitEvaluationMutation.isPending ? 'Submitting...' : 'Submit Evaluation'}
                      </Button>
                    </Grid>
                  </Grid>
                </Box>
              )}

              {submitEvaluationMutation.isSuccess && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  Evaluation submitted successfully!
                </Alert>
              )}

              {submitEvaluationMutation.error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  Failed to submit evaluation: {submitEvaluationMutation.error.message}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Evaluations */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <HistoryIcon />
                Recent Evaluations
              </Typography>

              {evaluations && evaluations.length > 0 ? (
                <List>
                  {evaluations.slice(0, 10).map((evaluation, index) => (
                    <React.Fragment key={evaluation.result_id + evaluation.timestamp}>
                      <ListItem sx={{ px: 0 }}>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Chip label={`R: ${evaluation.relevance_score}`} size="small" />
                              <Chip label={`U: ${evaluation.usefulness_score}`} size="small" />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="caption" display="block">
                                Query: {evaluation.query}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {formatRelativeTime(evaluation.timestamp)}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < evaluations.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                  No evaluations yet
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default EvaluationInterface;