import React, { useState, useEffect, useCallback } from 'react';
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
  Tab,
  Tabs,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Switch,
  FormControlLabel,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  Star as StarIcon,
  Save as SaveIcon,
  History as HistoryIcon,
  Compare as CompareIcon,
  ExpandMore as ExpandMoreIcon,
  Download as DownloadIcon,
  Flag as FlagIcon,
  Timer as TimerIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';

import { Context7Evaluation, ToolEvaluation, SearchResult, EvaluationSession } from '../../types';

interface Context7EvaluationInterfaceProps {
  searchResults?: {
    query: string;
    query_context?: string;
    original_results: SearchResult[];
    reranked_results: SearchResult[];
    search_session_id: string;
  };
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const Context7EvaluationInterface: React.FC<Context7EvaluationInterfaceProps> = ({
  searchResults,
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [evaluation, setEvaluation] = useState<Partial<Context7Evaluation>>({
    overall_relevance: 0,
    completeness_score: 0,
    precision_score: 0,
    contextual_appropriateness: 0,
    improvement_rating: 0,
    tool_evaluations: [],
    strengths: '',
    weaknesses: '',
    suggestions: '',
    edge_cases: [],
  });

  const [currentSession, setCurrentSession] = useState<EvaluationSession | null>(null);
  const [evaluationStartTime, setEvaluationStartTime] = useState<Date>(new Date());
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showValidationErrors, setShowValidationErrors] = useState(false);

  // Initialize tool evaluations when search results change
  useEffect(() => {
    if (searchResults) {
      const allTools = new Set([
        ...searchResults.original_results.map(r => r.tool.id),
        ...searchResults.reranked_results.map(r => r.tool.id),
      ]);

      const toolEvaluations: ToolEvaluation[] = Array.from(allTools).map(toolId => {
        const originalResult = searchResults.original_results.find(r => r.tool.id === toolId);
        const rerankedResult = searchResults.reranked_results.find(r => r.tool.id === toolId);
        const toolName = originalResult?.tool.name || rerankedResult?.tool.name || 'Unknown';

        return {
          tool_id: toolId,
          tool_name: toolName,
          relevance_score: 0,
          confidence_score: 0,
          found_in_original: !!originalResult,
          found_in_reranked: !!rerankedResult,
          original_rank: originalResult?.rank,
          reranked_rank: rerankedResult?.rank,
          notes: '',
        };
      });

      setEvaluation(prev => ({
        ...prev,
        query: searchResults.query,
        query_context: searchResults.query_context,
        search_session_id: searchResults.search_session_id,
        original_results: searchResults.original_results,
        reranked_results: searchResults.reranked_results,
        tool_evaluations: toolEvaluations,
      }));
    }
  }, [searchResults]);

  const updateToolEvaluation = useCallback((toolId: string, field: keyof ToolEvaluation, value: any) => {
    setEvaluation(prev => ({
      ...prev,
      tool_evaluations: prev.tool_evaluations?.map(te =>
        te.tool_id === toolId ? { ...te, [field]: value } : te
      ) || [],
    }));
  }, []);

  const validateEvaluation = (): boolean => {
    if (!evaluation.overall_relevance || !evaluation.completeness_score ||
        !evaluation.precision_score || !evaluation.contextual_appropriateness ||
        !evaluation.improvement_rating) {
      return false;
    }

    // Check that at least some tools have been rated
    const ratedTools = evaluation.tool_evaluations?.filter(te => te.relevance_score > 0) || [];
    return ratedTools.length > 0;
  };

  const handleSubmitEvaluation = async () => {
    if (!validateEvaluation()) {
      setShowValidationErrors(true);
      return;
    }

    setIsSubmitting(true);
    try {
      const evaluationTime = Math.floor((new Date().getTime() - evaluationStartTime.getTime()) / 1000);

      const completeEvaluation: Context7Evaluation = {
        ...evaluation as Context7Evaluation,
        id: `eval_${Date.now()}`,
        timestamp: new Date(),
        evaluation_time: evaluationTime,
      };

      // Here you would call the API to submit the evaluation
      console.log('Submitting evaluation:', completeEvaluation);

      // Reset form
      setEvaluation({
        overall_relevance: 0,
        completeness_score: 0,
        precision_score: 0,
        contextual_appropriateness: 0,
        improvement_rating: 0,
        tool_evaluations: [],
        strengths: '',
        weaknesses: '',
        suggestions: '',
        edge_cases: [],
      });
      setEvaluationStartTime(new Date());
      setCurrentTab(0);

    } catch (error) {
      console.error('Failed to submit evaluation:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const exportEvaluationData = () => {
    // Export evaluation data as JSON
    const dataStr = JSON.stringify(evaluation, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

    const exportFileDefaultName = `context7_evaluation_${Date.now()}.json`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  if (!searchResults) {
    return (
      <Paper elevation={1} sx={{ p: 3, textAlign: 'center' }}>
        <AssessmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          No Search Results to Evaluate
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Perform a search query to begin Context7 evaluation.
        </Typography>
      </Paper>
    );
  }

  const isValid = validateEvaluation();
  const evaluationProgress = [
    evaluation.overall_relevance,
    evaluation.completeness_score,
    evaluation.precision_score,
    evaluation.contextual_appropriateness,
    evaluation.improvement_rating,
  ].filter(score => score && score > 0).length / 5 * 100;

  return (
    <Box>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AssessmentIcon />
              Context7 Evaluation Interface
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Systematic evaluation according to Context7 standards for tool search quality assessment.
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Chip
              icon={<TimerIcon />}
              label={`${Math.floor((new Date().getTime() - evaluationStartTime.getTime()) / 1000)}s`}
              variant="outlined"
            />
            <Tooltip title="Export evaluation data">
              <IconButton onClick={() => setShowExportDialog(true)}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Progress Indicator */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2">Evaluation Progress</Typography>
            <Typography variant="body2">{Math.round(evaluationProgress)}%</Typography>
          </Box>
          <LinearProgress variant="determinate" value={evaluationProgress} />
        </Box>

        {/* Query Display */}
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>Query Context</Typography>
            <Typography variant="body1" sx={{ fontWeight: 'medium', mb: 1 }}>
              "{searchResults.query}"
            </Typography>
            {searchResults.query_context && (
              <Typography variant="body2" color="text.secondary">
                Context: {searchResults.query_context}
              </Typography>
            )}
            <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
              <Chip label={`${searchResults.original_results.length} Original Results`} />
              <Chip label={`${searchResults.reranked_results.length} Reranked Results`} />
            </Box>
          </CardContent>
        </Card>
      </Paper>

      {/* Evaluation Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={(_, newValue) => setCurrentTab(newValue)}>
            <Tab label="Overview Metrics" />
            <Tab label="Tool Comparison" />
            <Tab label="Individual Tools" />
            <Tab label="Qualitative Feedback" />
          </Tabs>
        </Box>

        {/* Overview Metrics Tab */}
        <TabPanel value={currentTab} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>Context7 Core Metrics</Typography>

              {[
                { key: 'overall_relevance', label: 'Overall Relevance', description: 'How well do selected tools match query intent?' },
                { key: 'completeness_score', label: 'Completeness', description: 'Were all necessary tools found?' },
                { key: 'precision_score', label: 'Precision', description: 'Were irrelevant tools excluded?' },
                { key: 'contextual_appropriateness', label: 'Contextual Fit', description: 'Do tools fit the specific use case?' },
                { key: 'improvement_rating', label: 'Reranking Improvement', description: 'How much better are reranked vs original results?' },
              ].map(({ key, label, description }) => (
                <Box key={key} sx={{ mb: 3 }}>
                  <Typography component="legend" gutterBottom>
                    {label}
                  </Typography>
                  <Rating
                    name={key}
                    value={evaluation[key as keyof Context7Evaluation] as number || 0}
                    onChange={(_, newValue) => setEvaluation(prev => ({ ...prev, [key]: newValue || 0 }))}
                    size="large"
                    icon={<StarIcon fontSize="inherit" />}
                  />
                  <Typography variant="caption" display="block" color="text.secondary">
                    {description}
                  </Typography>
                </Box>
              ))}
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>Results Summary</Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell>Original</TableCell>
                      <TableCell>Reranked</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Total Results</TableCell>
                      <TableCell>{searchResults.original_results.length}</TableCell>
                      <TableCell>{searchResults.reranked_results.length}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Unique Tools</TableCell>
                      <TableCell>{new Set(searchResults.original_results.map(r => r.tool.id)).size}</TableCell>
                      <TableCell>{new Set(searchResults.reranked_results.map(r => r.tool.id)).size}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>

              {showValidationErrors && !isValid && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Please complete all core metrics and rate at least one tool before submitting.
                  </Typography>
                </Alert>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Tool Comparison Tab */}
        <TabPanel value={currentTab} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CompareIcon />
                Original Results
              </Typography>
              <List>
                {searchResults.original_results.map((result, index) => (
                  <ListItem key={result.tool.id} divider>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Chip label={`#${index + 1}`} size="small" />
                          <Typography variant="body1">{result.tool.name}</Typography>
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" display="block">
                            Score: {result.score.toFixed(3)}
                          </Typography>
                          <Typography variant="body2" sx={{ mt: 0.5 }}>
                            {result.tool.description}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CompareIcon sx={{ transform: 'scaleX(-1)' }} />
                Reranked Results
              </Typography>
              <List>
                {searchResults.reranked_results.map((result, index) => (
                  <ListItem key={result.tool.id} divider>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Chip label={`#${index + 1}`} size="small" color="primary" />
                          <Typography variant="body1">{result.tool.name}</Typography>
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" display="block">
                            Score: {result.score.toFixed(3)}
                          </Typography>
                          <Typography variant="body2" sx={{ mt: 0.5 }}>
                            {result.tool.description}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Individual Tools Tab */}
        <TabPanel value={currentTab} index={2}>
          <Typography variant="h6" gutterBottom>Individual Tool Evaluations</Typography>
          {evaluation.tool_evaluations?.map((toolEval) => (
            <Accordion key={toolEval.tool_id}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                  <Typography variant="subtitle1">{toolEval.tool_name}</Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {toolEval.found_in_original && <Chip label="Original" size="small" />}
                    {toolEval.found_in_reranked && <Chip label="Reranked" size="small" color="primary" />}
                  </Box>
                  {toolEval.relevance_score > 0 && (
                    <Chip
                      icon={<CheckCircleIcon />}
                      label={`R: ${toolEval.relevance_score}`}
                      size="small"
                      color="success"
                    />
                  )}
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Typography component="legend" gutterBottom>
                      Relevance Score
                    </Typography>
                    <Rating
                      value={toolEval.relevance_score}
                      onChange={(_, newValue) => updateToolEvaluation(toolEval.tool_id, 'relevance_score', newValue || 0)}
                      size="large"
                      icon={<StarIcon fontSize="inherit" />}
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Typography component="legend" gutterBottom>
                      Confidence Score
                    </Typography>
                    <Rating
                      value={toolEval.confidence_score}
                      onChange={(_, newValue) => updateToolEvaluation(toolEval.tool_id, 'confidence_score', newValue || 0)}
                      size="large"
                      icon={<StarIcon fontSize="inherit" />}
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" gutterBottom>Position Comparison</Typography>
                    <Box>
                      {toolEval.original_rank && (
                        <Typography variant="caption" display="block">
                          Original: #{toolEval.original_rank}
                        </Typography>
                      )}
                      {toolEval.reranked_rank && (
                        <Typography variant="caption" display="block">
                          Reranked: #{toolEval.reranked_rank}
                        </Typography>
                      )}
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      multiline
                      rows={2}
                      label="Notes"
                      value={toolEval.notes || ''}
                      onChange={(e) => updateToolEvaluation(toolEval.tool_id, 'notes', e.target.value)}
                      placeholder="Any specific observations about this tool's relevance..."
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          ))}
        </TabPanel>

        {/* Qualitative Feedback Tab */}
        <TabPanel value={currentTab} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Strengths"
                value={evaluation.strengths}
                onChange={(e) => setEvaluation(prev => ({ ...prev, strengths: e.target.value }))}
                placeholder="What worked well in the search results?"
                sx={{ mb: 3 }}
              />

              <TextField
                fullWidth
                multiline
                rows={4}
                label="Weaknesses"
                value={evaluation.weaknesses}
                onChange={(e) => setEvaluation(prev => ({ ...prev, weaknesses: e.target.value }))}
                placeholder="What could be improved?"
                sx={{ mb: 3 }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Suggestions for Improvement"
                value={evaluation.suggestions}
                onChange={(e) => setEvaluation(prev => ({ ...prev, suggestions: e.target.value }))}
                placeholder="Specific recommendations for better results..."
                sx={{ mb: 3 }}
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={evaluation.flagged_for_review || false}
                    onChange={(e) => setEvaluation(prev => ({ ...prev, flagged_for_review: e.target.checked }))}
                  />
                }
                label="Flag for Review"
              />

              {evaluation.flagged_for_review && (
                <Alert severity="info" sx={{ mt: 1 }}>
                  This evaluation will be marked for additional review.
                </Alert>
              )}
            </Grid>
          </Grid>
        </TabPanel>
      </Card>

      {/* Submit Button */}
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
        <Button
          variant="outlined"
          onClick={() => setCurrentTab(0)}
          disabled={isSubmitting}
        >
          Reset
        </Button>

        <Button
          variant="contained"
          onClick={handleSubmitEvaluation}
          disabled={!isValid || isSubmitting}
          startIcon={isSubmitting ? <TimerIcon /> : <SaveIcon />}
          size="large"
        >
          {isSubmitting ? 'Submitting...' : 'Submit Context7 Evaluation'}
        </Button>
      </Box>

      {/* Export Dialog */}
      <Dialog open={showExportDialog} onClose={() => setShowExportDialog(false)}>
        <DialogTitle>Export Evaluation Data</DialogTitle>
        <DialogContent>
          <Typography variant="body2" gutterBottom>
            Export current evaluation data as JSON for analysis or backup.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowExportDialog(false)}>Cancel</Button>
          <Button onClick={exportEvaluationData} startIcon={<DownloadIcon />}>
            Export JSON
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Context7EvaluationInterface;