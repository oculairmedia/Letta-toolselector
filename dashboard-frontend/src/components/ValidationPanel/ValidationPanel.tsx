import React, { useState } from 'react';
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
  IconButton,
  Tooltip,
  LinearProgress,
  Collapse,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
} from '@mui/icons-material';
import { useFormValidation } from '../../hooks/useFormValidation';

interface ValidationPanelProps {
  title?: string;
  visible?: boolean;
  onToggleVisibility?: () => void;
  showSummaryOnly?: boolean;
  autoValidate?: boolean;
  configFields?: Array<{
    fieldId: string;
    configType: string;
    field: string;
    label: string;
    value: any;
    context?: Record<string, any>;
  }>;
}

const ValidationPanel: React.FC<ValidationPanelProps> = ({
  title = "Configuration Validation",
  visible = true,
  onToggleVisibility,
  showSummaryOnly = false,
  autoValidate = false,
  configFields = []
}) => {
  const {
    validating,
    results,
    overallValid,
    lastValidated,
    validateFields,
    getFieldValidation,
    getFieldErrors,
    getFieldWarnings,
    getFieldSuggestions
  } = useFormValidation();

  const [detailsExpanded, setDetailsExpanded] = useState(false);

  // Calculate summary statistics
  const totalFields = Object.keys(results).length;
  const validFields = Object.values(results).filter(r => r.valid).length;
  const fieldsWithErrors = Object.values(results).filter(r => r.errors.length > 0).length;
  const fieldsWithWarnings = Object.values(results).filter(r => r.warnings.length > 0).length;
  const fieldsWithSuggestions = Object.values(results).filter(r => r.suggestions.length > 0).length;

  const handleValidateAll = async () => {
    if (configFields.length === 0) return;

    const validations = configFields.map(field => ({
      field_id: field.fieldId,
      config_type: field.configType,
      field: field.field,
      value: field.value,
      context: field.context
    }));

    try {
      await validateFields(validations);
    } catch (error) {
      console.error('Bulk validation failed:', error);
    }
  };

  // Auto-validate when fields change
  React.useEffect(() => {
    if (autoValidate && configFields.length > 0) {
      handleValidateAll();
    }
  }, [autoValidate, configFields]);

  if (!visible) {
    return null;
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {overallValid ? (
              <CheckIcon color="success" />
            ) : fieldsWithErrors > 0 ? (
              <ErrorIcon color="error" />
            ) : fieldsWithWarnings > 0 ? (
              <WarningIcon color="warning" />
            ) : (
              <InfoIcon color="info" />
            )}
            {title}
          </Typography>

          <Box display="flex" alignItems="center" gap={1}>
            {lastValidated && (
              <Typography variant="caption" color="text.secondary">
                Last validated: {lastValidated.toLocaleTimeString()}
              </Typography>
            )}

            {configFields.length > 0 && (
              <Button
                size="small"
                onClick={handleValidateAll}
                disabled={validating}
                startIcon={validating ? <LinearProgress /> : <RefreshIcon />}
              >
                {validating ? 'Validating...' : 'Validate All'}
              </Button>
            )}

            {onToggleVisibility && (
              <IconButton size="small" onClick={onToggleVisibility}>
                <VisibilityOffIcon />
              </IconButton>
            )}
          </Box>
        </Box>

        {validating && <LinearProgress sx={{ mb: 2 }} />}

        {/* Summary Statistics */}
        <Grid container spacing={2} mb={2}>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 1 }}>
                <Typography variant="h6" color={overallValid ? 'success.main' : 'text.primary'}>
                  {validFields}/{totalFields}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Valid Fields
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 1 }}>
                <Typography variant="h6" color={fieldsWithErrors > 0 ? 'error.main' : 'text.primary'}>
                  {fieldsWithErrors}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Errors
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 1 }}>
                <Typography variant="h6" color={fieldsWithWarnings > 0 ? 'warning.main' : 'text.primary'}>
                  {fieldsWithWarnings}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Warnings
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 1 }}>
                <Typography variant="h6" color={fieldsWithSuggestions > 0 ? 'info.main' : 'text.primary'}>
                  {fieldsWithSuggestions}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Suggestions
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Overall Status */}
        {totalFields > 0 && (
          <Alert
            severity={overallValid ? 'success' : fieldsWithErrors > 0 ? 'error' : 'warning'}
            sx={{ mb: 2 }}
          >
            {overallValid ? (
              'All configuration fields are valid'
            ) : fieldsWithErrors > 0 ? (
              `${fieldsWithErrors} field${fieldsWithErrors > 1 ? 's' : ''} have errors that must be fixed`
            ) : (
              `${fieldsWithWarnings} field${fieldsWithWarnings > 1 ? 's' : ''} have warnings`
            )}
          </Alert>
        )}

        {/* Detailed Results (if not summary only) */}
        {!showSummaryOnly && totalFields > 0 && (
          <Accordion expanded={detailsExpanded} onChange={() => setDetailsExpanded(!detailsExpanded)}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">
                Validation Details ({totalFields} field{totalFields > 1 ? 's' : ''})
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {Object.entries(results).map(([fieldId, result]) => {
                  const fieldConfig = configFields.find(f => f.fieldId === fieldId);
                  const fieldLabel = fieldConfig?.label || fieldId;

                  return (
                    <ListItem key={fieldId} divider>
                      <ListItemIcon>
                        {result.valid ? (
                          <CheckIcon color="success" />
                        ) : result.errors.length > 0 ? (
                          <ErrorIcon color="error" />
                        ) : (
                          <WarningIcon color="warning" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography variant="body1">{fieldLabel}</Typography>
                            <Chip
                              label={result.valid ? 'Valid' : 'Invalid'}
                              size="small"
                              color={result.valid ? 'success' : 'error'}
                            />
                          </Box>
                        }
                        secondary={
                          <Box>
                            {result.errors.map((error, idx) => (
                              <Alert key={idx} severity="error" sx={{ mt: 0.5 }}>
                                {error}
                              </Alert>
                            ))}
                            {result.warnings.map((warning, idx) => (
                              <Alert key={idx} severity="warning" sx={{ mt: 0.5 }}>
                                {warning}
                              </Alert>
                            ))}
                            {result.suggestions.map((suggestion, idx) => (
                              <Alert key={idx} severity="info" sx={{ mt: 0.5 }}>
                                {suggestion}
                              </Alert>
                            ))}
                          </Box>
                        }
                      />
                    </ListItem>
                  );
                })}
              </List>
            </AccordionDetails>
          </Accordion>
        )}

        {/* No validation results message */}
        {totalFields === 0 && !validating && (
          <Alert severity="info">
            No fields have been validated yet. Configure fields to enable validation.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default ValidationPanel;