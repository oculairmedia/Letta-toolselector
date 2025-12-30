import React, { useState, useEffect, useCallback } from 'react';
import {
  TextField,
  Box,
  Alert,
  Chip,
  CircularProgress,
  IconButton,
  Tooltip,
  TextFieldProps,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { debounce } from 'lodash';

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
  field: string;
  value: any;
  config_type: string;
}

interface ValidatedTextFieldProps extends Omit<TextFieldProps, 'error' | 'helperText'> {
  configType: 'tool_selector' | 'embedding' | 'weaviate' | 'letta_api';
  fieldName: string;
  validationContext?: Record<string, any>;
  onValidationChange?: (result: ValidationResult | null) => void;
  validateOnMount?: boolean;
  debounceMs?: number;
  testConnection?: boolean;
  serviceType?: 'ollama' | 'weaviate' | 'letta_api' | 'openai';
}

const ValidatedTextField: React.FC<ValidatedTextFieldProps> = ({
  configType,
  fieldName,
  value,
  onChange,
  validationContext = {},
  onValidationChange,
  validateOnMount = true,
  debounceMs = 500,
  testConnection = false,
  serviceType,
  ...textFieldProps
}) => {
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [connectionResult, setConnectionResult] = useState<any>(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);

  const validateValue = useCallback(async (valueToValidate: any) => {
    if (valueToValidate === '' || valueToValidate === null || valueToValidate === undefined) {
      setValidationResult(null);
      onValidationChange?.(null);
      return;
    }

    setIsValidating(true);
    try {
      const response = await fetch('/api/v1/config/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config_type: configType,
          field: fieldName,
          value: valueToValidate,
          context: validationContext
        })
      });

      const result = await response.json();
      if (result.success) {
        setValidationResult(result.data);
        onValidationChange?.(result.data);
      } else {
        setValidationResult({
          valid: false,
          errors: [result.error || 'Validation failed'],
          warnings: [],
          suggestions: [],
          field: fieldName,
          value: valueToValidate,
          config_type: configType
        });
      }
    } catch (error) {
      setValidationResult({
        valid: false,
        errors: ['Failed to validate: ' + (error as Error).message],
        warnings: [],
        suggestions: [],
        field: fieldName,
        value: valueToValidate,
        config_type: configType
      });
    } finally {
      setIsValidating(false);
    }
  }, [configType, fieldName, validationContext, onValidationChange]);

  const debouncedValidate = useCallback(
    debounce(validateValue, debounceMs),
    [validateValue, debounceMs]
  );

  const testServiceConnection = useCallback(async () => {
    if (!serviceType || !value) return;

    setIsTestingConnection(true);
    try {
      let config = {};

      // Build config based on service type and field
      if (serviceType === 'ollama') {
        if (fieldName === 'ollama_host') {
          config = { host: value };
        } else if (fieldName === 'ollama_port') {
          config = { port: parseInt(value as string) };
        }
      } else if (serviceType === 'weaviate') {
        if (fieldName === 'url') {
          config = { url: value };
        }
      } else if (serviceType === 'openai') {
        if (fieldName === 'openai_api_key') {
          config = { api_key: value };
        }
      }

      const response = await fetch('/api/v1/config/validate/connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          service_type: serviceType,
          config: config
        })
      });

      const result = await response.json();
      if (result.success) {
        setConnectionResult(result.data);
      }
    } catch (error) {
      setConnectionResult({
        available: false,
        error: 'Connection test failed: ' + (error as Error).message
      });
    } finally {
      setIsTestingConnection(false);
    }
  }, [serviceType, fieldName, value]);

  useEffect(() => {
    if (validateOnMount && value) {
      validateValue(value);
    }
  }, [validateOnMount, validateValue]);

  useEffect(() => {
    if (value !== '') {
      debouncedValidate(value);
    }
  }, [value, debouncedValidate]);

  // Determine field state
  const hasErrors = validationResult && validationResult.errors.length > 0;
  const hasWarnings = validationResult && validationResult.warnings.length > 0;
  const hasInfo = validationResult && validationResult.suggestions.length > 0;

  // Get status icon
  const getStatusIcon = () => {
    if (isValidating) {
      return <CircularProgress size={20} />;
    }

    if (validationResult === null) {
      return null;
    }

    if (hasErrors) {
      return <ErrorIcon color="error" />;
    } else if (hasWarnings) {
      return <WarningIcon color="warning" />;
    } else if (validationResult.valid) {
      return <CheckIcon color="success" />;
    }

    return null;
  };

  // Get helper text with validation messages
  const getHelperContent = () => {
    const messages = [];

    if (hasErrors) {
      messages.push(...validationResult!.errors.map(error => (
        <Alert key={error} severity="error" sx={{ mt: 0.5, mb: 0.5 }}>
          {error}
        </Alert>
      )));
    }

    if (hasWarnings) {
      messages.push(...validationResult!.warnings.map(warning => (
        <Alert key={warning} severity="warning" sx={{ mt: 0.5, mb: 0.5 }}>
          {warning}
        </Alert>
      )));
    }

    if (hasInfo) {
      messages.push(...validationResult!.suggestions.map(suggestion => (
        <Alert key={suggestion} severity="info" sx={{ mt: 0.5, mb: 0.5 }}>
          {suggestion}
        </Alert>
      )));
    }

    // Connection test results
    if (connectionResult) {
      if (connectionResult.available) {
        messages.push(
          <Alert key="connection-success" severity="success" sx={{ mt: 0.5, mb: 0.5 }}>
            Connection test successful
            {connectionResult.model && ` - Model: ${connectionResult.model}`}
          </Alert>
        );
      } else {
        messages.push(
          <Alert key="connection-error" severity="error" sx={{ mt: 0.5, mb: 0.5 }}>
            Connection failed: {connectionResult.error}
          </Alert>
        );
      }
    }

    return messages.length > 0 ? <Box>{messages}</Box> : null;
  };

  return (
    <Box>
      <TextField
        {...textFieldProps}
        value={value}
        onChange={onChange}
        error={hasErrors || false}
        helperText={getHelperContent()}
        InputProps={{
          ...textFieldProps.InputProps,
          endAdornment: (
            <Box display="flex" alignItems="center" gap={0.5}>
              {getStatusIcon()}
              {testConnection && serviceType && (
                <Tooltip title="Test connection">
                  <IconButton
                    size="small"
                    onClick={testServiceConnection}
                    disabled={isTestingConnection || !value}
                  >
                    {isTestingConnection ? (
                      <CircularProgress size={16} />
                    ) : (
                      <RefreshIcon fontSize="small" />
                    )}
                  </IconButton>
                </Tooltip>
              )}
              {textFieldProps.InputProps?.endAdornment}
            </Box>
          )
        }}
      />
    </Box>
  );
};

export default ValidatedTextField;