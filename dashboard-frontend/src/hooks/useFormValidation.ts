import { useState, useCallback, useRef } from 'react';

interface ValidationRequest {
  field_id: string;
  config_type: string;
  field: string;
  value: any;
  context?: Record<string, any>;
}

interface ValidationResult {
  field_id: string;
  valid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

interface BulkValidationResult {
  results: ValidationResult[];
  overall_valid: boolean;
}

interface FormValidationState {
  validating: boolean;
  results: Record<string, ValidationResult>;
  overallValid: boolean;
  lastValidated: Date | null;
}

export const useFormValidation = () => {
  const [state, setState] = useState<FormValidationState>({
    validating: false,
    results: {},
    overallValid: true,
    lastValidated: null
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const validateFields = useCallback(async (validations: ValidationRequest[]): Promise<BulkValidationResult> => {
    // Cancel any existing validation
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController();

    setState(prev => ({ ...prev, validating: true }));

    try {
      const response = await fetch('/api/v1/config/validate/bulk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ validations }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`Validation failed: ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Bulk validation failed');
      }

      const bulkResult: BulkValidationResult = result.data;

      // Update state with results
      const newResults: Record<string, ValidationResult> = {};
      bulkResult.results.forEach(result => {
        newResults[result.field_id] = result;
      });

      setState({
        validating: false,
        results: newResults,
        overallValid: bulkResult.overall_valid,
        lastValidated: new Date()
      });

      return bulkResult;

    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        // Request was aborted, don't update state
        return { results: [], overall_valid: false };
      }

      setState(prev => ({
        ...prev,
        validating: false,
        overallValid: false
      }));

      throw error;
    }
  }, []);

  const validateSingleField = useCallback(async (
    fieldId: string,
    configType: string,
    field: string,
    value: any,
    context?: Record<string, any>
  ): Promise<ValidationResult> => {
    const validations: ValidationRequest[] = [{
      field_id: fieldId,
      config_type: configType,
      field: field,
      value: value,
      context: context
    }];

    const result = await validateFields(validations);
    return result.results[0];
  }, [validateFields]);

  const getFieldValidation = useCallback((fieldId: string): ValidationResult | null => {
    return state.results[fieldId] || null;
  }, [state.results]);

  const isFieldValid = useCallback((fieldId: string): boolean => {
    const result = getFieldValidation(fieldId);
    return result ? result.valid : true;
  }, [getFieldValidation]);

  const getFieldErrors = useCallback((fieldId: string): string[] => {
    const result = getFieldValidation(fieldId);
    return result ? result.errors : [];
  }, [getFieldValidation]);

  const getFieldWarnings = useCallback((fieldId: string): string[] => {
    const result = getFieldValidation(fieldId);
    return result ? result.warnings : [];
  }, [getFieldValidation]);

  const getFieldSuggestions = useCallback((fieldId: string): string[] => {
    const result = getFieldValidation(fieldId);
    return result ? result.suggestions : [];
  }, [getFieldValidation]);

  const clearValidation = useCallback((fieldId?: string) => {
    if (fieldId) {
      setState(prev => {
        const newResults = { ...prev.results };
        delete newResults[fieldId];
        return {
          ...prev,
          results: newResults,
          overallValid: Object.values(newResults).every(r => r.valid)
        };
      });
    } else {
      setState({
        validating: false,
        results: {},
        overallValid: true,
        lastValidated: null
      });
    }
  }, []);

  const clearAllValidation = useCallback(() => {
    clearValidation();
  }, [clearValidation]);

  return {
    // State
    validating: state.validating,
    results: state.results,
    overallValid: state.overallValid,
    lastValidated: state.lastValidated,

    // Actions
    validateFields,
    validateSingleField,

    // Field-specific getters
    getFieldValidation,
    isFieldValid,
    getFieldErrors,
    getFieldWarnings,
    getFieldSuggestions,

    // Clear functions
    clearValidation,
    clearAllValidation
  };
};

// Helper hook for individual field validation with debouncing
export const useFieldValidation = (
  fieldId: string,
  configType: string,
  field: string,
  debounceMs: number = 500
) => {
  const { validateSingleField, getFieldValidation, isFieldValid, clearValidation } = useFormValidation();
  const [isValidating, setIsValidating] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const debouncedValidate = useCallback((value: any, context?: Record<string, any>) => {
    // Clear existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Clear validation if value is empty
    if (value === '' || value === null || value === undefined) {
      clearValidation(fieldId);
      setIsValidating(false);
      return;
    }

    setIsValidating(true);

    // Set new timeout
    timeoutRef.current = setTimeout(async () => {
      try {
        await validateSingleField(fieldId, configType, field, value, context);
      } catch (error) {
        console.error('Field validation error:', error);
      } finally {
        setIsValidating(false);
      }
    }, debounceMs);
  }, [fieldId, configType, field, debounceMs, validateSingleField, clearValidation]);

  const immediateValidate = useCallback(async (value: any, context?: Record<string, any>) => {
    setIsValidating(true);
    try {
      return await validateSingleField(fieldId, configType, field, value, context);
    } catch (error) {
      console.error('Field validation error:', error);
      return null;
    } finally {
      setIsValidating(false);
    }
  }, [fieldId, configType, field, validateSingleField]);

  return {
    isValidating,
    validationResult: getFieldValidation(fieldId),
    isValid: isFieldValid(fieldId),
    debouncedValidate,
    immediateValidate,
    clearValidation: () => clearValidation(fieldId)
  };
};