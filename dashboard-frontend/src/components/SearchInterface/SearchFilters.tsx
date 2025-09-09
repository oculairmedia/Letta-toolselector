import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Autocomplete,
  Chip,
  Button,
  Box,
  IconButton,
  Divider,
} from '@mui/material';
import {
  Close as CloseIcon,
  Clear as ClearIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';

import { useTools } from '../../hooks/useApi';

interface SearchFiltersProps {
  filters: {
    category: string;
    source: string;
    tags: string[];
    limit: number;
  };
  onChange: (filters: SearchFiltersProps['filters']) => void;
  onClose: () => void;
}

const SearchFilters: React.FC<SearchFiltersProps> = ({
  filters,
  onChange,
  onClose,
}) => {
  const { data: tools } = useTools();

  // Extract unique values for autocomplete options
  const categories = React.useMemo(() => {
    if (!tools) return [];
    const uniqueCategories = [...new Set(tools
      .map(tool => tool.category)
      .filter(Boolean)
    )].sort();
    return uniqueCategories;
  }, [tools]);

  const sources = React.useMemo(() => {
    if (!tools) return [];
    const uniqueSources = [...new Set(tools.map(tool => tool.source))].sort();
    return uniqueSources;
  }, [tools]);

  const allTags = React.useMemo(() => {
    if (!tools) return [];
    const uniqueTags = [...new Set(tools
      .flatMap(tool => tool.tags || [])
    )].sort();
    return uniqueTags;
  }, [tools]);

  const handleFilterChange = (field: string, value: any) => {
    onChange({
      ...filters,
      [field]: value,
    });
  };

  const handleClearFilters = () => {
    onChange({
      category: '',
      source: '',
      tags: [],
      limit: 10,
    });
  };

  const hasActiveFilters = filters.category || filters.source || filters.tags.length > 0;

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FilterIcon />
            Search Filters
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            {hasActiveFilters && (
              <Button
                size="small"
                onClick={handleClearFilters}
                startIcon={<ClearIcon />}
              >
                Clear All
              </Button>
            )}
            <IconButton onClick={onClose}>
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>

        <Divider sx={{ mb: 2 }} />

        <Grid container spacing={3}>
          {/* Category Filter */}
          <Grid item xs={12} md={4}>
            <Autocomplete
              options={categories}
              value={filters.category}
              onChange={(_, value) => handleFilterChange('category', value || '')}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Category"
                  placeholder="Select category"
                  variant="outlined"
                  size="small"
                />
              )}
              clearOnEscape
            />
          </Grid>

          {/* Source Filter */}
          <Grid item xs={12} md={4}>
            <Autocomplete
              options={sources}
              value={filters.source}
              onChange={(_, value) => handleFilterChange('source', value || '')}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Source"
                  placeholder="Select source"
                  variant="outlined"
                  size="small"
                />
              )}
              clearOnEscape
            />
          </Grid>

          {/* Result Limit */}
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Result Limit"
              type="number"
              value={filters.limit}
              onChange={(e) => handleFilterChange('limit', Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))}
              inputProps={{
                min: 1,
                max: 100,
              }}
              size="small"
              variant="outlined"
            />
          </Grid>

          {/* Tags Filter */}
          <Grid item xs={12}>
            <Autocomplete
              multiple
              options={allTags}
              value={filters.tags}
              onChange={(_, value) => handleFilterChange('tags', value)}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    variant="filled"
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                    key={option}
                  />
                ))
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Tags"
                  placeholder="Select tags"
                  variant="outlined"
                  size="small"
                />
              )}
              clearOnEscape
            />
          </Grid>
        </Grid>

        {/* Active Filters Summary */}
        {hasActiveFilters && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              Active filters:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {filters.category && (
                <Chip
                  label={`Category: ${filters.category}`}
                  size="small"
                  onDelete={() => handleFilterChange('category', '')}
                  color="primary"
                  variant="outlined"
                />
              )}
              {filters.source && (
                <Chip
                  label={`Source: ${filters.source}`}
                  size="small"
                  onDelete={() => handleFilterChange('source', '')}
                  color="primary"
                  variant="outlined"
                />
              )}
              {filters.tags.map((tag) => (
                <Chip
                  key={tag}
                  label={`Tag: ${tag}`}
                  size="small"
                  onDelete={() => handleFilterChange('tags', filters.tags.filter(t => t !== tag))}
                  color="secondary"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SearchFilters;