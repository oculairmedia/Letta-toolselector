import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Alert,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Tune as TuneIcon,
  Storage as StorageIcon,
  Bookmark as BookmarkIcon,
  AppRegistration as RegistryIcon,
} from '@mui/icons-material';

import RerankerConfig from './RerankerConfig';
import PresetsManager from './PresetsManager';
import SystemSettings from './SystemSettings';
import ModelRegistryManager from './ModelRegistryManager';
import ModelIndicator from '../ModelIndicator';
import { useRerankerConfig } from '../../hooks/useApi';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`config-tabpanel-${index}`}
      aria-labelledby={`config-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `config-tab-${index}`,
    'aria-controls': `config-tabpanel-${index}`,
  };
}

const ConfigurationPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const { data: rerankerConfig, isLoading, error } = useRerankerConfig();

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Box>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SettingsIcon />
          Configuration Panel
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Configure reranker models, manage presets, and adjust system settings for optimal tool search performance.
        </Typography>
        
        {/* Current Model Status */}
        <ModelIndicator showDetails={false} />
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Failed to load configuration: {error instanceof Error ? error.message : 'Unknown error'}
        </Alert>
      )}

      {/* Configuration Tabs */}
      <Paper elevation={1}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange} 
            aria-label="configuration tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab 
              label="Reranker Settings" 
              icon={<TuneIcon />} 
              iconPosition="start"
              {...a11yProps(0)} 
            />
            <Tab 
              label="Configuration Presets" 
              icon={<BookmarkIcon />} 
              iconPosition="start"
              {...a11yProps(1)} 
            />
            <Tab 
              label="Model Registry" 
              icon={<RegistryIcon />} 
              iconPosition="start"
              {...a11yProps(2)} 
            />
            <Tab 
              label="System Settings" 
              icon={<StorageIcon />} 
              iconPosition="start"
              {...a11yProps(3)} 
            />
          </Tabs>
        </Box>

        <TabPanel value={activeTab} index={0}>
          <RerankerConfig 
            config={rerankerConfig}
            isLoading={isLoading}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <PresetsManager />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <ModelRegistryManager />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <SystemSettings />
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default ConfigurationPanel;