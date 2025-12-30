# LDTS Reranker Testing Dashboard Frontend

A React TypeScript dashboard for testing and evaluating the Letta Tool Selector reranking system.

## Features

### Phase 1 (MVP Foundation) - Implemented ✅

- **LDTS-18**: Main search interface with real-time results
  - Semantic search with query expansion
  - Real-time search as you type (debounced)
  - Configurable search filters (category, source, tags, limit)
  - Recent queries history
  - Search result cards with detailed tool information
  - Reranker toggle with configuration display

- **LDTS-19**: Configuration panel with model selection controls
  - Reranker configuration (provider, model, parameters)
  - Configuration presets manager
  - System settings and health monitoring
  - Connection testing functionality

- **LDTS-20**: Results comparison view (original vs reranked)
  - Side-by-side comparison of search results
  - Ranking change analysis (improvements/declines)
  - Performance metrics and charts
  - Score correlation analysis

- **LDTS-25**: Responsive UI/UX with Material-UI components
  - Dark/light theme support with system preference detection
  - Responsive design for mobile and desktop
  - Material-UI component library integration
  - Navigation drawer with mobile support

### Additional Components

- **Evaluation Interface**: Manual result rating system for building evaluation datasets
- **Analytics Dashboard**: Performance metrics, usage trends, and model comparison charts

## Technical Stack

- **React 19** with **TypeScript** for type safety
- **Material-UI (MUI) v7** for consistent UI components
- **React Query** for efficient data fetching and caching
- **Recharts** for analytics visualizations
- **Axios** for API communication

## Project Structure

```
src/
├── components/
│   ├── SearchInterface/          # LDTS-18: Main search functionality
│   │   ├── SearchInterface.tsx
│   │   ├── SearchResultCard.tsx
│   │   └── SearchFilters.tsx
│   ├── ConfigurationPanel/       # LDTS-19: Settings and configuration
│   │   ├── ConfigurationPanel.tsx
│   │   ├── RerankerConfig.tsx
│   │   ├── PresetsManager.tsx
│   │   └── SystemSettings.tsx
│   ├── ResultsComparison/        # LDTS-20: Compare original vs reranked
│   │   ├── ResultsComparison.tsx
│   │   ├── ComparisonMetrics.tsx
│   │   └── ComparisonChart.tsx
│   ├── EvaluationInterface/      # Manual evaluation tools
│   │   └── EvaluationInterface.tsx
│   └── Analytics/                # Performance analytics
│       └── Analytics.tsx
├── hooks/                        # Custom React hooks
│   └── useApi.ts                # React Query hooks for API calls
├── services/                     # API service layer
│   └── api.ts                   # Axios-based API client
├── types/                        # TypeScript type definitions
│   └── index.ts                 # Shared interfaces and types
├── utils/                        # Utility functions
│   └── index.ts                 # Formatters, validators, helpers
└── App.tsx                      # Main app with routing and theme
```

## API Integration

The dashboard integrates with the LDTS backend API at `/api/v1/` endpoints:

- `POST /tools/search` - Standard tool search
- `POST /tools/search/rerank` - Search with reranking
- `GET/PUT /config/reranker` - Reranker configuration
- `GET/POST/PUT/DELETE /config/presets` - Configuration presets
- `POST /evaluations` - Submit manual evaluations
- `GET /analytics` - Usage analytics and metrics
- `GET /health` - System health check

## Key Features

### Search Interface (LDTS-18)
- Real-time semantic search with debouncing
- Advanced filtering (category, source, tags)
- Search result cards with scores and reasoning
- Recent queries persistence
- Reranker enable/disable toggle

### Configuration Panel (LDTS-19)
- Multi-provider reranker support (OpenAI, Anthropic, Ollama, Cohere)
- Model-specific parameter configuration
- Configuration presets for different scenarios
- Connection testing and validation
- System health monitoring

### Results Comparison (LDTS-20)
- Side-by-side original vs reranked results
- Ranking improvement/decline analysis
- Performance metrics (response time, accuracy gain)
- Interactive charts and visualizations
- Score correlation analysis

### Responsive Design (LDTS-25)
- Mobile-first responsive layout
- Collapsible navigation drawer
- Dark/light theme switching
- Material-UI design system
- Touch-friendly mobile interface

## Development Commands

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## Environment Setup

The dashboard expects the backend API to be available at `/api/v1/`. For development, configure a proxy in `package.json` or use environment variables to set the API base URL.

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Future Enhancements

- Real-time WebSocket updates for live search metrics
- Advanced analytics with custom date ranges
- Export functionality for evaluation data
- Bulk evaluation tools
- A/B testing framework integration
- Custom embedding model support
- Advanced query expansion strategies