"""
LDTS-46: Custom Metrics and Reporting System
Extensible framework for creating custom metrics, visualizations, and reports
"""

import asyncio
import json
import uuid
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import logging
import numpy as np
from collections import defaultdict
import base64
import io

from metrics_calculation_engine import MetricsCalculationEngine, MetricResult, MetricType
from evaluation_history_tracking import EvaluationHistoryTracker, EvaluationRun
from performance_monitoring import PerformanceMonitor

class ChartType(Enum):
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    PIE = "pie"
    BOX_PLOT = "box_plot"
    RADAR = "radar"

class AggregationFunction(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "std"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"

class ReportFormat(Enum):
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"

@dataclass
class CustomMetricDefinition:
    """Definition of a custom metric"""
    id: str
    name: str
    description: str
    
    # Calculation logic
    calculation_function: str  # Python code or function name
    input_metrics: List[str]  # Required input metric types
    output_type: str = "float"  # float, int, bool, string
    
    # Display settings
    display_name: str = ""
    unit: str = ""
    format_string: str = "{:.4f}"
    
    # Aggregation settings
    aggregation_function: AggregationFunction = AggregationFunction.MEAN
    time_window_hours: int = 24
    
    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VisualizationConfig:
    """Configuration for data visualization"""
    id: str
    name: str
    description: str
    
    # Chart configuration
    chart_type: ChartType
    data_source: str  # metric_id or query
    
    # Axes and labels
    x_axis_label: str = ""
    y_axis_label: str = ""
    title: str = ""
    
    # Styling
    color_scheme: str = "default"  # default, viridis, plasma, etc.
    width: int = 800
    height: int = 600
    
    # Data processing
    aggregation: Optional[AggregationFunction] = None
    time_grouping: str = "hour"  # minute, hour, day, week, month
    limit: int = 100
    
    # Filters
    filters: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReportTemplate:
    """Template for generating reports"""
    id: str
    name: str
    description: str
    
    # Report structure
    sections: List[Dict[str, Any]] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)  # visualization IDs
    custom_metrics: List[str] = field(default_factory=list)  # custom metric IDs
    
    # Report settings
    format: ReportFormat = ReportFormat.HTML
    auto_refresh_hours: Optional[int] = None
    
    # Styling
    template_css: str = ""
    header_html: str = ""
    footer_html: str = ""
    
    # Distribution
    recipients: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedReport:
    """Generated report instance"""
    id: str
    template_id: str
    name: str
    
    # Content
    content: str  # HTML, JSON, etc.
    format: ReportFormat
    
    # Generation details
    generated_at: datetime
    data_period_start: datetime
    data_period_end: datetime
    
    # File info
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    
    # Status
    generation_time_seconds: float = 0.0
    error_message: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class CustomMetricsEngine:
    """Engine for calculating custom metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.registered_functions: Dict[str, Callable] = {}
        self.metrics_engine = MetricsCalculationEngine()
        
        # Register built-in functions
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """Register built-in custom metric functions"""
        
        # Success rate = (successful_queries / total_queries) * 100
        def success_rate(metrics_data: Dict[str, List[float]]) -> float:
            successful = sum(metrics_data.get('successful_queries', [0]))
            total = sum(metrics_data.get('total_queries', [1]))
            return (successful / total) * 100 if total > 0 else 0.0
        
        # Average improvement = mean of relative improvements
        def avg_improvement(metrics_data: Dict[str, List[float]]) -> float:
            improvements = metrics_data.get('relative_improvements', [])
            return np.mean(improvements) * 100 if improvements else 0.0
        
        # Query complexity score based on length and type
        def query_complexity_score(metrics_data: Dict[str, List[float]]) -> float:
            lengths = metrics_data.get('query_lengths', [])
            if not lengths:
                return 0.0
            
            # Simple complexity based on query length distribution
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            return (avg_length + std_length) / 20.0  # Normalized score
        
        # Performance efficiency = throughput / resource_usage
        def performance_efficiency(metrics_data: Dict[str, List[float]]) -> float:
            throughput = sum(metrics_data.get('throughput', [0]))
            resource_usage = np.mean(metrics_data.get('cpu_usage', [1]))
            return throughput / resource_usage if resource_usage > 0 else 0.0
        
        # Register functions
        self.registered_functions['success_rate'] = success_rate
        self.registered_functions['avg_improvement'] = avg_improvement
        self.registered_functions['query_complexity_score'] = query_complexity_score
        self.registered_functions['performance_efficiency'] = performance_efficiency
    
    def register_function(self, name: str, function: Callable):
        """Register a custom metric function"""
        self.registered_functions[name] = function
        self.logger.info(f"Registered custom metric function: {name}")
    
    async def calculate_custom_metric(
        self,
        metric_def: CustomMetricDefinition,
        data_period_start: datetime,
        data_period_end: datetime
    ) -> Optional[float]:
        """Calculate a custom metric value"""
        
        try:
            # Get input data based on required metrics
            metrics_data = await self._gather_input_data(
                metric_def.input_metrics,
                data_period_start,
                data_period_end
            )
            
            # Execute calculation function
            if metric_def.calculation_function in self.registered_functions:
                result = self.registered_functions[metric_def.calculation_function](metrics_data)
            else:
                # Try to execute as Python code (be very careful with this in production!)
                result = self._execute_calculation_code(metric_def.calculation_function, metrics_data)
            
            # Validate result
            if metric_def.min_value is not None and result < metric_def.min_value:
                self.logger.warning(f"Custom metric {metric_def.id} result {result} below minimum {metric_def.min_value}")
                return metric_def.min_value
            
            if metric_def.max_value is not None and result > metric_def.max_value:
                self.logger.warning(f"Custom metric {metric_def.id} result {result} above maximum {metric_def.max_value}")
                return metric_def.max_value
            
            return float(result)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate custom metric {metric_def.id}: {e}")
            return None
    
    async def _gather_input_data(
        self,
        input_metrics: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, List[float]]:
        """Gather input data for custom metric calculation"""
        
        metrics_data = {}
        
        # This is a simplified implementation - in practice, you'd query
        # the actual metric databases/stores based on the input_metrics
        
        # Mock data for demonstration
        for metric_name in input_metrics:
            if metric_name == 'successful_queries':
                metrics_data[metric_name] = [95, 98, 97, 96, 99]  # Mock success counts
            elif metric_name == 'total_queries':
                metrics_data[metric_name] = [100, 100, 100, 100, 100]  # Mock total counts
            elif metric_name == 'relative_improvements':
                metrics_data[metric_name] = [0.05, 0.03, 0.08, 0.02, 0.06]  # Mock improvements
            elif metric_name == 'query_lengths':
                metrics_data[metric_name] = [5, 8, 12, 3, 15, 7, 9, 11, 6, 4]  # Mock query lengths
            elif metric_name == 'throughput':
                metrics_data[metric_name] = [1000, 1200, 950, 1100, 1050]  # Mock throughput
            elif metric_name == 'cpu_usage':
                metrics_data[metric_name] = [45.2, 52.1, 38.7, 49.3, 41.8]  # Mock CPU usage
            else:
                metrics_data[metric_name] = [0.0]  # Default empty data
        
        return metrics_data
    
    def _execute_calculation_code(self, code: str, metrics_data: Dict[str, List[float]]) -> float:
        """Execute calculation code safely (simplified implementation)"""
        
        # WARNING: In production, this should use a sandboxed environment
        # and have strict security controls
        
        allowed_functions = {
            'sum': sum,
            'len': len,
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'math': math,
            'np': np
        }
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': {},
            'metrics_data': metrics_data,
            **allowed_functions
        }
        
        try:
            result = eval(code, safe_globals)
            return float(result)
        except Exception as e:
            raise ValueError(f"Calculation code execution failed: {e}")

class ReportGenerator:
    """Generator for custom reports"""
    
    def __init__(self, storage_path: str = "reports_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.templates_path = self.storage_path / "templates"
        self.generated_path = self.storage_path / "generated"
        self.assets_path = self.storage_path / "assets"
        
        for path in [self.templates_path, self.generated_path, self.assets_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.custom_metrics = CustomMetricsEngine()
    
    async def generate_chart_data(self, viz_config: VisualizationConfig) -> Dict[str, Any]:
        """Generate data for visualization"""
        
        try:
            # This is a simplified mock implementation
            # In practice, this would query actual data sources
            
            if viz_config.chart_type == ChartType.LINE:
                # Generate time series data
                timestamps = []
                values = []
                
                start_time = datetime.utcnow() - timedelta(hours=24)
                for i in range(24):
                    timestamps.append((start_time + timedelta(hours=i)).isoformat())
                    # Mock data with some trend and noise
                    base_value = 0.8 + 0.1 * math.sin(i * math.pi / 12)
                    noise = np.random.normal(0, 0.05)
                    values.append(max(0, base_value + noise))
                
                return {
                    'chart_type': viz_config.chart_type.value,
                    'data': {
                        'x': timestamps,
                        'y': values,
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'name': viz_config.name
                    },
                    'layout': {
                        'title': viz_config.title or viz_config.name,
                        'xaxis': {'title': viz_config.x_axis_label},
                        'yaxis': {'title': viz_config.y_axis_label},
                        'width': viz_config.width,
                        'height': viz_config.height
                    }
                }
            
            elif viz_config.chart_type == ChartType.BAR:
                # Generate categorical data
                categories = ['Config A', 'Config B', 'Config C', 'Config D']
                values = [0.85, 0.79, 0.91, 0.76]
                
                return {
                    'chart_type': viz_config.chart_type.value,
                    'data': {
                        'x': categories,
                        'y': values,
                        'type': 'bar',
                        'name': viz_config.name
                    },
                    'layout': {
                        'title': viz_config.title or viz_config.name,
                        'xaxis': {'title': viz_config.x_axis_label},
                        'yaxis': {'title': viz_config.y_axis_label},
                        'width': viz_config.width,
                        'height': viz_config.height
                    }
                }
            
            elif viz_config.chart_type == ChartType.HEATMAP:
                # Generate heatmap data
                x_labels = ['Hour 0-6', 'Hour 6-12', 'Hour 12-18', 'Hour 18-24']
                y_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                
                # Generate random correlation matrix
                z_values = np.random.rand(len(y_labels), len(x_labels)).tolist()
                
                return {
                    'chart_type': viz_config.chart_type.value,
                    'data': {
                        'z': z_values,
                        'x': x_labels,
                        'y': y_labels,
                        'type': 'heatmap',
                        'colorscale': 'Viridis'
                    },
                    'layout': {
                        'title': viz_config.title or viz_config.name,
                        'width': viz_config.width,
                        'height': viz_config.height
                    }
                }
            
            else:
                # Default empty chart
                return {
                    'chart_type': viz_config.chart_type.value,
                    'data': {},
                    'layout': {'title': f"Chart type {viz_config.chart_type.value} not implemented"}
                }
                
        except Exception as e:
            self.logger.error(f"Failed to generate chart data for {viz_config.id}: {e}")
            return {
                'chart_type': viz_config.chart_type.value,
                'data': {},
                'layout': {'title': f'Error generating chart: {str(e)}'}
            }
    
    async def generate_html_report(
        self,
        template: ReportTemplate,
        data_period_start: datetime,
        data_period_end: datetime
    ) -> str:
        """Generate HTML report from template"""
        
        try:
            start_time = datetime.utcnow()
            
            # Generate report content
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{template.name}</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .footer {{
            border-top: 1px solid #ddd;
            padding-top: 10px;
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        {template.template_css}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{template.name}</h1>
            <p><strong>Description:</strong> {template.description}</p>
            <p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            <p><strong>Data Period:</strong> {data_period_start.strftime('%Y-%m-%d %H:%M')} to {data_period_end.strftime('%Y-%m-%d %H:%M')}</p>
            {template.header_html}
        </div>
"""
            
            # Add custom metrics section
            if template.custom_metrics:
                html_content += '<div class="section"><h2>Custom Metrics</h2>'
                
                for metric_id in template.custom_metrics:
                    # Load custom metric definition (mock implementation)
                    metric_name = f"Custom Metric {metric_id[:8]}"
                    metric_value = 0.8542  # Mock value
                    
                    html_content += f'''
                    <div class="metric-card">
                        <h3>{metric_name}</h3>
                        <div style="font-size: 2em; font-weight: bold; color: #007bff;">
                            {metric_value:.4f}
                        </div>
                        <p>Description of the custom metric and what it represents.</p>
                    </div>
                    '''
                
                html_content += '</div>'
            
            # Add visualizations section
            if template.visualizations:
                html_content += '<div class="section"><h2>Visualizations</h2>'
                
                chart_counter = 0
                for viz_id in template.visualizations:
                    # Create mock visualization config
                    viz_config = VisualizationConfig(
                        id=viz_id,
                        name=f"Chart {viz_id[:8]}",
                        description="Sample visualization",
                        chart_type=ChartType.LINE if chart_counter % 2 == 0 else ChartType.BAR,
                        data_source="mock_data",
                        title=f"Performance Trend {chart_counter + 1}",
                        x_axis_label="Time",
                        y_axis_label="Value"
                    )
                    
                    # Generate chart data
                    chart_data = await self.generate_chart_data(viz_config)
                    
                    # Add chart to HTML
                    html_content += f'''
                    <div class="chart-container">
                        <div id="chart-{chart_counter}" style="width: 100%; height: 400px;"></div>
                        <script>
                            var data = [{json.dumps(chart_data['data'])}];
                            var layout = {json.dumps(chart_data['layout'])};
                            Plotly.newPlot('chart-{chart_counter}', data, layout);
                        </script>
                    </div>
                    '''
                    
                    chart_counter += 1
                
                html_content += '</div>'
            
            # Add sections
            for section in template.sections:
                section_title = section.get('title', 'Untitled Section')
                section_content = section.get('content', 'No content provided.')
                
                html_content += f'''
                <div class="section">
                    <h2>{section_title}</h2>
                    <p>{section_content}</p>
                </div>
                '''
            
            # Add footer
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            html_content += f'''
        <div class="footer">
            <p>Report generated by LDTS Dashboard in {generation_time:.2f} seconds</p>
            <p>LDTS Reranker Testing Dashboard - Custom Metrics & Reporting System</p>
            {template.footer_html}
        </div>
    </div>
</body>
</html>
            '''
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
    
    async def generate_report(
        self,
        template: ReportTemplate,
        data_period_start: datetime,
        data_period_end: datetime,
        save_to_disk: bool = True
    ) -> GeneratedReport:
        """Generate report from template"""
        
        try:
            start_time = datetime.utcnow()
            
            # Generate content based on format
            if template.format == ReportFormat.HTML:
                content = await self.generate_html_report(template, data_period_start, data_period_end)
            elif template.format == ReportFormat.JSON:
                content = await self.generate_json_report(template, data_period_start, data_period_end)
            else:
                raise ValueError(f"Report format {template.format} not implemented")
            
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create report record
            report_id = str(uuid.uuid4())
            report = GeneratedReport(
                id=report_id,
                template_id=template.id,
                name=f"{template.name} - {data_period_start.strftime('%Y-%m-%d')}",
                content=content,
                format=template.format,
                generated_at=start_time,
                data_period_start=data_period_start,
                data_period_end=data_period_end,
                generation_time_seconds=generation_time
            )
            
            # Save to disk if requested
            if save_to_disk:
                file_extension = "html" if template.format == ReportFormat.HTML else "json"
                file_path = self.generated_path / f"{report_id}.{file_extension}"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                report.file_path = str(file_path)
                report.file_size_bytes = file_path.stat().st_size
                
                # Also save report metadata
                metadata_path = self.generated_path / f"{report_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
            
            self.logger.info(f"Generated report {report_id} in {generation_time:.2f}s")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate report from template {template.id}: {e}")
            
            # Return error report
            return GeneratedReport(
                id=str(uuid.uuid4()),
                template_id=template.id,
                name=f"ERROR: {template.name}",
                content=f"Error generating report: {str(e)}",
                format=template.format,
                generated_at=datetime.utcnow(),
                data_period_start=data_period_start,
                data_period_end=data_period_end,
                error_message=str(e)
            )
    
    async def generate_json_report(
        self,
        template: ReportTemplate,
        data_period_start: datetime,
        data_period_end: datetime
    ) -> str:
        """Generate JSON report from template"""
        
        try:
            report_data = {
                'template_id': template.id,
                'template_name': template.name,
                'description': template.description,
                'generated_at': datetime.utcnow().isoformat(),
                'data_period': {
                    'start': data_period_start.isoformat(),
                    'end': data_period_end.isoformat()
                },
                'custom_metrics': {},
                'visualizations': {},
                'sections': []
            }
            
            # Add custom metrics
            for metric_id in template.custom_metrics:
                # Mock metric calculation
                report_data['custom_metrics'][metric_id] = {
                    'value': 0.8542,  # Mock value
                    'name': f"Custom Metric {metric_id[:8]}",
                    'calculated_at': datetime.utcnow().isoformat()
                }
            
            # Add visualizations
            chart_counter = 0
            for viz_id in template.visualizations:
                viz_config = VisualizationConfig(
                    id=viz_id,
                    name=f"Chart {viz_id[:8]}",
                    description="Sample visualization",
                    chart_type=ChartType.LINE if chart_counter % 2 == 0 else ChartType.BAR,
                    data_source="mock_data"
                )
                
                chart_data = await self.generate_chart_data(viz_config)
                report_data['visualizations'][viz_id] = chart_data
                chart_counter += 1
            
            # Add sections
            for section in template.sections:
                report_data['sections'].append(section)
            
            return json.dumps(report_data, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            return json.dumps({'error': str(e)}, indent=2)

class CustomMetricsReportingSystem:
    """Main system for custom metrics and reporting"""
    
    def __init__(self, storage_path: str = "custom_metrics_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.metrics_path = self.storage_path / "metrics"
        self.visualizations_path = self.storage_path / "visualizations"
        
        for path in [self.metrics_path, self.visualizations_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.custom_metrics = CustomMetricsEngine()
        self.report_generator = ReportGenerator()
        
        # Scheduled tasks
        self.scheduled_reports: Dict[str, asyncio.Task] = {}
    
    # Custom Metric Management
    async def create_custom_metric(self, metric_def: CustomMetricDefinition) -> bool:
        """Create a new custom metric definition"""
        try:
            metric_path = self.metrics_path / f"{metric_def.id}.json"
            with open(metric_path, 'w') as f:
                json.dump(asdict(metric_def), f, indent=2, default=str)
            
            self.logger.info(f"Created custom metric definition {metric_def.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create custom metric {metric_def.id}: {e}")
            return False
    
    async def load_custom_metric(self, metric_id: str) -> Optional[CustomMetricDefinition]:
        """Load custom metric definition"""
        try:
            metric_path = self.metrics_path / f"{metric_id}.json"
            if not metric_path.exists():
                return None
            
            with open(metric_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings and enums
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            data['aggregation_function'] = AggregationFunction(data['aggregation_function'])
            
            return CustomMetricDefinition(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load custom metric {metric_id}: {e}")
            return None
    
    # Visualization Management
    async def create_visualization(self, viz_config: VisualizationConfig) -> bool:
        """Create a new visualization configuration"""
        try:
            viz_path = self.visualizations_path / f"{viz_config.id}.json"
            with open(viz_path, 'w') as f:
                json.dump(asdict(viz_config), f, indent=2, default=str)
            
            self.logger.info(f"Created visualization configuration {viz_config.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization {viz_config.id}: {e}")
            return False
    
    # Report Template Management
    async def create_report_template(self, template: ReportTemplate) -> bool:
        """Create a new report template"""
        return await self.report_generator.create_report_template(template)
    
    # Scheduled Reporting
    async def schedule_report(
        self,
        template_id: str,
        interval_hours: int,
        start_time: Optional[datetime] = None
    ):
        """Schedule automatic report generation"""
        
        if template_id in self.scheduled_reports:
            self.scheduled_reports[template_id].cancel()
        
        async def report_task():
            while True:
                try:
                    # Load template (simplified)
                    template = ReportTemplate(
                        id=template_id,
                        name=f"Scheduled Report {template_id[:8]}",
                        description="Automatically generated report",
                        format=ReportFormat.HTML
                    )
                    
                    # Generate report for last 24 hours
                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(hours=24)
                    
                    report = await self.report_generator.generate_report(
                        template, start_time, end_time
                    )
                    
                    self.logger.info(f"Generated scheduled report {report.id}")
                    
                    # Wait for next interval
                    await asyncio.sleep(interval_hours * 3600)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in scheduled report task {template_id}: {e}")
                    await asyncio.sleep(3600)  # Retry after 1 hour on error
        
        task = asyncio.create_task(report_task())
        self.scheduled_reports[template_id] = task
        
        self.logger.info(f"Scheduled report {template_id} every {interval_hours} hours")

# Global instance
custom_metrics_system = CustomMetricsReportingSystem()

# Example usage
async def create_example_custom_metric():
    """Create an example custom metric"""
    
    metric_def = CustomMetricDefinition(
        id=str(uuid.uuid4()),
        name="Search Success Rate",
        description="Percentage of successful search queries",
        calculation_function="success_rate",
        input_metrics=["successful_queries", "total_queries"],
        display_name="Search Success Rate",
        unit="%",
        format_string="{:.2f}%",
        aggregation_function=AggregationFunction.MEAN,
        time_window_hours=24
    )
    
    success = await custom_metrics_system.create_custom_metric(metric_def)
    if success:
        print(f"Created custom metric: {metric_def.id}")
        
        # Test calculation
        result = await custom_metrics_system.custom_metrics.calculate_custom_metric(
            metric_def,
            datetime.utcnow() - timedelta(hours=24),
            datetime.utcnow()
        )
        print(f"Calculated value: {result}")
        return metric_def.id
    else:
        print("Failed to create custom metric")
        return None

async def create_example_report():
    """Create and generate an example report"""
    
    # Create custom metric
    metric_id = await create_example_custom_metric()
    
    if metric_id:
        # Create report template
        template = ReportTemplate(
            id=str(uuid.uuid4()),
            name="Performance Dashboard",
            description="Comprehensive performance and metrics report",
            sections=[
                {
                    'title': 'Executive Summary',
                    'content': 'This report provides an overview of system performance and key metrics.'
                },
                {
                    'title': 'Key Findings',
                    'content': 'Analysis shows consistent performance improvements across all metrics.'
                }
            ],
            visualizations=[str(uuid.uuid4()), str(uuid.uuid4())],
            custom_metrics=[metric_id],
            format=ReportFormat.HTML
        )
        
        # Generate report
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        report = await custom_metrics_system.report_generator.generate_report(
            template, start_time, end_time
        )
        
        print(f"Generated report: {report.id}")
        if report.file_path:
            print(f"Report saved to: {report.file_path}")
        
        return report.id

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_example_report())