"""
Integration Layer Component - LDTS-62 through LDTS-65
Unified API and integration layer for all dashboard components
"""

from .api_integration import APIIntegrationLayer, api_integration

__all__ = [
    'APIIntegrationLayer',
    'api_integration'
]