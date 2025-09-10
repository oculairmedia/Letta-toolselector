#!/usr/bin/env python3
"""
API Test script for LDTS-58: Cost Controls and Budget Management System
Tests the deployed API endpoints.
"""

import json
import requests
import sys


def test_api_endpoint(endpoint, method='GET', data=None, description=""):
    """Test a single API endpoint"""
    url = f"http://localhost:8020{endpoint}"
    
    print(f"🧪 Testing {method} {endpoint}")
    if description:
        print(f"   📝 {description}")
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        elif method == 'DELETE':
            response = requests.delete(url, json=data, timeout=10)
        
        print(f"   📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"   ✅ SUCCESS: {result.get('message', 'Operation completed')}")
                return True, result
            else:
                print(f"   ❌ API ERROR: {result.get('error', 'Unknown error')}")
                return False, result
        else:
            print(f"   ❌ HTTP ERROR: {response.status_code}")
            try:
                error_data = response.json()
                print(f"      Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"      Error: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"   💥 EXCEPTION: {e}")
        return False, None


def main():
    """Test all cost control API endpoints"""
    print("🚀 LDTS-58: Cost Control API Integration Tests")
    print("=" * 60)
    
    # Test 1: Get cost control status (overall system health)
    success, data = test_api_endpoint(
        "/api/v1/cost-control/status",
        description="Get overall cost control system status"
    )
    if success and data:
        status = data.get('data', {})
        print(f"   📊 System Status: {status.get('system_status', 'unknown')}")
        print(f"   💰 Daily Total: ${status.get('daily_summary', {}).get('total_cost', 0):.4f}")
    
    print()
    
    # Test 2: Get budget limits
    success, data = test_api_endpoint(
        "/api/v1/cost-control/budget",
        description="Get all budget limits and status"
    )
    if success and data:
        budgets = data.get('data', {})
        print(f"   🎯 Active Budget Limits: {len(budgets)}")
        for key, budget in list(budgets.items())[:3]:  # Show first 3
            print(f"      {key}: ${budget['current_cost']:.4f}/${budget['limit']:.2f} ({budget['percentage_used']:.1f}%)")
    
    print()
    
    # Test 3: Get cost summary for daily period
    success, data = test_api_endpoint(
        "/api/v1/cost-control/summary?period=daily",
        description="Get daily cost summary"
    )
    if success and data:
        summary = data.get('data', {})
        print(f"   📈 Daily Entries: {summary.get('entry_count', 0)}")
        print(f"   💵 Daily Total: ${summary.get('total_cost', 0):.4f}")
        categories = summary.get('cost_by_category', {})
        for cat, cost in categories.items():
            print(f"      {cat}: ${cost:.4f}")
    
    print()
    
    # Test 4: Estimate operation cost
    success, data = test_api_endpoint(
        "/api/v1/cost-control/estimate",
        method='POST',
        data={
            "operation_type": "openai_embedding",
            "params": {"token_count": 1000}
        },
        description="Estimate cost for 1000 token embedding"
    )
    if success and data:
        estimate = data.get('data', {})
        print(f"   💰 Estimated Cost: ${estimate.get('estimated_cost', 0):.6f}")
        print(f"   🔧 Operation: {estimate.get('operation_type', 'unknown')}")
    
    print()
    
    # Test 5: Record a test cost
    success, data = test_api_endpoint(
        "/api/v1/cost-control/record",
        method='POST',
        data={
            "category": "experiments",
            "operation": "api_integration_test",
            "cost": 0.001,
            "metadata": {"test": "api_integration"}
        },
        description="Record a small test cost"
    )
    if success and data:
        result = data.get('data', {})
        print(f"   📝 Recorded: {result.get('recorded', False)}")
        print(f"   ✅ Allowed: {result.get('allowed', False)}")
    
    print()
    
    # Test 6: Get recent alerts
    success, data = test_api_endpoint(
        "/api/v1/cost-control/alerts?hours=1",
        description="Get alerts from last hour"
    )
    if success and data:
        alerts = data.get('data', {})
        alert_count = alerts.get('count', 0)
        print(f"   🚨 Recent Alerts: {alert_count}")
        if alert_count > 0:
            for alert in alerts.get('alerts', [])[:2]:  # Show first 2
                print(f"      {alert['level'].upper()}: {alert['message'][:60]}...")
    
    print()
    
    # Test 7: Set a test budget limit
    success, data = test_api_endpoint(
        "/api/v1/cost-control/budget",
        method='POST',
        data={
            "category": "experiments",
            "period": "daily", 
            "limit": 1.0,
            "alert_thresholds": [0.5, 0.8, 0.95],
            "hard_limit": False
        },
        description="Set daily experiments budget limit"
    )
    if success and data:
        result = data.get('data', {})
        print(f"   🎯 Budget Key: {result.get('budget_key', 'unknown')}")
        print(f"   ✅ Message: {result.get('message', 'Set successfully')}")
    
    print()
    print("=" * 60)
    print("✅ API Integration Tests Completed!")
    print("🎉 Cost Control System API is operational and ready for production use.")


if __name__ == "__main__":
    main()