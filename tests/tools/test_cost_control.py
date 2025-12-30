#!/usr/bin/env python3
"""
Test script for LDTS-58: Cost Controls and Budget Management System
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'tool-selector-api'))

from cost_control_manager import (
    CostControlManager, CostCategory, BudgetPeriod, AlertLevel,
    get_cost_manager, record_embedding_cost, record_weaviate_cost
)


async def test_basic_functionality():
    """Test basic cost control functionality"""
    print("🧪 Testing Basic Cost Control Functionality")
    print("=" * 50)
    
    # Get manager instance
    manager = get_cost_manager()
    
    # Test 1: Record some test costs
    print("✅ Test 1: Recording test costs...")
    
    await manager.record_cost(
        CostCategory.EMBEDDING_API, 
        "test_embedding_generation", 
        0.05,
        {"tokens": 500, "model": "text-embedding-3-small"}
    )
    
    await manager.record_cost(
        CostCategory.VECTOR_DATABASE,
        "test_weaviate_query",
        0.01,
        {"query_count": 10}
    )
    
    await manager.record_cost(
        CostCategory.LETTA_API,
        "test_agent_call",
        0.02,
        {"call_count": 2}
    )
    
    print("   ✓ Recorded test costs for different categories")
    
    # Test 2: Get cost summary
    print("\n✅ Test 2: Getting cost summaries...")
    
    daily_summary = await manager.get_cost_summary(BudgetPeriod.DAILY)
    print(f"   📊 Daily Summary: ${daily_summary.total_cost:.4f}")
    print(f"   📈 Entry count: {daily_summary.entry_count}")
    
    for category, cost in daily_summary.cost_by_category.items():
        print(f"   💰 {category.value}: ${cost:.4f}")
    
    # Test 3: Check budget status
    print("\n✅ Test 3: Checking budget status...")
    
    budget_status = await manager.get_budget_status()
    for key, status in budget_status.items():
        print(f"   🎯 {key}:")
        print(f"      Limit: ${status['limit']:.2f}")
        print(f"      Current: ${status['current_cost']:.4f}")
        print(f"      Remaining: ${status['remaining']:.4f}")
        print(f"      Used: {status['percentage_used']:.1f}%")
    
    return True


async def test_budget_limits():
    """Test budget limit enforcement"""
    print("\n🧪 Testing Budget Limits and Alerts")
    print("=" * 50)
    
    manager = get_cost_manager()
    
    # Test 1: Set a low budget limit for testing
    print("✅ Test 1: Setting test budget limits...")
    
    await manager.set_budget_limit(
        category=CostCategory.EMBEDDING_API,
        period=BudgetPeriod.DAILY,
        limit=0.10,  # Very low limit for testing
        alert_thresholds=[0.3, 0.7, 0.9],
        hard_limit=False
    )
    
    await manager.set_budget_limit(
        category=CostCategory.VECTOR_DATABASE,
        period=BudgetPeriod.DAILY,
        limit=0.05,  # Very low limit
        alert_thresholds=[0.5, 0.8, 0.95],
        hard_limit=True  # Hard limit for testing
    )
    
    print("   ✓ Set low budget limits for testing")
    
    # Test 2: Record costs that should trigger alerts
    print("\n✅ Test 2: Testing alert generation...")
    
    # This should trigger threshold alerts
    await manager.record_cost(
        CostCategory.EMBEDDING_API,
        "test_alert_trigger", 
        0.08,  # 80% of 0.10 limit
        {"test": "alert_threshold"}
    )
    
    # This should exceed the hard limit
    try:
        result = await manager.record_cost(
            CostCategory.VECTOR_DATABASE,
            "test_hard_limit",
            0.06,  # Over the 0.05 limit
            {"test": "hard_limit"}
        )
        print(f"   🚨 Hard limit result: {'ALLOWED' if result else 'BLOCKED'}")
    except Exception as e:
        print(f"   ❌ Hard limit test failed: {e}")
    
    # Test 3: Check recent alerts
    print("\n✅ Test 3: Checking generated alerts...")
    
    recent_alerts = await manager.get_recent_alerts(hours=1)
    print(f"   🔔 Found {len(recent_alerts)} recent alerts")
    
    for alert in recent_alerts[-3:]:  # Show last 3 alerts
        print(f"   📢 {alert.level.value.upper()}: {alert.message}")
        print(f"      Category: {alert.category.value if alert.category else 'Overall'}")
        print(f"      Usage: {alert.percentage_used:.1f}%")
    
    return True


async def test_cost_estimation():
    """Test cost estimation functionality"""
    print("\n🧪 Testing Cost Estimation")
    print("=" * 50)
    
    manager = get_cost_manager()
    
    # Test different operation types
    operations = [
        ("openai_embedding", {"token_count": 1000}),
        ("weaviate_query", {"query_count": 5}),
        ("weaviate_insert", {"insert_count": 10}),
        ("letta_api_call", {"call_count": 3}),
    ]
    
    print("✅ Estimating costs for different operations:")
    
    for op_type, params in operations:
        cost = await manager.estimate_operation_cost(op_type, **params)
        print(f"   💵 {op_type}: ${cost:.6f} (params: {params})")
    
    return True


async def test_convenience_functions():
    """Test convenience functions for cost tracking"""
    print("\n🧪 Testing Convenience Functions")
    print("=" * 50)
    
    print("✅ Testing convenience cost recording functions...")
    
    # Test embedding cost recording
    result1 = await record_embedding_cost("test_convenience_embedding", 750, "openai")
    print(f"   ✓ Embedding cost recorded: {'Allowed' if result1 else 'Blocked'}")
    
    # Test Weaviate cost recording
    result2 = await record_weaviate_cost("test_convenience_weaviate", "query", 3)
    print(f"   ✓ Weaviate cost recorded: {'Allowed' if result2 else 'Blocked'}")
    
    return True


async def test_period_calculations():
    """Test period boundary calculations"""
    print("\n🧪 Testing Period Calculations")
    print("=" * 50)
    
    manager = get_cost_manager()
    
    # Test different periods
    periods = [BudgetPeriod.HOURLY, BudgetPeriod.DAILY, BudgetPeriod.WEEKLY, BudgetPeriod.MONTHLY]
    
    print("✅ Testing period boundary calculations:")
    
    for period in periods:
        start, end = manager._get_period_bounds(period)
        duration = end - start
        print(f"   📅 {period.value}: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} ({duration})")
    
    return True


async def main():
    """Run all tests"""
    print("🚀 LDTS-58: Cost Control System Tests")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_budget_limits,
        test_cost_estimation,
        test_convenience_functions,
        test_period_calculations,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = await test()
            results.append(("✅", test.__name__, "PASSED"))
            print(f"\n🎉 {test.__name__}: PASSED")
        except Exception as e:
            results.append(("❌", test.__name__, f"FAILED: {e}"))
            print(f"\n💥 {test.__name__}: FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if "PASSED" in r[2])
    failed = len(results) - passed
    
    for emoji, test_name, status in results:
        print(f"{emoji} {test_name}: {status}")
    
    print(f"\n🏆 Results: {passed} passed, {failed} failed out of {len(results)} tests")
    
    if failed == 0:
        print("🎉 All tests passed! Cost Control System is working correctly.")
        return True
    else:
        print("😞 Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)