#!/usr/bin/env python3
"""
Embedding Version Management CLI

Command-line interface for managing embedding versions and tracking migrations.

Usage:
    python version_cli.py versions list
    python version_cli.py versions register --provider ollama --model qwen3 --version 1.0 --dimensions 768
    python version_cli.py migrations list
    python version_cli.py migrations plan --from openai_v1 --to ollama_v1
    python version_cli.py analyze --collection Tool
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from embedding_version_manager import EmbeddingVersionManager, MigrationStatus

def handle_versions_list(manager: EmbeddingVersionManager, args):
    """List all registered versions"""
    versions = manager.list_versions(provider=args.provider if hasattr(args, 'provider') else None)

    if not versions:
        print("No embedding versions registered.")
        return

    print(f"\n{'ID':<25} {'Provider':<12} {'Model':<25} {'Dimensions':<12} {'Created'}")
    print("-" * 95)

    for version_id, version in versions:
        created = version.created_at.split('T')[0] if 'T' in version.created_at else version.created_at
        print(f"{version_id:<25} {version.provider:<12} {version.model:<25} {version.dimensions:<12} {created}")

    # Show current environment version
    current = manager.get_current_version_from_env()
    if current:
        print(f"\n🔍 Current Environment: {current[0]} ({current[1].provider}/{current[1].model})")

def handle_versions_register(manager: EmbeddingVersionManager, args):
    """Register a new embedding version"""
    try:
        performance_metrics = {}
        if args.accuracy:
            performance_metrics["search_accuracy"] = args.accuracy
        if args.speed:
            performance_metrics["speed"] = args.speed

        version_id = manager.register_version(
            provider=args.provider,
            model=args.model,
            version=args.version,
            dimensions=args.dimensions,
            description=args.description or "",
            performance_metrics=performance_metrics if performance_metrics else None
        )

        print(f"✅ Registered embedding version: {version_id}")
        print(f"   Provider: {args.provider}")
        print(f"   Model: {args.model}")
        print(f"   Version: {args.version}")
        print(f"   Dimensions: {args.dimensions}")

    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return 1

    return 0

def handle_versions_info(manager: EmbeddingVersionManager, args):
    """Show detailed information about a version"""
    version = manager.get_version(args.version_id)

    if not version:
        print(f"❌ Version '{args.version_id}' not found")
        return 1

    print(f"\n📦 Version Details: {args.version_id}")
    print(f"{'='*50}")
    print(f"Provider: {version.provider}")
    print(f"Model: {version.model}")
    print(f"Version: {version.version}")
    print(f"Dimensions: {version.dimensions}")
    print(f"Created: {version.created_at}")
    print(f"Compatibility Hash: {version.compatibility_hash}")

    if version.description:
        print(f"Description: {version.description}")

    if version.performance_metrics:
        print(f"\n📊 Performance Metrics:")
        for metric, value in version.performance_metrics.items():
            print(f"  {metric}: {value}")

    return 0

def handle_migrations_list(manager: EmbeddingVersionManager, args):
    """List migration history"""
    status_filter = None
    if hasattr(args, 'status') and args.status:
        try:
            status_filter = MigrationStatus(args.status.lower())
        except ValueError:
            print(f"❌ Invalid status: {args.status}")
            return 1

    migrations = manager.list_migrations(status=status_filter)

    if not migrations:
        print("No migrations recorded.")
        return 0

    print(f"\n{'ID':<20} {'Status':<12} {'From':<20} {'To':<20} {'Started'}")
    print("-" * 95)

    for migration_id, migration in migrations:
        from_version = f"{migration.from_version.provider}/{migration.from_version.model}"
        to_version = f"{migration.to_version.provider}/{migration.to_version.model}"
        started = migration.started_at.split('T')[0] if 'T' in migration.started_at else migration.started_at

        print(f"{migration_id:<20} {migration.status.value:<12} {from_version:<20} {to_version:<20} {started}")

    return 0

def handle_migrations_info(manager: EmbeddingVersionManager, args):
    """Show detailed migration information"""
    migration = manager.get_migration(args.migration_id)

    if not migration:
        print(f"❌ Migration '{args.migration_id}' not found")
        return 1

    print(f"\n🔄 Migration Details: {args.migration_id}")
    print(f"{'='*50}")
    print(f"Status: {migration.status.value}")
    print(f"Started: {migration.started_at}")
    if migration.completed_at:
        print(f"Completed: {migration.completed_at}")

    print(f"\nFrom Version:")
    print(f"  {migration.from_version.provider}/{migration.from_version.model} v{migration.from_version.version}")
    print(f"  Dimensions: {migration.from_version.dimensions}")

    print(f"\nTo Version:")
    print(f"  {migration.to_version.provider}/{migration.to_version.model} v{migration.to_version.version}")
    print(f"  Dimensions: {migration.to_version.dimensions}")

    if migration.tools_migrated:
        print(f"\nProgress:")
        print(f"  Tools Migrated: {migration.tools_migrated}")
        print(f"  Tools Failed: {migration.tools_failed}")
        success_rate = migration.tools_migrated / (migration.tools_migrated + migration.tools_failed) * 100
        print(f"  Success Rate: {success_rate:.1f}%")

    if migration.performance_improvement:
        print(f"\nPerformance Improvement:")
        for metric, improvement in migration.performance_improvement.items():
            print(f"  {metric}: {improvement:+.2f}%")

    if migration.notes:
        print(f"\nNotes: {migration.notes}")

    return 0

def handle_migrations_plan(manager: EmbeddingVersionManager, args):
    """Generate migration plan between versions"""
    try:
        plan = manager.generate_migration_plan(
            from_version_id=args.from_version,
            to_version_id=args.to_version,
            estimated_tools=args.tools if hasattr(args, 'tools') else None
        )

        print(f"\n📋 Migration Plan: {args.from_version} → {args.to_version}")
        print(f"{'='*60}")

        # Compatibility
        compat = plan['compatibility']
        print(f"Compatibility: {'✅ Compatible' if compat['compatible'] else '❌ Incompatible'}")

        if compat['warnings']:
            print(f"Warnings:")
            for warning in compat['warnings']:
                print(f"  ⚠️  {warning}")

        # Estimates
        if plan['estimated_tools']:
            print(f"\nEstimates:")
            print(f"  Tools to migrate: {plan['estimated_tools']}")
            print(f"  Estimated time: {plan['estimated_time_range']}")

        # Prerequisites
        print(f"\nPrerequisites:")
        for prereq in plan['prerequisites']:
            print(f"  - {prereq}")

        # Risks
        if plan['risks']:
            print(f"\nRisks:")
            for risk in plan['risks']:
                print(f"  ⚠️  {risk}")

        # Rollback strategy
        print(f"\nRollback Strategy:")
        print(f"  Method: {plan['rollback_strategy']['method']}")
        for step in plan['rollback_strategy']['steps']:
            print(f"  - {step}")

        # Validation
        print(f"\nValidation Steps:")
        for step in plan['validation_steps']:
            print(f"  - {step}")

        # Migration recommendation
        if compat.get('migration_recommended'):
            print(f"\n✅ Migration recommended based on performance metrics")
        elif compat['compatible']:
            print(f"\n✅ Migration is technically feasible")
        else:
            print(f"\n❌ Migration not recommended due to compatibility issues")

    except Exception as e:
        print(f"❌ Failed to generate migration plan: {e}")
        return 1

    return 0

async def handle_analyze_collection(manager: EmbeddingVersionManager, args):
    """Analyze embedding versions in a collection"""
    try:
        analysis = await manager.analyze_collection_version(args.collection)

        if 'error' in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return 1

        print(f"\n🔍 Collection Analysis: {args.collection}")
        print(f"{'='*50}")
        print(f"Total Tools: {analysis['total_tools']}")

        if analysis['version_distribution']:
            print(f"\nVersion Distribution:")
            for version, count in analysis['version_distribution'].items():
                percentage = (count / analysis['total_tools']) * 100
                print(f"  {version}: {count} tools ({percentage:.1f}%)")

        if analysis.get('oldest_migration'):
            print(f"\nMigration Timeline:")
            print(f"  Oldest: {analysis['oldest_migration']}")
            print(f"  Newest: {analysis['newest_migration']}")

    except Exception as e:
        print(f"❌ Collection analysis failed: {e}")
        return 1

    return 0

def handle_compatibility_check(manager: EmbeddingVersionManager, args):
    """Check compatibility between two versions"""
    try:
        compat = manager.check_version_compatibility(args.version1, args.version2)

        print(f"\n🔗 Compatibility Check: {args.version1} ↔ {args.version2}")
        print(f"{'='*60}")

        print(f"Overall Compatible: {'✅ Yes' if compat['compatible'] else '❌ No'}")
        print(f"Dimension Match: {'✅' if compat['dimension_match'] else '❌'}")
        print(f"Provider Match: {'✅' if compat['provider_match'] else '❌'}")
        print(f"Migration Recommended: {'✅' if compat['migration_recommended'] else '❌'}")

        if compat['warnings']:
            print(f"\nWarnings:")
            for warning in compat['warnings']:
                print(f"  ⚠️  {warning}")

    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return 1

    return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Embedding Version Management CLI"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Versions subcommands
    versions_parser = subparsers.add_parser('versions', help='Manage embedding versions')
    versions_subparsers = versions_parser.add_subparsers(dest='versions_command')

    # versions list
    list_parser = versions_subparsers.add_parser('list', help='List all versions')
    list_parser.add_argument('--provider', help='Filter by provider')

    # versions register
    register_parser = versions_subparsers.add_parser('register', help='Register new version')
    register_parser.add_argument('--provider', required=True, help='Provider name')
    register_parser.add_argument('--model', required=True, help='Model name')
    register_parser.add_argument('--version', required=True, help='Version string')
    register_parser.add_argument('--dimensions', type=int, required=True, help='Embedding dimensions')
    register_parser.add_argument('--description', help='Version description')
    register_parser.add_argument('--accuracy', type=float, help='Search accuracy metric')
    register_parser.add_argument('--speed', type=float, help='Speed metric')

    # versions info
    info_parser = versions_subparsers.add_parser('info', help='Show version details')
    info_parser.add_argument('version_id', help='Version ID to show')

    # Migrations subcommands
    migrations_parser = subparsers.add_parser('migrations', help='Manage migrations')
    migrations_subparsers = migrations_parser.add_subparsers(dest='migrations_command')

    # migrations list
    mig_list_parser = migrations_subparsers.add_parser('list', help='List migrations')
    mig_list_parser.add_argument('--status', help='Filter by status')

    # migrations info
    mig_info_parser = migrations_subparsers.add_parser('info', help='Show migration details')
    mig_info_parser.add_argument('migration_id', help='Migration ID to show')

    # migrations plan
    plan_parser = migrations_subparsers.add_parser('plan', help='Generate migration plan')
    plan_parser.add_argument('--from', dest='from_version', required=True, help='Source version ID')
    plan_parser.add_argument('--to', dest='to_version', required=True, help='Target version ID')
    plan_parser.add_argument('--tools', type=int, help='Estimated number of tools')

    # Analysis commands
    analyze_parser = subparsers.add_parser('analyze', help='Analyze collection versions')
    analyze_parser.add_argument('--collection', required=True, help='Collection name to analyze')

    # Compatibility command
    compat_parser = subparsers.add_parser('compatibility', help='Check version compatibility')
    compat_parser.add_argument('version1', help='First version ID')
    compat_parser.add_argument('version2', help='Second version ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize manager
    manager = EmbeddingVersionManager()

    # Route commands
    try:
        if args.command == 'versions':
            if args.versions_command == 'list':
                return handle_versions_list(manager, args)
            elif args.versions_command == 'register':
                return handle_versions_register(manager, args)
            elif args.versions_command == 'info':
                return handle_versions_info(manager, args)
            else:
                versions_parser.print_help()
                return 1

        elif args.command == 'migrations':
            if args.migrations_command == 'list':
                return handle_migrations_list(manager, args)
            elif args.migrations_command == 'info':
                return handle_migrations_info(manager, args)
            elif args.migrations_command == 'plan':
                return handle_migrations_plan(manager, args)
            else:
                migrations_parser.print_help()
                return 1

        elif args.command == 'analyze':
            return asyncio.run(handle_analyze_collection(manager, args))

        elif args.command == 'compatibility':
            return handle_compatibility_check(manager, args)

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())