#!/usr/bin/env python3
"""
Clean up stale tools from Letta that are no longer provided by their MCP servers.

For each active MCP server:
1. Get the current tools it provides
2. Find all tools in Letta that claim to be from that server
3. Delete tools that exist in Letta but not in the MCP server's current list
"""

import requests
import os
import sys
from collections import defaultdict

def main(dry_run=True):
    letta_url = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1')
    letta_pass = os.getenv('LETTA_PASSWORD', '')
    headers = {'Authorization': f'Bearer {letta_pass}'}
    
    print(f"Connecting to Letta at {letta_url}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE - WILL DELETE TOOLS'}\n")
    
    # Step 1: Get all MCP servers
    print("Fetching MCP servers...")
    resp = requests.get(f'{letta_url}/tools/mcp/servers', headers=headers)
    if resp.status_code != 200:
        print(f"Error fetching MCP servers: {resp.status_code}")
        return
    
    mcp_servers = resp.json()
    print(f"Found {len(mcp_servers)} MCP servers\n")
    
    # Step 2: For each MCP server, get its current tools
    mcp_current_tools = {}  # server_name -> set of tool names
    
    for server_name in mcp_servers.keys():
        resp = requests.get(f'{letta_url}/tools/mcp/servers/{server_name}/tools', headers=headers)
        if resp.status_code == 200:
            tools = resp.json()
            tool_names = {t['name'] for t in tools}
            mcp_current_tools[server_name] = tool_names
            print(f"  {server_name}: {len(tool_names)} current tools")
        else:
            print(f"  {server_name}: ERROR {resp.status_code} (server may be down)")
            mcp_current_tools[server_name] = set()  # Empty set for unreachable servers
    
    # Step 3: Get all tools from Letta
    print("\nFetching all tools from Letta...")
    resp = requests.get(f'{letta_url}/tools/', headers=headers, params={'limit': 1000})
    if resp.status_code != 200:
        print(f"Error fetching tools: {resp.status_code}")
        return
    
    all_letta_tools = resp.json()
    print(f"Found {len(all_letta_tools)} total tools in Letta\n")
    
    # Step 4: Find stale tools (in Letta but not in MCP server's current list)
    stale_tools = []
    tools_by_server = defaultdict(list)
    
    for tool in all_letta_tools:
        tool_name = tool.get('name', '')
        tool_id = tool.get('id', '')
        tool_type = tool.get('tool_type', '')
        
        # Get MCP server name from metadata
        metadata = tool.get('metadata_', {}) or {}
        mcp_info = metadata.get('mcp', {}) or {}
        server_name = mcp_info.get('server_name', '')
        
        # Also check top-level mcp_server_name if metadata doesn't have it
        if not server_name:
            server_name = tool.get('mcp_server_name', '')
        
        if tool_type == 'external_mcp' and server_name:
            tools_by_server[server_name].append(tool)
            
            # Check if this tool is still provided by its MCP server
            if server_name in mcp_current_tools:
                current_tools = mcp_current_tools[server_name]
                if tool_name not in current_tools:
                    stale_tools.append({
                        'id': tool_id,
                        'name': tool_name,
                        'server': server_name,
                        'reason': 'not in current MCP tool list'
                    })
    
    # Print summary by server
    print("Tools per MCP server in Letta:")
    for server_name, tools in sorted(tools_by_server.items()):
        current_count = len(mcp_current_tools.get(server_name, set()))
        letta_count = len(tools)
        status = "OK" if current_count == letta_count else f"MISMATCH (MCP has {current_count})"
        print(f"  {server_name}: {letta_count} in Letta - {status}")
    
    # Print stale tools
    print(f"\n{'='*60}")
    print(f"Found {len(stale_tools)} stale tools to delete:")
    print(f"{'='*60}\n")
    
    stale_by_server = defaultdict(list)
    for tool in stale_tools:
        stale_by_server[tool['server']].append(tool)
    
    for server_name, tools in sorted(stale_by_server.items()):
        print(f"\n{server_name} ({len(tools)} stale tools):")
        for tool in sorted(tools, key=lambda x: x['name']):
            print(f"  - {tool['name']}")
    
    # Step 5: Delete stale tools (if not dry run)
    if not dry_run and stale_tools:
        print(f"\n{'='*60}")
        print("DELETING STALE TOOLS...")
        print(f"{'='*60}\n")
        
        deleted = 0
        failed = 0
        
        for tool in stale_tools:
            tool_id = tool['id']
            tool_name = tool['name']
            
            resp = requests.delete(f'{letta_url}/tools/{tool_id}', headers=headers)
            if resp.status_code in [200, 204]:
                print(f"  ✓ Deleted: {tool_name}")
                deleted += 1
            else:
                print(f"  ✗ Failed to delete {tool_name}: {resp.status_code}")
                failed += 1
        
        print(f"\nDeletion complete: {deleted} deleted, {failed} failed")
    elif stale_tools:
        print(f"\nDRY RUN - No tools were deleted. Run with --live to delete.")

if __name__ == "__main__":
    dry_run = '--live' not in sys.argv
    main(dry_run=dry_run)
