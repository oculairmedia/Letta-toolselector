import aiohttp
import asyncio
import json
import time
from datetime import datetime
from pprint import pprint
# import re  # Currently unused
import os
from dotenv import load_dotenv
import logging # Added logging import
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Async Helper Function ---
async def fetch_tools_from_server_async(session, url, headers, limit=1000):
    """
    Asynchronously fetch all tools from a specific server URL using pagination.
    """
    all_tools = []
    after = None
    retries = 3
    delay = 1

    while True:
        params = {"limit": limit}
        if after:
            params["after"] = after

        current_retries = retries
        while current_retries > 0:
            try:
                print(f"Fetching tools from {url} (limit={limit}, after={after})...")
                async with session.get(url, headers=headers, params=params, timeout=60) as response:
                    response.raise_for_status()
                    # Handle potential non-JSON responses gracefully
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        tools = await response.json()
                    else:
                        text_response = await response.text()
                        print(f"Warning: Received non-JSON response ({content_type}) from {url}. Content: {text_response[:200]}...")
                        tools = [] # Treat as empty list if not JSON

                    # If tools is not a list, wrap it in one (or handle error)
                    if isinstance(tools, dict):
                        # Sometimes a single object might be returned instead of a list
                        tools = [tools]
                    elif not isinstance(tools, list):
                         print(f"Warning: Expected list but got {type(tools)} from {url}. Treating as empty.")
                         tools = []

                    # Add tools to our list
                    all_tools.extend(tools)

                    print(f"Retrieved {len(tools)} tools in this batch from {url}.")

                    # If we got fewer tools than the limit, we're done for this URL
                    if len(tools) < limit:
                        return all_tools # Return successfully fetched tools for this server

                    # Get the last tool ID for pagination
                    if tools and 'id' in tools[-1]:
                        after = tools[-1]['id']
                        await asyncio.sleep(0.2) # Shorter delay for async
                    else:
                        # No more tools or last tool has no ID for pagination
                        return all_tools # Return successfully fetched tools

                    break # Success, break retry loop

            except aiohttp.ClientResponseError as e:
                print(f"HTTP Error fetching from {url} (Status: {e.status}): {e.message}")
                # Don't retry on 404 Not Found or client errors (4xx)
                if 400 <= e.status < 500:
                    return all_tools # Return what we have so far, skip this server/endpoint
                current_retries -= 1
                if current_retries == 0:
                    print(f"Max retries reached for {url}. Skipping.")
                    return all_tools # Return what we have so far
                print(f"Retrying in {delay}s... ({retries - current_retries}/{retries})")
                await asyncio.sleep(delay)
                delay *= 2 # Exponential backoff
            except aiohttp.ClientError as e: # Catch other client errors (timeouts, connection issues)
                print(f"Client Error fetching from {url}: {e}")
                current_retries -= 1
                if current_retries == 0:
                    print(f"Max retries reached for {url}. Skipping.")
                    return all_tools # Return what we have so far
                print(f"Retrying in {delay}s... ({retries - current_retries}/{retries})")
                await asyncio.sleep(delay)
                delay *= 2 # Exponential backoff
            except Exception as e: # Catch unexpected errors
                print(f"Unexpected error fetching from {url}: {e}")
                # Depending on the error, might want to retry or just break
                return all_tools # Safer to return what we have

    return all_tools # Should technically be unreachable due to returns in loop

# --- Keep the original sync function for now, we'll replace its internals ---
def fetch_tools_from_server(url, headers, limit=1000):
    """
    Fetch all tools from a specific server URL.
    (This is the original sync version - will be replaced)
    """
    all_tools = []
    after = None

    while True:
        params = {"limit": limit}
        if after:
            params["after"] = after

        try:
            # This part will be replaced by async logic later
            print(f"[SYNC] Fetching tools from {url} (limit={limit}, after={after})...")
            # Placeholder for sync request logic
            # import re  # Currently unusedquests # Keep requests import local to this placeholder
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tools = response.json()

            # If tools is not a list, wrap it in one
            if isinstance(tools, dict):
                tools = [tools]
            
            # Add tools to our list
            all_tools.extend(tools)
            
            print(f"Retrieved {len(tools)} tools in this batch.")
            
            # If we got fewer tools than the limit, we're done
            if len(tools) < limit:
                break
            
            # Get the last tool ID for pagination
            if tools:
                after = tools[-1].get('id')
                time.sleep(0.5)  # Small delay to avoid rate limiting
            else:
                break
                
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
            break
    
    return all_tools

# --- Main Async Function ---
async def fetch_all_tools_async():
    """
    Asynchronously fetch all tools from Letta and MCP servers,
    attempt registration, and save them to a JSON file.
    """
    # Load environment variables if not already loaded
    load_dotenv()

    # Set up headers using environment variable for password/key
    api_key = os.getenv('LETTA_PASSWORD') # Get password from environment
    if not api_key:
        print("Error: LETTA_PASSWORD environment variable not set.")
        return [] # Cannot proceed without authentication

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Using the potentially incorrect X-BARE-PASSWORD header for now
        "X-BARE-PASSWORD": f'password {api_key}'
    }

    base_url = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1').replace('http://', 'https://')
    if not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'

    tools_by_name = {}
    mcp_servers = {}

    async with aiohttp.ClientSession() as session:
        # --- Fetch Main Letta Tools ---
        print("Fetching main Letta tools...")
        main_tools_url = f"{base_url}/tools"
        try:
            letta_tools = await fetch_tools_from_server_async(session, main_tools_url, headers)
            tools_by_name = {tool['name']: tool for tool in letta_tools if 'name' in tool}
            print(f"Fetched {len(tools_by_name)} main Letta tools.")
            
            # --- Extract mcp_server_name from metadata_.mcp.server_name for tools that have it ---
            # This handles tools that were registered from MCP servers but aren't in the current MCP server tool lists
            extracted_count = 0
            for tool_name, tool_data in tools_by_name.items():
                if 'mcp_server_name' not in tool_data or not tool_data.get('mcp_server_name'):
                    # Try to extract from metadata_.mcp.server_name
                    metadata = tool_data.get('metadata_', {})
                    if isinstance(metadata, dict):
                        mcp_info = metadata.get('mcp', {})
                        if isinstance(mcp_info, dict) and mcp_info.get('server_name'):
                            tool_data['mcp_server_name'] = mcp_info['server_name']
                            extracted_count += 1
            if extracted_count > 0:
                print(f"Extracted mcp_server_name from metadata for {extracted_count} tools.")
        except Exception as e:
            print(f"Error fetching main Letta tools: {e}")
            # Decide if we should continue without main tools or stop

        # --- Fetch MCP Servers List ---
        try:
            mcp_servers_url = f"{base_url}/tools/mcp/servers"
            print(f"\nFetching MCP servers list from {mcp_servers_url}...")
            async with session.get(mcp_servers_url, headers=headers, timeout=30) as response:
                response.raise_for_status()
                mcp_servers = await response.json()
                # Log the actual server data received
                logger.info(f"Found {len(mcp_servers)} MCP servers. Data: {mcp_servers}")
        except Exception as e:
            print(f"Error fetching MCP servers list: {e}")
            # Decide if we should stop if server list fails

        # --- Fetch Tools from each MCP Server Concurrently ---
        mcp_fetch_tasks = []
        if mcp_servers:
             print("\nFetching tools from MCP servers concurrently...")
             for server_name in mcp_servers.keys():
                 mcp_tools_url = f"{base_url}/tools/mcp/servers/{server_name}/tools"
                 task = asyncio.create_task(
                     fetch_tools_from_server_async(session, mcp_tools_url, headers),
                     name=f"fetch-{server_name}" # Name task for easier debugging
                 )
                 mcp_fetch_tasks.append((server_name, task))

             # Wait for all MCP server tool fetches to complete
             mcp_results = await asyncio.gather(*[task for _, task in mcp_fetch_tasks], return_exceptions=True)

             # --- Process MCP Tools & Attempt Registration ---
             print("\nProcessing fetched MCP tools and attempting registration...")
             registration_tasks = []
             tools_to_register = []

             for i, result in enumerate(mcp_results):
                 server_name, _ = mcp_fetch_tasks[i]
                 if isinstance(result, Exception):
                     print(f"Error fetching tools from MCP server '{server_name}': {result}")
                     continue # Skip this server if fetching failed

                 mcp_tools_list = result
                 print(f"Processing {len(mcp_tools_list)} tools from '{server_name}'...")
                 for tool_data in mcp_tools_list: # Rename loop variable
                     # Basic validation
                     if not isinstance(tool_data, dict) or 'name' not in tool_data:
                         print(f"Warning: Skipping invalid tool data from '{server_name}': {str(tool_data)[:100]}")
                         continue

                     # --- Add origin server name to ALL tools fetched from MCP endpoints ---
                     tool_data['mcp_server_name'] = server_name
                     # Ensure tool_type is set correctly if missing
                     if 'tool_type' not in tool_data:
                         tool_data['tool_type'] = 'external_mcp'
                     # ---

                     name = tool_data['name']
                     if name not in tools_by_name:
                         # Tool not in main Letta list, needs registration.
                         # Add to registration list (tool_data already includes mcp_server_name)
                         tools_to_register.append((server_name, name, tool_data))
                     else:
                         # Tool *is* in main Letta list (tools_by_name).
                         # Ensure this existing entry gets tagged with its MCP origin if found via MCP list.
                         if 'mcp_server_name' not in tools_by_name[name]:
                             # logger.debug(f"Tagging existing tool '{name}' with MCP origin '{server_name}'.")
                             tools_by_name[name]['mcp_server_name'] = server_name
                         elif tools_by_name[name]['mcp_server_name'] != server_name:
                             # Log if tool is found via multiple MCP servers - might indicate config issue
                             logger.warning(f"Tool '{name}' found via multiple MCP servers: '{tools_by_name[name]['mcp_server_name']}' and '{server_name}'. Using first found.")
                         # Optionally update other fields from the MCP fetch if desired:
                         # tools_by_name[name].update(tool_data) # Example: Overwrite with MCP data


             # --- Parallel Registration ---
             print(f"\nAttempting to register {len(tools_to_register)} new MCP tools concurrently...")
             async def register_tool_async(session, server_name, tool_name, tool_data, headers, base_url):
                 register_url = f"{base_url}/tools/mcp/servers/{server_name}/{tool_name}"
                 try:
                     # print(f"Registering: {tool_name} from {server_name}") # Verbose log
                     # Log the tool definition being sent for registration - KEEP THIS LOG
                     print(f"Tool definition for '{tool_name}': {json.dumps(tool_data, indent=2)}")
                     async with session.post(register_url, headers=headers, timeout=60) as register_response:
                         if register_response.status == 200:
                             registered_tool = await register_response.json()
                             if registered_tool and 'id' in registered_tool and 'name' in registered_tool:
                                 print(f"Successfully registered MCP tool '{registered_tool['name']}' with ID {registered_tool['id']}")
                                 return registered_tool # Return the registered tool data
                             else:
                                 print(f"Warning: Registration response for '{tool_name}' missing tool ID or name. Response: {await register_response.text()}")
                                 return None # Indicate failure but don't raise exception
                         else:
                             # Log detailed error
                             error_text = await register_response.text()
                             print(f"Warning: Failed to register '{tool_name}' from '{server_name}'. Status: {register_response.status}. Response: {error_text[:500]}")
                             return None # Indicate failure
                 except asyncio.TimeoutError:
                     print(f"Error: Timeout registering tool '{tool_name}' from '{server_name}'")
                     return None
                 except Exception as e:
                     print(f"Error registering tool '{tool_name}' from '{server_name}': {e}")
                     return None

             # Create tasks for registration
             registration_tasks = [
                 register_tool_async(session, s_name, t_name, t_data, headers, base_url)
                 for s_name, t_name, t_data in tools_to_register
             ]

             # Execute registration tasks concurrently
             registration_results = await asyncio.gather(*registration_tasks, return_exceptions=True)

             # Update tools_by_name with successfully registered tools
             registered_count = 0
             failed_count = 0
             for result in registration_results:
                 if isinstance(result, Exception):
                     # This shouldn't happen often due to try/except in register_tool_async
                     print(f"Unexpected gather error during registration: {result}")
                     failed_count += 1
                 elif result and isinstance(result, dict) and 'name' in result:
                     # Successfully registered, result is the registered_tool data from Letta API
                     registered_tool_name = result['name']
                     # Find the original tool_data sent for registration to get mcp_server_name
                     original_tool_data = next((t_data for srv, t_name, t_data in tools_to_register if t_name == registered_tool_name), None)

                     if original_tool_data:
                         # Update the original data with the ID from the successful registration result
                         original_tool_data['id'] = result.get('id') or result.get('tool_id')
                         # Store the original data (which includes mcp_server_name) in the final map
                         tools_by_name[registered_tool_name] = original_tool_data
                         registered_count += 1
                     else:
                         # This case should be rare
                         logger.warning(f"Could not find original data for successfully registered tool: {registered_tool_name}. Storing raw API result.")
                         tools_by_name[registered_tool_name] = result # Store raw result as fallback
                         registered_count += 1
                 else:
                     # Registration failed or returned invalid data (already logged in register_tool_async)
                     failed_count += 1

             print(f"Registration summary: {registered_count} successful, {failed_count} failed.")

             # Add tools that failed registration with defaults, marking their origin
             for s_name, t_name, t_data in tools_to_register:
                 if t_name not in tools_by_name:
                     print(f"Adding unregistered tool {t_name} with defaults and origin '{s_name}'")
                     t_data['source_type'] = t_data.get('source_type', 'python')
                     t_data['tool_type'] = t_data.get('tool_type', 'external_mcp')
                     # Store origin server for tools that failed registration
                     t_data['mcp_server_name'] = s_name
                     tools_by_name[t_name] = t_data


    # --- Filter out MCP tools whose servers are no longer active ---
    active_server_names = set(mcp_servers.keys()) # Get names of currently active servers
    filtered_tools_by_name = {}
    removed_count = 0
    logger.info(f"Filtering tools against {len(active_server_names)} active MCP servers: {active_server_names}")
    for name, tool_data in tools_by_name.items():
        # Log the tool being checked during filtering
        logger.debug(f"Filtering check for tool '{name}': Type='{tool_data.get('tool_type')}', Server='{tool_data.get('mcp_server_name')}'")
        if tool_data.get("tool_type") == "external_mcp":
            server_origin = tool_data.get("mcp_server_name")
            # --- Stricter Filter ---
            # Exclude if server_origin is missing OR if it's present but not in the active list
            if not server_origin:
                logger.warning(f"Excluding tool '{name}' because it is external_mcp but has no mcp_server_name recorded (likely obsolete).")
                removed_count += 1
                continue # Skip adding this tool
            elif server_origin not in active_server_names:
                logger.warning(f"Excluding tool '{name}' because its MCP server '{server_origin}' is no longer listed as active.")
                removed_count += 1
                continue # Skip adding this tool
            # --- End Stricter Filter ---
        # Keep the tool if it's not external_mcp or if its server is active
        filtered_tools_by_name[name] = tool_data

    logger.info(f"Removed {removed_count} tools belonging to inactive MCP servers.")

    # --- Final Steps (outside async session) ---
    # Prepare final list using the FILTERED map
    all_tools_with_origin = []
    logger.debug("Constructing final tool list from filtered_tools_by_name map...")
    for name, tool_data in filtered_tools_by_name.items(): # Use the filtered map
        # Add detailed log to check final tool data before returning
        # logger.debug(f"Final data for tool '{name}': {tool_data}")
        if tool_data.get("tool_type") == "external_mcp" and "mcp_server_name" not in tool_data:
             # This warning might still appear if a tool was kept despite missing server name
             logger.warning(f"Tool '{name}' is external_mcp but missing mcp_server_name in final list construction!")
        all_tools_with_origin.append(tool_data)
    logger.debug("Finished constructing final tool list.")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_letta_tools_{timestamp}.json"

    try:
        with open(filename, 'w') as f:
            # Save the list that potentially includes mcp_server_name
            json.dump(all_tools_with_origin, f, indent=2)
        print(f"\nTotal unique tools retrieved/registered: {len(all_tools_with_origin)}")
        print(f"Tools saved to {filename}")
    except IOError as e:
        print(f"Error saving tools to file {filename}: {e}")

    categorize_tools(all_tools_with_origin) # Pass the potentially richer list

    return all_tools_with_origin # Return the list that includes origin info

# --- Keep categorize_tools synchronous for now ---
def categorize_tools(tools):
    """
    Categorize tools and print a summary.
    """
    # Group tools by prefix/category
    categories = {}
    source_types = set()
    tool_types = set()
    
    for tool in tools:
        name = tool.get('name', '')
        source_type = tool.get('source_type', '')
        tool_type = tool.get('tool_type', '')
        
        source_types.add(source_type)
        tool_types.add(tool_type)
        
        # Try to determine category from name
        category = "Other"
        if "__" in name:
            parts = name.split("__")
            category = parts[0]
        elif name.startswith("archival_memory") or name.startswith("core_memory"):
            category = "Memory Management"
        elif "agent" in name:
            category = "Agent Management"
        elif "memory_block" in name:
            category = "Memory Blocks"
        elif "ghost" in name:
            category = "Ghost CMS"
        elif "plane" in name:
            category = "Plane"
        elif "graphiti" in name:
            category = "Graphiti"
        elif "book" in name or name.startswith("read_") or name.startswith("list_"):
            category = "Knowledge Base"
        elif "gmail" in name:
            category = "Gmail"
        elif "repomix" in name:
            category = "Repomix"
            
        # Add to categories
        if category not in categories:
            categories[category] = []
        categories[category].append({
            'name': name,
            'source_type': source_type,
            'tool_type': tool_type
        })
    
    # Print summary
    print("\nTools by Category:")
    for category, tools in sorted(categories.items()):
        print(f"\n{category} ({len(tools)}):")
        for tool in sorted(tools, key=lambda x: x['name']):
            print(f"  - {tool['name']}")
            print(f"    Source Type: {tool['source_type']}")
            print(f"    Tool Type: {tool['tool_type']}")
    
    print("\nUnique Source Types:")
    pprint(sorted(source_types))
    
    print("\nUnique Tool Types:")
    pprint(sorted(tool_types))

# --- Main execution block ---
if __name__ == "__main__":
    # Run the async function using asyncio.run()
    start_time = time.time()
    asyncio.run(fetch_all_tools_async())
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")