import os
import time # Ensure time is imported (already present, just confirming)
import schedule
import requests
from datetime import datetime
from dotenv import load_dotenv
import logging
import json
import weaviate
import asyncio
import aiohttp # Import aiohttp
import aiofiles # Import aiofiles
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
import weaviate.classes.query as wq
# Import the new async function
from embedding_config import OPENAI_EMBEDDING_MODEL
from fetch_all_tools import fetch_all_tools_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Define Cache Directory and File Path ---
CACHE_DIR = "/app/runtime_cache" # Changed cache directory
TOOL_CACHE_FILE_PATH = os.path.join(CACHE_DIR, "tool_cache.json")
MCP_SERVERS_CACHE_FILE_PATH = os.path.join(CACHE_DIR, "mcp_servers_cache.json")

# --- Helper function to write tool cache ---
async def write_tool_cache(tools_data):
    """Writes the provided tools data to the tool cache file asynchronously."""
    try:
        # Ensure cache directory exists (synchronous is fine here as it's usually fast)
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Write the file asynchronously
        async with aiofiles.open(TOOL_CACHE_FILE_PATH, mode='w') as f: # Use renamed variable
            await f.write(json.dumps(tools_data, indent=2)) # Dump to string first
        logger.info(f"Successfully updated tool cache file: {TOOL_CACHE_FILE_PATH}") # Use renamed variable
    except Exception as e:
        logger.error(f"Error writing tool cache file {TOOL_CACHE_FILE_PATH}: {e}") # Use renamed variable

# --- Helper function to write MCP servers cache ---
async def write_mcp_servers_cache(servers_data):
    """Writes the provided MCP servers data to the cache file asynchronously."""
    try:
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Attempting to write MCP servers cache. Data: {json.dumps(servers_data, indent=2)}") # Log data being written
        async with aiofiles.open(MCP_SERVERS_CACHE_FILE_PATH, mode='w') as f:
            await f.write(json.dumps(servers_data, indent=2))
        logger.info(f"Successfully updated MCP servers cache file: {MCP_SERVERS_CACHE_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error writing MCP servers cache file {MCP_SERVERS_CACHE_FILE_PATH}. Exception: {e}", exc_info=True) # Log full exception


# --- Helper function to get Weaviate tools ---
async def get_weaviate_tools(client): # Make async
    """Fetch all tools from Weaviate."""
    try:
        # Wrap synchronous calls for async context
        collection = await asyncio.to_thread(client.collections.get, "Tool")
        result = await asyncio.to_thread(collection.query.fetch_objects, limit=10000) # Increased limit
        # Return dict mapping name to full object properties for consistency
        return {obj.properties["name"]: obj.properties for obj in result.objects if "name" in obj.properties}
    except Exception as e:
        logger.error(f"Error fetching tools from Weaviate: {e}")
        return {}

# --- Helper function to get or create schema ---
def get_vectorizer_config():
    """Get the appropriate vectorizer configuration based on environment settings."""
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'openai').lower()
    
    if embedding_provider == 'ollama':
        # Use Ollama vectorizer with proper endpoint configuration
        vectorizer_config = weaviate.classes.config.Configure.Vectorizer.text2vec_ollama(
            api_endpoint=f"http://{os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')}:11434",
            model=os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
        )
        logger.info(f"Using Ollama vectorizer with endpoint: http://{os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')}:11434")
    else:
        # Default to OpenAI vectorizer
        vectorizer_config = weaviate.classes.config.Configure.Vectorizer.text2vec_openai(
            model=OPENAI_EMBEDDING_MODEL
        )
        logger.info("Using OpenAI vectorizer")
    
    return vectorizer_config

async def get_or_create_tool_schema(client) -> weaviate.collections.Collection: # Make async
    """Get existing schema or create new one if it doesn't exist."""
    try:
        # Wrap synchronous calls for async context
        collection = await asyncio.to_thread(client.collections.get, "Tool")
        logger.info("Using existing Tool schema")
        return collection
    except Exception as e_get: # Catch specific exception if possible, e.g., weaviate.exceptions.UnexpectedStatusCodeError
        logger.warning(f"Tool schema not found (or error checking): {e_get}. Attempting creation.")
        try:
            # Wrap synchronous create call
            collection = await asyncio.to_thread(
                client.collections.create,
                name="Tool",
                description="A Letta tool with its metadata and description",
                vectorizer_config=get_vectorizer_config(),
                properties=[
                    weaviate.classes.config.Property(name="tool_id", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="name", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="description", data_type=weaviate.classes.config.DataType.TEXT, vectorize_property_name=False),
                    weaviate.classes.config.Property(name="source_type", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="tool_type", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="tags", data_type=weaviate.classes.config.DataType.TEXT_ARRAY),
                    weaviate.classes.config.Property(name="json_schema", data_type=weaviate.classes.config.DataType.TEXT, vectorize_property_name=False),
                    # Add field for originating MCP server name
                    weaviate.classes.config.Property(name="mcp_server_name", data_type=weaviate.classes.config.DataType.TEXT, skip_vectorization=True) # Optional, non-vectorized
                ]
            ) # End of args for client.collections.create
            logger.info("Schema created successfully")
            return collection
        except Exception as e_create:
            logger.error(f"Failed to create Tool schema: {e_create}")
            raise

async def sync_tools(): # Make async
    """
    Synchronize tools between Letta and Weaviate:
    1. Fetch tools from Letta (this becomes the cache).
    2. Fetch tools from Weaviate.
    3. Remove tools from Weaviate that are no longer in Letta.
    4. Add tools to Weaviate that are in Letta but not Weaviate.
    5. Update the tool cache file.
    """
    logger.info("Starting tool synchronization...")
    load_dotenv() # Ensure env vars are loaded

    # Verify required environment variables (removed WEAVIATE_API_KEY and WEAVIATE_URL for local connection)
    required_vars = ["OPENAI_API_KEY", "LETTA_API_URL", "LETTA_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables for sync: {', '.join(missing_vars)}")
        return

    weaviate_conn = None # Rename for clarity
    http_session = None # For Letta API calls
    letta_tools = None # Initialize letta_tools
    try:
        # --- Create HTTP Session ---
        # Reusing session logic similar to api_server.py
        http_session = aiohttp.ClientSession()
        logger.info("Sync service HTTP session created.")

        # --- Fetch Letta Tools (using the new async function) ---
        logger.info("Fetching tools from Letta (async)...")
        letta_tools = await fetch_all_tools_async() # Await the async function directly
        if not letta_tools:
             logger.error("Failed to fetch tools from Letta. Aborting sync cycle.")
             return # Stop this sync cycle if fetching failed

        # --- Update Tool Cache File ---
        # Write the fetched data to the cache immediately after fetching (asynchronously)
        await write_tool_cache(letta_tools)

        # Process the results for internal use in this function
        letta_tool_names = {tool["name"] for tool in letta_tools if 'name' in tool}
        letta_tools_dict = {tool["name"]: tool for tool in letta_tools if 'name' in tool}
        logger.info(f"Found {len(letta_tool_names)} tools in Letta (cache updated).")


        # --- Fetch MCP Servers ---
        logger.info("Fetching MCP servers from Letta API...")
        mcp_servers = []
        letta_url_base = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1').replace('http://', 'https://')
        if not letta_url_base.endswith('/v1'):
            letta_url_base = letta_url_base.rstrip('/') + '/v1'
        mcp_servers_url = f"{letta_url_base}/tools/mcp/servers"
        letta_api_key = os.getenv('LETTA_PASSWORD')
        mcp_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-BARE-PASSWORD": f"password {letta_api_key}"
        }
        try:
            async with http_session.get(mcp_servers_url, headers=mcp_headers) as response:
                response.raise_for_status()
                mcp_servers = await response.json()
                logger.info(f"Successfully fetched {len(mcp_servers)} MCP servers. Data: {json.dumps(mcp_servers, indent=2)}") # Log fetched data
                # Write MCP servers to their cache file
                await write_mcp_servers_cache(mcp_servers)
        except aiohttp.ClientError as e:
            logger.error(f"ClientError fetching MCP servers: {e}. Skipping MCP cache update.", exc_info=True) # Log full exception
        except Exception as e:
            logger.error(f"Unexpected error fetching MCP servers. Exception: {e}. Skipping MCP cache update.", exc_info=True) # Log full exception


        # --- Connect to Weaviate ---
        logger.info("Connecting to local Weaviate...")
        weaviate_conn = weaviate.connect_to_local(
            host="weaviate",  # Use the service name from docker-compose
            port=8080,
            grpc_port=50051,
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
            skip_init_checks=True  # Skip initialization checks
        )
        logger.info("Connected to Weaviate.")

        # --- Get/Create Schema ---
        collection = await get_or_create_tool_schema(weaviate_conn) # Pass connection

        # --- Fetch Weaviate Tools ---
        logger.info("Fetching tools from Weaviate...")
        weaviate_tools_dict = await get_weaviate_tools(weaviate_conn) # Pass connection
        weaviate_tool_names = set(weaviate_tools_dict.keys())
        logger.info(f"Found {len(weaviate_tool_names)} tools in Weaviate.")

        # --- Remove Obsolete Tools ---
        obsolete_tool_names = weaviate_tool_names - letta_tool_names
        if obsolete_tool_names:
            logger.info(f"Found {len(obsolete_tool_names)} obsolete tools to remove from Weaviate.")
            # Log the actual names identified as obsolete for debugging
            logger.debug(f"Obsolete tool names identified: {obsolete_tool_names}")
            removed_count = 0
            failed_removal_details = {}
            for tool_name in obsolete_tool_names:
                try:
                    name_filter = wq.Filter.by_property("name").equal(tool_name)
                    # Check if it exists before attempting delete, reduces noise (already wrapped)
                    query_result = await asyncio.to_thread(collection.query.fetch_objects, limit=1, filters=name_filter)
                    if query_result.objects:
                        # Log the filter being used
                        logger.debug(f"Attempting deletion for '{tool_name}' using filter: {name_filter}")
                        # Wrap sync call (already wrapped)
                        delete_result = await asyncio.to_thread(collection.data.delete_many, where=name_filter)
                        # Log the raw result object for detailed inspection
                        logger.debug(f"Raw delete_many result for '{tool_name}': {delete_result}")

                        # Check results more thoroughly
                        successful_deletes = getattr(delete_result, 'successful', 0)
                        failed_deletes = getattr(delete_result, 'failed', 0)
                        matches = getattr(delete_result, 'matches', 0) # How many objects matched the filter

                        if successful_deletes > 0:
                            removed_count += successful_deletes
                            logger.info(f"- Successfully removed {successful_deletes} object(s) for obsolete tool: {tool_name} (Matches: {matches})")
                        elif matches > 0 and failed_deletes > 0:
                             logger.warning(f"- Failed to remove obsolete tool: {tool_name} (Matches: {matches}, Failed: {failed_deletes}, Errors: {getattr(delete_result, 'errors', 'N/A')})")
                             failed_removal_details[tool_name] = getattr(delete_result, 'errors', 'Unknown failure')
                        elif matches == 0:
                             logger.info(f"- No objects matched filter for obsolete tool: {tool_name}. Assumed already removed.")
                        else: # Should not happen if matches > 0, but good to catch
                             logger.warning(f"- Unexpected delete result for obsolete tool: {tool_name} (Matches: {matches}, Successful: {successful_deletes}, Failed: {failed_deletes}, Errors: {getattr(delete_result, 'errors', 'N/A')})")
                             failed_removal_details[tool_name] = getattr(delete_result, 'errors', 'Unexpected result')

                    else:
                         logger.info(f"- Obsolete tool already removed or never existed: {tool_name}")

                except Exception as e_del:
                    logger.error(f"Error removing obsolete tool {tool_name}: {e_del}")
            logger.info(f"Finished removing obsolete tools. Total objects removed: {removed_count}")
            if failed_removal_details:
                logger.error(f"Failures occurred during obsolete tool removal: {failed_removal_details}")
        else:
            logger.info("No obsolete tools found in Weaviate.")

        # --- Add New Tools ---
        new_tool_names = letta_tool_names - weaviate_tool_names
        if new_tool_names:
            logger.info(f"Found {len(new_tool_names)} new tools to add to Weaviate.")
            added_count = 0
            failed_count = 0
            with collection.batch.dynamic() as batch:
                for tool_name in new_tool_names:
                    tool = letta_tools_dict[tool_name]
                    try:
                        properties = {
                            "tool_id": tool.get("id") or tool.get("tool_id", ""), # Use id or tool_id
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "source_type": tool.get("source_type", "python"),
                            "tool_type": tool.get("tool_type", "external_mcp"),
                            "tags": tool.get("tags", []),
                            "json_schema": json.dumps(tool.get("json_schema", {})) if tool.get("json_schema") else "",
                            # Add mcp_server_name if present in the fetched tool data
                            "mcp_server_name": tool.get("mcp_server_name")
                        }
                        # Filter out None values before adding
                        properties = {k: v for k, v in properties.items() if v is not None}

                        batch.add_object(properties=properties)
                        added_count += 1
                        logger.debug(f"- Added new tool to batch: {tool_name}")
                    except Exception as e_add:
                        failed_count += 1
                        logger.error(f"Error adding new tool {tool_name} to batch: {e_add}")
            logger.info(f"Finished adding new tools. Added: {added_count}, Failed: {failed_count}")
        else:
            logger.info("No new tools found in Letta to add to Weaviate.")

        logger.info("Tool synchronization process completed (Weaviate updates done).")

        # --- Backfill mcp_server_name for existing entries ---
        # This part remains the same, operating on Weaviate data
        logger.info("Checking for and backfilling missing 'mcp_server_name' in Weaviate...")
        backfilled_count = 0
        try:
            # Wrap sync call (already wrapped)
            all_weaviate_tools_raw = await asyncio.to_thread(collection.query.fetch_objects, limit=10000) # Fetch all with properties
            if all_weaviate_tools_raw and all_weaviate_tools_raw.objects:
                 # Rebuild tool_origin_map based on the full fetched list from fetch_all_tools_async run
                 tool_origin_map = {}
                 # Use the already fetched letta_tools list
                 for tool_data in letta_tools:
                     if tool_data.get("tool_type") == "external_mcp" and "mcp_server_name" in tool_data and "name" in tool_data:
                         tool_origin_map[tool_data["name"]] = tool_data["mcp_server_name"]

                 logger.info(f"Built origin map with {len(tool_origin_map)} MCP tools for backfill check.")

                 updates_to_make = []
                 for obj in all_weaviate_tools_raw.objects:
                     props = obj.properties
                     tool_name = props.get("name")
                     # Check if it's an MCP tool missing the server name in Weaviate
                     if props.get("tool_type") == "external_mcp" and "mcp_server_name" not in props:
                         # Check if we know its origin from our full fetch
                         origin_server = tool_origin_map.get(tool_name)
                         if origin_server:
                             updates_to_make.append({
                                 "uuid": obj.uuid,
                                 "properties": {"mcp_server_name": origin_server}
                             })
                             # logger.debug(f"Will backfill mcp_server_name='{origin_server}' for tool '{tool_name}' (UUID: {obj.uuid})")

                 if updates_to_make:
                     logger.info(f"Found {len(updates_to_make)} existing Weaviate entries missing 'mcp_server_name'. Attempting individual updates...")
                     # Cannot use batch update, must update individually
                     update_success_count = 0
                     update_fail_count = 0
                     for update_item in updates_to_make:
                         try:
                             # Wrap sync call
                             await asyncio.to_thread(
                                 collection.data.update,
                                 uuid=update_item["uuid"],
                                 properties=update_item["properties"]
                             )
                             update_success_count += 1
                         except Exception as e_update:
                             logger.error(f"Failed to backfill mcp_server_name for UUID {update_item['uuid']}: {e_update}")
                             update_fail_count += 1

                     backfilled_count = update_success_count
                     logger.info(f"Backfill complete. Successfully updated: {update_success_count}, Failed: {update_fail_count}")
                 else:
                     logger.info("No existing Weaviate entries found requiring 'mcp_server_name' backfill.")

        except Exception as e_backfill:
            logger.error(f"Error during Weaviate backfill check/update: {e_backfill}")


    except Exception as e:
        logger.error(f"Error during sync process: {e}", exc_info=True)
    finally:
        # Close Weaviate connection
        if weaviate_conn and weaviate_conn.is_connected():
            weaviate_conn.close()
            logger.info("Weaviate connection closed.")
        # Close HTTP session
        if http_session:
            await http_session.close()
            logger.info("Sync service HTTP session closed.")

def run_sync_job():
    """Synchronous wrapper to run the async sync_tools function."""
    logger.info("Scheduler triggered sync job.")
    try:
        # Ensure a new event loop is created for each run if needed,
        # or manage a single loop if running within an async context already.
        # asyncio.run() handles loop creation/closing.
        asyncio.run(sync_tools())
    except Exception as e:
        logger.error(f"Error running scheduled sync job: {e}", exc_info=True)

def main():
    """
    Main service function that schedules and runs tool synchronization.
    """
    # Load environment variables
    load_dotenv()

    # Get sync interval from environment (default to 5 minutes)
    sync_interval = int(os.getenv('SYNC_INTERVAL', 300))

    logger.info(f"Starting sync service (interval: {sync_interval} seconds)")

    # Optionally clear Weaviate collection on startup
    if os.getenv('CLEAR_WEAVIATE_ON_STARTUP', 'false').lower() == 'true':
        logger.warning("CLEAR_WEAVIATE_ON_STARTUP is true. Attempting to delete 'Tool' collection...")
        client = None
        deleted = False
        try:
            # Need to connect client temporarily to delete
            logger.info("Connecting to Weaviate for deletion...")
            client = weaviate.connect_to_local(
                host="weaviate",  # Use the service name from docker-compose
                port=8080,
                grpc_port=50051,
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
                skip_init_checks=True  # Skip initialization checks
            )
            logger.info("Connected. Checking if 'Tool' collection exists...")
            if client.collections.exists("Tool"):
                logger.info("'Tool' collection exists. Deleting...")
                client.collections.delete("Tool")
                # Add a small delay and re-check to be more certain
                time.sleep(2) # Make sure time is imported
                if not client.collections.exists("Tool"):
                    logger.info("'Tool' collection successfully deleted.")
                    deleted = True
                else:
                    logger.error("!!! Failed to verify 'Tool' collection deletion after attempt.")
            else:
                logger.info("'Tool' collection does not exist, skipping deletion.")
                deleted = True # Effectively cleared as it didn't exist
        except Exception as e:
            logger.error(f"Error during Weaviate collection deletion: {e}")
        finally:
            if client and client.is_connected():
                logger.info("Closing Weaviate connection after deletion attempt.")
                client.close()

        if deleted:
             logger.warning("Weaviate 'Tool' collection cleared or did not exist. Proceeding with initial sync.")
             # Also clear the cache files if Weaviate was cleared
             if os.path.exists(TOOL_CACHE_FILE_PATH):
                 try:
                     os.remove(TOOL_CACHE_FILE_PATH)
                     logger.info(f"Removed existing tool cache file: {TOOL_CACHE_FILE_PATH}")
                 except Exception as e_rem:
                     logger.error(f"Error removing tool cache file {TOOL_CACHE_FILE_PATH}: {e_rem}")
             if os.path.exists(MCP_SERVERS_CACHE_FILE_PATH):
                 try:
                     os.remove(MCP_SERVERS_CACHE_FILE_PATH)
                     logger.info(f"Removed existing MCP servers cache file: {MCP_SERVERS_CACHE_FILE_PATH}")
                 except Exception as e_rem:
                     logger.error(f"Error removing MCP servers cache file {MCP_SERVERS_CACHE_FILE_PATH}: {e_rem}")
        else:
             logger.error("Failed to clear Weaviate 'Tool' collection. Sync might use stale data.")

    # Schedule the synchronous wrapper function
    schedule.every(sync_interval).seconds.do(run_sync_job)

    # Do initial sync on startup using the wrapper
    logger.info("Performing initial sync...")
    run_sync_job() # Run the first sync via the wrapper

    # Keep running the scheduler
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()