from fastapi import APIRouter, Depends, HTTPException, Query, Path, Response
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, Literal
import time
import logging
import json
import csv
from io import StringIO

from app.services.ldts_client import LDTSClient
from app.models.tools import (
    Tool, ExtendedTool, ToolDetailResponse, ToolBrowseRequest, ToolBrowseResponse,
    ExportRequest, RefreshResponse, CategoryListResponse, SourceListResponse, ErrorResponse
)
from app.models.search import SearchRequest
from config.settings import settings

router = APIRouter(tags=["tools"])
logger = logging.getLogger(__name__)

async def get_ldts_client() -> LDTSClient:
    """Dependency to get LDTS client."""
    from app.main import ldts_client
    if ldts_client is None:
        raise HTTPException(status_code=503, detail="LDTS client not initialized")
    return ldts_client

async def fetch_all_tools(ldts_client: LDTSClient) -> List[Dict[str, Any]]:
    """Fetch all tools from LDTS API server."""
    try:
        if not ldts_client.session:
            raise RuntimeError("LDTS client not initialized")
        
        # Call the LDTS API server's /api/v1/tools endpoint
        async with ldts_client.session.get(f"{ldts_client.api_url}/api/v1/tools") as response:
            response.raise_for_status()
            tools_data = await response.json()
            
            # Handle both direct array response and wrapped response
            if isinstance(tools_data, list):
                return tools_data
            elif isinstance(tools_data, dict) and "tools" in tools_data:
                return tools_data["tools"]
            else:
                logger.warning(f"Unexpected tools data format: {type(tools_data)}")
                return []
                
    except Exception as e:
        logger.error(f"Failed to fetch tools: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch tools: {str(e)}"
        )

def convert_to_extended_tool(tool_data: Dict[str, Any]) -> ExtendedTool:
    """Convert tool data dictionary to ExtendedTool model."""
    return ExtendedTool(
        id=tool_data.get("id", tool_data.get("tool_id", "unknown")),
        name=tool_data.get("name", "Unknown Tool"),
        description=tool_data.get("description", ""),
        source=tool_data.get("source", "unknown"),
        category=tool_data.get("category"),
        tags=tool_data.get("tags", []),
        mcp_server_name=tool_data.get("mcp_server_name"),
        last_updated=tool_data.get("last_updated"),
        registered_in_letta=tool_data.get("registered_in_letta"),
        embedding_id=tool_data.get("embedding_id")
    )

def filter_tools(tools: List[ExtendedTool], params: ToolBrowseRequest) -> List[ExtendedTool]:
    """Apply filters to tools list."""
    filtered = tools
    
    # Search filter
    if params.search:
        search_lower = params.search.lower()
        filtered = [
            tool for tool in filtered
            if (search_lower in tool.name.lower() or
                search_lower in tool.description.lower())
        ]
    
    # Category filter
    if params.category:
        filtered = [
            tool for tool in filtered
            if tool.category == params.category
        ]
    
    # Source filter
    if params.source:
        filtered = [
            tool for tool in filtered
            if tool.source == params.source
        ]
    
    # MCP server filter
    if params.mcp_server:
        filtered = [
            tool for tool in filtered
            if tool.mcp_server_name == params.mcp_server
        ]
    
    return filtered

def sort_tools(tools: List[ExtendedTool], sort_by: str, order: str) -> List[ExtendedTool]:
    """Sort tools by specified field and order."""
    reverse = (order == 'desc')
    
    if sort_by == 'name':
        return sorted(tools, key=lambda t: t.name.lower(), reverse=reverse)
    elif sort_by == 'category':
        return sorted(tools, key=lambda t: t.category or "", reverse=reverse)
    elif sort_by == 'updated':
        return sorted(tools, key=lambda t: t.last_updated or "", reverse=reverse)
    else:  # Default to name
        return sorted(tools, key=lambda t: t.name.lower(), reverse=reverse)

@router.get("/tools/browse", response_model=ToolBrowseResponse)
async def browse_tools(
    page: int = Query(0, ge=0, description="Page number (0-based)"),
    limit: int = Query(25, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    source: Optional[str] = Query(None, description="Filter by source"),
    mcp_server: Optional[str] = Query(None, description="Filter by MCP server"),
    sort: Literal['name', 'category', 'updated', 'relevance'] = Query('name', description="Sort field"),
    order: Literal['asc', 'desc'] = Query('asc', description="Sort order"),
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> ToolBrowseResponse:
    """Browse tools with pagination and filtering."""
    start_time = time.time()
    
    try:
        logger.info(f"Browsing tools: page={page}, limit={limit}, search='{search}'")
        
        # Create request object
        params = ToolBrowseRequest(
            page=page,
            limit=limit,
            search=search,
            category=category,
            source=source,
            mcp_server=mcp_server,
            sort=sort,
            order=order
        )
        
        # Fetch all tools
        tools_data = await fetch_all_tools(ldts_client)
        
        # Convert to ExtendedTool objects
        tools = [convert_to_extended_tool(tool_data) for tool_data in tools_data]
        
        # Apply filters
        filtered_tools = filter_tools(tools, params)
        
        # Sort tools
        sorted_tools = sort_tools(filtered_tools, params.sort, params.order)
        
        # Apply pagination
        total = len(sorted_tools)
        start_idx = page * limit
        end_idx = start_idx + limit
        paginated_tools = sorted_tools[start_idx:end_idx]
        
        has_more = end_idx < total
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Browse completed: {len(paginated_tools)} tools in {processing_time:.2f}ms")
        
        return ToolBrowseResponse(
            tools=paginated_tools,
            total=total,
            page=page,
            limit=limit,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Browse tools failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Browse tools failed: {str(e)}"
        )

@router.get("/tools/{tool_id}/detail", response_model=ToolDetailResponse)
async def get_tool_detail(
    tool_id: str = Path(..., description="Tool ID"),
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> ToolDetailResponse:
    """Get detailed information about a specific tool."""
    try:
        logger.info(f"Getting tool details for: {tool_id}")
        
        # Fetch all tools and find the specific one
        tools_data = await fetch_all_tools(ldts_client)
        
        tool_data = None
        for tool in tools_data:
            if tool.get("id") == tool_id or tool.get("tool_id") == tool_id:
                tool_data = tool
                break
        
        if not tool_data:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")
        
        # Convert to detailed response
        detail_response = ToolDetailResponse(
            id=tool_data.get("id", tool_data.get("tool_id", tool_id)),
            name=tool_data.get("name", "Unknown Tool"),
            description=tool_data.get("description", ""),
            source=tool_data.get("source", "unknown"),
            category=tool_data.get("category"),
            tags=tool_data.get("tags", []),
            mcp_server_name=tool_data.get("mcp_server_name"),
            last_updated=tool_data.get("last_updated"),
            registered_in_letta=tool_data.get("registered_in_letta"),
            embedding_id=tool_data.get("embedding_id"),
            json_schema=tool_data.get("json_schema"),
            parameters=tool_data.get("parameters"),
            metadata=tool_data.get("metadata")
        )
        
        logger.info(f"Tool details retrieved for: {tool_id}")
        return detail_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get tool detail failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Get tool detail failed: {str(e)}"
        )

@router.get("/tools/categories", response_model=CategoryListResponse)
async def get_tool_categories(
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> CategoryListResponse:
    """Get list of all available tool categories."""
    try:
        logger.info("Getting tool categories")
        
        # Fetch all tools
        tools_data = await fetch_all_tools(ldts_client)
        
        # Extract unique categories
        categories = set()
        for tool in tools_data:
            category = tool.get("category")
            if category:
                categories.add(category)
        
        sorted_categories = sorted(list(categories))
        
        logger.info(f"Found {len(sorted_categories)} categories")
        return CategoryListResponse(
            categories=sorted_categories,
            count=len(sorted_categories)
        )
        
    except Exception as e:
        logger.error(f"Get categories failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Get categories failed: {str(e)}"
        )

@router.get("/tools/sources", response_model=SourceListResponse)
async def get_tool_sources(
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> SourceListResponse:
    """Get list of all available tool sources."""
    try:
        logger.info("Getting tool sources")
        
        # Fetch all tools
        tools_data = await fetch_all_tools(ldts_client)
        
        # Extract unique sources
        sources = set()
        for tool in tools_data:
            source = tool.get("source")
            if source:
                sources.add(source)
        
        sorted_sources = sorted(list(sources))
        
        logger.info(f"Found {len(sorted_sources)} sources")
        return SourceListResponse(
            sources=sorted_sources,
            count=len(sorted_sources)
        )
        
    except Exception as e:
        logger.error(f"Get sources failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Get sources failed: {str(e)}"
        )

@router.get("/tools/export")
async def export_tools(
    format: Literal['json', 'csv'] = Query('json', description="Export format"),
    search: Optional[str] = Query(None, description="Search filter"),
    category: Optional[str] = Query(None, description="Category filter"),
    source: Optional[str] = Query(None, description="Source filter"),
    mcp_server: Optional[str] = Query(None, description="MCP server filter"),
    ldts_client: LDTSClient = Depends(get_ldts_client)
):
    """Export tools data in specified format."""
    try:
        logger.info(f"Exporting tools in {format} format")
        
        # Create filter params (without using ToolBrowseRequest to avoid limit validation)
        # We'll handle filtering manually for export
        pass
        
        # Fetch and filter tools manually
        tools_data = await fetch_all_tools(ldts_client)
        tools = [convert_to_extended_tool(tool_data) for tool_data in tools_data]
        
        # Apply filters manually for export
        filtered_tools = tools
        if search:
            search_lower = search.lower()
            filtered_tools = [
                tool for tool in filtered_tools
                if (search_lower in tool.name.lower() or
                    search_lower in tool.description.lower())
            ]
        
        if category:
            filtered_tools = [
                tool for tool in filtered_tools
                if tool.category == category
            ]
        
        if source:
            filtered_tools = [
                tool for tool in filtered_tools
                if tool.source == source
            ]
        
        if mcp_server:
            filtered_tools = [
                tool for tool in filtered_tools
                if tool.mcp_server_name == mcp_server
            ]
        
        if format == 'json':
            # Export as JSON
            export_data = [tool.dict() for tool in filtered_tools]
            content = json.dumps(export_data, indent=2, default=str)
            
            return Response(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=tools_export_{int(time.time())}.json"
                }
            )
        
        elif format == 'csv':
            # Export as CSV
            output = StringIO()
            if filtered_tools:
                fieldnames = [
                    'id', 'name', 'description', 'source', 'category',
                    'mcp_server_name', 'last_updated', 'registered_in_letta'
                ]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for tool in filtered_tools:
                    writer.writerow({
                        'id': tool.id,
                        'name': tool.name,
                        'description': tool.description,
                        'source': tool.source,
                        'category': tool.category or '',
                        'mcp_server_name': tool.mcp_server_name or '',
                        'last_updated': tool.last_updated or '',
                        'registered_in_letta': tool.registered_in_letta or False
                    })
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=tools_export_{int(time.time())}.csv"
                }
            )
        
    except Exception as e:
        logger.error(f"Export tools failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Export tools failed: {str(e)}"
        )

@router.get("/tools", response_model=List[Tool])
async def get_all_tools(
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> List[Tool]:
    """Get all tools (basic list without pagination)."""
    try:
        logger.info("Getting all tools")
        
        # Fetch all tools
        tools_data = await fetch_all_tools(ldts_client)
        
        # Convert to basic Tool objects
        tools = []
        for tool_data in tools_data:
            tool = Tool(
                id=tool_data.get("id", tool_data.get("tool_id", "unknown")),
                name=tool_data.get("name", "Unknown Tool"),
                description=tool_data.get("description", ""),
                source=tool_data.get("source", "unknown"),
                category=tool_data.get("category"),
                tags=tool_data.get("tags", [])
            )
            tools.append(tool)
        
        logger.info(f"Returning {len(tools)} tools")
        return tools
        
    except Exception as e:
        logger.error(f"Get all tools failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Get all tools failed: {str(e)}"
        )

@router.post("/tools/refresh", response_model=RefreshResponse)
async def refresh_tool_index(
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> RefreshResponse:
    """Refresh the tool index cache."""
    try:
        logger.info("Refreshing tool index")
        
        if not ldts_client.session:
            raise RuntimeError("LDTS client not initialized")
        
        # Call the LDTS API server's refresh endpoint
        async with ldts_client.session.post(f"{ldts_client.api_url}/api/v1/tools/refresh") as response:
            response.raise_for_status()
            refresh_result = await response.json()
        
        # Extract information from response
        success = refresh_result.get("success", True)
        tools_count = refresh_result.get("tools_refreshed", 0)
        
        logger.info(f"Tool index refresh completed: {tools_count} tools")
        
        return RefreshResponse(
            success=success,
            tools_refreshed=tools_count,
            message=f"Successfully refreshed {tools_count} tools"
        )
        
    except Exception as e:
        logger.error(f"Refresh tool index failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Refresh tool index failed: {str(e)}"
        )

@router.post("/tools/search")
async def search_tools(
    request: Dict[str, Any],
    ldts_client: LDTSClient = Depends(get_ldts_client)
):
    """Search for tools using LDTS API."""
    try:
        logger.info(f"Tool search request: {request}")
        
        if not ldts_client.session:
            raise HTTPException(status_code=503, detail="LDTS client not initialized")
        
        # Forward the request to LDTS API server's tools search endpoint
        async with ldts_client.session.post(
            f"{ldts_client.api_url}/api/v1/tools/search",
            json=request
        ) as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
                media_type="application/json"
            )
            
    except Exception as e:
        logger.error(f"Tool search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool search failed: {str(e)}")

@router.post("/tools/search/rerank")
async def search_with_reranking(
    request: Dict[str, Any],
    ldts_client: LDTSClient = Depends(get_ldts_client)
):
    """Search with reranking using the API server's dedicated rerank endpoint."""
    try:
        logger.info(f"Search with reranking request: {request}")
        
        if not ldts_client.session:
            raise HTTPException(status_code=503, detail="LDTS client not initialized")
        
        # Forward the request directly to the API server's rerank endpoint
        async with ldts_client.session.post(
            f"{ldts_client.api_url}/api/v1/tools/search/rerank",
            json=request
        ) as response:
            if response.status == 200:
                # Parse the nested response and convert to direct array format
                data = await response.json()
                if data.get("success") and data.get("data"):
                    results = data["data"].get("results", [])
                    # Convert to direct array format matching regular search
                    converted_results = []
                    for item in results:
                        tool = item.get("tool", {})
                        converted_item = {
                            "tool_id": tool.get("id", ""),
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "source_type": tool.get("source", "unknown"),
                            "mcp_server_name": tool.get("category", ""),
                            "tags": tool.get("tags", []),
                            "score": item.get("score", 0),
                            "distance": 1.0 - item.get("score", 0),  # Convert score to distance
                            "reasoning": item.get("reasoning", ""),
                            "rank": item.get("rank", 0),
                            # Add rerank-specific metadata
                            "rerank_score": item.get("score", 0),
                            "reranked": True
                        }
                        converted_results.append(converted_item)
                    
                    return JSONResponse(content=converted_results)
                else:
                    # Fallback to original response if parsing fails
                    content = await response.read()
                    return Response(
                        content=content,
                        status_code=response.status,
                        headers=dict(response.headers),
                        media_type="application/json"
                    )
            else:
                # Handle non-200 responses
                content = await response.read()
                return Response(
                    content=content,
                    status_code=response.status,
                    headers=dict(response.headers),
                    media_type="application/json"
                )
            
    except Exception as e:
        logger.error(f"Tool search with reranking failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool search with reranking failed: {str(e)}")