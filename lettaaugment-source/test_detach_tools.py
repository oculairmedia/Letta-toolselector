import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detach_mcp_tools import detach_mcp_tools

agent_id = "agent-d5d91a6a-cc16-47dd-97be-07101cdbd49d"
print(f"Testing detach_mcp_tools with agent ID: {agent_id}")

result = detach_mcp_tools(agent_id=agent_id, request_heartbeat=True)
print("\nResult:")
print(result)