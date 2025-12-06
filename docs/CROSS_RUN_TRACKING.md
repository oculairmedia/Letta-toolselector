# Cross-Run Tracking for Matrix Bridge

## Problem Statement

When a Letta agent calls `find_tools`, tools are attached server-side but aren't available until a new run starts. This is due to Letta's architecture where tools are loaded into the LLM context at the **start** of each run.

Our current solution sends a system message to trigger a new run with the tools available. However, this breaks conversation tracking:

```
Run A: User asks "list my resumes" 
       → Agent calls find_tools 
       → Tools attached 
       → Run A ends (exit_loop rule)

Run B: System trigger message 
       → Agent has resume tools 
       → Agent responds with resume list
       → Response is in Run B

Problem: Matrix/chat apps tracking Run A never see the response from Run B
```

## Solution Overview

Based on Letta's background mode patterns (used for HITL approval), the Matrix bridge should track conversations across multiple runs. This is the same pattern Letta uses internally - even their approval responses create new runs.

---

## Architecture

### Conversation State Model

```python
conversation_state = {
    "matrix_room_id": "!roomId:matrix.oculair.ca",
    "matrix_event_id": "$originalEventId",      # Original user message
    "agent_id": "agent-597b5756-...",
    "runs": [
        {
            "run_id": "run-abc123",
            "seq_id": 0,
            "status": "completed",
            "triggered_by": "user_message"
        },
        {
            "run_id": "run-def456", 
            "seq_id": 0,
            "status": "completed",
            "triggered_by": "system_trigger",
            "parent_run_id": "run-abc123"
        }
    ],
    "active_run_id": "run-def456",
    "original_run_id": "run-abc123",
    "status": "completed"
}
```

### Message Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CROSS-RUN TRACKING FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  USER                    MATRIX BRIDGE              LETTA         TOOL SEL   │
│   │                           │                       │               │      │
│   │  1. Send message          │                       │               │      │
│   │ ─────────────────────────>│                       │               │      │
│   │                           │                       │               │      │
│   │                           │  2. Forward to agent  │               │      │
│   │                           │ ─────────────────────>│               │      │
│   │                           │                       │               │      │
│   │                           │  3. Return run_id     │               │      │
│   │                           │ <─────────────────────│               │      │
│   │                           │                       │               │      │
│   │                           │  [Store run_id as     │               │      │
│   │                           │   original_run_id]    │               │      │
│   │                           │                       │               │      │
│   │                           │                       │  4. Agent     │      │
│   │                           │                       │  calls        │      │
│   │                           │                       │  find_tools   │      │
│   │                           │                       │ ─────────────>│      │
│   │                           │                       │               │      │
│   │                           │                       │  5. Tools     │      │
│   │                           │                       │  attached     │      │
│   │                           │                       │ <─────────────│      │
│   │                           │                       │               │      │
│   │                           │                       │  6. Run A     │      │
│   │                           │                       │  ends         │      │
│   │                           │                       │               │      │
│   │                           │  7. Webhook:          │               │      │
│   │                           │  "run_triggered"      │               │      │
│   │                           │ <─────────────────────────────────────│      │
│   │                           │                       │               │      │
│   │                           │  [Add new run_id      │  8. System    │      │
│   │                           │   to conversation]    │  trigger      │      │
│   │                           │                       │ <─────────────│      │
│   │                           │                       │               │      │
│   │                           │                       │  9. Run B     │      │
│   │                           │                       │  starts,      │      │
│   │                           │                       │  tools        │      │
│   │                           │                       │  available    │      │
│   │                           │                       │               │      │
│   │                           │  10. Monitor Run B    │               │      │
│   │                           │ ─────────────────────>│               │      │
│   │                           │                       │               │      │
│   │                           │  11. Agent response   │               │      │
│   │                           │ <─────────────────────│               │      │
│   │                           │                       │               │      │
│   │  12. Post response to     │                       │               │      │
│   │  original thread          │                       │               │      │
│   │ <─────────────────────────│                       │               │      │
│   │                           │                       │               │      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Components

### 1. Tool Selector: Webhook Notification

After triggering a new run, the Tool Selector notifies the Matrix bridge:

**Environment Variable:**
```bash
MATRIX_BRIDGE_WEBHOOK_URL=http://matrix-bridge:8080/webhook/tool-selector
```

**Webhook Payload:**
```json
{
    "event": "run_triggered",
    "agent_id": "agent-597b5756-2915-4560-ba6b-91005f085166",
    "trigger_type": "tool_attachment",
    "tools_attached": [
        "get_resume",
        "download_resume_pdf",
        "update_experience"
    ],
    "query": "list my resumes",
    "timestamp": "2025-12-06T22:15:00Z"
}
```

**Code Addition to `api_server.py`:**
```python
MATRIX_BRIDGE_WEBHOOK_URL = os.getenv('MATRIX_BRIDGE_WEBHOOK_URL')

async def _send_trigger_message(agent_id: str, tool_names: list, query: str = None):
    # ... existing trigger code ...
    
    async with http_session.post(messages_url, headers=HEADERS, json=payload) as response:
        if response.status in (200, 201, 202):
            logger.info(f"[BACKGROUND] Trigger completed for {agent_id}")
            
            # Notify Matrix bridge of new run
            if MATRIX_BRIDGE_WEBHOOK_URL:
                await _notify_matrix_bridge(agent_id, tool_names, query)
            return

async def _notify_matrix_bridge(agent_id: str, tool_names: list, query: str = None):
    """Notify Matrix bridge that a new run was triggered"""
    webhook_payload = {
        "event": "run_triggered",
        "agent_id": agent_id,
        "trigger_type": "tool_attachment",
        "tools_attached": tool_names,
        "query": query,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        async with http_session.post(
            MATRIX_BRIDGE_WEBHOOK_URL,
            json=webhook_payload,
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status == 200:
                logger.info(f"Notified Matrix bridge of run trigger for {agent_id}")
            else:
                logger.warning(f"Matrix bridge webhook returned {resp.status}")
    except Exception as e:
        logger.warning(f"Failed to notify Matrix bridge: {e}")
```

---

### 2. Matrix Bridge: Conversation Tracker

**ConversationTracker Class:**
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
import asyncio

@dataclass
class RunInfo:
    run_id: str
    triggered_by: str  # "user_message" | "system_trigger"
    timestamp: str
    status: str = "active"
    parent_run_id: Optional[str] = None

@dataclass 
class ConversationState:
    matrix_room_id: str
    matrix_event_id: str
    agent_id: str
    runs: List[RunInfo] = field(default_factory=list)
    status: str = "pending"  # pending | active | completed
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def original_run_id(self) -> Optional[str]:
        return self.runs[0].run_id if self.runs else None
    
    @property
    def active_run_id(self) -> Optional[str]:
        active = [r for r in self.runs if r.status == "active"]
        return active[-1].run_id if active else None
    
    def add_run(self, run_id: str, triggered_by: str, parent_run_id: str = None):
        self.runs.append(RunInfo(
            run_id=run_id,
            triggered_by=triggered_by,
            timestamp=datetime.utcnow().isoformat(),
            parent_run_id=parent_run_id
        ))
        self.status = "active"
    
    def complete_run(self, run_id: str):
        for run in self.runs:
            if run.run_id == run_id:
                run.status = "completed"
                break


class ConversationTracker:
    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self.agent_conversations: Dict[str, str] = {}  # agent_id -> matrix_event_id
    
    def start_conversation(
        self, 
        matrix_event_id: str, 
        matrix_room_id: str, 
        agent_id: str
    ) -> ConversationState:
        """Start tracking a new conversation"""
        conv = ConversationState(
            matrix_room_id=matrix_room_id,
            matrix_event_id=matrix_event_id,
            agent_id=agent_id
        )
        self.conversations[matrix_event_id] = conv
        self.agent_conversations[agent_id] = matrix_event_id
        return conv
    
    def add_run(
        self, 
        matrix_event_id: str, 
        run_id: str, 
        triggered_by: str
    ):
        """Add a run to an existing conversation"""
        conv = self.conversations.get(matrix_event_id)
        if conv:
            parent_run = conv.active_run_id
            conv.add_run(run_id, triggered_by, parent_run)
            return True
        return False
    
    def get_conversation(self, matrix_event_id: str) -> Optional[ConversationState]:
        """Get conversation by Matrix event ID"""
        return self.conversations.get(matrix_event_id)
    
    def get_conversation_by_agent(self, agent_id: str) -> Optional[ConversationState]:
        """Get active conversation for an agent"""
        event_id = self.agent_conversations.get(agent_id)
        if event_id:
            conv = self.conversations.get(event_id)
            if conv and conv.status == "active":
                return conv
        return None
    
    def complete_conversation(self, matrix_event_id: str):
        """Mark conversation as completed"""
        conv = self.conversations.get(matrix_event_id)
        if conv:
            conv.status = "completed"
            # Clean up agent mapping
            if conv.agent_id in self.agent_conversations:
                del self.agent_conversations[conv.agent_id]
    
    def cleanup_old_conversations(self, max_age_seconds: int = 300):
        """Remove conversations older than max_age"""
        now = datetime.utcnow()
        to_remove = []
        
        for event_id, conv in self.conversations.items():
            created = datetime.fromisoformat(conv.created_at.replace('Z', '+00:00'))
            age = (now - created.replace(tzinfo=None)).total_seconds()
            if age > max_age_seconds:
                to_remove.append(event_id)
        
        for event_id in to_remove:
            conv = self.conversations.pop(event_id, None)
            if conv and conv.agent_id in self.agent_conversations:
                del self.agent_conversations[conv.agent_id]
```

---

### 3. Matrix Bridge: Webhook Handler

```python
from quart import Quart, request, jsonify

app = Quart(__name__)
conversation_tracker = ConversationTracker()

@app.route("/webhook/tool-selector", methods=["POST"])
async def handle_tool_selector_webhook():
    """Handle webhooks from Tool Selector"""
    data = await request.get_json()
    
    if data.get("event") == "run_triggered":
        agent_id = data["agent_id"]
        
        # Find the active conversation for this agent
        conv = conversation_tracker.get_conversation_by_agent(agent_id)
        
        if conv:
            logger.info(
                f"Tool attachment triggered new run for agent {agent_id}, "
                f"conversation {conv.matrix_event_id}"
            )
            
            # Mark previous run as potentially complete
            # (the new run will be added when we detect it)
            
            # Start monitoring for the agent's response
            asyncio.create_task(
                monitor_agent_response(conv, data.get("query"))
            )
            
            return jsonify({"status": "ok", "tracking": True})
        else:
            logger.warning(
                f"Received run_triggered for agent {agent_id} "
                f"but no active conversation found"
            )
            return jsonify({"status": "ok", "tracking": False})
    
    return jsonify({"status": "unknown_event"})
```

---

### 4. Matrix Bridge: Response Monitoring

```python
async def monitor_agent_response(
    conv: ConversationState, 
    original_query: str = None,
    max_wait: int = 60,
    poll_interval: int = 2
):
    """Monitor for agent response after tool attachment"""
    
    agent_id = conv.agent_id
    elapsed = 0
    last_message_count = 0
    
    while elapsed < max_wait:
        try:
            # Get recent messages for the agent
            messages = await letta_client.agents.messages.list(
                agent_id=agent_id,
                limit=10
            )
            
            # Look for new assistant messages
            for msg in messages:
                if msg.message_type == "assistant_message":
                    # Check if this is a new message (after our trigger)
                    msg_time = datetime.fromisoformat(
                        msg.date.replace('Z', '+00:00')
                    )
                    conv_time = datetime.fromisoformat(
                        conv.created_at.replace('Z', '+00:00')
                    )
                    
                    if msg_time > conv_time:
                        # New response found - post to Matrix
                        await post_response_to_matrix(conv, msg.content)
                        
                        # Mark conversation complete
                        conversation_tracker.complete_conversation(
                            conv.matrix_event_id
                        )
                        return
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
        except Exception as e:
            logger.error(f"Error monitoring agent response: {e}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
    
    # Timeout - notify user
    logger.warning(
        f"Timeout waiting for agent response, "
        f"conversation {conv.matrix_event_id}"
    )
    await post_response_to_matrix(
        conv, 
        "I'm still processing your request. Please wait or try again."
    )


async def post_response_to_matrix(conv: ConversationState, content: str):
    """Post agent response back to the original Matrix thread"""
    
    # Use Matrix API to send message as reply to original
    await matrix_client.room_send(
        room_id=conv.matrix_room_id,
        message_type="m.room.message",
        content={
            "msgtype": "m.text",
            "body": content,
            "m.relates_to": {
                "m.in_reply_to": {
                    "event_id": conv.matrix_event_id
                }
            }
        }
    )
```

---

### 5. Matrix Bridge: Initial Message Handling

```python
async def handle_user_message(
    room_id: str,
    event_id: str, 
    sender: str,
    content: str,
    agent_id: str
):
    """Handle incoming user message and track the conversation"""
    
    # Start tracking this conversation
    conv = conversation_tracker.start_conversation(
        matrix_event_id=event_id,
        matrix_room_id=room_id,
        agent_id=agent_id
    )
    
    # Send message to Letta agent
    response = await letta_client.agents.messages.create(
        agent_id=agent_id,
        messages=[{
            "role": "user",
            "content": content
        }]
    )
    
    # Extract and store the run_id
    if response.messages:
        run_id = response.messages[0].run_id
        conv.add_run(run_id, "user_message")
        logger.info(f"Started conversation {event_id} with run {run_id}")
    
    # Check for immediate response (no tool attachment needed)
    for msg in response.messages:
        if msg.message_type == "assistant_message":
            await post_response_to_matrix(conv, msg.content)
            conversation_tracker.complete_conversation(event_id)
            return
    
    # If no immediate response, the agent may have called find_tools
    # The webhook handler will pick up the continuation
    logger.info(
        f"No immediate response for {event_id}, "
        f"waiting for potential tool attachment"
    )
```

---

## Alternative: Background Mode with Resumable Streams

For more robust tracking, use Letta's background mode:

```python
async def handle_user_message_background(
    room_id: str,
    event_id: str,
    content: str,
    agent_id: str
):
    """Handle message with background mode for resilience"""
    
    conv = conversation_tracker.start_conversation(
        matrix_event_id=event_id,
        matrix_room_id=room_id,
        agent_id=agent_id
    )
    
    # Use background mode
    stream = letta_client.agents.messages.create(
        agent_id=agent_id,
        messages=[{"role": "user", "content": content}],
        streaming=True,
        background=True
    )
    
    run_id = None
    last_seq_id = 0
    
    for chunk in stream:
        # Track run info
        if hasattr(chunk, "run_id"):
            if chunk.run_id != run_id:
                # New run detected
                run_id = chunk.run_id
                conv.add_run(run_id, "stream_chunk")
            last_seq_id = chunk.seq_id
        
        # Handle different message types
        if chunk.message_type == "assistant_message":
            await post_response_to_matrix(conv, chunk.content)
        
        elif chunk.message_type == "system_message":
            if "[SYSTEM] New tools attached" in chunk.content:
                # Tool attachment happened
                # Continue streaming - next messages will have tools
                logger.info(f"Tools attached in conversation {event_id}")
    
    # Stream complete
    conversation_tracker.complete_conversation(event_id)
```

---

## Configuration

### Tool Selector Environment Variables

```bash
# Add to compose.yaml for api-server
MATRIX_BRIDGE_WEBHOOK_URL=http://matrix-bridge:8080/webhook/tool-selector
```

### Matrix Bridge Environment Variables

```bash
# Letta API connection
LETTA_API_URL=http://192.168.50.90:8283/v1
LETTA_API_KEY=your_api_key

# Conversation tracking
CONVERSATION_TIMEOUT_SECONDS=300
RESPONSE_POLL_INTERVAL=2
MAX_RESPONSE_WAIT=60
```

---

## Summary

| Component | Responsibility |
|-----------|---------------|
| **Tool Selector** | Send webhook after triggering new run |
| **Matrix Bridge** | Track conversation state across runs |
| **Matrix Bridge** | Handle webhook notifications |
| **Matrix Bridge** | Monitor for responses and post to Matrix |
| **Matrix Bridge** | Clean up stale conversations |

This architecture ensures that:
1. User messages are tracked from the original Matrix event
2. Tool attachments trigger webhook notifications
3. Responses from any subsequent run are posted to the original thread
4. The user sees a seamless conversation despite multiple Letta runs
