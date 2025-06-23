#!/usr/bin/env python
import requests
import time
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv
import json

load_dotenv()

class TimeMemoryService:
    def __init__(self):
        self.host = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca').replace('http://', 'https://')
        api_key = os.getenv('LETTA_PASSWORD', 'lettaSecurePass123')
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-BARE-PASSWORD': f'password {api_key}',
            'Content-Type': 'application/json'
        }
        self.block_id = None
        self.block_name = "watch"
        self.timezone = "America/Toronto"
        
    def list_agents(self):
        """Get list of all agents in the system"""
        try:
            response = requests.get(
                f"{self.host}/v1/agents",
                headers=self.headers
            )
            response.raise_for_status()
            agents = response.json()
            return agents
        except requests.exceptions.RequestException as e:
            print(f"Error listing agents: {e}")
            return []

    def create_time_block(self):
        """Create the initial time memory block"""
        # First check if watch block already exists for any agent
        agents = self.list_agents()
        for agent in agents:
            existing_block = self.get_block_by_label(agent['id'], 'watch')
            if existing_block:
                print(f"Found existing watch block: {existing_block['id']}")
                self.block_id = existing_block['id']
                return True

        # Create new block if none found
        block_data = {
            "name": self.block_name,
            "label": "watch",  # Use watch as the label
            "value": json.dumps(self._get_time_info()),
            "metadata": {
                "type": "time_service",
                "version": "1.0",
                "timezone": self.timezone,
                "update_frequency": "1s"
            }
        }
        
        try:
            response = requests.post(
                f"{self.host}/v1/blocks",
                json=block_data,
                headers=self.headers
            )
            response.raise_for_status()
            self.block_id = response.json()['id']
            print(f"Created time block with ID: {self.block_id}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error creating time block: {e}")
            return False

    def get_block_by_label(self, agent_id, label):
        """Get a block by its label from an agent's core memory"""
        # This endpoint retrieves a block *if* it's attached to the agent
        try:
            response = requests.get(
                f"{self.host}/v1/agents/{agent_id}/core-memory/blocks/{label}",
                headers=self.headers
            )
            if response.status_code == 200:
                # Block exists and is attached
                return response.json()
            elif response.status_code == 404:
                # Block not found (or not attached with this label)
                return None
            else:
                # Other error
                response.raise_for_status()
                return None # Should not reach here if raise_for_status works
        except requests.exceptions.RequestException as e:
            # Handle connection errors or non-404 HTTP errors
            print(f"Error getting block by label '{label}' for agent {agent_id}: {e}")
            # Check if the error response has details
            try:
                error_details = e.response.json()
                print(f"Server response: {json.dumps(error_details, indent=2)}")
            except: # Ignore if response is not JSON or doesn't exist
                pass
            return None

    def get_block_by_name(self, block_name):
        """Get a block by name"""
        try:
            response = requests.get(
                f"{self.host}/v1/blocks",
                headers=self.headers
            )
            response.raise_for_status()
            blocks = response.json()
            for block in blocks:
                if block['name'] == block_name:
                    return block
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting block: {e}")
            return None

    def create_agent_card(self, agent_id, agent_name):
        """Create an agent card memory block"""
        block_data = {
            "name": f"agent_card_{agent_id}",
            "label": "agent_card",
            "value": json.dumps({
                "agent_id": agent_id,
                "agent_name": agent_name,
                "type": "agent_card",
                "created_at": datetime.now(pytz.UTC).isoformat()
            }),
            "metadata": {
                "type": "agent_card",
                "version": "1.0",
                "last_updated": datetime.now(pytz.UTC).isoformat()
            }
        }

        try:
            # Check if agent card block exists by label
            if not self.get_block_by_label(agent_id, 'agent_card'):
                response = requests.post(
                    f"{self.host}/v1/blocks",
                    json=block_data,
                    headers=self.headers
                )
                response.raise_for_status()
                block_id = response.json()['id']
                self.attach_block_to_agent(agent_id, agent_name, block_id)
                print(f"Created agent card for {agent_name} ({agent_id})")
            else:
                print(f"Agent card already exists for {agent_name} ({agent_id})")
        except requests.exceptions.RequestException as e:
            print(f"Error creating agent card: {e}")

    def attach_block_to_agent(self, agent_id, agent_name, block_id=None):
        """Attach the time block to a specific agent"""
        try:
            # First check if block is already attached with the watch label
            existing_block = self.get_block_by_label(agent_id, 'watch')
            if existing_block:
                print(f"Block already attached to agent {agent_name} with label 'watch'")
                return True

            block_id_to_attach = block_id or self.block_id
            attach_url = f"{self.host}/v1/agents/{agent_id}/core-memory/blocks/attach/{block_id_to_attach}?label=watch"
            response = requests.patch(
                attach_url,
                json={},
                headers=self.headers
            )
            response.raise_for_status()
            print(f"Attached time block to agent: {agent_name} ({agent_id})")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error attaching block to agent {agent_name}: {e}")
            return False

    def _get_time_info(self):
        """Generate current time information"""
        tz = pytz.timezone(self.timezone)
        now = datetime.now(tz)
        
        return {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": {
                "name": self.timezone,
                "offset": now.strftime("%z"),
                "abbreviation": now.strftime("%Z")
            },
            "unix_timestamp": int(now.timestamp())
        }

    def update_time_block(self):
        """Update the time memory block"""
        if not self.block_id:
            return
            
        update_data = {
            "value": json.dumps(self._get_time_info())
        }
        
        try:
            response = requests.patch(
                f"{self.host}/v1/blocks/{self.block_id}",
                json=update_data,
                headers=self.headers
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error updating time block: {e}")

    def run(self):
        """Main service loop"""
        print("Starting Time Memory Service...")
        
        # Get list of agents first
        print("Fetching list of agents...")
        agents = self.list_agents()
        
        if not agents:
            print("No agents found or failed to fetch agents")
            return

        # Try to find existing watch block or create new one
        if not self.create_time_block():
            print("Failed to create/find watch block")
            return

        # Process all agents
        print(f"Processing {len(agents)} agents...")
        for agent in agents:
            # Create agent card if needed
            self.create_agent_card(agent['id'], agent['name'])
            
            # Check if watch block is already attached by label
            existing_block = self.get_block_by_label(agent['id'], 'watch')
            if not existing_block:
                print(f"Attaching watch block to {agent['name']}")
                self.attach_block_to_agent(agent['id'], agent['name'])
            else:
                print(f"Watch block already attached to {agent['name']}")

        print("\nTime Memory Service is running...")
        print(f"Block ID: {self.block_id}")
        print("Updating time every second...")
        
        while True:
            try:
                self.update_time_block()
                time.sleep(1)  # Update every second
            except KeyboardInterrupt:
                print("\nShutting down Time Memory Service...")
                break
            except Exception as e:
                print(f"Error in service loop: {e}")
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    service = TimeMemoryService()
    service.run()