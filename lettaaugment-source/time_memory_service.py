#!/usr/bin/env python
import requests
import time
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'   
)
logger = logging.getLogger(__name__)

load_dotenv()

class TimeMemoryService:
    def __init__(self):
        # Ensure the base URL includes /v1
        base_url = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1').replace('http://', 'https://')
        if not base_url.endswith('/v1'):
             # Add /v1 if it's missing, handling potential trailing slash
            base_url = base_url.rstrip('/') + '/v1'
        self.host = base_url

        api_key = os.getenv('LETTA_PASSWORD', 'lettaSecurePass123')
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-BARE-PASSWORD': f'password {api_key}'
        }
        self.block_id = None
        self.block_name = "watch"
        self.timezone = "America/Toronto"

    def list_agents(self):
        """Get list of all agents in the system"""
        try:
            response = requests.get(
                f"{self.host}/agents",
                headers=self.headers
            )
            response.raise_for_status()
            agents = response.json()
            return agents
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing agents: {e}")
            return []

    def create_time_block(self):
        """Create the initial time memory block"""
        # First check if watch block already exists for any agent
        agents = self.list_agents()
        for agent in agents:
            existing_block = self.get_block_by_label(agent['id'], 'watch')
            if existing_block:
                logger.info(f"Found existing watch block: {existing_block['id']}")
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
                f"{self.host}/blocks",
                json=block_data,
                headers=self.headers
            )
            response.raise_for_status()
            self.block_id = response.json()['id']
            logger.info(f"Created time block with ID: {self.block_id}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating time block: {e}")
            return False

    def get_block_by_label(self, agent_id, label):
        """Get a block by its label from an agent's core memory"""
        try:
            response = requests.get(
                f"{self.host}/agents/{agent_id}/core-memory/blocks/{label}",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                response.raise_for_status()
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting block by label '{label}' for agent {agent_id}: {e}")
            try:
                error_details = e.response.json()
                logger.error(f"Server response: {json.dumps(error_details, indent=2)}")
            except:
                pass
            return None

    def get_block_by_name(self, block_name):
        """Get a block by name"""
        try:
            response = requests.get(
                f"{self.host}/blocks",
                headers=self.headers
            )
            response.raise_for_status()
            blocks = response.json()
            for block in blocks:
                if block['name'] == block_name:
                    return block
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting block: {e}")
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
                    f"{self.host}/blocks",
                    json=block_data,
                    headers=self.headers
                )
                response.raise_for_status()
                block_id = response.json()['id']
                self.attach_block_to_agent(agent_id, agent_name, block_id)
                logger.info(f"Created agent card for {agent_name} ({agent_id})")
            else:
                logger.info(f"Agent card already exists for {agent_name} ({agent_id})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating agent card: {e}")

    def attach_block_to_agent(self, agent_id, agent_name, block_id=None):
        """Attach the time block to a specific agent"""
        try:
            # First check if block is already attached with the watch label
            existing_block = self.get_block_by_label(agent_id, 'watch')
            if existing_block:
                logger.info(f"Block already attached to agent {agent_name} with label 'watch'")
                return True

            block_id_to_attach = block_id or self.block_id
            # Construct URL carefully, ensuring no double slashes if self.host already has /v1
            attach_url = f"{self.host}/agents/{agent_id}/core-memory/blocks/attach/{block_id_to_attach}?label=watch"
            response = requests.patch(
                attach_url,
                json={},
                headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"Attached time block to agent: {agent_name} ({agent_id})")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error attaching block to agent {agent_name}: {e}")
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
                f"{self.host}/blocks/{self.block_id}",
                json=update_data,
                headers=self.headers
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating time block: {e}")

    def run(self):
        """Main service loop"""
        logger.info("Starting Time Memory Service...")

        # Get list of agents first
        logger.info("Fetching list of agents...")
        agents = self.list_agents()

        if not agents:
            logger.error("No agents found or failed to fetch agents")
            return

        # Try to find existing watch block or create new one
        if not self.create_time_block():
            logger.error("Failed to create/find watch block")
            return

        # Process all agents
        logger.info(f"Processing {len(agents)} agents...")
        for agent in agents:
            # Create agent card if needed
            self.create_agent_card(agent['id'], agent['name'])

            # Check if watch block is already attached by label
            existing_block = self.get_block_by_label(agent['id'], 'watch')
            if not existing_block:
                logger.info(f"Attaching watch block to {agent['name']}")
                self.attach_block_to_agent(agent['id'], agent['name'])
            else:
                logger.info(f"Watch block already attached to {agent['name']}")

        logger.info("\nTime Memory Service is running...")
        logger.info(f"Block ID: {self.block_id}")
        logger.info("Updating time every second...")

        while True:
            try:
                self.update_time_block()
                time.sleep(1)  # Update every second
            except KeyboardInterrupt:
                logger.info("\nShutting down Time Memory Service...")
                break
            except Exception as e:
                logger.error(f"Error in service loop: {e}")
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    service = TimeMemoryService()
    service.run()