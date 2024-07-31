import asyncio

from openai import AsyncOpenAI
from config import OPENAI_API_KEY_DICT
from Modules.openai_assistant.event_handler import EventHandler
import boto3
import json
OPENAI_API_KEY = OPENAI_API_KEY_DICT['QG']

class AssistantService:
    client: AsyncOpenAI
    assistant_id: str

    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.assistant_id = ""

    async def get_assistant(self):
        assistant = await self.client.beta.assistants.retrieve(self.assistant_id)
        return assistant

    async def create_thread(self):
        thread = await self.client.beta.threads.create()
        return thread

    async def retrieve_thread(self, thread_id: str):
        thread = await self.client.beta.threads.retrieve(thread_id)
        return thread

    async def create_message(self, thread_id, content):
        message = await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
        )
        return message

    async def run_stream(self, thread, stream_it):
        async with self.client.beta.threads.runs.stream(
            thread_id=thread,
            assistant_id=self.assistant_id,
            event_handler=stream_it,
        ) as stream:
            await stream.until_done()

    async def create_gen(self, thread, connection_id, stream_it, gatewayapi):
        task = asyncio.create_task(self.run_stream(thread, stream_it))
        
        counting = 0
        try:
            async for token in stream_it.aiter():
                print(token)
                gatewayapi.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({'message': token,'type': 'message',"counting":counting})
                )
                counting = counting + 1
                yield token
        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            stream_it.done.set()

        await task