from openai import AsyncOpenAI, AssistantEventHandler
from config import OPENAI_API_KEY_DICT
import time, json
import asyncio
from typing import AsyncIterator, Literal, Union, cast
OPENAI_API_KEY = OPENAI_API_KEY_DICT['QG']
MATH_ASSISTANT_ID = ""
FITB_ASSISTANT_ID = ""
VALIDATION_ASSISTANT_ID = ""
QGEN_ASSISTANT_ID=""
AF_ASSISTANT_ID=""
PASSAGE_ASSISTANT_ID=""
QUESTION_ASSISTANT_ID=""

  # or a hard-coded ID like "asst-..."
# asst_aEavCfUvUEC8YFdqUlqpup3n
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

class EventHandler(AssistantEventHandler):
    queue: asyncio.Queue[str]
    done: asyncio.Event

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    async def on_text_created(self, text) -> None:
        print("created")

        print(f"\nassistant > ", end="", flush=True)
        self.done.clear()

    async def on_text_delta(self, delta, snapshot) -> None:
        print("delta")

        print(delta.value, end="", flush=True)
        if delta.value is not None and delta.value != "":
            self.queue.put_nowait(delta.value)

    async def on_end(self) -> None:
        print("end")
        """Fires when stream ends or when exception is thrown"""
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        while not self.queue.empty() or not self.done.is_set():
            done, other = await asyncio.wait(
                [
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if other:
                other.pop().cancel()

            token_or_done = cast(Union[str, Literal[True]], done.pop().result())

            if token_or_done is True:
                break

            yield token_or_done

async def assistant_stream(thread_id,assistant_id,stream_it):
    print('assistant id')
    print(assistant_id)
    print('thread id')
    print(thread_id)
    async with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=stream_it,
    ) as stream:
        stream.until_done()

async def create_gen(thread, assistant_id, stream_it: EventHandler):
    task = asyncio.create_task(assistant_stream(thread, assistant_id, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task

async def submit_message(assistant_id, thread, user_message):
    user_json = json.loads(user_message)
    user_prompt = ""
    for item in user_json:
        if item["type"] == "text":
            text = item["text"]
            if text == "question :":
                text = "main question :"
            user_prompt += text
            user_prompt += '\n'

    await client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_prompt
    )
    return await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

async def submit_message_validation(assistant_id, thread, user_message):
    await client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

async def submit_message_qgen(assistant_id, thread, user_message):
    # await client.beta.threads.messages.create(
    #     thread_id=thread.id, role="user", content=user_message
    # )
    return await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

async def submit_message_kb(assistant_id, thread_id, user_message):
    await client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_message
    )
    return await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

async def submit_message_kb_simple(assistant_id, thread_id, message = ""):
    if (message is not None and message != ""):
        await client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=message
        )

    return await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

async def get_response(thread):
    return await client.beta.threads.messages.list(thread_id=thread.id, order="desc")

async def get_response_kb(thread):
    return await client.beta.threads.messages.list(thread_id=thread, order="desc")

async def create_thread_and_run(user_input,question_type_id):
    thread = await client.beta.threads.create()
    if question_type_id in [5, 6, 7]:
        run = await submit_message(FITB_ASSISTANT_ID, thread, user_input)
    else:
        run = await submit_message(MATH_ASSISTANT_ID, thread, user_input)

    return thread, run

async def create_thread_and_run_kb(user_input,file_id):
    thread = await client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
                # Attach the new file to the message.
                "attachments": [
                    { "file_id": file_id, "tools": [{"type": "file_search"}] }
                ],
            }
        ]
    )
    print(thread.tool_resources.file_search)
    run = await submit_message_qgen(QGEN_ASSISTANT_ID, thread, user_input)

    return thread, run

async def create_thread_and_run_kb_chat(user_input,vector_id):
    thread = await client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    # print(thread.tool_resources.file_search)
    # run = await submit_message_qgen(QGEN_ASSISTANT_ID, thread, user_input)

    return thread, AF_ASSISTANT_ID

async def create_thread_and_run_passage(user_input,file_id):
    thread = await client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
                # Attach the new file to the message.
                "attachments": [
                    { "file_id": file_id, "tools": [{"type": "file_search"}] }
                ],
            }
        ]
    )
    print(thread.tool_resources.file_search)
    run = await submit_message_qgen(PASSAGE_ASSISTANT_ID, thread, user_input)

    return thread, run

async def create_thread_and_run_question(user_input,file_id):
    thread = await client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
                # Attach the new file to the message.
                "attachments": [
                    { "file_id": file_id, "tools": [{"type": "file_search"}] }
                ],
            }
        ]
    )
    print(thread.tool_resources.file_search)
    run = await submit_message_qgen(QUESTION_ASSISTANT_ID, thread, user_input)

    return thread, run

async def validate(user_input):
    print("validation")
    print(user_input)
    thread = await client.beta.threads.create()
    run = await submit_message_validation(VALIDATION_ASSISTANT_ID, thread, user_input)

    return thread, run

def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


# Waiting in a loop
async def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

async def wait_on_run_kb(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run