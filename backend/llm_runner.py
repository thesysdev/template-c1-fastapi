from pydantic import BaseModel
from typing import (
    List,
    AsyncIterator,
    TypedDict,
    Literal,
)
import os
from openai import OpenAI
from dotenv import load_dotenv  # type: ignore

from thread_store import Message, thread_store
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

# define the client
client = OpenAI(
    api_key=os.getenv("THESYS_API_KEY"),
    base_url="https://api.thesys.dev/v1/embed",
)

# define the prompt type in request
class Prompt(TypedDict):
    role: Literal["user"]
    content: str
    id: str

# define the request type
class ChatRequest(BaseModel):
    prompt: Prompt
    threadId: str
    responseId: str

    class Config:
        extra = "allow"  # Allow extra fields

async def generate_stream(chat_request: ChatRequest) -> AsyncIterator[str]:
    conversation_history: List[ChatCompletionMessageParam] = thread_store.get_messages(chat_request.threadId)
    conversation_history.append(chat_request.prompt)
    thread_store.append_message(chat_request.threadId, Message(
        openai_message=chat_request.prompt,
        id=chat_request.prompt['id']
    ))

    assistant_response_content = ""
    assistant_message_for_history: dict | None = None

    stream = client.chat.completions.create(
        messages=conversation_history,
        model="c1-nightly",
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        if delta and delta.content:
            assistant_response_content += delta.content
            yield delta.content

        if finish_reason:
            assistant_message_for_history = {"role": "assistant", "content": assistant_response_content or None}

    if assistant_message_for_history:
        conversation_history.append(assistant_message_for_history)

        # Store the assistant message with the responseId
        thread_store.append_message(chat_request.threadId, Message(
            openai_message=assistant_message_for_history,
            id=chat_request.responseId # Assign responseId to the final assistant message
        ))