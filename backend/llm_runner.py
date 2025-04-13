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
from tools import openai_tools, exec_tool

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

    while True:
        tool_calls_details = []
        assistant_response_content = ""
        assistant_message_for_history: dict | None = None 

        stream = client.chat.completions.create(
            messages=conversation_history,
            model="c1-nightly",
            tools=openai_tools,
            stream=True,
        )
        accumulated_tool_calls_args = {}

        for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                assistant_response_content += delta.content
                if not tool_calls_details:
                    yield delta.content

            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    if tool_call_chunk.id:
                        if tool_call_chunk.id not in accumulated_tool_calls_args:
                            accumulated_tool_calls_args[tool_call_chunk.id] = {
                                "id": tool_call_chunk.id,
                                "function_name": tool_call_chunk.function.name,
                                "arguments": ""
                            }
                        accumulated_tool_calls_args[tool_call_chunk.id]["arguments"] += tool_call_chunk.function.arguments


            if finish_reason:
                assistant_message_for_history = {"role": "assistant", "content": assistant_response_content or None}
                if accumulated_tool_calls_args:
                    assistant_message_for_history["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["function_name"], "arguments": tc["arguments"]},
                        }
                        for tc in accumulated_tool_calls_args.values()
                    ]
                    tool_calls_details = list(accumulated_tool_calls_args.values())


        conversation_history.append(assistant_message_for_history)

        # if this is the final response from the assistant, we need to store the responseId
        assistant_id = chat_request.responseId if len(tool_calls_details) > 0 else None
        thread_store.append_message(chat_request.threadId, Message(
            openai_message=assistant_message_for_history, 
            id=assistant_id
        ))


        if tool_calls_details:
            tool_messages_for_history = []
            for tool_call_info in tool_calls_details:
                tool_call_id = tool_call_info["id"]
                tool_arguments = tool_call_info["arguments"]
                tool_result = exec_tool(tool_call_info["function_name"], tool_arguments)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_call_info["function_name"],
                    "content": tool_result,
                }

                tool_messages_for_history.append(tool_message)
                thread_store.append_message(chat_request.threadId, Message(
                    openai_message=tool_message,
                    id=tool_call_id
                ))

            conversation_history.extend(tool_messages_for_history)
            continue
        else:
            break