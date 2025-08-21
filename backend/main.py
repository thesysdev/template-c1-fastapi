from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llm_runner import generate_stream, ChatRequest
from thesys_genui_sdk.fast_api import with_c1_response


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/chat")
@with_c1_response()
async def chat_endpoint(request: ChatRequest):
    await generate_stream(request)
