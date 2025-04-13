from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llm_runner import generate_stream, ChatRequest


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream",
    )
