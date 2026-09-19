"""FastAPI HTTP/SSE adapter for the LangGraph customer-support workflow."""
import json
import logging
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from graph import app as workflow_app
from auth import require_api_token
from nodes import human_fallback
from schemas import ServiceReply

logger = logging.getLogger(__name__)
app = FastAPI(title="E-commerce Support API", version="1.0.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    """Validated /chat request; stream=True returns Server-Sent Events."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., max_length=1000, description="用户咨询内容")
    thread_id: str = Field(default="default", min_length=1, max_length=128)
    stream: bool = Field(default=True, description="是否使用 SSE 流式返回")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


def result_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Convert LangGraph state to JSON-safe data without exposing internals."""
    result = state.get("result")
    if isinstance(result, ServiceReply):
        return {"result": result.model_dump(mode="json")}
    return {"result": human_fallback("系统未生成有效结果").model_dump(mode="json")}


def sse_event(payload: dict[str, Any], event: str = "message") -> str:
    """Format one SSE frame; ensure_ascii=False keeps Chinese readable."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_chat(question: str, thread_id: str) -> Iterator[str]:
    """Stream workflow state updates and convert runtime failures to an SSE error."""
    emitted = False
    try:
        config = {"configurable": {"thread_id": thread_id}}
        for state in workflow_app.stream(
            {"question": question}, config=config, stream_mode="values"
        ):
            if "result" in state:
                emitted = True
                yield sse_event(result_payload(state))
        if not emitted:
            yield sse_event(result_payload({}), event="error")
    except Exception:
        # Exceptions after the response starts must be represented as an SSE event.
        logger.exception("LangGraph streaming failed")
        yield sse_event(
            {"result": human_fallback("系统异常").model_dump(mode="json")},
            event="error",
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a stable validation response instead of FastAPI's default shape."""
    # Pydantic v2 may place a ValueError in ctx; encode it before JSON serialization.
    details = jsonable_encoder(exc.errors())
    return JSONResponse(
        status_code=422,
        content={"error": "请求参数校验失败", "details": details},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected route errors and prevent raw tracebacks reaching clients."""
    logger.exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"error": "服务内部异常"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
async def chat(request: ChatRequest, _: None = Depends(require_api_token)):
    """Run the existing graph as JSON or Server-Sent Events."""
    if request.stream:
        return StreamingResponse(
            stream_chat(request.question, request.thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    try:
        state = workflow_app.invoke(
            {"question": request.question},
            config={"configurable": {"thread_id": request.thread_id}},
        )
        return result_payload(state)["result"]
    except Exception:
        logger.exception("LangGraph request failed")
        return JSONResponse(
            status_code=500,
            content=human_fallback("系统异常").model_dump(mode="json"),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
