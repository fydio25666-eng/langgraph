"""LangGraph 电商售前售后客服 Demo。"""
import os
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

class ServiceReply(BaseModel):
    """所有分支统一返回的结构化客服结果。"""

    intent: Literal["presales", "after_sales", "other"]
    reply: str = Field(min_length=1, max_length=300)
    need_human: bool
    risk_level: Literal["low", "medium", "high"]

class SupportState(TypedDict, total=False):
    question: str
    result: ServiceReply

def human_fallback(reason: str) -> ServiceReply:
    return ServiceReply(
        intent="other", reply=f"系统暂时无法处理：{reason}，已为您转人工客服。",
        need_human=True, risk_level="medium"
    )


def load_model() -> ChatOpenAI | None:
    """关键节点：读取配置，缺失时不让图启动后崩溃。"""
    load_dotenv()
    # 缺少任一连接参数时返回 None，后续节点将改走人工兜底。
    if not all(os.getenv(key) for key in ("BASE_URL", "API_KEY", "MODEL")):
        return None
    return ChatOpenAI(
        base_url=os.environ["BASE_URL"], api_key=os.environ["API_KEY"],
        # 限制请求时长和重试次数，避免单次咨询长时间阻塞。
        model=os.environ["MODEL"], temperature=0, timeout=20, max_retries=1
    )


model = load_model()  # 启动时仅初始化客户端，不发起模型请求。

def classify(state: SupportState) -> dict:
    """关键节点：模型意图识别，异常或结构校验失败均返回标准兜底结果。"""
    # 标准化输入，保证模型接收到的文本无首尾空白。
    question = state.get("question", "").strip()
    if not question or len(question) > 1000:
        return {"result": human_fallback("请输入 1 至 1000 个字符的咨询内容")}
    if model is None:
        return {"result": human_fallback("服务配置不完整")}
    prompt = (
        "你是电商客服。将用户问题分类为 presales（商品、价格、库存、配送）、"
        "after_sales（退款、退货、换货、物流异常）或 other。回复简洁且不承诺无法确认的信息。\n"
        f"用户问题：{question}"
    )
    try:
        # 让模型直接生成 ServiceReply，降低文本解析失败的概率。
        reply = model.with_structured_output(ServiceReply, method="json_mode").invoke(prompt)
        return {"result": ServiceReply.model_validate(reply)}
    except (ValidationError, Exception):
        return {"result": human_fallback("模型服务繁忙")}


def route(state: SupportState) -> str:
    """关键节点：由 Pydantic 校验后的意图驱动 LangGraph 条件分支。"""
    return state["result"].intent


def finish(state: SupportState) -> dict:
    # 保持当前结构化结果，作为各业务分支的统一出口。
    return state


# 关键节点：构建售前、售后与人工兜底三条可观测流程分支。
graph = StateGraph(SupportState)
graph.add_node("classify", classify)
graph.add_node("presales", finish)
graph.add_node("after_sales", finish)
graph.add_node("other", finish)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route, {
    "presales": "presales", "after_sales": "after_sales", "other": "other"
})
for node in ("presales", "after_sales", "other"):
    graph.add_edge(node, END)
app = graph.compile()  # 将流程定义编译为可调用的 LangGraph 应用。


if __name__ == "__main__":
    try:  # 最外层保护：图运行意外失败时仍向用户返回结构化结果。
        output = app.invoke({"question": " ".join(sys.argv[1:])})
        print(output["result"].model_dump_json(indent=2))
    except Exception:
        print(human_fallback("系统异常").model_dump_json(indent=2))
