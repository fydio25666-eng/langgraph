"""LangGraph nodes for classification, response generation, and fallback."""
import re
import hashlib
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from config import model
from order_service import order_service
from rag import retrieve
from refund_service import refund_service
from schemas import (
    OrderInfo,
    OrderQueryInput,
    RefundRequest,
    RefundResult,
    ServiceReply,
    Message,
    SupportState,
)

MAX_RETRIES = 3
MAX_HISTORY_MESSAGES = 10


def query_order(request: OrderQueryInput) -> OrderInfo:
    """订单工具兼容入口：委托独立适配器，便于以后切换真实订单服务。"""
    return order_service.query_order(request)


def apply_refund(request: RefundRequest, order: OrderInfo) -> RefundResult:
    """退款工具兼容入口：委托独立适配器创建退款申请。"""
    return refund_service.apply_refund(request, order)


def refund_idempotency_key(order_id: str, amount: Decimal, reason: str) -> str:
    """Create a stable key so identical user requests are not submitted twice."""
    raw = f"{order_id}|{amount:.2f}|{reason}".encode("utf-8")
    return "refund-" + hashlib.sha256(raw).hexdigest()[:32]


def human_fallback(reason: str) -> ServiceReply:
    return ServiceReply(
        intent="other",
        reply=f"系统暂时无法处理：{reason}，已为您转人工客服。",
        need_human=True,
        risk_level="medium",
        sources=[],
    )


def _failure(state: SupportState) -> dict[str, Any]:
    """Record one failed model validation/call for graph-level retry routing."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "retryable_error": True,
    }


def _invoke(prompt: str) -> ServiceReply:
    """Generate and validate one structured response from the configured model."""
    if model is None:
        raise RuntimeError("服务配置不完整")
    reply = model.with_structured_output(ServiceReply, method="json_mode").invoke(prompt)
    return ServiceReply.model_validate(reply)


def prepare_history(state: SupportState) -> dict[str, Any]:
    """Append the current user turn and keep only the latest ten messages."""
    question = state.get("question", "").strip()
    history = list(state.get("messages", []))
    history.append({"role": "user", "content": question})
    return {
        "question": question,
        "messages": history[-MAX_HISTORY_MESSAGES:],
        "retry_count": 0,
        "retryable_error": False,
    }


def history_context(state: SupportState) -> str:
    """Format the sliding-window conversation for model prompts."""
    messages = state.get("messages", [])[-MAX_HISTORY_MESSAGES:]
    if not messages:
        return "暂无历史对话。"
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def finalize_history(state: SupportState) -> dict[str, Any]:
    """Append the assistant turn and persist the bounded conversation window."""
    result = state.get("result")
    if not result:
        return {"messages": list(state.get("messages", []))[-MAX_HISTORY_MESSAGES:]}
    history: list[Message] = list(state.get("messages", []))
    history.append({"role": "assistant", "content": result.reply})
    return {"messages": history[-MAX_HISTORY_MESSAGES:]}


def retrieve_knowledge(state: SupportState) -> dict[str, Any]:
    """Retrieve local manual context before classification and business nodes."""
    context, sources = retrieve(state.get("question", ""))
    return {"knowledge_context": context, "knowledge_sources": sources}


def classify(state: SupportState) -> dict[str, Any]:
    """Classify the top-level intent and prepare a response candidate."""
    question = state.get("question", "").strip()
    if not question or len(question) > 1000:
        return {"result": human_fallback("请输入 1 至 1000 个字符的咨询内容"), "retryable_error": False}
    if model is None:
        return {"result": human_fallback("服务配置不完整"), "retryable_error": False}
    context = history_context(state)
    prompt = (
        "你是电商客服。识别用户问题的一级意图：presales（售前商品咨询）、"
        "after_sales（售后退款或物流）或 other。请严格返回 JSON 对象，字段为 "
        "intent、sub_intent、reply、need_human、risk_level；sub_intent 只能是 "
        "product、refund、logistics、general，risk_level 只能是 low、medium、high。\n"
        f"最近对话（最多 10 条）：\n{context}\n当前用户问题：{question}"
    )
    try:
        return {"result": _invoke(prompt), "retryable_error": False}
    except (ValidationError, Exception):
        return _failure(state)


def presales(state: SupportState) -> dict[str, Any]:
    """独立售前节点：补全商品、价格、库存和配送咨询的应答。"""
    question = state.get("question", "").strip()
    context = state.get("knowledge_context", "暂无可用手册内容。")
    sources = state.get("knowledge_sources", [])
    conversation = history_context(state)
    prompt = (
        "你是售前商品顾问。请根据用户问题生成准确、简洁的商品咨询答复，"
        "只回答能够确认的商品、价格、库存或配送信息，不要编造数据。"
        "必须返回 JSON，intent 固定为 presales，sub_intent 固定为 product，"
        "并包含 reply、need_human、risk_level 字段。\n"
        "只能依据知识库内容回答；回复中应注明引用来源。\n"
        f"知识库上下文：{context}\n"
        f"最近对话：\n{conversation}\n"
        f"用户问题：{question}"
    )
    try:
        result = _invoke(prompt).model_copy(update={"sources": sources[:5]})
        return {"result": result, "retryable_error": False}
    except (ValidationError, Exception):
        return _failure(state)


def after_sales(state: SupportState) -> dict[str, Any]:
    """独立售后节点：细分退款与物流意图并生成对应答复。"""
    question = state.get("question", "").strip()
    context = state.get("knowledge_context", "暂无可用手册内容。")
    sources = state.get("knowledge_sources", [])
    conversation = history_context(state)
    tool_context = "未提供可识别的订单号，暂不执行订单工具。"
    order_match = re.search(r"ORD-\d{6,20}", question, re.IGNORECASE)
    if order_match:
        order_id = order_match.group(0).upper()
        try:
            order = query_order(OrderQueryInput(order_id=order_id))
            tool_context = f"订单查询结果：{order.model_dump_json()}"
            amount_match = re.search(r"(\d+(?:\.\d{1,2})?)\s*元", question)
            if amount_match and any(word in question for word in ("退款", "退费", "退钱")):
                refund = apply_refund(
                    RefundRequest(
                        order_id=order_id,
                        customer_id="demo-user",
                        amount=Decimal(amount_match.group(1)),
                        reason="用户售后退款申请",
                        idempotency_key=refund_idempotency_key(
                            order_id, Decimal(amount_match.group(1)), "用户售后退款申请"
                        ),
                    ),
                    order,
                )
                tool_context += f"；退款工具结果：{refund.model_dump_json()}"
        except ValidationError:
            tool_context = "订单号或退款金额未通过安全校验，未执行退款操作。"
    prompt = (
        "你是售后客服。处理退款或物流问题：退款相关将 sub_intent 设为 refund，"
        "物流、配送、签收异常相关将 sub_intent 设为 logistics；无法判断时设为 general。"
        "必须返回 JSON，intent 固定为 after_sales，并包含 reply、need_human、risk_level。"
        "不承诺未经订单系统确认的退款结果或物流时效。\n"
        f"工具上下文：{tool_context}\n"
        f"知识库上下文：{context}\n"
        f"最近对话：\n{conversation}\n"
        f"用户问题：{question}"
    )
    try:
        result = _invoke(prompt).model_copy(update={"sources": sources[:5]})
        return {"result": result, "retryable_error": False}
    except (ValidationError, Exception):
        return _failure(state)


def route(state: SupportState) -> str:
    """Route classification failures to retry or circuit-breaker branches."""
    if state.get("retryable_error"):
        return "human" if state.get("retry_count", 0) >= MAX_RETRIES else "retry"
    return state["result"].intent


def response_route(state: SupportState) -> str:
    """Retry response generation, or open the human fallback circuit."""
    if state.get("retryable_error"):
        return "human" if state.get("retry_count", 0) >= MAX_RETRIES else "retry"
    return "done"


def finish(state: SupportState) -> dict[str, Any]:
    """Preserve a valid result as the branch's common exit."""
    return state


def human_node(state: SupportState) -> dict[str, Any]:
    """熔断节点：累计失败达到上限后强制转人工。"""
    return {"result": human_fallback("模型输出连续校验失败"), "retryable_error": False}
