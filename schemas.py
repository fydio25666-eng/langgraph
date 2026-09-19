"""Typed state and validated service response models."""
from decimal import Decimal
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class ServiceReply(BaseModel):
    """Unified structured response returned by every workflow branch."""

    intent: Literal["presales", "after_sales", "other"]
    sub_intent: Literal["product", "refund", "logistics", "general"] = "general"
    reply: str = Field(min_length=1, max_length=300)
    need_human: bool
    risk_level: Literal["low", "medium", "high"]
    sources: list[str] = Field(default_factory=list, max_length=5)


class OrderQueryInput(BaseModel):
    """订单查询工具的安全入参。"""

    order_id: str = Field(pattern=r"^ORD-\d{6,20}$")


class OrderInfo(BaseModel):
    """订单查询工具的结构化结果。"""

    order_id: str
    customer_id: str = Field(min_length=1, max_length=64)
    status: Literal["paid", "shipped", "delivered", "refunding", "refunded"]
    total_amount: Decimal = Field(gt=Decimal("0"), max_digits=12, decimal_places=2)
    refundable_amount: Decimal = Field(ge=Decimal("0"), max_digits=12, decimal_places=2)


class RefundRequest(BaseModel):
    """退款申请入参，包含归属校验所需用户和幂等请求编号。"""

    order_id: str = Field(pattern=r"^ORD-\d{6,20}$")
    customer_id: str = Field(min_length=1, max_length=64)
    amount: Decimal = Field(gt=Decimal("0"), le=Decimal("100000"), max_digits=12, decimal_places=2)
    reason: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(pattern=r"^refund-[A-Za-z0-9_-]{16,128}$")


class RefundResult(BaseModel):
    """退款工具结果，明确表示是否实际受理。"""

    order_id: str
    idempotency_key: str
    approved: bool
    amount: Decimal = Field(ge=Decimal("0"), max_digits=12, decimal_places=2)
    message: str = Field(min_length=1, max_length=200)


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class SupportState(TypedDict, total=False):
    question: str
    result: ServiceReply
    messages: list[Message]
    retry_count: int
    retryable_error: bool
    knowledge_context: str
    knowledge_sources: list[str]
