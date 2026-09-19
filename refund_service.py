"""Refund-application adapter; this layer deliberately does not call a payment system."""
from typing import Protocol

from schemas import OrderInfo, RefundRequest, RefundResult


class RefundService(Protocol):
    """Contract for a future payment/OMS refund-application client."""

    def apply_refund(self, request: RefundRequest, order: OrderInfo) -> RefundResult:
        """Validate and create a refund application."""


class MockRefundService:
    """In-memory adapter with ownership, status, amount, and idempotency checks."""

    _applications: dict[str, RefundResult] = {}

    def apply_refund(self, request: RefundRequest, order: OrderInfo) -> RefundResult:
        request = RefundRequest.model_validate(request)
        order = OrderInfo.model_validate(order)
        existing = self._applications.get(request.idempotency_key)
        if existing is not None:
            return existing
        if request.order_id != order.order_id or request.customer_id != order.customer_id:
            return self._reject(request, "订单不属于当前用户，无法申请退款")
        if order.status not in {"paid", "shipped", "delivered"}:
            return self._reject(request, "当前订单状态不支持退款申请")
        if request.amount > order.refundable_amount:
            return self._reject(
                request,
                f"可退款金额不足，当前最多可退 {order.refundable_amount} 元",
            )
        result = RefundResult(
            order_id=request.order_id,
            idempotency_key=request.idempotency_key,
            approved=True,
            amount=request.amount,
            message="退款申请已创建，当前未调用支付系统，最终结果以审核为准",
        )
        self._applications[request.idempotency_key] = result
        return result

    @staticmethod
    def _reject(request: RefundRequest, message: str) -> RefundResult:
        return RefundResult(
            order_id=request.order_id,
            idempotency_key=request.idempotency_key,
            approved=False,
            amount=0,
            message=message,
        )


def build_refund_service() -> RefundService:
    """Return the mock now; replace this factory with a real payment adapter later."""
    return MockRefundService()


refund_service = build_refund_service()
