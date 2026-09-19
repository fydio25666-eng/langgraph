"""Order-service adapter boundary with a local mock implementation."""
from decimal import Decimal
from typing import Protocol

from schemas import OrderInfo, OrderQueryInput


class OrderService(Protocol):
    """Stable contract shared by mock and future OMS/HTTP implementations."""

    def query_order(self, request: OrderQueryInput) -> OrderInfo:
        """Return one validated order record."""


class MockOrderService:
    """Local deterministic adapter used until a real order API is available."""

    def query_order(self, request: OrderQueryInput) -> OrderInfo:
        request = OrderQueryInput.model_validate(request)
        return OrderInfo(
            order_id=request.order_id,
            customer_id="demo-user",
            status="shipped",
            total_amount=Decimal("199.00"),
            refundable_amount=Decimal("199.00"),
        )


def build_order_service() -> OrderService:
    """Create the active adapter; replace this factory with a real API adapter later."""
    return MockOrderService()


order_service = build_order_service()
