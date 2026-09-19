"""Minimal API and safety tests without real model or network calls."""
import os
import unittest
from decimal import Decimal
from unittest.mock import patch

os.environ["API_TOKEN"] = "test-token"

from fastapi.testclient import TestClient

import app as api
from nodes import apply_refund, query_order
from schemas import RefundRequest, ServiceReply


class ChatApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(api.app)
        cls.headers = {"Authorization": "Bearer test-token"}

    def mock_reply(self, intent: str, text: str) -> dict:
        return {
            "result": ServiceReply(
                intent=intent,
                sub_intent="product" if intent == "presales" else "general",
                reply=text,
                need_human=False,
                risk_level="low",
            )
        }

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_presales_consultation(self) -> None:
        with patch.object(api.workflow_app, "invoke", return_value=self.mock_reply("presales", "有现货")):
            response = self.client.post(
                "/chat",
                headers=self.headers,
                json={"question": "请问有现货吗？", "stream": False},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "presales")

    def test_after_sales_consultation(self) -> None:
        with patch.object(api.workflow_app, "invoke", return_value=self.mock_reply("after_sales", "请提供订单号")):
            response = self.client.post(
                "/chat",
                headers=self.headers,
                json={"question": "我的订单怎么查物流？", "stream": False},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "after_sales")

    def test_refund_amount_exceeded(self) -> None:
        order = query_order({"order_id": "ORD-123456"})
        result = apply_refund(
            RefundRequest(
                order_id=order.order_id,
                customer_id=order.customer_id,
                amount=Decimal("250.00"),
                reason="商品问题",
                idempotency_key="refund-test-amount-exceeded",
            ),
            order,
        )
        self.assertFalse(result.approved)
        self.assertIn("可退款金额", result.message)

    def test_empty_input(self) -> None:
        response = self.client.post(
            "/chat",
            headers=self.headers,
            json={"question": "   ", "stream": False},
        )
        self.assertEqual(response.status_code, 422)

    def test_model_exception(self) -> None:
        with patch.object(api.workflow_app, "invoke", side_effect=RuntimeError("model unavailable")):
            response = self.client.post(
                "/chat",
                headers=self.headers,
                json={"question": "请介绍商品", "stream": False},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["reply"], "系统暂时无法处理：系统异常，已为您转人工客服。")


if __name__ == "__main__":
    unittest.main()
