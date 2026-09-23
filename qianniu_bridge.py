"""Bridge the local support API to the Qianniu desktop chat window."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from dotenv import load_dotenv

try:
    import pyperclip
    import uiautomation as auto
except ImportError as exc:  # pragma: no cover - Windows desktop dependencies
    raise SystemExit("请先安装桌面桥接依赖：pip install uiautomation pyperclip") from exc

LOG = logging.getLogger("qianniu_bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

API_URL = os.getenv("QIANNIU_API_URL", "http://127.0.0.1:8000/chat")
POLL_SECONDS = float(os.getenv("QIANNIU_POLL_SECONDS", "1.0"))
HUMAN_DELAY_SECONDS = float(os.getenv("QIANNIU_HUMAN_DELAY", "1.5"))
WINDOW_KEYWORDS = tuple(
    value.strip()
    for value in os.getenv("QIANNIU_WINDOW_KEYWORDS", "千牛,接待").split(",")
    if value.strip()
)
INPUT_NAME = os.getenv("QIANNIU_INPUT_NAME", "")
NICKNAME_CONTROL_NAME = os.getenv("QIANNIU_NICKNAME_CONTROL_NAME", "")
EXCLUDED_TEXT = (
    "千牛工单",
    "全店工单",
    "指派给我",
    "千牛商家工作台",
    "工单中心",
)
IGNORED_TEXT = {
    "千牛", "接待", "发送", "表情", "图片", "文件", "转人工",
    "快捷回复", "搜索", "订单", "商品", "输入消息", "请输入消息",
}
last_sent_text = ""
MESSAGE_BLACKLIST = tuple(
    value.encode("ascii").decode("unicode_escape")
    for value in (
        r"\u5168\u5e97\u5de5\u5355", r"\u5343\u725b\u5de5\u5355",
        r"\u5343\u725b\u5546\u5bb6\u5de5\u4f5c\u53f0", r"\u6682\u65e0\u5de5\u5355",
        r"\u667a\u80fd\u5ba2\u670d", r"\u5ba2\u670d", r"\u8bbe\u7f6e",
        r"\u672a\u8bfb", r"\u8f7b\u94a2\u9f99\u9aa8\u6709\u54ea\u4e9b\u89c4\u683c\uff1f",
    )
)


@dataclass
class BuyerMessage:
    nickname: str
    text: str
    window: Any


def load_api_token() -> str:
    load_dotenv()
    token = os.getenv("API_TOKEN", "").strip()
    if token:
        return token
    try:
        import config

        return str(getattr(config, "API_TOKEN", "") or "").strip()
    except (ImportError, AttributeError):
        return ""


def control_name(control: Any) -> str:
    try:
        return str(control.Name or "").strip()
    except Exception:
        return ""


def descendants(control: Any, max_depth: int = 14) -> Iterable[Any]:
    pending = [(control, 0)]
    while pending:
        current, depth = pending.pop()
        if depth >= max_depth:
            continue
        try:
            children = current.GetChildren()
        except Exception:
            continue
        for child in children:
            yield child
            pending.append((child, depth + 1))


def find_window() -> Any | None:
    """Find a top-level Qianniu window and resolve it through its native handle."""
    root = auto.GetRootControl()
    try:
        windows = root.GetChildren()
    except Exception:
        return None

    for window in windows:
        title = control_name(window)
        if not title or not any(word.lower() in title.lower() for word in WINDOW_KEYWORDS):
            continue
        try:
            if window.IsOffscreen:
                continue
        except Exception:
            pass
        handle = getattr(window, "NativeWindowHandle", 0)
        resolver = getattr(auto, "ControlFromHandle", None)
        if handle and callable(resolver):
            try:
                return resolver(handle)
            except Exception:
                LOG.debug("无法通过原生句柄解析窗口，继续使用顶层控件", exc_info=True)
        return window
    return None


def control_rect(control: Any) -> tuple[int, int, int, int] | None:
    try:
        rect = control.BoundingRectangle
        left, top, right, bottom = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom
    except Exception:
        return None


def is_visible(control: Any) -> bool:
    try:
        return not control.IsOffscreen and control_rect(control) is not None
    except Exception:
        return control_rect(control) is not None


def find_named_control(window: Any, name: str, control_type: str | None = None) -> Any | None:
    if not name:
        return None
    for candidate in descendants(window):
        if control_name(candidate) != name:
            continue
        if control_type and getattr(candidate, "ControlTypeName", "") != control_type:
            continue
        return candidate
    return None


def find_input(window: Any) -> Any | None:
    """Find a likely lower chat EditControl without requiring it to exist."""
    named = find_named_control(window, INPUT_NAME, "EditControl") if INPUT_NAME else None
    if named and is_visible(named):
        return named

    window_rect = control_rect(window)
    if not window_rect:
        return None
    wl, wt, wr, wb = window_rect
    width, height = wr - wl, wb - wt
    candidates: list[tuple[int, Any]] = []
    for candidate in descendants(window):
        if getattr(candidate, "ControlTypeName", "") != "EditControl" or not is_visible(candidate):
            continue
        rect = control_rect(candidate)
        if not rect:
            continue
        left, top, right, bottom = rect
        editor_width = right - left
        center_x = (left + right) // 2
        if top < wt + height * 0.42 or editor_width < max(120, width * 0.12):
            continue
        if not (wl + width * 0.12 < center_x < wr - width * 0.12):
            continue
        candidates.append((editor_width * 10 + bottom, candidate))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def read_control_text(control: Any) -> str:
    name = control_name(control)
    if name:
        return name
    try:
        value = control.GetValuePattern().Value
        return str(value or "").strip()
    except Exception:
        return ""


def is_valid_message_text(text: str) -> bool:
    text = re.sub(r"[\s\x00-\x1f\x7f-\x9f\ufffc]+", "", text or "")
    text = re.sub(r"\[(?:表情|图片|emoji|emoticon)(?::[^\]]*)?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:<表情>|<emoji>|:emoji:|回车|Enter)", "", text, flags=re.IGNORECASE)
    if not text or text in IGNORED_TEXT:
        return False
    return True


def visible_text_blocks(window: Any) -> list[tuple[Any, str]]:
    """Collect buyer-aligned visible text blocks only from the 20%-70% column."""
    window_rect = control_rect(window)
    if not window_rect:
        return []
    wl, wt, wr, wb = window_rect
    accepted_types = {"TextControl", "DocumentControl", "ListItemControl", "DataItemControl"}
    blocks: list[tuple[Any, str]] = []
    for candidate in descendants(window):
        if getattr(candidate, "ControlTypeName", "") not in accepted_types or not is_visible(candidate):
            continue
        rect = control_rect(candidate)
        text = read_control_text(candidate)
        if not rect or not text or len(text) > 2000:
            continue
        left, top, right, bottom = rect
        if right <= wl or left >= wr or bottom <= wt or top >= wb:
            continue
        center_ratio = (((left + right) / 2) - wl) / (wr - wl)
        left_ratio = (left - wl) / (wr - wl)
        if not 0.20 <= center_ratio <= 0.70:
            continue
        if left_ratio > 0.48:
            continue
        if is_valid_message_text(text):
            blocks.append((candidate, text))
    return blocks


def find_message_text(window: Any) -> str | None:
    """Return the latest buyer bubble, or None when it is not safe to process."""
    blocks = visible_text_blocks(window)
    if not blocks:
        return None
    seen: set[str] = set()
    ordered: list[tuple[int, int, str]] = []
    for index, (control, text) in enumerate(blocks):
        if text == last_sent_text or text in seen:
            continue
        seen.add(text)
        rect = control_rect(control)
        ordered.append((rect[3] if rect else index, rect[0] if rect else index, text))
    ordered.sort(key=lambda item: (item[0], item[1]))
    if not ordered:
        return None
    text = ordered[-1][2]
    if any(term in text for term in MESSAGE_BLACKLIST):
        LOG.info("丢弃命中黑名单的聊天文本：%r", text)
        return None
    return text


def find_nickname(window: Any) -> str:
    named = find_named_control(window, NICKNAME_CONTROL_NAME) if NICKNAME_CONTROL_NAME else None
    if named:
        nickname = read_control_text(named)
        if nickname:
            return nickname[:128]

    title = control_name(window)
    parts = [part.strip() for part in re.split(r"[-|—–]", title) if part.strip()]
    for part in reversed(parts):
        if not any(word.lower() in part.lower() for word in WINDOW_KEYWORDS):
            return part[:128]
    return "qianniu-unknown"


def post_chat(question: str, thread_id: str) -> dict[str, Any]:
    token = load_api_token()
    if not token:
        raise RuntimeError("未读取到 API_TOKEN，请在 .env 或环境变量中配置")
    payload = json.dumps(
        {"question": question, "thread_id": thread_id, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"本地接口返回 HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接本地接口 {API_URL}: {exc.reason}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("本地接口返回的不是有效 JSON") from exc
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        result = result["result"]
    if not isinstance(result, dict):
        raise RuntimeError("本地接口返回缺少结构化结果")
    return result


def focus_composer(window: Any, editor: Any | None) -> Any | None:
    """Focus the discovered EditControl, or use the lower-center coordinate fallback."""
    if editor is not None and getattr(editor, "ControlTypeName", "") == "EditControl" and is_visible(editor):
        try:
            rect = control_rect(window)
            if not rect:
                return None
            left, top, right, bottom = rect
            auto.Click(left + int((right - left) * 0.45), bottom - 80)
            return window
        except Exception:
            LOG.warning("EditControl 点击失败，改用窗口坐标聚焦")

    rect = control_rect(window)
    if not rect:
        LOG.error("无法读取千牛主窗口 Rect，跳过本次发送")
        return None
    left, top, right, bottom = rect
    try:
        window.SetFocus()
    except Exception:
        pass
    try:
        auto.Click(left + int((right - left) * 0.45), bottom - 80)
        return window
    except Exception:
        LOG.exception("窗口底部坐标点击失败")
        return None


def paste_into_chat(window: Any, editor: Any | None, text: str, send: bool) -> bool:
    keyboard_target = focus_composer(window, editor)
    if keyboard_target is None and control_rect(window) is None:
        return False
    time.sleep(0.15)
    pyperclip.copy(text)
    try:
        if keyboard_target is not None:
            keyboard_target.SendKeys("{Ctrl}v")
        else:
            auto.SendKeys("{Ctrl}v")
        if send:
            time.sleep(0.15)
            shortcut = os.getenv("QIANNIU_SEND_KEY", "enter").lower()
            key = "{Alt}s" if shortcut == "alt_s" else "{Enter}"
            if keyboard_target is not None:
                keyboard_target.SendKeys(key)
            else:
                auto.SendKeys(key)
    except Exception:
        LOG.exception("粘贴或发送快捷键执行失败")
        return False
    return True


def handle_message(message: BuyerMessage) -> None:
    global last_sent_text
    LOG.info("检测到有效买家文本 nickname=%s text=%r", message.nickname, message.text)
    result = post_chat(message.text, message.nickname)
    reply = str(result.get("reply", "")).strip()
    if result.get("need_human") is True:
        warning = "您好，当前问题需要人工客服介入，请稍候。"
        LOG.warning("需要人工介入 nickname=%s", message.nickname)
        paste_into_chat(message.window, find_input(message.window), warning, send=False)
        return
    if not reply:
        LOG.warning("接口返回空 reply，跳过发送")
        return

    time.sleep(HUMAN_DELAY_SECONDS)
    if paste_into_chat(message.window, find_input(message.window), reply, send=True):
        last_sent_text = reply
        LOG.info("已发送自动回复 nickname=%s", message.nickname)


def run(once: bool = False) -> None:
    processed: dict[str, str] = {}
    LOG.info("千牛桥接已启动，轮询间隔 %.1fs", POLL_SECONDS)
    while True:
        window = find_window()
        if window is None:
            LOG.debug("未找到千牛主窗口")
        else:
            extracted = find_message_text(window)
            text = extracted.strip() if extracted is not None else ""
            if text and is_valid_message_text(text):
                nickname = find_nickname(window)
                if text == last_sent_text:
                    LOG.debug("跳过机器人刚发送的回复")
                elif processed.get(nickname) != text:
                    processed[nickname] = text
                    LOG.info("提取到聊天消息：%r", text)
                    try:
                        handle_message(BuyerMessage(nickname, text, window))
                    except Exception:
                        LOG.exception("处理买家消息失败 nickname=%s", nickname)
                else:
                    LOG.debug("跳过重复消息 nickname=%s", nickname)
        if once:
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="千牛桌面聊天自动回复桥接")
    parser.add_argument("--once", action="store_true", help="仅轮询一次")
    args = parser.parse_args()
    try:
        run(once=args.once)
    except KeyboardInterrupt:
        LOG.info("千牛桥接已停止")


if __name__ == "__main__":
    main()
