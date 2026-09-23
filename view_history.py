"""Local browser for LangGraph conversation history."""
from __future__ import annotations
import os, sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ROOT = Path(__file__).resolve().parent
APP = FastAPI(title="LangGraph history viewer")
SERDE = JsonPlusSerializer()

def db_path() -> Path:
    configured = os.getenv("SQLITE_CHECKPOINT_PATH")
    path = Path(configured) if configured else ROOT / "data" / "checkpoints.sqlite"
    return path if path.is_absolute() else ROOT / path

def decode(type_name: str, value: bytes) -> Any:
    try:
        return SERDE.loads_typed((type_name, value))
    except Exception:
        return None

def plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if hasattr(value, "model_dump"): return plain(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"): return plain(vars(value))
    return str(value)

def parsed_time(value: Any) -> datetime | None:
    try: return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone()
    except (TypeError, ValueError): return None

def format_time(value: Any) -> str:
    parsed = parsed_time(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else "未知时间"

def all_rows() -> list[tuple[Any, ...]]:
    path = db_path()
    if not path.exists(): return []
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT thread_id, checkpoint, type, rowid FROM checkpoints ORDER BY rowid ASC").fetchall()

def snapshot_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    thread_id, blob, type_name, row_id = row
    checkpoint = decode(type_name, blob) or {}
    values = checkpoint.get("channel_values", {}) or {}
    messages = []
    for item in plain(values.get("messages", []) or []):
        if isinstance(item, dict) and item.get("content"):
            messages.append({"role": item.get("role", "assistant"), "content": str(item["content"])})
    return {"thread_id": thread_id, "row_id": row_id, "ts": checkpoint.get("ts") or "", "messages": messages, "result": plain(values.get("result")) or {}, "sources": plain(values.get("knowledge_sources")) or []}

def thread_snapshots(thread_id: str) -> list[dict[str, Any]]:
    return [snapshot_from_row(row) for row in all_rows() if row[0] == thread_id]

def merge_message_windows(snapshots: list[dict[str, Any]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for snapshot in snapshots:
        current = snapshot["messages"]
        if not current: continue
        overlap = 0
        for size in range(min(len(merged), len(current)), 0, -1):
            if merged[-size:] == current[:size]: overlap = size; break
        merged.extend(current[overlap:])
    return merged

def date_group(value: Any) -> str:
    current = parsed_time(value)
    if not current: return "earlier"
    if current.date() == date.today(): return "today"
    if current.date() == date.today() - timedelta(days=1): return "yesterday"
    return "earlier"

def summarize_thread(thread_id: str, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    latest = snapshots[-1] if snapshots else {}
    messages = merge_message_windows(snapshots)
    title = next((m["content"] for m in messages if m["role"] == "user"), "未命名会话")
    return {"thread_id": thread_id, "title": title[:80], "updated_at": format_time(latest.get("ts")), "updated_ts": latest.get("ts", ""), "row_id": latest.get("row_id", 0), "kind": "eval" if thread_id.startswith("eval-") else "business", "date_group": date_group(latest.get("ts")), "message_count": len(messages)}

def all_thread_summaries() -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows():
        snapshot = snapshot_from_row(row)
        grouped.setdefault(snapshot["thread_id"], []).append(snapshot)
    items = [summarize_thread(thread_id, snapshots) for thread_id, snapshots in grouped.items()]
    return sorted(items, key=lambda item: (item["updated_ts"], item["row_id"]), reverse=True)

def read_thread(thread_id: str) -> dict[str, Any]:
    snapshots = thread_snapshots(thread_id)
    if not snapshots: raise HTTPException(status_code=404, detail="未找到该会话")
    latest = snapshots[-1]
    result = latest.get("result") or {}
    return {**summarize_thread(thread_id, snapshots), "messages": merge_message_windows(snapshots), "result": {"risk_level": result.get("risk_level", "unknown"), "intent": result.get("intent", "unknown"), "sub_intent": result.get("sub_intent", "unknown"), "sources": result.get("sources") or latest.get("sources") or []}}

@APP.get("/api/threads")
def threads() -> list[dict[str, Any]]: return all_thread_summaries()

@APP.get("/api/threads/{thread_id:path}")
def thread(thread_id: str) -> dict[str, Any]: return read_thread(thread_id)

@APP.get("/api/status")
def status() -> dict[str, Any]:
    path = db_path()
    return {"database": str(path), "exists": path.exists(), "thread_count": len(all_thread_summaries()), "message_limit": 10}

PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>会话历史</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;color:#1f2937;background:#f3f4f6}*{box-sizing:border-box}body{margin:0;height:100vh;overflow:hidden}.app{display:flex;height:100vh}.side{width:350px;flex:0 0 350px;background:#fff;border-right:1px solid #e5e7eb;display:flex;flex-direction:column}.head{padding:20px 18px 14px;border-bottom:1px solid #eef0f2}.head h1{font-size:18px;margin:0 0 5px}.head p{font-size:12px;color:#6b7280;margin:0}.search{width:100%;margin-top:13px;padding:9px 11px;border:1px solid #dbe1e6;border-radius:8px;font-size:13px;outline:none}.filters{display:flex;gap:6px;margin-top:9px}.filter{flex:1;padding:7px 4px;border:1px solid #dbe1e6;border-radius:7px;background:#fff;color:#52605b;cursor:pointer;font-size:12px}.filter.active{background:#e8f7f1;border-color:#10a37f;color:#087f63}.refresh{width:100%;margin-top:9px;padding:8px;border:1px solid #dbe1e6;border-radius:8px;background:#fff;color:#374151;cursor:pointer}.db-status{margin-top:8px;font-size:11px;color:#9ca3af}.threads{flex:1;overflow:auto;padding:8px}.group-label{padding:11px 10px 5px;color:#9ca3af;font-size:11px;font-weight:650}.thread{display:block;width:100%;padding:11px 10px;border:0;border-radius:9px;background:transparent;text-align:left;cursor:pointer}.thread:hover,.thread.active{background:#eef8f5}.thread-title{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;color:#263238}.thread-row{display:flex;align-items:center;gap:7px;margin-top:6px}.thread-time{font-size:11px;color:#9ca3af}.badge{padding:2px 5px;border-radius:4px;font-size:10px}.badge.eval{background:#fff3df;color:#a16207}.badge.business{background:#eaf2ff;color:#315a96}.empty{padding:22px 10px;color:#9ca3af;font-size:13px}.main{display:flex;flex:1;min-width:0;flex-direction:column}.main-head{min-height:70px;padding:14px 28px;border-bottom:1px solid #e5e7eb;background:#fff;display:flex;align-items:center;justify-content:space-between;gap:16px}.main-title{font-size:15px;font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.main-meta{font-size:12px;color:#6b7280;white-space:nowrap}.chat{flex:1;overflow:auto;padding:28px 5vw 40px}.placeholder{height:100%;display:grid;place-items:center;color:#9ca3af}.message{display:flex;gap:12px;max-width:900px;margin:0 auto 22px}.message.user{flex-direction:row-reverse}.avatar{width:32px;height:32px;flex:0 0 32px;border-radius:9px;display:grid;place-items:center;color:#fff;background:#10a37f;font-size:11px;font-weight:700}.user .avatar{background:#64748b}.wrap{max-width:78%}.role{font-size:11px;color:#9ca3af;margin:1px 0 5px}.user .role{text-align:right}.bubble{padding:11px 14px;border-radius:12px;background:#fff;border:1px solid #e7eaee;box-shadow:0 2px 6px #1111;white-space:pre-wrap;word-break:break-word}.user .bubble{background:#eaf2ff;border-color:#d8e5fa}.detail{max-width:900px;margin:12px auto 0;padding:16px;border-radius:12px;background:#fff;border:1px solid #e5e7eb}.detail h2{margin:0 0 12px;font-size:13px}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{padding:5px 8px;border-radius:6px;background:#f1f5f9;color:#475569;font-size:12px}.risk-low{color:#087f63;background:#e8f7f1}.risk-medium{color:#a16207;background:#fff7df}.risk-high{color:#b42318;background:#fff0ee}.source{margin-top:9px;padding:8px 10px;border-left:3px solid #10a37f;background:#f7faf9;color:#52605b;font-size:12px}@media(max-width:760px){.side{width:275px;flex-basis:275px}.main-head{padding:0 16px}.chat{padding:20px 13px}.wrap{max-width:82%}}</style></head><body><div class="app"><aside class="side"><div class="head"><h1>会话历史</h1><p>按时间和类型整理的 LangGraph 记录</p><input class="search" id="search" placeholder="搜索标题或 thread_id"><div class="filters"><button class="filter active" data-kind="all">全部</button><button class="filter" data-kind="business">业务会话</button><button class="filter" data-kind="eval">评测会话</button></div><button class="refresh" id="refresh">刷新会话列表</button><div class="db-status" id="dbStatus">正在读取数据库...</div></div><div class="threads" id="threads"></div></aside><main class="main"><header class="main-head"><div class="main-title" id="title">选择左侧会话查看详情</div><div class="main-meta" id="meta"></div></header><section class="chat" id="chat"><div class="placeholder">请从左侧选择一个会话</div></section></main></div><script>
const threadsEl=document.getElementById('threads'),chatEl=document.getElementById('chat'),titleEl=document.getElementById('title'),metaEl=document.getElementById('meta'),searchEl=document.getElementById('search');let active='',allItems=[],kind='all';
function renderThreads(){threadsEl.replaceChildren();const q=searchEl.value.trim().toLowerCase();const filtered=allItems.filter(x=>(kind==='all'||x.kind===kind)&&(!q||(x.title+' '+x.thread_id).toLowerCase().includes(q)));if(!filtered.length){threadsEl.innerHTML='<div class="empty">没有匹配的会话</div>';return}const labels={today:'今天',yesterday:'昨天',earlier:'更早'};for(const group of ['today','yesterday','earlier']){const rows=filtered.filter(x=>x.date_group===group);if(!rows.length)continue;const label=document.createElement('div');label.className='group-label';label.textContent=labels[group];threadsEl.appendChild(label);rows.forEach(item=>{const b=document.createElement('button');b.className='thread'+(item.thread_id===active?' active':'');const t=document.createElement('span');t.className='thread-title';t.textContent=item.title;const row=document.createElement('span');row.className='thread-row';const time=document.createElement('span');time.className='thread-time';time.textContent=item.updated_at;const badge=document.createElement('span');badge.className='badge '+item.kind;badge.textContent=item.kind==='eval'?'评测':'业务';row.append(time,badge);b.append(t,row);b.onclick=()=>loadThread(item.thread_id);threadsEl.appendChild(b)})}}
function renderMessage(item){const role=item.role==='user'?'user':'assistant',row=document.createElement('div');row.className='message '+role;const avatar=document.createElement('div');avatar.className='avatar';avatar.textContent=role==='user'?'用户':'客服';const wrap=document.createElement('div');wrap.className='wrap';const label=document.createElement('div');label.className='role';label.textContent=role==='user'?'用户':'客服';const bubble=document.createElement('div');bubble.className='bubble';bubble.textContent=item.content;wrap.append(label,bubble);row.append(avatar,wrap);chatEl.appendChild(row)}
function renderDetail(data){chatEl.replaceChildren();titleEl.textContent=data.title;metaEl.textContent='最后更新：'+data.updated_at+' · '+(data.kind==='eval'?'评测会话':'业务会话');data.messages.forEach(renderMessage);const result=data.result||{},detail=document.createElement('div');detail.className='detail';const h=document.createElement('h2');h.textContent='会话元数据';detail.appendChild(h);const chips=document.createElement('div');chips.className='chips';[['风险等级',result.risk_level,'risk-'+result.risk_level],['一级意图',result.intent,''],['二级意图',result.sub_intent,'']].forEach(x=>{const c=document.createElement('span');c.className='chip '+x[2];c.textContent=x[0]+'：'+x[1];chips.appendChild(c)});detail.appendChild(chips);(result.sources||[]).forEach(source=>{const s=document.createElement('div');s.className='source';s.textContent='知识库来源：'+source;detail.appendChild(s)});if(!result.sources?.length){const s=document.createElement('div');s.className='source';s.textContent='知识库来源：本次记录未保存来源';detail.appendChild(s)}chatEl.appendChild(detail)}
async function loadThread(id){active=id;const response=await fetch('/api/threads/'+encodeURIComponent(id));if(!response.ok){chatEl.innerHTML='<div class="placeholder">读取会话失败</div>';return}renderDetail(await response.json());renderThreads()}
async function loadStatus(){const data=await(await fetch('/api/status')).json();document.getElementById('dbStatus').textContent=(data.exists?'数据库已连接':'数据库不存在')+' · '+data.thread_count+' 个会话'}
async function loadThreads(){const list=await(await fetch('/api/threads')).json();allItems=list;renderThreads();if(active&&list.some(x=>x.thread_id===active))return;if(list.length)await loadThread(list[0].thread_id)}
document.querySelectorAll('.filter').forEach(button=>button.onclick=()=>{document.querySelectorAll('.filter').forEach(x=>x.classList.remove('active'));button.classList.add('active');kind=button.dataset.kind;renderThreads()});searchEl.oninput=renderThreads;document.getElementById('refresh').onclick=()=>Promise.all([loadThreads(),loadStatus()]);loadStatus().catch(()=>{});loadThreads().catch(()=>{threadsEl.innerHTML='<div class="empty">无法读取检查点数据库</div>'});setInterval(()=>Promise.all([loadThreads(),loadStatus()]),5000);
</script></body></html>'''

@APP.get("/", response_class=HTMLResponse)
def index() -> str: return PAGE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="127.0.0.1", port=8080)

