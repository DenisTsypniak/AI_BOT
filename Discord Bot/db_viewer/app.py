from __future__ import annotations

import os
from contextlib import asynccontextmanager
import csv
from datetime import datetime
from io import StringIO
import json
import logging
from pathlib import Path
import time

try:
    import aiohttp
except Exception:  # pragma: no cover - optional in minimal viewer env
    aiohttp = None  # type: ignore[assignment]

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

try:
    from zoneinfo import ZoneInfo
except ImportError:
    pass

from . import queries

logger = logging.getLogger(__name__)

# Load .env from the parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

POSTGRES_DSN = os.getenv("MEMORY_POSTGRES_DSN")


class AppState:
    pool: asyncpg.Pool | None = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not POSTGRES_DSN:
        raise RuntimeError("MEMORY_POSTGRES_DSN is not set in .env")
    app_state.pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=5)
    yield
    if app_state.pool:
        await app_state.pool.close()

app = FastAPI(title="Discord Bot DB Viewer", lifespan=lifespan)

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
# Override environment loader to strictly enforce utf-8 on Windows
import jinja2
templates.env.loader = jinja2.FileSystemLoader(str(templates_dir), encoding='utf-8')

def format_kyiv_time(dt: datetime | None, format_str: str = '%Y-%m-%d %H:%M', default: str = '') -> str:
    if not dt:
        return default
    try:
        try:
            kyiv_tz = ZoneInfo("Europe/Kyiv")
        except NameError:
            import pytz
            kyiv_tz = pytz.timezone("Europe/Kyiv")
            
        if dt.tzinfo is None:
            try:
                utc = ZoneInfo("UTC")
            except NameError:
                utc = pytz.UTC
            dt = dt.replace(tzinfo=utc)
            
        dt_kyiv = dt.astimezone(kyiv_tz)
        return dt_kyiv.strftime(format_str)
    except Exception:
        logger.debug("Failed to format datetime with Kyiv timezone", exc_info=True)
        if isinstance(dt, datetime):
            return dt.strftime(format_str)
        return str(dt)

templates.env.filters["kyiv_time"] = format_kyiv_time

def json_pretty(val):
    if isinstance(val, (dict, list)):
        return json.dumps(val, indent=2, ensure_ascii=False)
    elif isinstance(val, str):
        try:
            parsed = json.loads(val)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return val
    return val

templates.env.filters["json_pretty"] = json_pretty


def render(request: Request, name: str, context: dict) -> HTMLResponse:
    """Helper to ensure UTF-8 encoding the response."""
    context["request"] = request
    return templates.TemplateResponse(
        name, 
        context,
        media_type="text/html; charset=utf-8",
        headers={"Content-Type": "text/html; charset=utf-8"}
    )


async def _ollama_healthcheck_snapshot() -> dict[str, object]:
    base_url = str(os.getenv("MEMORY_OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").strip().rstrip("/")
    model = str(os.getenv("MEMORY_OLLAMA_MODEL", "") or "").strip()
    backend = str(os.getenv("MEMORY_EXTRACTOR_BACKEND", "gemini") or "gemini").strip().lower()
    snapshot: dict[str, object] = {
        "configured_backend": backend,
        "base_url": base_url,
        "model": model,
        "ping_ok": False,
        "model_available": False,
        "latency_ms": None,
        "error": "",
    }
    if backend != "ollama":
        snapshot["error"] = "MEMORY_EXTRACTOR_BACKEND is not ollama"
        return snapshot
    if not model:
        snapshot["error"] = "MEMORY_OLLAMA_MODEL is empty"
        return snapshot
    if aiohttp is None:
        snapshot["error"] = "aiohttp not installed in db_viewer environment"
        return snapshot

    started = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for body in ({"model": model}, {"name": model}):
                async with session.post(f"{base_url}/api/show", json=body) as response:
                    snapshot["latency_ms"] = max(0, int((time.perf_counter() - started) * 1000))
                    text = await response.text()
                    if response.status == 200:
                        snapshot["ping_ok"] = True
                        snapshot["model_available"] = True
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            payload = {}
                        if isinstance(payload, dict):
                            details = payload.get("details")
                            if isinstance(details, dict):
                                snapshot["format"] = details.get("format")
                                snapshot["family"] = details.get("family")
                                snapshot["parameter_size"] = details.get("parameter_size")
                                snapshot["quantization_level"] = details.get("quantization_level")
                        return snapshot
                    snapshot["error"] = f"Ollama HTTP {response.status}: {text[:180]}"
                    if response.status not in {400, 404}:
                        return snapshot
                    # retry with alternate field name if version expects "name" instead of "model"
            return snapshot
    except Exception as exc:
        snapshot["latency_ms"] = max(0, int((time.perf_counter() - started) * 1000))
        snapshot["error"] = f"{exc.__class__.__name__}: {exc}"
        return snapshot


def _normalize_paging(limit: int, offset: int, *, default_limit: int, max_limit: int) -> tuple[int, int]:
    try:
        safe_limit = int(limit)
    except (TypeError, ValueError):
        safe_limit = default_limit
    try:
        safe_offset = int(offset)
    except (TypeError, ValueError):
        safe_offset = 0
    if safe_limit <= 0:
        safe_limit = default_limit
    safe_limit = min(safe_limit, max_limit)
    safe_offset = max(0, safe_offset)
    return safe_limit, safe_offset


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


async def _fetch_paginated_rows(
    sql: str,
    *,
    limit: int,
    offset: int,
    label: str,
) -> tuple[list[dict], str | None]:
    if not app_state.pool:
        return [], "Database pool not initialized"
    async with app_state.pool.acquire() as conn:
        try:
            records = await conn.fetch(sql, limit, offset)
            return [dict(r) for r in records], None
        except Exception as exc:
            logger.exception("DB viewer query failed for %s (limit=%s offset=%s)", label, limit, offset)
            return [], f"DB Error: {str(exc)}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    stats = await queries.get_dashboard_stats(app_state.pool)
    return render(request, "index.html", {"stats": stats})


@app.get("/users", response_class=HTMLResponse)
async def users(request: Request, limit: int = 50, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=50, max_limit=500)
    users_list = await queries.get_users(app_state.pool, limit=limit, offset=offset)
    return render(request, "users.html", {
        "users": users_list, 
        "limit": limit, 
        "offset": offset
    })

@app.get("/users/{guild_id}/{user_id}", response_class=HTMLResponse)
async def user_detail(request: Request, guild_id: str, user_id: str):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    data = await queries.get_user_details(app_state.pool, guild_id, user_id)
    if not data:
        return HTMLResponse("User not found", status_code=404)
    return render(request, "user_detail.html", {"data": data, "user_id": user_id, "guild_id": guild_id})

@app.post("/users/{guild_id}/{user_id}/delete")
async def delete_user_route(guild_id: str, user_id: str):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    
    deleted = await queries.delete_user(app_state.pool, guild_id, user_id)
    if not deleted:
        return HTMLResponse("Failed to delete user or user not found", status_code=400)
    
    # Use 303 See Other for redirecting after a POST
    return RedirectResponse(url="/users", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/users/{guild_id}/{user_id}/facts/{fact_id}/delete")
async def delete_fact_route(guild_id: str, user_id: str, fact_id: int):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    
    deleted = await queries.delete_fact(app_state.pool, guild_id, user_id, fact_id)
    if not deleted:
        return HTMLResponse("Failed to delete fact or fact not found", status_code=400)
    
    return RedirectResponse(url=f"/users/{guild_id}/{user_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/messages", response_class=HTMLResponse)
async def messages(
    request: Request, 
    limit: int = 100, 
    offset: int = 0,
    guild_id: str = "",
    channel_id: str = "",
    user_id: str = "",
    role: str = "",
    modality: str = "",
    source: str = "",
    q: str = "",
    from_date: str = "",
    to_date: str = ""
):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=100, max_limit=500)
        
    filters = {
        "guild_id": guild_id.strip() if guild_id else None,
        "channel_id": channel_id.strip() if channel_id else None,
        "user_id": user_id.strip() if user_id else None,
        "role": role.strip() if role else None,
        "modality": modality.strip() if modality else None,
        "source": source.strip() if source else None,
        "q": q.strip() if q else None,
        "from_date": from_date.strip() if from_date else None,
        "to_date": to_date.strip() if to_date else None,
    }
    
    msgs = await queries.get_recent_messages(app_state.pool, limit=limit, offset=offset, filters=filters)
    return render(request, "messages.html", {
        "messages": msgs, 
        "limit": limit, 
        "offset": offset,
        "filters": filters
    })

@app.get("/messages/{message_id:int}", response_class=HTMLResponse)
async def message_detail(request: Request, message_id: int):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    msg = await queries.get_message_by_id(app_state.pool, message_id)
    if not msg:
        return HTMLResponse("Message not found", status_code=404)
    return render(request, "detail_generic.html", {"title": f"Повідомлення #{message_id}", "data": msg, "back_url": "/messages"})


@app.get("/messages/csv")
async def messages_csv(
    request: Request, 
    guild_id: str = "", channel_id: str = "", user_id: str = "", 
    role: str = "", modality: str = "", source: str = "", 
    q: str = "", from_date: str = "", to_date: str = ""
):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    filters = {
        "guild_id": guild_id.strip() if guild_id else None,
        "channel_id": channel_id.strip() if channel_id else None,
        "user_id": user_id.strip() if user_id else None,
        "role": role.strip() if role else None,
        "modality": modality.strip() if modality else None,
        "source": source.strip() if source else None,
        "q": q.strip() if q else None,
        "from_date": from_date.strip() if from_date else None,
        "to_date": to_date.strip() if to_date else None,
    }
    msgs = await queries.get_recent_messages(app_state.pool, limit=10000, offset=0, filters=filters)
    
    stream = StringIO()
    writer = csv.writer(stream)
    if msgs:
        writer.writerow(msgs[0].keys())
        for m in msgs:
            writer.writerow(m.values())
            
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=messages_export.csv"
    return response


@app.get("/memory", response_class=HTMLResponse)
async def memory(request: Request):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    facts = await queries.get_user_facts(app_state.pool)
    episodes = await queries.get_persona_episodes(app_state.pool)
    return render(request, "memory.html", {"facts": facts, "episodes": episodes})


@app.get("/memory/provenance", response_class=HTMLResponse)
async def memory_provenance(request: Request, limit: int = 80, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=80, max_limit=300)
    snapshot = await queries.get_memory_provenance_snapshot(app_state.pool, limit=limit, offset=offset)
    return render(
        request,
        "memory_provenance.html",
        {
            "limit": limit,
            "offset": offset,
            "prev_offset": max(0, offset - limit),
            "global_facts": snapshot.get("global_facts", []),
            "persona_facts": snapshot.get("persona_facts", []),
            "local_facts": snapshot.get("local_facts", []),
            "biographies": snapshot.get("biographies", []),
        },
    )


@app.get("/memory/extractor", response_class=HTMLResponse)
async def memory_extractor_diagnostics(request: Request, run_limit: int = 80, candidate_limit: int = 120):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    run_limit, _ = _normalize_paging(run_limit, 0, default_limit=80, max_limit=300)
    candidate_limit, _ = _normalize_paging(candidate_limit, 0, default_limit=120, max_limit=500)
    snapshot = await queries.get_memory_extractor_diagnostics_snapshot(
        app_state.pool,
        run_limit=run_limit,
        candidate_limit=candidate_limit,
    )
    ollama_health = await _ollama_healthcheck_snapshot()
    return render(
        request,
        "memory_extractor_diagnostics.html",
        {
            "run_limit": run_limit,
            "candidate_limit": candidate_limit,
            "summary": snapshot.get("summary", {}),
            "by_backend": snapshot.get("by_backend", []),
            "recent_runs": snapshot.get("recent_runs", []),
            "dry_run_candidates": snapshot.get("dry_run_candidates", []),
            "rejected_candidates": snapshot.get("rejected_candidates", []),
            "ollama_health": ollama_health,
        },
    )

@app.get("/memory/episodes/{episode_id}", response_class=HTMLResponse)
async def episode_detail(request: Request, episode_id: int): # Assuming pk is int or auto-castable
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    row = await queries.get_row_by_pk(app_state.pool, "persona_episodes", "episode_id", episode_id)
    if not row:
         return HTMLResponse("Episode not found", status_code=404)
    return render(request, "detail_generic.html", {"title": f"Епізод #{episode_id}", "data": row, "back_url": "/memory"})


@app.get("/persona", response_class=HTMLResponse)
async def persona(request: Request):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    traits = await queries.get_persona_traits(app_state.pool)
    return render(request, "persona.html", {"traits": traits})


@app.get("/persona/reflections", response_class=HTMLResponse)
async def persona_reflections(request: Request, limit: int = 50, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=50, max_limit=500)
    rows, error_msg = await _fetch_paginated_rows(
        "SELECT * FROM persona_reflections ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        limit=limit,
        offset=offset,
        label="persona_reflections",
    )
    return render(request, "persona_generic.html", {"title": "Рефлексії Персони", "rows": rows, "limit": limit, "offset": offset, "endpoint": "/persona/reflections", "error": error_msg, "can_view_detail": True})

@app.get("/persona/reflections/{reflection_id}", response_class=HTMLResponse)
async def reflection_detail(request: Request, reflection_id: int):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    row = await queries.get_row_by_pk(app_state.pool, "persona_reflections", "reflection_id", reflection_id)
    if not row:
         return HTMLResponse("Reflection not found", status_code=404)
    return render(request, "detail_generic.html", {"title": f"Рефлексія #{reflection_id}", "data": row, "back_url": "/persona/reflections"})


@app.get("/persona/relationships", response_class=HTMLResponse)
async def persona_relationships(request: Request, limit: int = 50, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=50, max_limit=500)
    rows, error_msg = await _fetch_paginated_rows(
        "SELECT * FROM persona_relationships ORDER BY updated_at DESC LIMIT $1 OFFSET $2",
        limit=limit,
        offset=offset,
        label="persona_relationships",
    )
    return render(request, "persona_generic.html", {"title": "Відносини (Relationships)", "rows": rows, "limit": limit, "offset": offset, "endpoint": "/persona/relationships", "error": error_msg, "can_view_detail": False})

@app.get("/persona/evidence", response_class=HTMLResponse)
async def persona_evidence(request: Request, limit: int = 50, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=50, max_limit=500)
    rows, error_msg = await _fetch_paginated_rows(
        "SELECT * FROM persona_trait_evidence ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        limit=limit,
        offset=offset,
        label="persona_trait_evidence",
    )
    return render(request, "persona_generic.html", {"title": "Докази (Evidence)", "rows": rows, "limit": limit, "offset": offset, "endpoint": "/persona/evidence", "error": error_msg, "can_view_detail": False})

@app.get("/persona/audit", response_class=HTMLResponse)
async def persona_audit(request: Request, limit: int = 50, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=50, max_limit=500)
    rows, error_msg = await _fetch_paginated_rows(
        "SELECT * FROM persona_audit_log ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        limit=limit,
        offset=offset,
        label="persona_audit_log",
    )
    return render(request, "persona_generic.html", {"title": "Аудит Лог", "rows": rows, "limit": limit, "offset": offset, "endpoint": "/persona/audit", "error": error_msg, "can_view_detail": False})

@app.get("/stt", response_class=HTMLResponse)
async def stt_diagnostics(request: Request, limit: int = 100, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=100, max_limit=500)
    rows, error_msg = await _fetch_paginated_rows(
        "SELECT * FROM stt_turns ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        limit=limit,
        offset=offset,
        label="stt_turns",
    )
    return render(request, "persona_generic.html", {"title": "STT Діагностика", "rows": rows, "limit": limit, "offset": offset, "endpoint": "/stt", "error": error_msg, "can_view_detail": False})


@app.get("/schema", response_class=HTMLResponse)
async def schema_index(request: Request):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    tables = await queries.get_tables(app_state.pool)
    return render(request, "schema.html", {"tables": tables})


@app.get("/schema/{table_name}", response_class=HTMLResponse)
async def schema_table(request: Request, table_name: str, limit: int = 50, offset: int = 0):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    limit, offset = _normalize_paging(limit, offset, default_limit=50, max_limit=200)
    
    # Validate table_name exists to prevent SQL Injection
    valid_tables = await queries.get_tables(app_state.pool)
    if not any(t["name"] == table_name for t in valid_tables):
        return HTMLResponse("Table not found", status_code=404)
        
    columns = await queries.get_table_columns(app_state.pool, table_name)
    rows = await queries.get_table_rows(app_state.pool, table_name, limit, offset)
    
    # Simple check for automatic Detail linking in the generic table viewer
    linkable_tables = {
        "users": "/users",
        "messages": "/messages",
        "persona_episodes": "/memory/episodes",
        "persona_reflections": "/persona/reflections"
    }
    validation_route = linkable_tables.get(table_name)
    
    return render(request, "schema_table.html", {
        "table_name": table_name,
        "columns": columns,
        "rows": rows,
        "limit": limit,
        "offset": offset,
        "validation_route_exists": validation_route is not None,
        "route_prefix": validation_route
    })

@app.get("/schema/{table_name}/csv")
async def schema_table_csv(request: Request, table_name: str):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
        
    valid_tables = await queries.get_tables(app_state.pool)
    if not any(t["name"] == table_name for t in valid_tables):
        return HTMLResponse("Table not found", status_code=404)
        
    rows = await queries.get_table_rows(app_state.pool, table_name, limit=10000, offset=0)
    
    stream = StringIO()
    writer = csv.writer(stream)
    if rows:
        writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow(r.values())
            
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={table_name}_export.csv"
    return response


@app.get("/query", response_class=HTMLResponse)
async def query_tool_get(request: Request):
    return render(request, "query.html", {"query": "", "result": None})


@app.post("/query", response_class=HTMLResponse)
async def query_tool_post(request: Request):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    
    form_data = await request.form()
    query_str = form_data.get("query", "")
    
    if not query_str.strip():
        return render(request, "query.html", {
            "query": query_str, 
            "result": {"success": False, "error": "Query cannot be empty"}
        })
        
    result = await queries.execute_readonly_query(app_state.pool, query_str)
    
    return render(request, "query.html", {
        "query": query_str,
        "result": result
    })


@app.get("/health", response_class=HTMLResponse)
async def health(request: Request):
    db_status = "unknown"
    ping_ms = 0
    pool_size = 0
    idle_size = 0
    
    if not app_state.pool:
        db_status = "Disconnected (No pool)"
    else:
        try:
            start = time.perf_counter()
            async with app_state.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            ping_ms = (time.perf_counter() - start) * 1000
            db_status = "Connected"
            pool_size = app_state.pool.get_size()
            idle_size = app_state.pool.get_idle_size()
        except Exception as exc:
            logger.exception("DB viewer health check failed")
            db_status = f"Error: {str(exc)}"
            
    return render(request, "health.html", {
        "db_status": db_status,
        "ping_ms": ping_ms,
        "pool_size": pool_size,
        "idle_size": idle_size
    })

@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    if not app_state.pool:
        return HTMLResponse("Database pool not initialized", status_code=500)
    data = await queries.get_analytics_data(app_state.pool)
    return render(request, "analytics.html", {"data": data})


def run():
    import uvicorn
    host = os.getenv("DB_VIEWER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int((os.getenv("DB_VIEWER_PORT") or "8000").strip())
    except ValueError:
        port = 8000
    port = max(1, min(port, 65535))
    reload_enabled = _env_bool("DB_VIEWER_RELOAD", True)
    uvicorn.run("db_viewer.app:app", host=host, port=port, reload=reload_enabled)

if __name__ == "__main__":
    run()
