from __future__ import annotations

import asyncpg


async def get_dashboard_stats(pool: asyncpg.Pool) -> dict[str, int]:
    async with pool.acquire() as conn:
        stats = {
            "total_users": 0,
            "total_messages": 0,
            "total_sessions": 0,
            "total_facts": 0,
            "total_episodes": 0,
            "total_reflections": 0,
        }
        
        try:
            stats["total_users"] = await conn.fetchval("SELECT COUNT(*) FROM users")
            stats["total_messages"] = await conn.fetchval("SELECT COUNT(*) FROM messages")
            stats["total_sessions"] = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        except asyncpg.exceptions.UndefinedTableError:
            pass
            
        try:
            stats["total_facts"] = await conn.fetchval("SELECT COUNT(*) FROM user_facts")
        except asyncpg.exceptions.UndefinedTableError:
            pass
            
        try:
            stats["total_episodes"] = await conn.fetchval(
                "SELECT COUNT(*) FROM persona_episodes"
            )
        except asyncpg.exceptions.UndefinedTableError:
            pass
            
        try:
            stats["total_reflections"] = await conn.fetchval(
                "SELECT COUNT(*) FROM persona_reflections"
            )
        except asyncpg.exceptions.UndefinedTableError:
            pass
            
        return stats


async def get_users(pool: asyncpg.Pool, limit: int = 50, offset: int = 0) -> list[dict]:
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT 
                    u.guild_id, u.user_id, u.discord_username, u.combined_label, u.first_seen, u.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.user_id = u.user_id AND m.guild_id = u.guild_id) as message_count
                FROM users u
                ORDER BY u.updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            return []


async def get_recent_messages(
    pool: asyncpg.Pool, limit: int = 100, offset: int = 0, filters: dict = None
) -> list[dict]:
    if filters is None:
        filters = {}
        
    async with pool.acquire() as conn:
        try:
            base_query = """
                SELECT m.message_id, m.session_id, m.guild_id, m.channel_id, m.user_id, 
                       m.author_label, m.role, m.modality, m.content_clean, m.source, m.created_at,
                       s.transcript, s.confidence
                FROM messages m
                LEFT JOIN stt_turns s ON m.message_id = s.message_id
            """
            
            where_clauses = []
            args = []
            idx = 1
            
            # Simple Exact Matches
            for field in ["guild_id", "channel_id", "user_id", "role", "modality", "source"]:
                if filters.get(field):
                    where_clauses.append(f"m.{field} = ${idx}")
                    args.append(filters[field])
                    idx += 1
                    
            # Text Search
            if filters.get("q"):
                where_clauses.append(f"m.content_clean ILIKE ${idx}")
                args.append(f"%{filters['q']}%")
                idx += 1
                
            # Date Range
            if filters.get("from_date"):
                try:
                    where_clauses.append(f"m.created_at >= ${idx}::timestamp")
                    args.append(filters["from_date"])
                    idx += 1
                except Exception:
                    pass
            if filters.get("to_date"):
                try:
                    where_clauses.append(f"m.created_at <= ${idx}::timestamp")
                    args.append(filters["to_date"])
                    idx += 1
                except Exception:
                    pass
                    
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
                
            base_query += f" ORDER BY m.message_id DESC LIMIT ${idx} OFFSET ${idx+1}"
            args.extend([limit, offset])
            
            rows = await conn.fetch(base_query, *args)
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            return []


async def get_user_facts(pool: asyncpg.Pool, limit: int = 100) -> list[dict]:
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT f.fact_id, f.guild_id, f.user_id, u.discord_username, 
                       f.fact_key, f.fact_value, f.fact_type, f.confidence, f.importance, 
                       f.status, f.pinned, f.evidence_count, f.updated_at
                FROM user_facts f
                LEFT JOIN users u ON f.user_id = u.user_id AND f.guild_id = u.guild_id
                ORDER BY f.importance DESC, f.updated_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            return []


async def get_persona_episodes(pool: asyncpg.Pool, limit: int = 50) -> list[dict]:
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT episode_id, persona_id, guild_id, channel_id, episode_type, 
                       title, summary, status, importance, vividness, confidence, 
                       source_message_count, updated_at
                FROM persona_episodes
                ORDER BY importance DESC, updated_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            return []


async def get_persona_traits(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT t.persona_id, t.trait_key, c.label, c.description,
                       t.anchor_value, t.current_value, t.confidence, t.drift_velocity,
                       t.evidence_count, t.status, t.updated_at
                FROM persona_traits t
                JOIN persona_trait_catalog c ON t.trait_key = c.trait_key
                ORDER BY t.current_value DESC
                """
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            return []

async def get_message_by_id(pool: asyncpg.Pool, message_id: int) -> dict | None:
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT m.*, s.transcript, s.confidence
                FROM messages m
                LEFT JOIN stt_turns s ON m.message_id = s.message_id
                WHERE m.message_id = $1
                """,
                message_id
            )
            return dict(row) if row else None
        except Exception:
            return None

async def get_user_details(pool: asyncpg.Pool, guild_id: str, user_id: str) -> dict | None:
    async with pool.acquire() as conn:
        try:
            user = await conn.fetchrow(
                "SELECT * FROM users WHERE guild_id = $1 AND user_id = $2",
                guild_id, user_id
            )
            if not user: return None
            
            messages = await conn.fetch(
                "SELECT message_id, content_clean, role, created_at FROM messages WHERE guild_id = $1 AND user_id = $2 ORDER BY created_at DESC LIMIT 50",
                guild_id, user_id
            )
            
            facts = await conn.fetch(
                "SELECT fact_id, fact_key, fact_value, status, importance FROM user_facts WHERE guild_id = $1 AND user_id = $2 ORDER BY updated_at DESC",
                guild_id, user_id
            )
            
            return {
                "user": dict(user),
                "messages": [dict(m) for m in messages],
                "facts": [dict(f) for f in facts]
            }
        except Exception:
            return None

async def get_row_by_pk(pool: asyncpg.Pool, table: str, pk_col: str, pk_val: int | str) -> dict | None:
    # Basic protection
    valid_tables = ["persona_episodes", "persona_reflections"]
    if table not in valid_tables: return None
    
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(f"SELECT * FROM {table} WHERE {pk_col} = $1", pk_val)
            return dict(row) if row else None
        except Exception:
            return None

async def get_analytics_data(pool: asyncpg.Pool) -> dict:
    data = {"daily_messages": [], "role_distribution": []}
    async with pool.acquire() as conn:
        try:
            msg_rows = await conn.fetch("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM messages
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 30
            """)
            data["daily_messages"] = [{"date": r["date"].isoformat() if r["date"] else "", "count": r["count"]} for r in msg_rows]
            
            role_rows = await conn.fetch("""
                SELECT role, COUNT(*) as count
                FROM messages
                GROUP BY role
            """)
            data["role_distribution"] = [dict(r) for r in role_rows]
        except Exception:
            pass
    return data

async def get_tables(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.relname as table_name,
                   c.reltuples::bigint AS estimated_count
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r'
              AND n.nspname = 'public'
            ORDER BY c.relname;
            """
        )
        tables = []
        for row in rows:
            t_name = row['table_name']
            count = row['estimated_count'] if row['estimated_count'] >= 0 else 0
            tables.append({"name": t_name, "count": count})
        return tables

async def get_table_columns(pool: asyncpg.Pool, table_name: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1
            ORDER BY ordinal_position;
            """,
            table_name
        )
        return [dict(row) for row in rows]

async def get_table_rows(pool: asyncpg.Pool, table_name: str, limit: int = 50, offset: int = 0) -> list[dict]:
    # Security: table_name must be validated before calling this
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                f"SELECT * FROM {table_name} LIMIT $1 OFFSET $2",
                limit, offset
            )
            return [dict(row) for row in rows]
        except Exception:
            return []

import re
import time

async def execute_readonly_query(pool: asyncpg.Pool, query: str) -> dict:
    clean_query = query.strip()
    
    # Strip basic block/line comments for regex checks
    comment_stripped = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL)
    comment_stripped = re.sub(r'--.*?\n', '', f"{comment_stripped}\n")
    upper_query = comment_stripped.upper()

    first_word = upper_query.strip().split()[0] if upper_query.strip() else ""
    
    if first_word not in ["SELECT", "WITH", "EXPLAIN"]:
        return {
            "success": False,
            "error": f"Security Policy blocks this statement. Must start with SELECT, WITH, or EXPLAIN.",
            "execution_time_ms": 0, "columns": [], "rows": []
        }

    # Explicit blocked keywords that could mutate state implicitly or bypass readonly
    blocked_keywords = [
        "ANALYZE", "INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "COMMIT", "ROLLBACK", "GRANT", "REVOKE", "CREATE ", "EXEC ", "EXECUTE "
    ]
    for b_kw in blocked_keywords:
        if b_kw in upper_query:
            return {
                "success": False,
                "error": f"Security Policy blocks the keyword '{b_kw}' to prevent mutations.",
                "execution_time_ms": 0, "columns": [], "rows": []
            }
            
    # Block multiple statements exactly
    if ";" in comment_stripped.strip().rstrip(";"):
         return {
            "success": False,
            "error": "Single statements only. Chained queries (;) are not allowed.",
            "execution_time_ms": 0, "columns": [], "rows": []
        }
    
    # Enforce a hard cap of 500 rows even if the user explicitly provided a large limit.
    # EXPLAIN cannot be wrapped as a subquery, so we skip wrapping it.
    if first_word in ["SELECT", "WITH"]:
        clean_query = f"SELECT * FROM ({clean_query.rstrip(';')}) AS __wrapped_query__ LIMIT 500"

    start_time = time.perf_counter()
    async with pool.acquire() as conn:
        try:
            async with conn.transaction():
                # Enforce server-side 5s timeout
                await conn.execute("SET LOCAL statement_timeout = '5000';")
                
                stmt = await conn.prepare(clean_query)
                records = await stmt.fetch(timeout=5.0)
                
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                if not records:
                    return {
                        "success": True, "execution_time_ms": execution_time_ms,
                        "columns": [], "rows": []
                    }
                    
                columns = list(records[0].keys())
                rows = [dict(record) for record in records]
                
                return {
                    "success": True,
                    "execution_time_ms": execution_time_ms,
                    "columns": columns,
                    "rows": rows,
                    "truncated": False,
                    "total_fetched": len(records)
                }
                
        except asyncpg.exceptions.QueryCanceledError:
             return {
                "success": False, "error": "Query timed out after 5 seconds.",
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                "columns": [], "rows": []
            }
        except Exception as e:
            return {
                "success": False, "error": str(e),
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                "columns": [], "rows": []
            }

