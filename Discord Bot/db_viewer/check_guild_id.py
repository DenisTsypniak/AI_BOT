import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv('.env')

async def main():
    pool = await asyncpg.create_pool(os.getenv('MEMORY_POSTGRES_DSN', ''))
    if not pool: return
    
    tables_with_user_id = ['users', 'messages', 'stt_turns', 'user_facts', 'dialogue_summaries', 'persona_relationships', 'persona_ingested_messages', 'persona_relationship_evidence', 'persona_user_memory_prefs', 'persona_episode_participants', 'persona_episode_evidence']
    
    async with pool.acquire() as conn:
        for t in tables_with_user_id:
            cols = await conn.fetch(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{t}'")
            col_names = [c['column_name'] for c in cols]
            print(f"{t}: guild_id in cols? {'guild_id' in col_names}, user_id in cols? {'user_id' in col_names}")
            
if __name__ == '__main__':
    asyncio.run(main())
