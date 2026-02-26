import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv('.env')

async def main():
    pool = await asyncpg.create_pool(os.getenv('MEMORY_POSTGRES_DSN', ''))
    if not pool:
        print("No pool")
        return
        
    async with pool.acquire() as conn:
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.columns 
            WHERE column_name = 'user_id' AND table_schema = 'public'
        """)
        print("Tables with user_id:", [t['table_name'] for t in tables])

if __name__ == '__main__':
    asyncio.run(main())
