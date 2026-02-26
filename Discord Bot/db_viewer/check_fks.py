import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv('.env')

async def main():
    pool = await asyncpg.create_pool(os.getenv('MEMORY_POSTGRES_DSN', ''))
    if not pool: return
    
    async with pool.acquire() as conn:
        fks = await conn.fetch("""
            SELECT
                tc.table_name, kcu.column_name, rc.delete_rule
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.referential_constraints AS rc
                  ON tc.constraint_name = rc.constraint_name
            WHERE constraint_type = 'FOREIGN KEY' AND kcu.column_name = 'user_id';
        """)
        for fk in fks:
            print(f"{fk['table_name']}.{fk['column_name']} -> {fk['delete_rule']}")
            
if __name__ == '__main__':
    asyncio.run(main())
