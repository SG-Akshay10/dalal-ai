import psycopg2

conn_str = 'postgresql://postgres:CskMssa%401010@db.usysqimimzxemrxqmxah.supabase.co:5432/postgres'

print('Connecting to Supabase PostgreSQL at: ' + conn_str.split('@')[-1])

try:
    conn = psycopg2.connect(conn_str)
    conn.autocommit = True
    cur = conn.cursor()

    with open('../.gemini/antigravity/brain/6c3f9065-8190-488e-aa83-1dd2025bde4f/supabase_schema.sql', 'r') as file:
        sql = file.read()
    
    print('Executing schema initialization script...')
    cur.execute(sql)
    print('Successfully applied supabase_schema.sql to the database!')

    cur.close()
    conn.close()

except Exception as e:
    print(f'Database error: {e}')
    exit(1)
