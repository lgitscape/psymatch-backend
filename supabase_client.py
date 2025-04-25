from supabase import create_client
import os

SUPABASE_URL = os.getenv("https://luexdfkqvvhrnxvgpcqs.supabase.co")
SUPABASE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1ZXhkZmtxdnZocm54dmdwY3FzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjczODI1NCwiZXhwIjoyMDU4MzE0MjU0fQ.Fwb4zcjUQ2CtMDrmVut4lHy1xGwDdDoqPcTrHcpKQEQ")  # gebruik de service role key
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
