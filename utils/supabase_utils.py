import time

def insert_with_retry(table, data, retries=3, delay=1):
    for attempt in range(retries):
        try:
            response = table.insert(data).execute()
            if response.status_code < 400:
                return response
        except Exception:
            pass
        time.sleep(delay * (2 ** attempt))  # Exponential backoff
    raise Exception("Supabase insert failed after retries")
