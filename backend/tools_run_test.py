from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)
payload = {"message":"test","session_id":"text2sql-test-tc","source":"postgres","route_mode":"sql"}
print('Sending payload:', payload)
resp = client.post('/nanobot/chat', json=payload, timeout=120)
print('Status:', resp.status_code)
print('Response body:', resp.text[:1000])
metrics = client.get('/metrics')
print('\nMetrics snippet:')
for line in metrics.text.splitlines():
    if 'postgres_query_latency_seconds' in line or 'nl2sql_requests_total' in line or 'nl2sql_active_requests' in line:
        print(line)
