#!/bin/bash
# End-to-end smoke test for deployed chatbot
# See build-spec Section 7.6
# Run after: docker compose up -d && sleep 15

set -euo pipefail

echo "=== Chatbot Permensos Smoke Test ==="

# 1. Health check
echo "1. Health check..."
curl -fsSL http://localhost:8000/health || { echo "FAIL: /health endpoint not responding"; exit 1; }
echo "OK"

# 2. Single query (requires FastAPI sidecar mounted at /api/query — see build-spec Section 7.6)
echo "2. Query test: Pasal 5 retrieval..."
RESP=$(curl -fsSL -X POST http://localhost:8000/api/query \
    -H "Content-Type: application/json" \
    -d '{"query":"Apa saja bentuk eksploitasi menurut Permensos 8/2023?"}')

echo "$RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert 'Pasal 5' in r.get('response', ''), 'FAIL: no Pasal 5 in response'
cv = r.get('citations_valid', [])
assert cv and all(cv), 'FAIL: missing or invalid citations'
print('OK — provider=' + str(r.get('llm_provider_used')) + ' latency=' + str(r.get('latency_ms', {}).get('total')) + 'ms')
" || exit 1

echo "=== Smoke test PASSED ==="
