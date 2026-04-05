import json
from pathlib import Path

files = sorted(Path("data/results").glob("structured_experiment_*.jsonl"))
with open(files[-1]) as f:
    records = [json.loads(line) for line in f if line.strip()]

# Show first failure per (model, schema) combination
seen = set()
for r in records:
    if r["success"]:
        continue
    key = (r["model"], r["schema_name"])
    if key in seen:
        continue
    seen.add(key)
    print("=" * 60)
    print("MODEL:   " + r["model"])
    print("SCHEMA:  " + r["schema_name"])
    print("TEMP:    " + str(r["temperature"]))
    print("RAW RESPONSE (first 600 chars):")
    print(r["raw_response"][:600])
    print()
