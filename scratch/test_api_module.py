import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.waveconf.ingestion.investing_api import (
    get_economic_calendar,
    is_high_risk,
    get_event_priority,
    parse_utc_to_local
)

# Test date parsing
utc_str = "2026-06-24T12:00:00Z"
local_dt = parse_utc_to_local(utc_str)
print(f"UTC: {utc_str} -> Local (UTC+7): {local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Test fetching calendar
now = datetime.now(timezone.utc)
start = now - timedelta(days=2)
end = now + timedelta(days=5)
print(f"Fetching from {start} to {end}...")
try:
    events = get_economic_calendar(start, end, "high")
    print(f"Successfully fetched {len(events)} events.")
    if events:
        print("Sample canonical event:")
        print(events[0])
        # Verify priority and high risk classification
        name = events[0]["event_name"]
        imp = events[0]["importance"]
        print(f"Name: {name}")
        print(f"High risk? {is_high_risk(name)}")
        print(f"Priority: {get_event_priority(name, imp)}")
except Exception as e:
    print(f"Failed to fetch or parse: {e}")
