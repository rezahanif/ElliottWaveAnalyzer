"""
investing_api.py
---------------------
Ingestion layer for Investing.com Economic Calendar occurrences API.
Bypasses Cloudflare protection using curl_cffi when available.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
import requests

try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False

# Set up logging
logger = logging.getLogger("investing_api")

# Define canonical HIGH_RISK_EVENTS
HIGH_RISK_EVENTS = {
    "Interest Rate Decision",
    "FOMC",
    "CPI",
    "Core CPI",
    "Nonfarm Payrolls",
    "GDP",
    "Unemployment Rate"
}

# Priority list for sorting (lower index = higher priority)
PRIORITY_ORDER = [
    "Interest Rate Decision",
    "FOMC",
    "CPI",
    "Core CPI",
    "Nonfarm Payrolls",
    "GDP",
    "Unemployment Rate"
]

def is_high_risk(event_name: str) -> bool:
    """Check if the event name matches any defined high risk event as a substring."""
    name_lower = event_name.lower()
    return any(hre.lower() in name_lower for hre in HIGH_RISK_EVENTS)

def get_event_priority(event_name: str, importance: str) -> int:
    """
    Get sorting priority of an event.
    Returns 1-7 for matching HIGH_RISK_EVENTS, 8 for other High Impact, and 9 otherwise.
    """
    name_lower = event_name.lower()
    
    # Check 'core cpi' specifically before general 'cpi'
    if "core cpi" in name_lower:
        return 4 # Priority 4
    elif "cpi" in name_lower:
        return 3 # Priority 3
        
    # Check other events by priority index
    order_without_cpi = [
        ("Interest Rate Decision", 1),
        ("FOMC", 2),
        ("Nonfarm Payrolls", 5),
        ("GDP", 6),
        ("Unemployment Rate", 7)
    ]
    for pattern, prio in order_without_cpi:
        if pattern.lower() in name_lower:
            return prio
            
    if importance.lower() == "high":
        return 8
    return 9


def parse_utc_to_local(utc_time_str: str) -> datetime:
    """Parse UTC ISO8601 string and convert to UTC+7 timezone."""
    # Handle Zulu timezone indicator
    t_str = utc_time_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(t_str)
    local_tz = timezone(timedelta(hours=7))
    return dt.astimezone(local_tz)

def get_economic_calendar(
    start_date: datetime,
    end_date: datetime,
    importance: str = "high"
) -> list[dict]:
    """
    Fetch economic calendar events and occurrences from Investing.com.
    Joins events[] and occurrences[] by event_id, converting to canonical structure.
    """
    url = "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/occurrences"
    
    # Format start and end date to ISO8601 UTC strings
    # Ensure they are in UTC
    start_utc = start_date.astimezone(timezone.utc)
    end_utc = end_date.astimezone(timezone.utc)
    
    params = {
        "domain_id": 1,
        "limit": 200,
        "start_date": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "importance": importance
    }
    
    headers = {
        "Origin": "https://www.investing.com",
        "Referer": "https://www.investing.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    data = None
    if CURL_CFFI_AVAILABLE:
        try:
            logger.info("Fetching economic calendar via curl_cffi...")
            resp = curl_requests.get(url, params=params, headers=headers, impersonate="chrome120", timeout=15)
            if resp.status_code == 200:
                data = resp.json()
            else:
                logger.warning(f"curl_cffi fetch returned status code {resp.status_code}. Falling back.")
        except Exception as e:
            logger.warning(f"curl_cffi fetch failed: {e}. Falling back to standard requests.")
            
    if data is None:
        try:
            logger.info("Fetching economic calendar via requests...")
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch economic calendar: {e}")
            raise RuntimeError(f"Could not retrieve economic calendar data: {e}")

    events = data.get("events", [])
    occurrences = data.get("occurrences", [])
    
    # Map event_id to event data
    event_map = {}
    for evt in events:
        if isinstance(evt, dict) and "event_id" in evt:
            event_map[evt["event_id"]] = evt
            
    canonical_list = []
    for occ in occurrences:
        if not isinstance(occ, dict):
            continue
        event_id = occ.get("event_id")
        event = event_map.get(event_id, {})
        
        # Build canonical dict
        canonical_list.append({
            "event_id": event_id,
            "occurrence_id": occ.get("occurrence_id"),
            "event_name": event.get("event_translated", ""),
            "currency": event.get("currency", ""),
            "importance": event.get("importance", ""),
            "occurrence_time": occ.get("occurrence_time", ""),
            "actual": occ.get("actual"),
            "forecast": occ.get("forecast"),
            "previous": occ.get("previous"),
            "actual_to_forecast": occ.get("actual_to_forecast"),
            "revised_to_previous": occ.get("revised_to_previous")
        })
        
    return canonical_list
