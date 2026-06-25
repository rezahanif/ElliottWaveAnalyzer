from curl_cffi import requests
from datetime import datetime, timedelta, timezone

url = "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/occurrences"

now = datetime.now(timezone.utc)
start_date = now - timedelta(days=3)
end_date = now + timedelta(days=7)

params = {
    "domain_id": 1,
    "limit": 200,
    "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "end_date": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "importance": "high"
}

headers = {
    "Origin": "https://www.investing.com",
    "Referer": "https://www.investing.com/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

try:
    resp = requests.get(url, params=params, headers=headers, impersonate="chrome120", timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        print("Success!")
        if "events" in data:
            print(f"Events is a {type(data['events'])} with {len(data['events'])} items.")
            if len(data['events']) > 0:
                print("First event item:", data['events'][0])
        if "occurrences" in data:
            print(f"Occurrences is a {type(data['occurrences'])} with {len(data['occurrences'])} items.")
            if len(data['occurrences']) > 0:
                print("First occurrence item:", data['occurrences'][0])
    else:
        print(f"Status Code: {resp.status_code}")
        print("Response Text:", resp.text[:500])
except Exception as e:
    print(f"Request failed: {e}")
