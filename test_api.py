import sys
import time

import requests


def wait_for_health(base_url: str, timeout_s: int = 20) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"API did not become healthy at {base_url}. Last error: {last_error}")


def main() -> int:
    base_url = "http://localhost:8000"
    wait_for_health(base_url)

    payload = {
        "step": 10,
        "type": "TRANSFER",
        "amount": 12000,
        "nameOrig": "C9000",
        "oldbalanceOrg": 15000,
        "newbalanceOrig": 3000,
        "nameDest": "M5000",
        "oldbalanceDest": 0,
        "newbalanceDest": 12000,
        "isFlaggedFraud": 0,
    }
    resp = requests.post(f"{base_url}/predict", json=payload, timeout=5)
    if resp.status_code != 200:
        print(f"Unexpected status: {resp.status_code} - {resp.text}")
        return 1
    data = resp.json()
    required = {"fraud_probability", "is_fraud", "reason_code"}
    if not required.issubset(set(data.keys())):
        print(f"Missing keys in response: {data}")
        return 1
    print("OK:", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
