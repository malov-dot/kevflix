"""Publish cache invalidations with provider backoff; never execute app code.

The CDN can report status=finished while a particular path is throttled. Only
matching delivered source counts as complete. Unchanged paths consume no purge.
"""
import argparse
import json
from pathlib import Path
import time
from urllib import request, error

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "malov-dot/kevflix"


def read(url):
    req = request.Request(url, headers={"User-Agent": "KevFLIX-Delivery-Publisher"})
    with request.urlopen(req, timeout=20) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Delivery response exceeds its allowed size")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wait", type=int, default=2100)
    args = parser.parse_args()
    paths = ["package.json", "tv/bootstrap/index.html", "tv/bootstrap/bootstrap.js", "tv/bootstrap/style.css"]
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / "tv/launchers").glob("*/index.html")]
    targets = {"/gh/" + REPOSITORY + "/" + p: (ROOT / p).read_bytes().replace(b"\r\n", b"\n") for p in paths}
    targets["/gh/" + REPOSITORY + "@main/package.json"] = targets["/gh/" + REPOSITORY + "/package.json"]
    pending = {path: 0.0 for path in targets}
    deadline = time.monotonic() + max(30, min(2400, args.max_wait))
    while pending:
        for path in list(pending):
            if time.monotonic() < pending[path]:
                continue
            try:
                delivered = read("https://cdn.jsdelivr.net" + path).replace(b"\r\n", b"\n")
                if delivered == targets[path]:
                    del pending[path]
                    print("Delivered source: " + path, flush=True)
                    continue
            except (OSError, ValueError):
                pass
            delay = 30
            try:
                result = json.loads(read("https://purge.jsdelivr.net" + path))
                status = result.get("paths", {}).get(path, {})
                if status.get("throttled"):
                    delay = max(30, int(status.get("throttlingReset", 60)) + 5)
                    print("Cache cooldown: " + path + "; retry in " + str(delay) + " seconds", flush=True)
                else:
                    print("Cache invalidation requested: " + path, flush=True)
            except error.HTTPError as exc:
                delay = max(60, int(exc.headers.get("Retry-After", "60")))
            except (OSError, ValueError):
                delay = 60
            pending[path] = time.monotonic() + delay
        if pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit("Release uploaded; CDN delivery still pending for: " + ", ".join(pending))
            # Runs in its own CI job. Packaging later releases is not blocked.
            time.sleep(min(remaining, max(1, min(pending.values()) - time.monotonic())))
    print(json.dumps({"source_delivery_complete": True, "paths": len(targets), "application_executed": False}))


if __name__ == "__main__":
    main()
