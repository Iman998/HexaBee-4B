"""Minimal OpenAI-compatible chat transport with validated SQLite caching.

There is no network activity unless a command constructs Client(execute=True).
Never logs credentials, endpoint URLs, request bodies, or server error bodies.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.error
import urllib.request

from .io import digest


class Client:
    def __init__(self, config, cache, execute=False, transport=None):
        self.config = dict(config)
        self.execute = execute
        self.transport = transport
        self.calls = 0
        self.cache_hits = 0
        self.base = os.environ.get(config.get("base_url_env", "PCT_API_BASE"), "").rstrip("/")
        self.key = os.environ.get(config.get("api_key_env", "PCT_API_KEY"), "")
        if execute and transport is None and (not self.base or not self.key):
            raise ValueError("Set the configured API endpoint/key environment variables first")
        if (
            execute
            and transport is None
            and not (
                self.base.startswith("https://")
                or self.base.startswith(("http://localhost:", "http://127.0.0.1:"))
            )
        ):
            raise ValueError("Use HTTPS, or HTTP on localhost for local inference")

        if not isinstance(config.get("model"), str) or not config["model"].strip():
            raise ValueError("API configuration requires a model identifier")
        for field, default in (("attempts", 3), ("max_tokens", 5000)):
            value = config.get(field, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if config.get("timeout", 120) <= 0:
            raise ValueError("timeout must be positive")
        self.db = sqlite3.connect(cache)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")

    def close(self):
        self.db.close()

    def complete(self, messages, validate=None):
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config.get("temperature", 0),
            "max_tokens": self.config.get("max_tokens", 5000),
        }
        for key in ("top_p", "seed"):
            if self.config.get(key) is not None:
                payload[key] = self.config[key]
        key = digest([self.base, self.config.get("revision"), payload])
        saved = self.db.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
        if saved:
            try:
                if validate:
                    validate(saved[0])
                self.cache_hits += 1
                return saved[0]
            except (ValueError, TypeError, KeyError):
                self.db.execute("DELETE FROM responses WHERE key=?", (key,))
                self.db.commit()
        if not self.execute:
            raise ValueError("API execution disabled. Review the command, then pass --execute")
        attempts = int(self.config.get("attempts", 3))
        for attempt in range(attempts):
            try:
                self.calls += 1
                if self.transport is not None:
                    content = self.transport(payload)
                else:
                    req = urllib.request.Request(
                        self.base + "/chat/completions",
                        data=json.dumps(payload, ensure_ascii=False).encode(),
                        headers={
                            "Authorization": "Bearer " + self.key,
                            "Content-Type": "application/json",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=self.config.get("timeout", 120)) as r:
                        result = json.load(r)
                    choice = result["choices"][0]
                    if choice.get("finish_reason") == "length":
                        raise ValueError("Completion hit its token limit; response not cached")
                    content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Service returned an empty completion")
                if validate:
                    validate(content)
                self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?)", (key, content))
                self.db.commit()
                return content
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise RuntimeError(f"Chat request failed with HTTP {exc.code}") from None
                error = f"HTTP {exc.code}"
            except (ValueError, TypeError, KeyError, IndexError):
                error = "invalid or incomplete response"
            except (urllib.error.URLError, TimeoutError, OSError):
                error = "connection failure"
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Chat request failed after {attempts} attempts ({error})") from None


def parse_json(raw):
    decoder = json.JSONDecoder()
    for i, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[i:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON value in response")
