"""Persistent provider admission; no transport or SDK dependency."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Callable


class Stop(RuntimeError):
    """A stable reason safe to persist or show to the planner."""


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stop(f"invalid_{name}")
    return value


@dataclass(frozen=True)
class Limits:
    model: str = "gpt-6.1-sol"
    effort: str = "high"
    batch_usd: str = "20"
    episode_usd: str = "5"
    # Deliberately covers even the documented 12h cache-write input price.
    input_usd_per_million: str = "5"
    output_usd_per_million: str = "10"
    max_requests: int = 30
    max_input_tokens: int = 64000
    max_output_tokens: int = 16384
    wall_seconds: int = 1200

    def __post_init__(self):
        for name in ("batch_usd", "episode_usd", "input_usd_per_million", "output_usd_per_million"):
            value = Decimal(getattr(self, name))
            if not value.is_finite() or value <= 0:
                raise ValueError(name)
        for name in ("max_requests", "max_input_tokens", "max_output_tokens", "wall_seconds"):
            if integer(getattr(self, name), name) == 0:
                raise ValueError(name)
        if self.max_input_tokens > self.input_ceiling:
            raise ValueError("long-context pricing is outside this policy")
        if Decimal(self.episode_usd) > Decimal(self.batch_usd):
            raise ValueError("episode budget exceeds batch")

    @property
    def input_ceiling(self):
        return 272000

    def cost(self, inputs: int, outputs: int) -> int:
        # Integer microdollars: 1 token * $/million equals microdollars.
        amount = (Decimal(integer(inputs, "input_tokens")) * Decimal(self.input_usd_per_million)
                  + Decimal(integer(outputs, "output_tokens")) * Decimal(self.output_usd_per_million))
        return int(amount.to_integral_value(rounding=ROUND_CEILING))

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class Ledger:
    def __init__(self, path: Path, limits: Limits, *, clock=time.time):
        self.path, self.limits, self.clock = Path(path), limits, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise Stop("ledger_symlink")
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        os.chmod(self.path, 0o600)
        self.lock = threading.RLock()
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS policy (digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS episode (id TEXT PRIMARY KEY, started REAL, stopped TEXT);
        CREATE TABLE IF NOT EXISTS call (id TEXT PRIMARY KEY, episode TEXT, reserved INTEGER,
            charged INTEGER, state TEXT, input_tokens INTEGER, output_tokens INTEGER);
        CREATE TABLE IF NOT EXISTS request_identity (digest TEXT PRIMARY KEY, call_id TEXT);
        """)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT digest FROM policy").fetchone()
            if row and row[0] != limits.digest:
                raise Stop("ledger_policy_mismatch")
            if row is None:
                self.db.execute("INSERT INTO policy VALUES (?)", (limits.digest,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def claim(self, episode: str):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                # This prepared authorization permits exactly one episode total.
                if self.db.execute("SELECT COUNT(*) FROM episode").fetchone()[0]:
                    raise Stop("single_episode_already_claimed")
                self.db.execute("INSERT INTO episode VALUES (?, ?, NULL)", (episode, self.clock()))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def stop(self, episode: str, reason: str):
        with self.lock:
            self.db.execute("UPDATE episode SET stopped=COALESCE(stopped, ?) WHERE id=?", (reason, episode))

    def check(self, episode: str):
        row = self.db.execute("SELECT started, stopped FROM episode WHERE id=?", (episode,)).fetchone()
        if row is None:
            raise Stop("episode_not_claimed")
        if row[1]:
            raise Stop(row[1])
        if self.clock() - row[0] >= self.limits.wall_seconds:
            raise Stop("wall_time_limit")

    def check_generation(self, episode: str):
        """Reject a used-up generation allowance before even counting input."""
        with self.lock:
            self.check(episode)
            count = self.db.execute("SELECT COUNT(*) FROM call WHERE episode=?", (episode,)).fetchone()[0]
            if count >= self.limits.max_requests:
                raise Stop("request_limit")

    def reserve(self, episode: str, inputs: int, outputs: int, *, fingerprint: str | None = None):
        integer(inputs, "input_tokens")
        integer(outputs, "output_tokens")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.check(episode)
                if inputs > self.limits.max_input_tokens or outputs > self.limits.max_output_tokens:
                    raise Stop("token_limit")
                count, used = self.db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(COALESCE(charged, reserved)),0) FROM call WHERE episode=?",
                    (episode,)).fetchone()
                if count >= self.limits.max_requests:
                    raise Stop("request_limit")
                # A crash leaves the outstanding reservation and prohibits recovery/replay.
                if self.db.execute("SELECT 1 FROM call WHERE state='reserved'").fetchone():
                    raise Stop("unsettled_request")
                if fingerprint and self.db.execute("SELECT 1 FROM request_identity WHERE digest=?", (fingerprint,)).fetchone():
                    raise Stop("generation_replay_refused")
                amount = self.limits.cost(inputs, outputs)
                batch = self.db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM call").fetchone()[0]
                if used + amount > Decimal(self.limits.episode_usd) * 1000000:
                    raise Stop("episode_budget_limit")
                if batch + amount > Decimal(self.limits.batch_usd) * 1000000:
                    raise Stop("batch_budget_limit")
                identity = uuid.uuid4().hex
                self.db.execute("INSERT INTO call VALUES (?, ?, ?, NULL, 'reserved', ?, ?)",
                                (identity, episode, amount, inputs, outputs))
                if fingerprint:
                    self.db.execute("INSERT INTO request_identity VALUES (?, ?)", (fingerprint, identity))
                self.db.execute("COMMIT")
                return identity
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def settle(self, identity: str, inputs: int, outputs: int, *, charged_microdollars=None):
        upper = self.limits.cost(inputs, outputs)
        charge = upper if charged_microdollars is None else integer(charged_microdollars, 'charge')
        if charge > upper:
            raise Stop('settlement_above_price_bound')
        with self.lock:
            row = self.db.execute("SELECT episode,reserved,state,input_tokens,output_tokens FROM call WHERE id=?", (identity,)).fetchone()
            if row is None or row[2] != "reserved":
                raise Stop("invalid_settlement")
            self.db.execute("UPDATE call SET charged=?,state='settled',input_tokens=?,output_tokens=? WHERE id=?",
                            (charge, inputs, outputs, identity))
            if charge > row[1]:
                self.stop(row[0], "usage_exceeded_reservation")
                raise Stop("usage_exceeded_reservation")
            if inputs > row[3] or outputs > row[4]:
                self.stop(row[0], "usage_exceeded_token_bounds")
                raise Stop("usage_exceeded_token_bounds")

    def snapshot(self):
        with self.lock:
            used, pending, count = self.db.execute("""SELECT
                COALESCE(SUM(COALESCE(charged,reserved)),0),
                COALESCE(SUM(CASE WHEN charged IS NULL THEN reserved ELSE 0 END),0),COUNT(*) FROM call""").fetchone()
            return {"accounting": "conservative_microdollars_not_provider_bill", "charged_or_reserved_usd": used / 1e6,
                    "unsettled_reserved_usd": pending / 1e6, "generation_attempts": count,
                    "episodes": [dict(zip(("id", "started", "stop_reason"), row))
                                 for row in self.db.execute("SELECT * FROM episode")], "policy_sha256": self.limits.digest}


def prepare_request(body: dict, limits: Limits):
    if not isinstance(body, dict) or body.get("model") != limits.model:
        raise Stop("model_mismatch")
    if body.get("reasoning", {}).get("effort") != limits.effort:
        raise Stop("effort_mismatch")
    if any(body.get(key) for key in ("previous_response_id", "conversation", "background")):
        raise Stop("hidden_history_or_background")
    robot_tools = {"actor_act", "cancel_tool", "capture_scene", "finish", "flush_queue",
                   "get_tool_result", "list_tool_jobs", "queue_status", "view_env_state",
                   "wait_for_tool", "wait_for_tools"}
    mode = getattr(limits, 'control_mode', None)
    direct = mode in ('direct_sync', 'adaptive_direct_sync', 'saved_state_probe')
    if direct:
        from .tools import DIRECT_TOOLS, SDK_NATIVE_TOOLS
        robot_tools = {'request_context'} if mode == 'saved_state_probe' else DIRECT_TOOLS
        if mode == 'adaptive_direct_sync':robot_tools=robot_tools | {'request_context'}
        body=dict(body,tools=[tool for tool in body.get('tools',[]) if not
            (isinstance(tool,dict) and tool.get('type')=='function' and tool.get('name') in SDK_NATIVE_TOOLS)])
    for tool in body.get("tools", []):
        if not isinstance(tool, dict):
            raise Stop("hosted_tools_disabled")
        if tool.get("type") in {"function", "custom"}:
            if direct and tool.get('name', '').rsplit('__', 1)[-1].rsplit('.',1)[-1] not in robot_tools:
                raise Stop('only_direct_robot_tools_allowed')
            continue
        # Codex groups local MCP functions in a Responses namespace. This does
        # not turn them into hosted services; preserve their names and schemas.
        if (tool.get("type") == "namespace" and tool.get("name") == "mcp__uni_robot_agent"
                and isinstance(tool.get("tools"), list) and tool["tools"]
                and all(isinstance(child, dict) and child.get("type") == "function"
                        and child.get("name") in robot_tools for child in tool["tools"])):
            continue
        raise Stop("hosted_tools_disabled")
    prepared = dict(body)
    requested = integer(body.get("max_output_tokens", limits.max_output_tokens), "max_output_tokens")
    if requested == 0:
        raise Stop("invalid_max_output_tokens")
    prepared["max_output_tokens"] = min(requested, limits.max_output_tokens)
    prepared.update(service_tier="default", store=False, truncation="disabled")
    # Never allow SDK settings to request paid cache retention/processing tiers.
    prepared.pop("prompt_cache_retention", None)
    count_body = {key: value for key, value in prepared.items() if key in {
        "model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls",
        "reasoning", "text", "personality", "truncation"}}
    return prepared, count_body


def response_terminal(raw: bytes):
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        value = None
        for line in raw.splitlines():
            if line.startswith(b"data:"):
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                    value = event.get("response")
    if not isinstance(value, dict):
        raise Stop("missing_terminal_response")
    return value


def completed_response(raw: bytes):
    value = response_terminal(raw)
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise Stop("missing_usage")
    inputs = integer(usage.get("input_tokens"), "input_tokens")
    outputs = integer(usage.get("output_tokens"), "output_tokens")
    return value.get("status"), inputs, outputs


class BudgetGate:
    """forward(path, body) returns status, headers, raw bytes; no retries."""
    path = "/v1/responses"
    count_path = "/responses/input_tokens"
    generation_path = "/responses"
    output_field = "max_output_tokens"

    def prepare(self, body):
        return prepare_request(body, self.ledger.limits)

    def complete(self, raw):
        return completed_response(raw)

    def input_bound(self, count):
        return count

    def settle(self, identity, inputs, outputs):
        self.ledger.settle(identity, inputs, outputs)

    def forward_request(self, path, body, headers):
        return self.forward(path, body)
    def __init__(self, ledger: Ledger, episode: str, forward: Callable):
        self.ledger, self.episode, self.forward = ledger, episode, forward
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def handle(self, body, headers=None, *, exchange=None):
        with self.lock:
            self._exchange = exchange
            try:
                self.ledger.check_generation(self.episode)
                prepared, count_body = self.prepare(body)
                status, _, raw = self.forward_request(self.count_path, count_body, headers or {})
                if status != 200:
                    raise Stop("input_count_failed")
                try:
                    inputs = integer(json.loads(raw).get("input_tokens"), "input_count")
                except (ValueError, AttributeError):
                    raise Stop("invalid_input_count") from None
                fingerprint = hashlib.sha256(json.dumps(count_body, sort_keys=True).encode()).hexdigest()
                identity = self.ledger.reserve(self.episode, self.input_bound(inputs), prepared[self.output_field],
                                               fingerprint=fingerprint)
                status, headers, raw = self.forward_request(self.generation_path, prepared, headers or {})
                if status != 200:
                    raise Stop("generation_http_error")
                state, actual_inputs, actual_outputs = self.complete(raw)
                self.settle(identity, actual_inputs, actual_outputs)
                if state != "completed":
                    raise Stop("generation_" + state)
                return status, headers, raw
            except Exception as error:
                reason = str(error) if isinstance(error, Stop) else "upstream_transport_error"
                self.ledger.stop(self.episode, reason)
                self.stopped.set()
                raise Stop(reason) from None
            finally:
                self._exchange = None


def read_private_key(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        meta = os.fstat(fd)
        if (not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.getuid()
                or stat.S_IMODE(meta.st_mode) != 0o600 or meta.st_size > 8192):
            raise Stop("key_requires_owned_regular_file_mode_0600")
        try:
            value = os.read(fd, 8193).decode('ascii').strip()
        except UnicodeDecodeError:
            raise Stop("invalid_key_file") from None
        if (not value.startswith('sk-') or len(value) > 8192
                or any(not (char.isalnum() or char in '-_') for char in value)):
            raise Stop("invalid_key_file")
        return value
    finally:
        os.close(fd)

