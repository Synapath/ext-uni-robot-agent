"""Optional trusted provider service using RUA's gateway and cancellable client."""
from __future__ import annotations

import json
from pathlib import Path

from uni_robot_agent.provider.gateway import ProviderGateway
from uni_robot_agent.provider.protocols.messages import MessagesProtocol
from uni_robot_agent.provider.protocols.responses import ResponsesProtocol
from uni_robot_agent.provider.retry import RetryPolicy
from .core import Stop, read_private_key


def serve_gate(gate, token, *, key_file=None, upstream_base_url=None,
               retry_policy=None, record=None, trace_dir=None, artifacts=None):
    """Start one RUA-owned provider. Inject forward only for offline fixtures.

    Real credentials stay in this process. Generation/count retry is handled by
    ModelClient, under the same ledger deadline, never by an SDK or an outer loop.
    """
    messages = gate.generation_path == '/messages'
    headers = None
    if key_file is not None:
        key = read_private_key(Path(key_file))
        headers = ({'x-api-key': key} if messages
                   else {'Authorization': 'Bearer '+key})
    if gate.forward is None and headers is None:
        raise Stop('trusted_provider_key_required')
    protocol = MessagesProtocol(keepalive_interval_s=60) if messages else ResponsesProtocol()
    gateway = ProviderGateway(
        upstream_base_url or ('https://api.anthropic.com/v1' if messages else 'https://api.openai.com/v1'),
        protocol=protocol, admission=gate, local_bearer_token=token,
        upstream_headers=headers, retry_policy=retry_policy or RetryPolicy(max_retries=0),
        request_guard=lambda: gate.ledger.check(gate.episode),
        attempt_recorder=record, trace_dir=trace_dir, artifacts=artifacts,
        request_timeout_s=min(180 if messages else 300, gate.ledger.limits.wall_seconds), max_retries=0,
    )
    if gate.forward is None:
        def forward(path, body, selected=None):
            if path not in (gate.count_path, gate.generation_path):
                raise Stop('unprepared_endpoint')
            outgoing = dict(selected or {})
            outgoing.update(headers or {})
            if messages:
                outgoing.setdefault('anthropic-version', '2023-06-01')
            outgoing.update({'Content-Type': 'application/json', 'Accept-Encoding': 'identity'})
            exchange = getattr(gate, '_exchange', None)
            if exchange is None:
                exchange = {'provider_attempts': []}
            status, response_headers, raw = gateway._client.send(
                target=gateway._adapter.target_url(gateway._upstream_base_url, path),
                request_bytes=json.dumps(body).encode(), headers=outgoing, exchange=exchange,
            )
            return status, response_headers, raw
        gate.forward = forward
    endpoint = gateway.start()
    return ProviderService(gateway, gate), endpoint


class ProviderService:
    def __init__(self, gateway, admission):
        self.gateway, self.admission = gateway, admission

    def cancel(self):
        self.gateway._client.cancel()

    def shutdown(self):
        self.gateway.stop(drain_timeout_s=15)

    def server_close(self):
        """Compatibility with old fixture cleanup; shutdown already owns close."""
