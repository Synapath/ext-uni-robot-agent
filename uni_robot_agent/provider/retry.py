"""Opt-in phase-aware network retries; one request/reservation, no robot replay."""
from dataclasses import dataclass
import hashlib
import json
import math
import threading
import time



@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    initial_delay_s: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_s: float = 8.0

    def __post_init__(self):
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        for name in ('initial_delay_s', 'backoff_multiplier', 'max_delay_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name)
        if self.initial_delay_s < 0 or self.backoff_multiplier < 1 or self.max_delay_s < self.initial_delay_s:
            raise ValueError('invalid network retry delay policy')

    def delay(self, attempt):
        return min(self.max_delay_s, self.initial_delay_s * self.backoff_multiplier**attempt)


def error_details(error):
    """No arbitrary exception text, URL or authentication headers enter logs."""
    chain, codes, seen = [], [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current)); chain.append(type(current).__name__)
        for name in ('errno', 'verify_code'):
            value = getattr(current, name, None)
            if type(value) is int:
                codes.append({name: value})
        current = current.__cause__ or current.__context__
    return {'error_type': type(error).__name__, 'cause_types': chain, 'cause_codes': codes}



def retryable(path, error):
    import httpx
    generation = path in ('/responses', '/messages', '/v1/responses', '/v1/messages')
    count = path in ('/responses/input_tokens', '/messages/count_tokens',
                     '/v1/responses/input_tokens', '/v1/messages/count_tokens')
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        return generation or count
    return count and isinstance(error, (
        httpx.ReadError, httpx.WriteError, httpx.ReadTimeout, httpx.WriteTimeout,
        httpx.RemoteProtocolError, httpx.ProxyError, httpx.PoolTimeout))
