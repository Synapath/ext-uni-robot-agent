"""First-party Responses transport and opt-in usage-aware Sol accounting."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import time

from .core import BudgetGate, Limits, Stop, integer, read_private_key, response_terminal


@dataclass(frozen=True)
class SolLimits(Limits):
    input_usd_per_million: str = '5.5'
    output_usd_per_million: str = '11'
    cached_input_usd_per_million: str = '0.11'

    def __post_init__(self):
        super().__post_init__()
        if self.model != 'gpt-6.1-sol' or self.effort != 'high':
            raise ValueError('Only frozen Sol/high is prepared')
        if (Decimal(self.input_usd_per_million) < Decimal('5.5')
                or Decimal(self.output_usd_per_million) < 11
                or Decimal(self.cached_input_usd_per_million) < Decimal('0.11')
                or Decimal(self.cached_input_usd_per_million) > Decimal(self.input_usd_per_million)):
            raise ValueError('Rates must preserve the conservative price bound')


class SolBudgetGate(BudgetGate):
    def complete(self, raw):
        response = response_terminal(raw)
        if response.get('model') != self.ledger.limits.model:
            raise Stop('response_model_mismatch')
        state, inputs, outputs = super().complete(raw)
        usage = response.get('usage', {})
        details = usage.get('input_tokens_details')
        if not isinstance(details, dict):
            raise Stop('missing_cache_usage')
        self.cached = integer(details.get('cached_tokens'), 'cached_tokens')
        if self.cached > inputs:
            raise Stop('invalid_cache_usage')
        return state, inputs, outputs

    def settle(self, identity, inputs, outputs):
        limits = self.ledger.limits
        amount = (Decimal(inputs - self.cached) * Decimal(limits.input_usd_per_million)
                  + Decimal(self.cached) * Decimal(limits.cached_input_usd_per_million)
                  + Decimal(outputs) * Decimal(limits.output_usd_per_million))
        self.ledger.settle(identity, inputs, outputs,
                           charged_microdollars=int(amount.to_integral_value(rounding=ROUND_CEILING)))


from .tools import DIRECT_TOOLS

@dataclass(frozen=True)
class DirectSolLimits(SolLimits):
    max_input_tokens: int = 922000
    max_output_tokens: int = 128000

    def __post_init__(self):
        super().__post_init__()
        if self.max_output_tokens > 128000 or self.max_input_tokens + self.max_output_tokens > 1050000:
            raise ValueError('Direct limits exceed the frozen Sol model capacity')
        if (self.input_usd_per_million,self.output_usd_per_million,self.cached_input_usd_per_million)!=('5.5','11','0.11'):
            raise ValueError('Direct accounting rates are frozen in its tier-aware settlement policy')

    @property
    def input_ceiling(self):
        return 922000

    @property
    def control_mode(self):
        return 'direct_sync'

    def cost(self, inputs, outputs):
        long = inputs > 272000
        amount = Decimal(inputs) * Decimal('5.5') * (2 if long else 1)
        amount += Decimal(outputs) * Decimal('11') * (Decimal('1.5') if long else 1)
        return int(amount.to_integral_value(rounding=ROUND_CEILING))


class DirectSolBudgetGate(SolBudgetGate):
    def prepare(self, body):
        if not any(isinstance(tool,dict) and (
                tool.get('type')=='namespace' and tool.get('name')=='mcp__uni_robot_agent'
                or tool.get('type')=='function' and tool.get('name') in DIRECT_TOOLS)
                for tool in body.get('tools',[])):
            raise Stop('direct_automatic_compaction_or_missing_robot_tools')
        return super().prepare(body)

    def complete(self, raw):
        self.direct_usage = response_terminal(raw).get('usage')
        return super().complete(raw)

    def settle(self, identity, inputs, outputs):
        # Unknown cache-write TTL retains the documented maximum write rate.
        details = self.direct_usage['input_tokens_details']
        cached = integer(details['cached_tokens'], 'cached_tokens')
        remaining = inputs - cached
        writes = integer(details.get('cache_write_tokens', remaining), 'cache_write_tokens')
        if not 0 <= writes <= remaining:
            raise Stop('invalid_cache_write_usage')
        breakdown = details.get('cache_write_tokens_details')
        if breakdown is None:
            write_amount = Decimal(writes) * 5
        else:
            short = integer(breakdown.get('ttl_30m_tokens', 0), 'ttl_30m_tokens')
            long = integer(breakdown.get('ttl_12h_tokens', 0), 'ttl_12h_tokens')
            if short + long != writes:
                raise Stop('cache_write_usage_inconsistent')
            write_amount = Decimal(short) * Decimal('2.5') + Decimal(long) * 5
        amount = Decimal(remaining - writes) * 2 + Decimal(cached) * Decimal('.1') + write_amount
        amount *= 2 if inputs > 272000 else 1
        amount += Decimal(outputs) * 10 * (Decimal('1.5') if inputs > 272000 else 1)
        amount *= Decimal('1.1')
        return self.ledger.settle(identity, inputs, outputs,
            charged_microdollars=int(amount.to_integral_value(rounding=ROUND_CEILING)))


class CacheStableDirectSolBudgetGate(DirectSolBudgetGate):
    """Optional trusted verification of append-only effective text and cache TTL."""
    def prepare(self, body):
        from uni_robot_agent.context.policies.append_only_observation import (
            validate_effective_request, without_cache_markers, fingerprint)
        validate_effective_request(body)
        prepared, count = super().prepare(body)
        if prepared.get('prompt_cache_options') != {'mode': 'explicit', 'ttl': '30m'}:
            raise Stop('cache_stable_policy_required')
        items = without_cache_markers(prepared['input'])
        if items and any(p.get('text', '').startswith('[RUA CURRENT OBSERVATION]\n')
                         for p in items[-1].get('content', []) if isinstance(p, dict)):
            items = items[:-1]
        hashes = [fingerprint(i) for i in items]
        previous = getattr(self, '_stable_text', [])
        header = fingerprint({k: prepared.get(k) for k in (
            'model', 'instructions', 'tools', 'reasoning', 'parallel_tool_calls', 'prompt_cache_key')})
        if (hashes[:len(previous)] != previous or
                getattr(self, '_stable_header', header) != header):
            raise Stop('trusted_append_only_history_changed')
        self._stable_text, self._stable_header = hashes, header
        return prepared, count

