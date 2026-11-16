"""Bounded first-party Anthropic Messages admission; no SDK cost-table dependency."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .core import BudgetGate, Limits, Stop, integer, read_private_key

ROBOT_TOOLS = frozenset({"actor_act", "capture_scene", "view_env_state", "finish",
    "wait_for_tool", "wait_for_tools", "get_tool_result", "list_tool_jobs",
    "queue_status", "cancel_tool", "flush_queue"})
from .tools import DIRECT_TOOLS

PREFIX = "mcp__uni_robot_agent__"


@dataclass(frozen=True)
class FableLimits(Limits):
    model: str = "claude-fable-5-2"
    batch_usd: str = "20"
    episode_usd: str = "20"
    input_usd_per_million: str = "17.6"
    output_usd_per_million: str = "44"
    max_requests: int = 80
    max_input_tokens: int = 128000
    wall_seconds: int = 3600

    def __post_init__(self):
        from decimal import Decimal
        super().__post_init__()
        if self.model != "claude-fable-5-2" or self.effort != "high":
            raise ValueError("Only frozen Fable 5.2/high is prepared")
        if Decimal(self.input_usd_per_million) < Decimal('17.6') or Decimal(self.output_usd_per_million) < 44:
            raise ValueError("Rates must cover one-hour cache writes and geography surcharge")


def prepare_messages(body, limits):
    if not isinstance(body, dict) or body.get('model') != limits.model:
        raise Stop('model_mismatch')
    if body.get('output_config', {}).get('effort') != limits.effort:
        raise Stop('effort_mismatch')
    if not isinstance(body.get('messages'), list) or not body['messages']:
        raise Stop('missing_messages')
    if body.get('thinking', {}).get('type', 'adaptive') != 'adaptive':
        raise Stop('adaptive_thinking_required')
    context=body.get('context_management')
    if context is not None and context != {'edits':[{'type':'clear_thinking_20251015','keep':'all'}]}:
        raise Stop('context_editing_disabled')
    for key in ('mcp_servers', 'container', 'speed', 'betas'):
        if body.get(key):
            raise Stop('unbudgeted_feature')
    allowed = {'model','messages','system','tools','tool_choice','max_tokens','stream',
               'thinking','output_config','metadata','service_tier','inference_geo','context_management'}
    if set(body) - allowed:
        raise Stop('unknown_messages_field')
    if set(body.get('output_config', {})) - {'effort'}:
        raise Stop('output_format_not_prepared')
    for tool in body.get('tools', []):
        if (not isinstance(tool, dict) or tool.get('type', 'custom') != 'custom'
            or not tool.get('name', '').startswith(PREFIX)
            or tool['name'][len(PREFIX):] not in (DIRECT_TOOLS if getattr(limits, 'control_mode', None) == 'direct_sync' else ROBOT_TOOLS)
            or not isinstance(tool.get('input_schema'), dict)):
            raise Stop('only_robot_mcp_tools_allowed')
    def check_sources(value):
        if isinstance(value, dict):
            if value.get('type') in {'image','document'} and value.get('source',{}).get('type') != 'base64':
                raise Stop('only_inline_media_allowed')
            for child in value.values(): check_sources(child)
        elif isinstance(value,list):
            for child in value: check_sources(child)
    check_sources(body['messages'])
    result=copy.deepcopy(body)
    requested=integer(body.get('max_tokens',limits.max_output_tokens),'max_tokens')
    if not requested: raise Stop('invalid_max_tokens')
    result['max_tokens']=min(requested,limits.max_output_tokens)
    result['service_tier']='standard_only'
    count={k:result[k] for k in ('model','messages','system','tools','thinking','tool_choice') if k in result}
    return result,count


def message_terminal(raw):
    """Reuse the native protocol parser; never rewrite signed thinking."""
    from uni_robot_agent.provider.protocols.messages import message_response
    try:
        message = message_response(raw)
        if message.get('type') != 'message':
            raise ValueError('Not a terminal message')
        return message
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        raise Stop('incomplete_messages_response') from None


def message_usage(message):
    usage=message.get('usage')
    if not isinstance(usage,dict): raise Stop('missing_usage')
    ordinary=integer(usage.get('input_tokens'),'input_tokens')
    writes=integer(usage.get('cache_creation_input_tokens',0),'cache_creation_input_tokens')
    reads=integer(usage.get('cache_read_input_tokens',0),'cache_read_input_tokens')
    output=integer(usage.get('output_tokens'),'output_tokens')
    breakdown=usage.get('cache_creation')
    if breakdown is not None:
        if not isinstance(breakdown,dict) or sum(integer(breakdown.get(k,0),k) for k in
                ('ephemeral_5m_input_tokens','ephemeral_1h_input_tokens')) != writes:
            raise Stop('cache_usage_inconsistent')
    return ordinary+writes+reads,output


def message_charge_upper(usage):
    """Known TTLs narrow settled usage; absent TTL stays at the one-hour bound."""
    from decimal import Decimal,ROUND_CEILING
    message_usage({'usage':usage})
    writes=usage.get('cache_creation_input_tokens',0)
    breakdown=usage.get('cache_creation') or {}
    five=breakdown.get('ephemeral_5m_input_tokens',0)
    amount=(Decimal(usage['input_tokens'])*Decimal('8.8')
            +Decimal(five)*11+Decimal(writes-five)*Decimal('17.6')
            +Decimal(usage.get('cache_read_input_tokens',0))*Decimal('.22')
            +Decimal(usage['output_tokens'])*44)
    return int(amount.to_integral_value(rounding=ROUND_CEILING))


class MessagesBudgetGate(BudgetGate):
    path='/v1/messages'
    count_path='/messages/count_tokens'
    generation_path='/messages'
    output_field='max_tokens'

    def prepare(self,body): return prepare_messages(body,self.ledger.limits)

    def input_bound(self,count): return count+max(1024,(count+19)//20)

    def forward_request(self,path,body,headers):
        selected={k.lower():v for k,v in headers.items() if k.lower() in {'anthropic-version','anthropic-beta'}}
        if 'fast-mode' in selected.get('anthropic-beta',''):
            raise Stop('fast_mode_disabled')
        return self.forward(path,body,selected)

    def complete(self,raw):
        message=message_terminal(raw)
        if message.get('model')!=self.ledger.limits.model: raise Stop('model_mismatch')
        inputs,outputs=message_usage(message)
        self._usage=message['usage']
        state='completed'
        if message['stop_reason'] not in {'end_turn','tool_use'}: state=str(message['stop_reason'])
        return state,inputs,outputs

    def settle(self,identity,inputs,outputs):
        # Only provider-confirmed five-minute writes get the lower rate.
        # Reservation remains17.6/M for every input token, geography still1.1x.
        self.ledger.settle(identity,inputs,outputs,
                           charged_microdollars=message_charge_upper(self._usage))



@dataclass(frozen=True)
class DirectFableLimits(FableLimits):
    """Restricted Direct tool profile; pricing remains the frozen Fable bound."""
    control_mode: str = field(default='direct_sync', init=False)
