"""Native Messages cache/history envelope around the shared robot projection.

Only user observations enter the Responses-shaped inspection view. Assistant
text, signed thinking and tool identities remain native Messages blocks.
"""
from copy import deepcopy
from dataclasses import replace

from uni_robot_agent.context.history import ContextRequest
from uni_robot_agent.context.messages import observation_view, project_observations
from uni_robot_agent.context.observation_history import estimate_context
from uni_robot_agent.context.robot_observation import CURRENT
from .append_only_observation import (
    AppendOnlyObservationPolicy, CACHE_HISTORY_PROMPT, fingerprint,
    validate_effective_request, without_cache_markers,
)

CACHE_CONTROL = {'type': 'ephemeral', 'ttl': '5m'}
HEADER_KEYS = ('model', 'system', 'tools', 'tool_choice', 'thinking',
               'output_config', 'context_management')


def without_cache_control(value):
    """Discard SDK-selected cache metadata, not signed text or image content."""
    if isinstance(value, dict):
        return {k: without_cache_control(v) for k, v in value.items() if k != 'cache_control'}
    if isinstance(value, list):
        return [without_cache_control(v) for v in value]
    return value


def system_text(body):
    system = body.get('system', '')
    return system if isinstance(system, str) else '\n'.join(p.get('text', '') for p in system)


def canonical_messages(body):
    """The CLI alternates equivalent string/list encodings for system text.

    Normalize only user/system text containers. Signed assistant blocks are not
    re-encoded or interpreted. The resulting upstream prefix stays byte stable.
    """
    body = without_cache_control(body)
    def parts(value):
        return [{'type': 'text', 'text': value}] if isinstance(value, str) else value
    if 'system' in body:
        body['system'] = parts(body['system'])
    for message in body['messages']:
        if message['role'] not in ('user', 'system'):
            continue
        message['content'] = parts(message['content'])
        for block in message['content']:
            if block.get('type') == 'tool_result':
                block['content'] = parts(block.get('content', ''))
    return body


def restore_archived_images(old, new):
    """Rebuild only previously recorded media pruned by the pinned SDK.

    Claude removes old inline images or substitutes a media-limit notice. No
    text, signature, tool argument or identity change is accepted. Recovered
    historical observations are still projected to text; only fixed ICL pixels
    and the current triplet reach the upstream provider.
    """
    def parts(expected, received):
        out, index = [], 0
        for value in expected:
            current = received[index] if index < len(received) else None
            if value.get('type') == 'image':
                if current and current.get('type') == 'image':
                    if value != current:raise ValueError('append_only_messages_history_or_header_changed')
                    index += 1
                elif current and current.get('type') == 'text' and current.get('text', '').startswith('[media removed: request limit.'):
                    index += 1
                out.append(deepcopy(value))
                continue
            if current is None:
                raise ValueError('append_only_messages_history_or_header_changed')
            candidate = deepcopy(current)
            if value.get('type') == candidate.get('type') == 'tool_result':
                candidate['content'] = parts(value['content'], candidate['content'])
            if candidate != value:
                raise ValueError('append_only_messages_history_or_header_changed')
            out.append(candidate);index += 1
        if index != len(received):raise ValueError('append_only_messages_history_or_header_changed')
        return out
    if old.get('role') == 'assistant':
        if old != new:raise ValueError('append_only_messages_history_or_header_changed')
        return deepcopy(new)
    candidate = deepcopy(new)
    candidate['content'] = parts(old['content'], new['content'])
    if candidate != old:raise ValueError('append_only_messages_history_or_header_changed')
    return candidate


def persistent_messages(body):
    messages = without_cache_control(body['messages'])
    if messages and any(p.get('text', '').startswith(CURRENT+'\n')
                        for p in messages[-1].get('content', []) if isinstance(p, dict)):
        return messages[:-1]
    return messages


def mark_boundaries(body):
    """Cache the fixed task and the last two receipts; never the dynamic tail."""
    fixed, receipts = [], []
    for mi, message in enumerate(body['messages']):
        if message['role'] != 'user':
            continue
        for bi, block in enumerate(message['content']):
            if block.get('text') == '[RUA FIXED CONTEXT END]':
                fixed.append((mi, bi))
            if block.get('type') == 'tool_result':
                receipts.append((mi, bi))
    if len(fixed) != 1:
        raise ValueError('fixed_context_message_missing')
    selected = fixed + receipts[-2:]
    for mi, bi in selected:
        body['messages'][mi]['content'][bi]['cache_control'] = dict(CACHE_CONTROL)
    return selected


def validate_messages_effective(body, *, profile):
    """Validate selected pixels/state plus exact native cache boundary positions."""
    clean = without_cache_control(body)
    expected = deepcopy(clean)
    mark_boundaries(expected)
    if expected != body:
        raise ValueError('messages_cache_boundaries_required')
    view = observation_view(clean)
    view['instructions'] = system_text(clean)
    validate_effective_request(view, profile=profile)


class AppendOnlyMessagesPolicy:
    name, version = 'append_only_observation', 'messages-1'

    def __init__(self, *, profile):
        self.projector = AppendOnlyObservationPolicy(profile=profile)
        self.profile = profile
        self._raw, self._header = [], None
        self._messages = []

    @property
    def options(self):
        return {**self.projector.options, 'ttl': '5m', 'wire_format': 'messages'}

    def select(self, request):
        if request.preserve_history:
            raise ValueError('append_only_requires_native_messages_without_compaction')
        body = canonical_messages(request.body)
        if len(body['messages']) < len(self._messages):
            raise ValueError('append_only_messages_history_or_header_changed')
        for index, old in enumerate(self._messages):
            body['messages'][index] = restore_archived_images(old, body['messages'][index])
        if any(k in body for k in ('prompt_cache_options', 'prompt_cache_breakpoint')):
            raise ValueError('responses_cache_fields_in_messages')
        context = body.get('context_management')
        if context not in (None, {'edits': [{'type': 'clear_thinking_20251015', 'keep': 'all'}]}):
            raise ValueError('messages_compaction_disabled')
        header = fingerprint({k: body.get(k) for k in HEADER_KEYS})
        hashes = [fingerprint(m) for m in body['messages']]
        if self._header is not None and (header != self._header or hashes[:len(self._raw)] != self._raw):
            raise ValueError('append_only_messages_history_or_header_changed')
        snapshot = (self.projector._raw, self.projector._text,
                    self.projector._header, self.projector._head)
        try:
            decisions = []
            def project(view):
                decision = self.projector.select(ContextRequest(view))
                decisions.append(decision)
                return without_cache_markers(decision.request.body), decision.options
            effective, stats = project_observations(body, project)
            note = CACHE_HISTORY_PROMPT+'\n'+self.profile.instruction()
            system = effective.get('system', [])
            if isinstance(system, str):
                system = [{'type': 'text', 'text': system}] if system else []
            effective['system'] = [{'type': 'text', 'text': note}, *system]
            boundaries = mark_boundaries(effective)
            validate_messages_effective(effective, profile=self.profile)
        except Exception:
            (self.projector._raw, self.projector._text,
             self.projector._header, self.projector._head) = snapshot
            raise
        self._raw, self._header = hashes, header
        self._messages = deepcopy(body['messages'])
        before, after = estimate_context(request.body), estimate_context(effective)
        return replace(decisions[0], request=replace(request, body=effective),
                       version=self.version, estimated_tokens_before=before['text']+before['images'],
                       estimated_tokens_after=after['text']+after['images'],
                       options={**stats, **self.options, 'cache_boundaries': boundaries,
                                'stable_prefix_sha256': fingerprint(persistent_messages(effective))})
