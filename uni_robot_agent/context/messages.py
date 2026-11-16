"""Project user observations without rewriting signed assistant Messages blocks.

The projection callback operates on a Responses-shaped observation view only.
This is not a vendor wire codec: native assistant content, signatures, tool IDs,
system/cache controls and other request fields are copied unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import json


def _to_parts(content):
    if isinstance(content, str):
        return [{'type': 'input_text', 'text': content}]
    out = []
    for block in content:
        part = deepcopy(block)
        if part['type'] == 'text':
            part['type'] = 'input_text'
        elif part['type'] == 'image':
            source = part.pop('source')
            if source.get('type') != 'base64':
                raise ValueError('Observation projection requires inline base64 images')
            part.update(type='input_image', image_url='data:'+source['media_type']+';base64,'+source['data'])
        else:
            raise ValueError('Unsupported user observation block: '+part['type'])
        out.append(part)
    return out


def _from_parts(parts):
    if isinstance(parts, str):
        return parts
    out = []
    for value in parts:
        part = deepcopy(value)
        if part['type'] == 'input_text':
            part['type'] = 'text'
        elif part['type'] == 'input_image':
            uri = part.pop('image_url')
            prefix, data = uri.split(';base64,', 1)
            if not prefix.startswith('data:'):
                raise ValueError('Projection returned a non-inline image')
            part.update(type='image', source={'type': 'base64', 'media_type': prefix[5:], 'data': data})
            part.pop('detail', None)
        else:
            raise ValueError('Unsupported projected observation block')
        out.append(part)
    return out


def observation_view(body):
    """Expose calls/results/user evidence for deterministic context inspection."""
    items = []
    for mi, message in enumerate(body['messages']):
        content = message['content']
        if message['role'] == 'assistant':
            for block in content if isinstance(content, list) else []:
                if block.get('type') == 'tool_use':
                    items.append({'type': 'function_call', 'call_id': block['id'],
                                  'name': block['name'], 'arguments': json.dumps(block['input'])})
            continue
        blocks = content if isinstance(content, list) else [{'type': 'text', 'text': content}]
        pending = []
        def flush():
            if pending:
                items.append({'type': 'message', 'role': 'user', 'id': f'user:{mi}:{len(items)}',
                              'content': _to_parts(pending)})
                pending.clear()
        for block in blocks:
            if block['type'] == 'tool_result':
                flush()
                items.append({'type': 'function_call_output', 'call_id': block['tool_use_id'],
                              'output': _to_parts(block.get('content', ''))})
            else:
                pending.append(block)
        flush()
    return {'input': items, 'tools': [{'type': 'namespace', 'name': 'mcp__uni_robot_agent',
        'tools': [{'type': 'function', 'name': t['name'].removeprefix('mcp__uni_robot_agent__'),
                   'parameters': t.get('input_schema', {})} for t in body.get('tools', [])]}]}


def project_observations(body, projector):
    """Retain every assistant message byte value and every tool-result identity."""
    original = deepcopy(body)
    view = observation_view(body)
    projected, stats = projector(view)
    old = view['input']
    new = projected['input']
    results = {i['call_id']: i['output'] for i in new if i.get('type') == 'function_call_output'}
    users = {i['id']: i['content'] for i in new if i.get('type') == 'message' and i.get('id')}
    # Reconstruct each original user segment in place; assistant blocks never pass
    # through the observation projection or Responses conversion.
    for mi, message in enumerate(original['messages']):
        if message['role'] != 'user':
            continue
        segments = [i for i in old if i.get('id', '').startswith(f'user:{mi}:')]
        segment = iter(segments)
        content = message['content']
        blocks = content if isinstance(content, list) else [{'type': 'text', 'text': content}]
        output, pending = [], []
        def flush():
            if pending:
                item = next(segment)
                output.extend(_from_parts(users[item['id']]))
                pending.clear()
        for block in blocks:
            if block['type'] == 'tool_result':
                flush()
                block['content'] = _from_parts(results[block['tool_use_id']])
                output.append(block)
            else:
                pending.append(block)
        flush()
        message['content'] = output or [{'type': 'text', 'text': '[Prior visual observation retired]'}]
    known = {i.get('id') for i in old if i.get('id')}
    for item in new:
        if item.get('type') == 'message' and not item.get('id') in known:
            original['messages'].append({'role': 'user', 'content': _from_parts(item['content'])})
    return original, stats
