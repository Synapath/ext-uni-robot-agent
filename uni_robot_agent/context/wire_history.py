"""Wire evidence decoding shared by historical audits and current context views."""
import json
import re

from uni_robot_agent.provider.budget.core import Stop

FRAME = re.compile(r'\[RUA RGB frame \d+\] camera=(\w+); frame_id=(.*?); '
                   r'simulation_time_s=([0-9.]+); captured_at_utc=(.*?); state_version=(\d+)\.')


def payload(output):
    if isinstance(output, str):
        text = re.sub(r'^Wall time: [0-9.]+ seconds\nOutput:\n', '', output)
        try:
            return payload(json.loads(text))
        except ValueError:
            return None
    if isinstance(output, dict):
        return output
    if isinstance(output, list):
        values = [payload(p['text']) for p in output
                  if isinstance(p, dict) and p.get('type') in ('text', 'input_text')]
        values = [v for v in values if isinstance(v, dict)]
        if len(values) > 1:
            raise Stop('ambiguous_recorded_payload')
        return values[0] if values else None
    return None


def records(body):
    calls, results = {}, {}
    for index, item in enumerate(body.get('input', [])):
        kind = item.get('type')
        if kind == 'function_call':
            identity = item['call_id']
            if identity in calls:
                raise Stop('duplicate_recorded_call')
            calls[identity] = (index, item)
        elif kind == 'function_call_output':
            identity = item['call_id']
            if identity not in calls or identity in results:
                raise Stop('orphan_recorded_result')
            results[identity] = (index, item)
    if set(calls) != set(results):
        raise Stop('incomplete_recorded_pair')
    rows = []
    for identity, (index, call) in calls.items():
        result_index, result = results[identity]
        rows.append({'call_id': identity, 'index': index, 'result_index': result_index,
                     'call': call, 'result': result,
                     'name': call['name'].rsplit('__', 1)[-1].rsplit('.', 1)[-1],
                     'arguments': json.loads(call.get('arguments', '{}')),
                     'payload': payload(result['output'])})
    return rows
