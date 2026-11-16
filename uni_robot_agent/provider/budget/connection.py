"""Direct native provider access through the shared RUA client (no HTTP server)."""
import json
from .core import Stop, read_private_key
from uni_robot_agent.provider.client import ModelClient
from uni_robot_agent.provider.retry import RetryPolicy


class ProviderConnection:
    def __init__(self, key_file, *, protocol='responses', base_url=None, record=None):
        if protocol not in ('responses', 'messages'):
            raise ValueError('Unsupported native provider protocol')
        key = read_private_key(key_file)
        self.base_url = base_url or ('https://api.openai.com/v1' if protocol == 'responses'
                                     else 'https://api.anthropic.com/v1')
        self._headers = ({'Authorization': 'Bearer '+key} if protocol == 'responses' else
                         {'x-api-key': key})
        self.paths = ({'/responses', '/responses/input_tokens'} if protocol == 'responses' else
                      {'/messages', '/messages/count_tokens'})
        self.client = ModelClient(self.base_url, max_retries=0, retry_http_errors=False,
                                  retry_response_errors=False, attempt_recorder=record,
                                  retry_policy=RetryPolicy(max_retries=0))

    def __call__(self, path, body, headers=None):
        if path not in self.paths:
            raise Stop('unprepared_endpoint')
        outgoing = {k:v for k,v in (headers or {}).items()
                    if k.lower() in {'anthropic-version', 'anthropic-beta'}}
        outgoing.update(self._headers)
        if '/messages' in self.paths:
            outgoing.setdefault('anthropic-version', '2023-06-01')
        outgoing['Content-Type'] = 'application/json'
        return self.client.send(target=self.base_url+path, request_bytes=json.dumps(body).encode(),
                                headers=outgoing, exchange={'provider_attempts':[]})

    def close(self):
        self.client.close()
