"""Provider ownership, billing ambiguity and native Messages regressions; CPU only."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from uni_robot_agent.provider.client import ModelClient
from uni_robot_agent.provider.retry import RetryPolicy, error_details
from uni_robot_agent.provider.transport import ModelRequestCancelled
from uni_robot_agent.provider.budget import Ledger, Limits, BudgetGate, Stop
from uni_robot_agent.provider.budget.messages import DirectFableLimits, MessagesBudgetGate, prepare_messages
from uni_robot_agent.context.messages import project_observations


class ProviderBudgetChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.tmp.name)/'ledger.db', Limits())
        self.ledger.claim('one')
        self.events = []
        self.client = ModelClient('http://127.0.0.1:1/v1',
            retry_policy=RetryPolicy(initial_delay_s=0, max_delay_s=0),
            request_guard=lambda: self.ledger.check('one'), attempt_recorder=self.events.append)

    def tearDown(self):
        self.client.close(); self.ledger.db.close(); self.tmp.cleanup()

    def send(self, path, body):
        return self.client.send(target='http://127.0.0.1:1/v1'+path,
            request_bytes=json.dumps(body).encode(), headers={}, exchange={'provider_attempts': []})

    def body(self):
        return {'model':'gpt-6.1-sol','reasoning':{'effort':'high'},'input':'fixed',
                'tools':[{'type':'function','name':'capture_scene'}]}

    def gate(self, fault):
        calls = []
        def network(transport, target, **kw):
            calls.append((target, kw['request_bytes']))
            result = fault(target, len(calls))
            if result: raise result
            raw = {'input_tokens':100} if target.endswith('input_tokens') else {
                'status':'completed','usage':{'input_tokens':100,'output_tokens':5}}
            return 200, {}, json.dumps(raw).encode()
        gate = BudgetGate(self.ledger,'one',self.send)
        return gate, calls, patch('uni_robot_agent.provider.client._send_provider_request', side_effect=network)

    def test_count_retry_uses_one_generation_and_reservation(self):
        gate,calls,mock=self.gate(lambda path,n:httpx.ReadTimeout('private') if n==1 else None)
        with mock: gate.handle(self.body())
        self.assertEqual(len(calls),3)
        self.assertEqual(self.ledger.snapshot()['generation_attempts'],1)
        self.assertEqual(self.ledger.snapshot()['unsettled_reserved_usd'],0)
        self.assertEqual(calls[0][1],calls[1][1])
        self.assertNotIn('private',json.dumps(self.events))

    def test_generation_connect_retry_same_body_and_single_slot(self):
        gate,calls,mock=self.gate(lambda path,n:httpx.ConnectError('private') if n==2 else None)
        with mock: gate.handle(self.body())
        self.assertEqual(len(calls),3);self.assertEqual(calls[1][1],calls[2][1])
        self.assertEqual(self.ledger.snapshot()['generation_attempts'],1)

    def test_generation_read_timeout_retains_reservation_and_stops(self):
        gate,calls,mock=self.gate(lambda path,n:httpx.ReadTimeout('private') if n==2 else None)
        with mock,self.assertRaises(Stop):gate.handle(self.body())
        self.assertEqual(len(calls),2)
        self.assertGreater(self.ledger.snapshot()['unsettled_reserved_usd'],0)
        self.assertTrue(gate.stopped.is_set())

    def test_additional_retries_and_zero(self):
        for count in (0,2):
            self.client._max_retries=count
            with patch('uni_robot_agent.provider.client._send_provider_request',side_effect=httpx.ConnectError('private')) as send:
                with self.assertRaises(httpx.ConnectError):self.send('/responses/input_tokens',{})
                self.assertEqual(send.call_count,count+1)

    def test_http_failure_is_never_retried(self):
        response=httpx.Response(429,request=httpx.Request('POST','http://localhost'))
        from uni_robot_agent.services.errors import HttpCallError
        with patch('uni_robot_agent.provider.client._send_provider_request',side_effect=HttpCallError(response)) as send:
            with self.assertRaises(HttpCallError):self.send('/responses',{})
            self.assertEqual(send.call_count,1)

    def test_cancel_interrupts_wait(self):
        self.client._retry_policy=RetryPolicy(initial_delay_s=5,max_delay_s=5)
        entered=threading.Event();errors=[]
        def fail(*a,**k):entered.set();raise httpx.ConnectError('private')
        def run():
            try:self.send('/responses',{})
            except Exception as exc:errors.append(exc)
        with patch('uni_robot_agent.provider.client._send_provider_request',side_effect=fail) as send:
            worker=threading.Thread(target=run);worker.start();self.assertTrue(entered.wait(2))
            self.client.cancel();worker.join(2)
            self.assertFalse(worker.is_alive());self.assertEqual(send.call_count,1)
        self.assertIsInstance(errors[0],ModelRequestCancelled)

    def test_wall_deadline_checked_during_backoff(self):
        self.client._retry_policy=RetryPolicy(initial_delay_s=5,max_delay_s=5)
        def guard():
            if self.events:raise Stop('wall_time_limit')
        self.client._request_guard=guard
        with patch('uni_robot_agent.provider.client._send_provider_request',side_effect=httpx.ConnectError('private')) as send:
            with self.assertRaisesRegex(Stop,'wall_time'):self.send('/responses',{})
            self.assertEqual(send.call_count,1)

    def test_safe_error_chain(self):
        error=httpx.ConnectError('secret');error.__cause__=OSError(111,'secret')
        self.assertNotIn('secret',json.dumps(error_details(error)))
        self.assertIn({'errno':111},error_details(error)['cause_codes'])

    def test_fixed_exponential_and_capped_delays(self):
        self.assertEqual([RetryPolicy().delay(i) for i in range(5)],[1,2,4,8,8])
        policy=RetryPolicy(initial_delay_s=2,backoff_multiplier=1,max_delay_s=2)
        self.assertEqual([policy.delay(i) for i in range(5)],[2]*5)

    def test_messages_count_and_generation_share_retry_boundary(self):
        for path in ('/messages/count_tokens','/messages'):
            with self.subTest(path=path),patch('uni_robot_agent.provider.client._send_provider_request',side_effect=[httpx.ConnectError('x'),(200,{},b'{}')]) as send:
                self.send(path,{});self.assertEqual(send.call_count,2)
        with patch('uni_robot_agent.provider.client._send_provider_request',side_effect=httpx.ReadTimeout('x')) as send:
            with self.assertRaises(httpx.ReadTimeout):self.send('/messages',{})
            self.assertEqual(send.call_count,1)

    def test_messages_direct_allowlist_and_native_signed_history(self):
        thinking={'type':'thinking','thinking':'opaque fixture','signature':'fixture-signature'}
        body={'model':'claude-fable-5-2','thinking':{'type':'adaptive'},'output_config':{'effort':'high'},
            'messages':[{'role':'assistant','content':[thinking,{'type':'tool_use','id':'t1','name':'mcp__uni_robot_agent__move','input':{}}]},
                        {'role':'user','content':[{'type':'tool_result','tool_use_id':'t1','content':'receipt'}]}],
            'tools':[{'name':'mcp__uni_robot_agent__move','input_schema':{'type':'object'}}]}
        before=deepcopy(body)
        prepared,count=prepare_messages(body,DirectFableLimits())
        self.assertEqual(prepared['messages'],before['messages']);self.assertEqual(body,before)
        self.assertEqual(count['messages'],before['messages'])
        with self.assertRaises(Stop):prepare_messages(dict(body,tools=[{'name':'Bash','input_schema':{}}]),DirectFableLimits())

    def test_observation_projection_keeps_assistant_signatures_and_tool_identity(self):
        body={'system':[{'type':'text','text':'fixed','cache_control':{'type':'ephemeral'}}], 'messages':[
            {'role':'user','content':[{'type':'text','text':'task'}]},
            {'role':'assistant','content':[{'type':'thinking','thinking':'opaque','signature':'signed'},
                {'type':'tool_use','id':'t1','name':'move','input':{'x':1}}]},
            {'role':'user','content':[{'type':'tool_result','tool_use_id':'t1','is_error':False,'content':'large receipt'}]}]}
        before=deepcopy(body)
        def compact(view):
            view=deepcopy(view);view['input'][-1]['output']=[{'type':'input_text','text':'compact receipt'}]
            return view,{}
        result,_=project_observations(body,compact)
        self.assertEqual(result['messages'][1],before['messages'][1]);self.assertEqual(result['system'],before['system'])
        self.assertEqual(result['messages'][2]['content'][0]['tool_use_id'],'t1');self.assertEqual(body,before)


class NativeLoopbackChecks(unittest.TestCase):
    def test_native_transport_credentials_paths_and_settlement(self):
        from fastapi import APIRouter, Request
        from starlette.responses import JSONResponse
        from uni_robot_agent.services.hosting import HttpServer, create_app
        from uni_robot_agent.provider.budget.service import serve_gate
        for kind in ('responses','messages'):
            with self.subTest(protocol=kind),tempfile.TemporaryDirectory() as tmp:
                seen=[];router=APIRouter()
                @router.post('/{path:path}')
                async def native(request:Request,path:str):
                    body=await request.json();seen.append((path,dict(request.headers),body))
                    if path.endswith(('input_tokens','count_tokens')):return {'input_tokens':100}
                    if kind=='responses':return {'id':'resp_fixture','model':'gpt-6.1-sol','status':'completed',
                        'output':[],'usage':{'input_tokens':100,'output_tokens':5}}
                    return {'id':'msg_fixture','type':'message','model':'claude-fable-5-2','role':'assistant',
                        'content':[{'type':'text','text':'fixture'}],'stop_reason':'end_turn',
                        'usage':{'input_tokens':100,'output_tokens':5}}
                vendor=HttpServer(create_app(router),'127.0.0.1');vendor.start_background()
                limits=Limits() if kind=='responses' else DirectFableLimits()
                ledger=Ledger(Path(tmp)/'ledger.db',limits);ledger.claim('one')
                gate=(BudgetGate if kind=='responses' else MessagesBudgetGate)(ledger,'one',None)
                key=Path(tmp)/'fixture.key';key.write_text('sk-unusable-test');key.chmod(0o600)
                service,url=serve_gate(gate,'local-only',key_file=key,
                    upstream_base_url=f'http://127.0.0.1:{vendor.server_port}/v1')
                body=({'model':limits.model,'reasoning':{'effort':'high'},'input':'fixture','stream':False}
                      if kind=='responses' else {'model':limits.model,'messages':[{'role':'user','content':'fixture'}],
                          'output_config':{'effort':'high'},'thinking':{'type':'adaptive'},'stream':False})
                try:
                    with httpx.Client(trust_env=False) as client:
                        rejected=client.post(url+'/'+kind,json=body,headers={'Authorization':'Bearer wrong'})
                        self.assertEqual(rejected.status_code,403);self.assertEqual(seen,[])
                        result=client.post(url+'/'+kind,json=body,headers={'Authorization':'Bearer local-only',
                            'anthropic-version':'2023-06-01'})
                        self.assertEqual(result.status_code,200,result.text)
                        self.assertEqual(client.post(url+'/unprepared',json=body,
                            headers={'Authorization':'Bearer local-only'}).status_code,403)
                    self.assertEqual(len(seen),2)
                    self.assertEqual(ledger.snapshot()['generation_attempts'],1)
                    self.assertEqual(ledger.snapshot()['unsettled_reserved_usd'],0)
                    for _,headers,_ in seen:
                        self.assertNotIn('local-only',json.dumps(headers))
                        self.assertEqual(headers.get('authorization') if kind=='responses' else headers.get('x-api-key'),
                                         'Bearer sk-unusable-test' if kind=='responses' else 'sk-unusable-test')
                    self.assertEqual(len(service.gateway.trace_records[0]['provider_attempts']),2)
                finally:
                    service.shutdown();vendor.stop();ledger.db.close()


if __name__=='__main__':unittest.main()
