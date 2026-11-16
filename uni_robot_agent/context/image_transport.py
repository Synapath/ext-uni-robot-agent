"""Content-addressed image transport between SDK and local context projection.

Raw tool events keep original RGB. Only the executor's MCP return is replaced;
selected images are restored byte-for-byte before any provider request/count.
"""
from base64 import b64encode
from dataclasses import replace
import hashlib
from pathlib import Path
import re
import struct
import threading
import zlib

from uni_robot_agent.provider.budget.core import Stop

TAG = '[RUA transport sha256='
REFERENCE = re.compile(r'\n?\[RUA transport sha256=([0-9a-f]{64})\]')


def _chunk(kind,data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)


MARKER = (b'\x89PNG\r\n\x1a\n'+_chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))
          +_chunk(b'IDAT',zlib.compress(b'\x00\x00\x00\x00'))+_chunk(b'IEND',b''))
MARKER_URL = 'data:image/png;base64,'+b64encode(MARKER).decode('ascii')
_blobs = {}
_lock = threading.RLock()


def marker_image(image,root):
    """Called after raw ToolResult recording; never mutate the original attachment."""
    from uni_robot_agent.tools.toolkit import ImageAttachment
    key=hashlib.sha256(image.data).hexdigest();root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.blob')
    with _lock:
        if not path.exists():
            with path.open('xb') as file:file.write(image.data)
        if key not in _blobs or _blobs[key]['path']!=path:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=key:raise Stop('transport_blob_changed')
            _blobs[key]={'path':path,'mime_type':image.mime_type,'url':None}
    label=(image.label+'\n' if image.label else '')+TAG+key+']'
    return ImageAttachment(MARKER,mime_type='image/png',name=image.name,label=label)


def _url(key):
    with _lock:
        value=_blobs.get(key)
        if value is None:raise Stop('transport_blob_unknown')
        if value['url'] is None:
            data=value['path'].read_bytes()
            if hashlib.sha256(data).hexdigest()!=key:raise Stop('transport_blob_changed')
            value['url']='data:'+value['mime_type']+';base64,'+b64encode(data).decode('ascii')
        return value['url']


def restore_selected(body):
    """Restore only selected images, not all SDK history; remove all transport tags."""
    def visit(value):
        if isinstance(value,dict):
            return {k:visit(v) for k,v in value.items()}
        if not isinstance(value,list):return value
        out=[];pending=None
        for part in value:
            if isinstance(part,dict) and part.get('type') in ('text','input_text'):
                matches=REFERENCE.findall(part.get('text',''))
                if matches:
                    if len(matches)!=1 or pending is not None:raise Stop('transport_reference_ambiguous')
                    pending=matches[0];part=dict(part,text=REFERENCE.sub('',part['text']))
                    if part['text']:out.append(part)
                    continue
            if isinstance(part,dict) and part.get('type')=='input_image' and pending:
                part=dict(part,image_url=_url(pending));pending=None
            if isinstance(part,dict) and part.get('type')=='image' and pending:
                prefix, data = _url(pending).split(';base64,', 1)
                part=dict(part,source={'type':'base64','media_type':prefix[5:],'data':data});pending=None
            out.append(visit(part))
        if pending is not None:raise Stop('transport_reference_without_image')
        return out
    result=visit(body)
    if has_references(result):raise Stop('transport_placeholder_not_restored')
    return result


def has_references(value):
    if isinstance(value,dict):return any(has_references(v) for v in value.values())
    if isinstance(value,list):return any(map(has_references,value))
    return isinstance(value,str) and (TAG in value or value==MARKER_URL)


def bind_executor(executor,session_root):
    original=executor.call;root=Path(session_root)/'transport-blobs'
    def call(name,arguments):
        result=original(name,arguments)
        if result.output is None or not result.output.images:return result
        # Toolkit/telemetry already recorded original images before this boundary.
        output=replace(result.output,images=[marker_image(image,root) for image in result.output.images])
        return replace(result,output=output)
    executor.call=call
