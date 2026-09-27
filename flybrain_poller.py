#!/usr/bin/env python3
import json,time,urllib.request
INBOX='https://raw.githubusercontent.com/gorics/js-deobfuscate/flybrain-chat/flybrain_inbox.json'
last=-1
print('RES_POLLER_START',flush=True)
while True:
    try:
        with urllib.request.urlopen(INBOX+'?t='+str(time.time_ns()),timeout=10) as r:
            msg=json.loads(r.read().decode('utf-8'))
        nonce=int(msg.get('nonce',0))
        if nonce!=last:
            action=msg.get('action','chat')
            endpoint='/teach' if action=='teach' else '/chat'
            payload={'text':msg.get('message',msg.get('text',''))}
            if action=='teach': payload['reply']=msg.get('reply','')
            req=urllib.request.Request('http://127.0.0.1:3001'+endpoint,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=120) as r:
                out=json.loads(r.read().decode('utf-8'))
            print('RES_DIALOGUE '+json.dumps({'nonce':nonce,'input':msg,'result':out},ensure_ascii=False),flush=True)
            last=nonce
    except Exception as e:
        print('RES_POLLER_WAIT '+repr(e),flush=True)
    time.sleep(2)
