#!/usr/bin/env python3
import json,time,urllib.request
INBOX='https://raw.githubusercontent.com/gorics/js-deobfuscate/flybrain-chat/flybrain_inbox.json'
STATUS='http://127.0.0.1:3001/status'
CHAT='http://127.0.0.1:3001'
last=-1
print('RES_POLLER_START',flush=True)

def brain_ready():
    try:
        with urllib.request.urlopen(STATUS,timeout=5) as r:
            s=json.loads(r.read().decode('utf-8'))
        return bool(s.get('ready')) and bool(s.get('training_done'))
    except Exception:
        return False

while True:
    try:
        if not brain_ready():
            print('RES_POLLER_WAIT brain_not_trained',flush=True)
            time.sleep(2)
            continue
        with urllib.request.urlopen(INBOX+'?t='+str(time.time_ns()),timeout=10) as r:
            msg=json.loads(r.read().decode('utf-8'))
        nonce=int(msg.get('nonce',0))
        if nonce!=last:
            action=msg.get('action','chat')
            endpoint='/teach' if action=='teach' else '/chat'
            payload={'text':msg.get('message',msg.get('text',''))}
            if action=='teach': payload['reply']=msg.get('reply','')
            req=urllib.request.Request(CHAT+endpoint,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=180) as r:
                out=json.loads(r.read().decode('utf-8'))
            print('RES_DIALOGUE '+json.dumps({'nonce':nonce,'input':msg,'result':out},ensure_ascii=False),flush=True)
            last=nonce
    except Exception as e:
        print('RES_POLLER_WAIT '+repr(e),flush=True)
    time.sleep(2)
