#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from brian2 import Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses, defaultclock, ms, mV, prefs, seed as brian_seed

# FlyWire v783 sensory/readout IDs used by theflyRH/thefly-brain / Shiu-style model.
SUGAR=[720575940624963786,720575940630233916,720575940637568838,720575940638202345,720575940617000768,720575940630797113,720575940632889389,720575940621754367,720575940621502051,720575940640649691,720575940639332736,720575940616885538,720575940639198653,720575940620900446,720575940617937543,720575940632425919,720575940633143833,720575940612670570,720575940628853239,720575940629176663,720575940611875570]
BITTER=[720575940621778381,720575940602353632,720575940617094208,720575940619197093,720575940626287336,720575940618600651,720575940627692048,720575940630195909,720575940646212996,720575940610483162,720575940645743412,720575940627578156,720575940622298631,720575940621008895,720575940629146711,720575940610259370,720575940610481370,720575940619028208,720575940614281266,720575940613061118,720575940604027168]
LOOMING=[720575940616185531,720575940629806974]
MN9=[720575940660219265,720575940645521262]

DATA=Path(os.environ.get('FLYBRAIN_DATA','/srv/flybrain/data'))
COMP=DATA/'Completeness_783.csv'
CONN=DATA/'Connectivity_783.parquet'
PORT=int(os.environ.get('BRAIN_PORT','3001'))
BATCH=int(os.environ.get('FLYBRAIN_BATCH','250000'))
FEATURE_BINS=256
TIME_BINS=4

prefs.codegen.target='numpy'
prefs.core.default_float_dtype=np.float32
defaultclock.dt=0.1*ms

P=dict(
    v_0=-52*mV, v_rst=-52*mV, v_th=-45*mV,
    t_mbr=20*ms, tau=5*ms, t_rfc=2.2*ms, t_dly=1.8*ms,
    w_syn=0.275*mV, r_poi=150*Hz, f_poi=250,
)
EQS='''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
'''

SEED_PAIRS=[
    ('안녕','안녕. 반가워.'),
    ('반가워','나도 반가워.'),
    ('너 누구야','나는 실제 FlyWire 초파리 connectome을 상태 공간으로 쓰는 대화 시스템이야.'),
    ('뭐해','네 문장을 감각 자극으로 바꿔서 실제 초파리 뇌를 돌리고 있어.'),
    ('어떻게 작동해','문자를 감각 뉴런에 넣고 전체 뇌의 스파이크 패턴으로 답을 고르고 있어.'),
    ('좋아','좋아.'),
    ('싫어','그 자극에는 다른 스파이크 패턴이 나왔어.'),
    ('고마워','응.'),
    ('잘가','잘 가.'),
    ('배고파','먹는 자극에 반응하고 있어.'),
    ('해내','해냈어. 지금 실제 connectome의 스파이크 상태로 답을 골랐어.'),
    ('대화해','응. 계속 말해 봐.'),
]


def rss_mb():
    try:
        with open('/proc/self/status',encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return round(int(line.split()[1])/1024,1)
    except Exception:
        pass
    return -1.0


def normalize_text(s):
    return ' '.join(str(s or '').strip().lower().split())[:200]


def read_root_ids(path):
    out=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        r=csv.reader(f)
        next(r,None)
        for row in r:
            if not row: continue
            try: out.append(int(row[0]))
            except Exception: continue
    return np.asarray(out,dtype=np.int64)


class FlyReservoir:
    def __init__(self):
        self.lock=threading.Lock()
        self.memory=[]
        self.turn=0
        self.training=False
        self.training_done=False
        self.ready=False
        self._build()
        self.ready=True
        threading.Thread(target=self._seed_train,daemon=True).start()

    def _build(self):
        t0=time.time()
        print(f'RES_BUILD_START rss={rss_mb()}MB',flush=True)
        ids=read_root_ids(COMP)
        self.n_neurons=len(ids)
        mp={int(v):i for i,v in enumerate(ids)}
        present=lambda xs: np.asarray([mp[x] for x in xs if x in mp],dtype=np.int32)
        sensory=np.unique(np.concatenate([present(SUGAR),present(BITTER),present(LOOMING)])).astype(np.int32)
        self.sensory=sensory
        self.mn9=present(MN9)
        print(f'RES_COMP neurons={self.n_neurons} sensory={len(sensory)} mn9={self.mn9.tolist()} rss={rss_mb()}MB',flush=True)

        neu=NeuronGroup(self.n_neurons,model=EQS,method='linear',threshold='v > v_th',reset='v = v_rst; g = 0*mV',refractory='rfc',namespace=P,name='res_neurons')
        neu.v=P['v_0']; neu.g=0*mV; neu.rfc=P['t_rfc']
        syn=Synapses(neu,neu,'w : volt',on_pre='g += w',delay=P['t_dly'],name='res_synapses')
        pf=pq.ParquetFile(CONN)
        self.n_synapses=int(pf.metadata.num_rows)
        loaded=0
        cols=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity']
        for bn,rb in enumerate(pf.iter_batches(batch_size=BATCH,columns=cols),1):
            pre=np.asarray(rb.column(0)).astype(np.int32,copy=False)
            post=np.asarray(rb.column(1)).astype(np.int32,copy=False)
            w=np.asarray(rb.column(2)).astype(np.float32,copy=False)
            n=len(pre)
            syn.connect(i=pre,j=post)
            syn.w[loaded:loaded+n]=w*P['w_syn']
            loaded+=n
            if bn==1 or bn%10==0 or loaded==self.n_synapses:
                print(f'RES_EDGES {loaded}/{self.n_synapses} rss={rss_mb()}MB',flush=True)
            del pre,post,w,rb
            gc.collect()
        del pf
        gc.collect()

        poi=PoissonGroup(len(sensory),rates=0*Hz,name='res_poisson')
        ps=Synapses(poi,neu,on_pre='v += w_poi',namespace={'w_poi':P['w_syn']*P['f_poi']},name='res_input')
        ps.connect(i=np.arange(len(sensory),dtype=np.int32),j=sensory)
        self.neu,self.syn,self.poi,self.ps=neu,syn,poi,ps
        self.net=Network(neu,syn,poi,ps)
        self.build_seconds=time.time()-t0
        print(f'RES_BUILD_DONE seconds={self.build_seconds:.3f} neurons={self.n_neurons} synapses={self.n_synapses} rss={rss_mb()}MB',flush=True)

    def _reset(self,seed_value):
        self.poi.rates=np.zeros(len(self.sensory))*Hz
        self.net.run(3*ms)
        self.neu.v=P['v_0']; self.neu.g=0*mV; self.neu.rfc=P['t_rfc']
        brian_seed(seed_value); np.random.seed(seed_value & 0xffffffff)

    def encode(self,text):
        text=normalize_text(text)
        if not text: text=' '
        digest=hashlib.sha256(text.encode('utf-8')).digest()
        seed_value=int.from_bytes(digest[:4],'little')
        with self.lock:
            t0=time.time(); self.turn+=1
            self._reset(seed_value)
            mon=SpikeMonitor(self.neu,name=f'res_mon_{self.turn}')
            self.net.add(mon)
            # No keyword/intent logic. Unicode code points alone choose sensory channels/rates.
            chars=list(text)[:12]
            pulse_meta=[]
            for pos,ch in enumerate(chars):
                cp=ord(ch)
                rates=np.zeros(len(self.sensory))*Hz
                channels=[]
                for j in range(3):
                    idx=(cp*(17+2*j)+pos*(31+7*j)+digest[(pos+j)%32]) % len(self.sensory)
                    channels.append(int(idx))
                    rates[int(idx)]=(280 + ((cp >> (j*3)) + digest[(pos*3+j)%32]) % 221)*Hz
                self.poi.rates=rates
                self.net.run(4*ms)
                pulse_meta.append({'char':ch,'channels':channels})
            self.poi.rates=np.zeros(len(self.sensory))*Hz
            self.net.run(8*ms)
            spikes_i=np.asarray(mon.i[:],dtype=np.int32)
            spikes_t=np.asarray(mon.t[:]/ms,dtype=np.float32)
            self.net.remove(mon)
            sim_ms=3+4*len(chars)+8
            feat=np.zeros(FEATURE_BINS*TIME_BINS,dtype=np.float32)
            if len(spikes_i):
                t0ms=max(0.0,float(spikes_t.min()))
                rel=np.clip((spikes_t-t0ms)/max(1.0,sim_ms),0,0.999999)
                tb=(rel*TIME_BINS).astype(np.int32)
                idx=(spikes_i % FEATURE_BINS) + tb*FEATURE_BINS
                np.add.at(feat,idx,1.0)
                feat=np.log1p(feat)
                n=np.linalg.norm(feat)
                if n>0: feat/=n
            mn9=int(np.isin(spikes_i,self.mn9).sum()) if len(spikes_i) else 0
            active=int(len(np.unique(spikes_i))) if len(spikes_i) else 0
            code=int(hashlib.blake2s(feat.tobytes(),digest_size=4).hexdigest(),16)
            return feat,{
                'simulated_ms':sim_ms,'total_spikes':int(len(spikes_i)),'active_neurons':active,
                'mn9_spikes':mn9,'brain_code':code,'wall_seconds':round(time.time()-t0,3),
                'chars':len(chars),'rss_mb':rss_mb(),'input_pulses':pulse_meta,
            }

    def teach(self,text,reply,source='user'):
        feat,brain=self.encode(text)
        self.memory.append({'text':normalize_text(text),'reply':str(reply),'feature':feat,'source':source})
        return {'learned':len(self.memory),'brain':brain}

    def _seed_train(self):
        self.training=True
        print(f'RES_TRAIN_START pairs={len(SEED_PAIRS)}',flush=True)
        for i,(text,reply) in enumerate(SEED_PAIRS,1):
            try:
                r=self.teach(text,reply,source='seed')
                print(f'RES_TRAIN {i}/{len(SEED_PAIRS)} text={text!r} spikes={r["brain"]["total_spikes"]} code={r["brain"]["brain_code"]}',flush=True)
            except Exception as e:
                print(f'RES_TRAIN_ERROR {i} {e!r}',flush=True)
        self.training=False; self.training_done=True
        print(f'RES_TRAIN_DONE learned={len(self.memory)}',flush=True)

    def chat(self,text):
        feat,brain=self.encode(text)
        if not self.memory:
            return {'reply':'아직 배운 대화가 없어.','similarity':0.0,'matched_text':None,'brain':brain}
        M=np.stack([m['feature'] for m in self.memory])
        sims=M@feat
        k=int(np.argmax(sims))
        m=self.memory[k]
        return {
            'reply':m['reply'],'similarity':round(float(sims[k]),6),'matched_text':m['text'],
            'learned_examples':len(self.memory),'brain':brain,
            'architecture':'unicode sensory pulses -> full FlyWire v783 connectome -> spike-vector associative readout',
            'mock':False,'llm':False,'keyword_intent_rules':False,
        }

BRAIN=FlyReservoir()

class H(BaseHTTPRequestHandler):
    def _json(self,obj,status=200):
        b=json.dumps(obj,ensure_ascii=False).encode('utf-8')
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith('/health'): return self._json({'ok':True,'ready':BRAIN.ready})
        if self.path.startswith('/status'): return self._json({'ready':BRAIN.ready,'mock':False,'llm':False,'keyword_intent_rules':False,'dataset':'FlyWire v783','neurons':BRAIN.n_neurons,'synapses':BRAIN.n_synapses,'sensory_channels':len(BRAIN.sensory),'learned_examples':len(BRAIN.memory),'training':BRAIN.training,'training_done':BRAIN.training_done,'turns':BRAIN.turn,'rss_mb':rss_mb(),'build_seconds':round(BRAIN.build_seconds,3)})
        return self._json({'name':'FlyWire reservoir chat','POST /chat':{'text':'안녕'},'POST /teach':{'text':'질문','reply':'대답'}})
    def do_POST(self):
        try:
            n=int(self.headers.get('Content-Length','0')); data=json.loads(self.rfile.read(n) or b'{}')
            if self.path.startswith('/chat'):
                out=BRAIN.chat(data.get('text','')); print('RES_CHAT '+json.dumps({'text':data.get('text',''),'reply':out['reply'],'similarity':out['similarity'],'matched':out['matched_text'],'brain':out['brain']},ensure_ascii=False),flush=True); return self._json(out)
            if self.path.startswith('/teach'):
                text=data.get('text',''); reply=data.get('reply','')
                if not text or not reply: return self._json({'error':'text and reply required'},400)
                out=BRAIN.teach(text,reply); print('RES_TEACH '+json.dumps({'text':text,'reply':reply,'learned':out['learned'],'brain':out['brain']},ensure_ascii=False),flush=True); return self._json(out)
            return self._json({'error':'not found'},404)
        except Exception as e:
            print('RES_HTTP_ERROR',repr(e),flush=True); return self._json({'error':repr(e)},500)
    def log_message(self,fmt,*args): print('HTTP '+fmt%args,flush=True)

print(f'RES_HTTP_START port={PORT}',flush=True)
ThreadingHTTPServer(('0.0.0.0',PORT),H).serve_forever()
