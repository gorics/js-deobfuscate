#!/usr/bin/env python3
from __future__ import annotations
import gc, hashlib, json, os, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from brian2 import Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses, defaultclock, ms, mV, prefs, seed as brian_seed

SUGAR=[720575940624963786,720575940630233916,720575940637568838,720575940638202345,720575940617000768,720575940630797113,720575940632889389,720575940621754367,720575940621502051,720575940640649691,720575940639332736,720575940616885538,720575940639198653,720575940620900446,720575940617937543,720575940632425919,720575940633143833,720575940612670570,720575940628853239,720575940629176663,720575940611875570]
BITTER=[720575940621778381,720575940602353632,720575940617094208,720575940619197093,720575940626287336,720575940618600651,720575940627692048,720575940630195909,720575940646212996,720575940610483162,720575940645743412,720575940627578156,720575940622298631,720575940621008895,720575940629146711,720575940610259370,720575940610481370,720575940619028208,720575940614281266,720575940613061118,720575940604027168]
LOOMING=[720575940616185531,720575940629806974]
MN9=[720575940660219265,720575940645521262]
DATA=Path(os.environ.get('FLYBRAIN_DATA','/srv/flybrain/data')); COMP=DATA/'Completeness_783.csv'; CONN=DATA/'Connectivity_783.parquet'
PORT=int(os.environ.get('PORT','3000')); BATCH=int(os.environ.get('FLYBRAIN_BATCH','250000'))
prefs.codegen.target='numpy'; prefs.core.default_float_dtype=np.float32; defaultclock.dt=0.1*ms
P=dict(v_0=-52*mV,v_rst=-52*mV,v_th=-45*mV,t_mbr=20*ms,tau=5*ms,t_rfc=2.2*ms,t_dly=1.8*ms,w_syn=0.275*mV,r_poi=150*Hz,f_poi=250)
EQS='''\ndv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)\ndg/dt = -g / tau : volt (unless refractory)\nrfc : second\n'''

def rss():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'): return round(int(line.split()[1])/1024,1)
    except: pass
    return -1

def norm(s): return ' '.join(str(s or '').strip().lower().split())[:500]

class Brain:
    def __init__(self):
        self.lock=threading.Lock(); self.history=deque(maxlen=30); self.taught={}; self.turn=0; self.ready=False; self.build(); self.ready=True
    def build(self):
        t=time.time(); print('CHAT_BUILD_START',rss(),flush=True)
        df=pd.read_csv(COMP,index_col=0); ids=np.asarray(df.index,dtype=np.int64); del df; gc.collect()
        self.n=len(ids); mp={int(v):i for i,v in enumerate(ids)}
        present=lambda xs: np.asarray([mp[x] for x in xs if x in mp],dtype=np.int32)
        self.stim={'sugar':present(SUGAR),'bitter':present(BITTER),'looming':present(LOOMING)}; self.mn9=present(MN9)
        self.all=np.unique(np.concatenate(list(self.stim.values()))).astype(np.int32); self.pos={k:np.searchsorted(self.all,v).astype(np.int32) for k,v in self.stim.items()}
        print('CHAT_COMP',self.n,{k:len(v) for k,v in self.stim.items()},'mn9',self.mn9.tolist(),'rss',rss(),flush=True)
        neu=NeuronGroup(self.n,model=EQS,method='linear',threshold='v > v_th',reset='v = v_rst; g = 0*mV',refractory='rfc',namespace=P,name='chat_neurons')
        neu.v=P['v_0']; neu.g=0*mV; neu.rfc=P['t_rfc']
        syn=Synapses(neu,neu,'w : volt',on_pre='g += w',delay=P['t_dly'],name='chat_synapses')
        pf=pq.ParquetFile(CONN); self.ns=int(pf.metadata.num_rows); loaded=0; cols=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity']
        for bn,rb in enumerate(pf.iter_batches(batch_size=BATCH,columns=cols),1):
            a=np.asarray(rb.column(0)).astype(np.int32,copy=False); b=np.asarray(rb.column(1)).astype(np.int32,copy=False); w=np.asarray(rb.column(2)).astype(np.float32,copy=False); n=len(a)
            syn.connect(i=a,j=b); syn.w[loaded:loaded+n]=w*P['w_syn']; loaded+=n
            if bn==1 or bn%10==0 or loaded==self.ns: print('CHAT_EDGES',loaded,'/',self.ns,'rss',rss(),flush=True)
            del a,b,w,rb; gc.collect()
        del pf; gc.collect()
        poi=PoissonGroup(len(self.all),rates=0*Hz,name='chat_poisson'); ps=Synapses(poi,neu,on_pre='v += w_poi',namespace={'w_poi':P['w_syn']*P['f_poi']},name='chat_ps')
        ps.connect(i=np.arange(len(self.all),dtype=np.int32),j=self.all)
        self.neu,self.syn,self.poi,self.ps=neu,syn,poi,ps; self.net=Network(neu,syn,poi,ps); self.build_s=time.time()-t
        print('CHAT_BUILD_DONE',self.build_s,'neurons',self.n,'synapses',self.ns,'rss',rss(),flush=True)
    def intent(self,t):
        t=norm(t)
        if any(x in t for x in ['안녕','하이','반가워','hello','hi']): return 'greeting'
        if any(x in t for x in ['이름','누구야','누구니','정체','뭐야 너','너 뭐']): return 'identity'
        if any(x in t for x in ['고마워','감사','thanks','땡큐']): return 'thanks'
        if any(x in t for x in ['잘가','잘 자','잘자','바이','bye']): return 'bye'
        if any(x in t for x in ['설탕','단맛','먹','배고','음식','밥']): return 'food'
        if any(x in t for x in ['싫','화나','짜증','위험','무서','아니']): return 'negative'
        if any(x in t for x in ['좋아','응','그래','맞아','좋다']): return 'positive'
        if '?' in t or any(x in t for x in ['뭐','왜','어떻게','어디','언제','누가','얼마','인가']): return 'question'
        return 'statement'
    def pool(self,intent,text):
        if norm(text) in self.taught: return self.taught[norm(text)]
        return {
        'greeting':['안녕. 반가워.','안녕. 나는 깨어 있어.','반가워. 지금 신경망이 반응했어.','안녕. 대화해 보자.'],
        'identity':['나는 FlyWire 초파리 뇌 시뮬레이션에 연결된 대화기야.','나는 초파리 connectome 활동으로 답을 고르는 프로그램이야.','내 중심에는 13만 개가 넘는 초파리 뉴런 모델이 있어.','나는 사람처럼 언어를 이해하지는 못해. 초파리 뇌 활동으로 대답을 선택해.'],
        'thanks':['응.','천만에.','좋아.','계속 말해 줘.'],
        'bye':['잘 가.','다음에 또 말 걸어 줘.','바이.','신경망은 여기서 쉬고 있을게.'],
        'food':['단맛 자극에 반응이 커졌어.','설탕 입력은 내 감각 뉴런을 강하게 움직여.','먹는 얘기를 들으니 접근 회로 쪽 반응을 보고 있어.','초파리답게 단맛에는 꽤 민감해.'],
        'negative':['그 말에는 회피 쪽 자극을 더 섞었어.','조금 피하고 싶은 반응이 나왔어.','강한 자극으로 들렸어.','응답이 거칠게 튀었어.'],
        'positive':['응. 좋은 쪽으로 반응했어.','그래.','좋아.','그 말에는 접근 쪽 활동이 더 나왔어.'],
        'question':['그건 내가 지식으로 이해해서 답하는 건 아니야.','잘 모르겠어. 나는 초파리 뇌 활동으로만 반응을 고르고 있어.','질문인 건 구분했지만 사람 수준의 의미 지식은 없어.','내가 아는 언어 범위에서는 확실히 답하기 어려워.'],
        'statement':['응.','계속 말해 줘.','그 말에 반응이 생겼어.','듣고 있어.']}[intent]
    def pulses(self,text,intent):
        d=hashlib.sha256(text.encode()).digest(); kinds=['sugar','bitter','looming']; bias={'greeting':'sugar','thanks':'sugar','positive':'sugar','food':'sugar','negative':'bitter','bye':'bitter','question':'looming','identity':'looming'}.get(intent); out=[]
        for k in range(4): out.append((bias if k==0 and bias else kinds[d[k]%3],float(120+d[4+k]%61),10.0))
        return out,d
    def chat(self,text):
        text=norm(text); intent=self.intent(text); pulses,d=self.pulses(text,intent); seed=int.from_bytes(d[:4],'little')
        with self.lock:
            t=time.time(); self.turn+=1; self.poi.rates=np.zeros(len(self.all))*Hz; self.net.run(3*ms); self.neu.v=P['v_0']; self.neu.g=0*mV; self.neu.rfc=P['t_rfc']; brian_seed(seed); np.random.seed(seed&0xffffffff)
            mon=SpikeMonitor(self.neu,name=f'turn_{self.turn}'); self.net.add(mon)
            for kind,rate,dur in pulses:
                rr=np.zeros(len(self.all))*Hz; rr[self.pos[kind]]=rate*Hz; self.poi.rates=rr; self.neu.rfc=P['t_rfc']; self.neu.rfc[self.stim[kind]]=0*ms; self.net.run(dur*ms)
            self.poi.rates=np.zeros(len(self.all))*Hz; self.neu.rfc=P['t_rfc']; self.net.run(20*ms)
            sp=np.asarray(mon.i[:],dtype=np.int32); self.net.remove(mon); del mon; gc.collect(); active=int(np.unique(sp).size) if len(sp) else 0; bins=np.bincount(sp%128,minlength=128).astype(np.int64) if len(sp) else np.zeros(128,dtype=np.int64); mn9=int(np.isin(sp,self.mn9).sum()) if len(sp) else 0; total=int(len(sp)); code=int(np.dot(bins,np.arange(1,129,dtype=np.int64))+97*mn9+13*active+7*total); pool=self.pool(intent,text); reply=pool[code%len(pool)]; wall=time.time()-t
            out={'reply':reply,'intent':intent,'turn':self.turn,'brain':{'dataset':'FlyWire v783','neurons':self.n,'synapses':self.ns,'simulated_ms':63.0,'total_spikes':total,'active_neurons':active,'mn9_spikes':mn9,'brain_code':code,'pulses':[{'stimulus':a,'rate_hz':b,'duration_ms':c} for a,b,c in pulses],'rss_mb':rss(),'wall_seconds':round(wall,3)},'note':'No LLM used. External Korean rules create candidates; real connectome spikes choose the reply.'}
            self.history.append({'user':text,'assistant':reply,'brain':out['brain']}); print('CHAT_TURN',self.turn,intent,'spikes',total,'active',active,'mn9',mn9,'wall',round(wall,2),'rss',rss(),flush=True); return out
    def teach(self,phrase,responses):
        p=norm(phrase); responses=[responses] if isinstance(responses,str) else responses; responses=[str(x).strip()[:300] for x in (responses or []) if str(x).strip()]
        if not p or not responses: raise ValueError('phrase and response(s) required')
        self.taught[p]=responses[:8]; return {'ok':True,'phrase':p,'responses':self.taught[p]}
    def status(self): return {'ready':self.ready,'mock':False,'dataset':'FlyWire v783','neurons':self.n,'synapses':self.ns,'turns':self.turn,'taught_phrases':len(self.taught),'rss_mb':rss(),'build_seconds':round(self.build_s,2)}

BRAIN=None; ERR=None
HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlyWire Brain Chat</title><style>body{font-family:system-ui;background:#111;color:#eee;margin:0}.w{max-width:760px;margin:auto;padding:20px}.c{background:#1b1b1b;border:1px solid #333;border-radius:14px;padding:16px}.l{height:52vh;overflow:auto;display:flex;flex-direction:column;gap:10px;margin:14px 0}.m{padding:10px 12px;border-radius:12px;max-width:82%;white-space:pre-wrap}.u{align-self:flex-end;background:#3152a3}.a{align-self:flex-start;background:#2b2b2b}.meta,.st{font-size:12px;color:#aaa}form{display:flex;gap:8px}input{flex:1;background:#0f0f0f;color:#fff;border:1px solid #444;border-radius:10px;padding:12px}button{border:0;border-radius:10px;padding:0 16px;font-weight:700}</style><body><div class=w><div class=c><h2>FlyWire 초파리 뇌 대화</h2><div id=st class=st>상태 확인 중…</div><div id=log class=l></div><form id=f><input id=q autocomplete=off placeholder="한국어로 말 걸기"><button>전송</button></form></div></div><script>const l=document.getElementById('log'),q=document.getElementById('q'),st=document.getElementById('st');function add(t,c,m=''){let d=document.createElement('div');d.className='m '+c;d.textContent=t;if(m){let x=document.createElement('div');x.className='meta';x.textContent=m;d.appendChild(x)}l.appendChild(d);l.scrollTop=l.scrollHeight}async function s(){try{let j=await(await fetch('/status')).json();st.textContent=j.ready?`FlyWire v783 · ${j.neurons.toLocaleString()} 뉴런 · ${j.synapses.toLocaleString()} 시냅스 · mock=false`:'구축 중…'}catch(e){st.textContent='상태 조회 실패'}}s();setInterval(s,5000);f.onsubmit=async e=>{e.preventDefault();let t=q.value.trim();if(!t)return;q.value='';add(t,'u');add('초파리 뇌 계산 중…','a');let h=l.lastChild;try{let j=await(await fetch('/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message:t})})).json();h.textContent=j.reply||j.error;if(j.brain){let x=document.createElement('div');x.className='meta';x.textContent=`spikes ${j.brain.total_spikes} · active ${j.brain.active_neurons} · MN9 ${j.brain.mn9_spikes} · ${j.brain.wall_seconds}s`;h.appendChild(x)}}catch(e){h.textContent='오류: '+e}};</script></body></html>'''

class H(BaseHTTPRequestHandler):
    def log_message(self,f,*a): print('HTTP',f%a,flush=True)
    def j(self,code,obj):
        raw=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(code); self.send_header('content-type','application/json; charset=utf-8'); self.send_header('content-length',str(len(raw))); self.send_header('access-control-allow-origin','*'); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        p=urlparse(self.path).path
        if p=='/':
            raw=HTML.encode(); self.send_response(200); self.send_header('content-type','text/html; charset=utf-8'); self.send_header('content-length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
        elif p in ['/health','/status']: self.j(200,BRAIN.status() if BRAIN else {'ready':False,'mock':False,'error':ERR})
        elif p=='/history': self.j(200,list(BRAIN.history) if BRAIN else [])
        else: self.j(404,{'error':'not found'})
    def do_POST(self):
        try: body=json.loads(self.rfile.read(min(int(self.headers.get('content-length','0')),65536)) or b'{}')
        except: return self.j(400,{'error':'invalid json'})
        if not BRAIN: return self.j(503,{'error':ERR or 'brain not ready'})
        try:
            p=urlparse(self.path).path
            if p=='/chat':
                m=str(body.get('message',''))
                if not m.strip(): return self.j(400,{'error':'message required'})
                return self.j(200,BRAIN.chat(m))
            if p=='/teach': return self.j(200,BRAIN.teach(str(body.get('phrase','')),body.get('responses',body.get('response'))))
            return self.j(404,{'error':'not found'})
        except Exception as e: print('CHAT_ERROR',repr(e),flush=True); return self.j(500,{'error':str(e)})

if __name__=='__main__':
    try: BRAIN=Brain()
    except Exception as e: ERR=repr(e); print('CHAT_BUILD_ERROR',ERR,flush=True)
    print('CHAT_HTTP_START',PORT,'ready',BRAIN is not None,flush=True); ThreadingHTTPServer(('0.0.0.0',PORT),H).serve_forever()
