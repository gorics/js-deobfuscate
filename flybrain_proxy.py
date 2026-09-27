#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,urllib.request,urllib.error
PORT=3000
UP='http://127.0.0.1:3001'
class H(BaseHTTPRequestHandler):
    def _send(self,status,body,ctype='application/json; charset=utf-8'):
        if isinstance(body,str): body=body.encode('utf-8')
        self.send_response(status); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def _proxy(self):
        n=int(self.headers.get('Content-Length','0')); data=self.rfile.read(n) if n else None
        try:
            req=urllib.request.Request(UP+self.path,data=data,method=self.command,headers={'Content-Type':self.headers.get('Content-Type','application/json')})
            with urllib.request.urlopen(req,timeout=180) as r:
                self._send(r.status,r.read(),r.headers.get('Content-Type','application/json'))
        except urllib.error.HTTPError as e:
            self._send(e.code,e.read(),e.headers.get('Content-Type','application/json'))
        except Exception as e:
            self._send(503,json.dumps({'ready':False,'error':repr(e)}))
    def do_GET(self):
        if self.path.startswith('/health'):
            return self._send(200,json.dumps({'ok':True,'proxy':True}))
        return self._proxy()
    def do_POST(self): return self._proxy()
    def log_message(self,fmt,*args): print('PROXY '+fmt%args,flush=True)
print('RES_PROXY_READY_3000',flush=True)
ThreadingHTTPServer(('0.0.0.0',PORT),H).serve_forever()
