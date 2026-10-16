"""Minimal nurse-side web client, in-process, stdlib only.

``WebClient`` is a :class:`Notifier`: the run loop hands it the same
notifications it gives the console or MQTT, and it fans them out to every
open browser over Server-Sent Events. The page shows the bed state, recent
events and heartbeat age, beeps while ALARM / FAULT is latched, and has an
ACK button that POSTs to ``/ack``; the registered ``on_ack`` callback then
acknowledges the state machine and alarm manager. When the state feed carries
``people`` / ``patient`` / ``locked`` it also shows who is being followed, with
one button per head in view (numbered left to right) and an "auto" button;
those POST to ``/select`` and reach the ``on_select`` callbacks.

Why in-process rather than via the broker: the prototype must work on a
laptop with no MQTT running, and the nurse station is the same machine for
now. When the edge box and station separate, the page can be fed by an MQTT
subscriber instead without changing the HTML.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from nurse24hr.notify import Notifier

AckCallback = Callable[[dict[str, Any]], None]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>nurse24hr __BED__</title>
<style>
:root{--bg:#111;--fg:#eee;--muted:#999;--idle:#2e9e4f;--watch:#d9b400;--warn:#f08a00;--alarm:#d62828;--fault:#8e2bb8}
body{margin:0;padding:16px;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{margin:0 0 8px;font-size:1.1rem;color:var(--muted)}
#state{border-radius:12px;padding:28px 16px;text-align:center;font-size:2.6rem;font-weight:700;letter-spacing:.04em;background:var(--idle);transition:background .2s}
#reason{font-size:.95rem;color:var(--muted);margin:8px 0 14px;min-height:1.2em;text-align:center}
#hb{font-size:.9rem;color:var(--muted);margin-bottom:14px}
#hb.stale{color:var(--alarm);font-weight:600}
.row{display:flex;gap:8px;margin-bottom:16px}
#people{font-size:.9rem;color:var(--muted);margin-bottom:14px}
#people .btns{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
#people button{flex:0 0 auto;padding:8px 14px;font-size:1rem;background:#333;color:var(--fg)}
#people button.on{background:#c0c;color:#fff}
input{flex:1;padding:14px;font-size:1rem;border-radius:8px;border:1px solid #444;background:#222;color:var(--fg)}
button{flex:0 0 40%;padding:14px;font-size:1.2rem;font-weight:700;border:0;border-radius:8px;background:#eee;color:#111}
button:disabled{opacity:.4}
ul{list-style:none;padding:0;margin:0}
li{padding:8px 0;border-bottom:1px solid #2a2a2a;font-size:.9rem}
li time{color:var(--muted);margin-right:8px}
</style></head><body>
<h1>nurse24hr &middot; __BED__</h1>
<div id="state">&mdash;</div>
<div id="reason"></div>
<div id="hb">no heartbeat yet</div>
<div id="people"></div>
<div class="row"><input id="user" placeholder="nurse name"><button id="ack" disabled>ACK</button></div>
<ul id="events"></ul>
<script>
const $=id=>document.getElementById(id);
const colours={IDLE:'var(--idle)',WATCH:'var(--watch)',WARN:'var(--warn)',ALARM:'var(--alarm)',FAULT:'var(--fault)'};
let state='IDLE', lastHb=null, audio=null, osc=null, armed=false, eventId=null;
try{$('user').value=localStorage.getItem('nurse')||''}catch(e){}
$('user').addEventListener('input',()=>{try{localStorage.setItem('nurse',$('user').value)}catch(e){}});
function setState(s,reason){state=s;$('state').textContent=s;$('state').style.background=colours[s]||'#555';
  if(reason!==undefined)$('reason').textContent=reason||'';
  const latched=(s==='ALARM'||s==='FAULT');$('ack').disabled=!latched;latched?beep(true):beep(false)}
function beep(on){if(!armed)return;if(on&&!osc){audio=audio||new (window.AudioContext||window.webkitAudioContext)();
  osc=audio.createOscillator();const g=audio.createGain();osc.type='square';osc.frequency.value=880;g.gain.value=0.2;
  osc.connect(g);g.connect(audio.destination);osc.start();
  const lfo=audio.createOscillator();lfo.frequency.value=3;const lg=audio.createGain();lg.gain.value=0.2;lfo.connect(lg);lg.connect(g.gain);lfo.start();osc.lfo=lfo}
  if(!on&&osc){try{osc.stop();osc.lfo.stop()}catch(e){}osc=null}}
document.addEventListener('pointerdown',()=>{if(!armed){armed=true;if(state==='ALARM'||state==='FAULT')beep(true)}},{once:true});
function addEvent(p){const li=document.createElement('li');const t=new Date().toLocaleTimeString();
  li.innerHTML='<time>'+t+'</time>'+(p.level||p.kind||p.topic||'')+' '+(p.reason||p.summary||'')+(p.user?' by '+p.user:'');
  const ul=$('events');ul.insertBefore(li,ul.firstChild);while(ul.children.length>20)ul.removeChild(ul.lastChild)}
let people=null;
function renderPeople(p){
  const key=JSON.stringify([p.people,p.patient,p.locked,p.survey]);if(key===people)return;people=key;
  const el=$('people');if(p.people===undefined){el.innerHTML='';return}
  const who=p.patient?('#'+p.patient):'not visible';
  let html='patient '+who+' ('+(p.locked?'locked':'auto')+') &middot; '+p.people+' in view'+(p.survey?' &middot; scanning whole bed':'')+'<div class="btns">';
  for(let i=1;i<=p.people;i++)html+='<button data-n="'+i+'" class="'+(i===p.patient&&p.locked?'on':'')+'">'+i+'</button>';
  html+='<button data-n="0" class="'+(p.locked?'':'on')+'">auto</button><button data-survey="1">'+(p.survey?'stop scan':'find all')+'</button></div>';
  el.innerHTML=html;
  el.querySelectorAll('button[data-n]').forEach(b=>b.onclick=()=>post('/select',{n:+b.dataset.n}));
  const sv=el.querySelector('button[data-survey]');if(sv)sv.onclick=()=>post('/select',{survey:!p.survey})}
function post(path,body){return fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})}
function handle(p){
  if(p.topic==='state'){setState(p.state,undefined);renderPeople(p)}
  else if(p.topic==='event'||p.topic==='escalation'){if(p.state)setState(p.state,p.reason);if(p.event_id)eventId=p.event_id;addEvent(p)}
  else if(p.topic==='ack'){setState(p.state||'IDLE','acknowledged');addEvent(p)}
  else if(p.topic==='heartbeat'){lastHb=Date.now()}
  else addEvent(p)}
function connect(){const es=new EventSource('/events');es.onmessage=e=>{try{handle(JSON.parse(e.data))}catch(err){}};
  es.onerror=()=>{es.close();setTimeout(connect,2000)}}
connect();
setInterval(()=>{const el=$('hb');if(lastHb===null)return;const age=Math.round((Date.now()-lastHb)/1000);
  el.textContent='last heartbeat '+age+' s ago';el.classList.toggle('stale',age>30)},1000);
$('ack').onclick=async()=>{const user=$('user').value.trim()||'nurse';await post('/ack',{user,event_id:eventId})};
</script></body></html>
"""


class WebClient(Notifier):
    """HTTP + SSE server. Call :meth:`start`, feed :meth:`notify`, :meth:`stop` at exit."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, *, bed: str = "bed-01",
                 on_ack: AckCallback | None = None, history: int = 50) -> None:
        self.host = host
        self.port = port
        self.bed = bed
        self._ack_callbacks: list[AckCallback] = [on_ack] if on_ack else []
        self._select_callbacks: list[AckCallback] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._subscribers: set[queue.Queue[dict[str, Any] | None]] = set()
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    # ------------------------------------------------------------ lifecycle

    def start(self) -> WebClient:
        client = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # quiet
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/events":
                    client._serve_events(self)
                elif self.path in ("/", "/index.html"):
                    body = PAGE.replace("__BED__", client.bed).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                if self.path not in ("/ack", "/select"):
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("body must be a JSON object")
                    if self.path == "/select":
                        reply = client._parse_select(data)
                except Exception as exc:  # noqa: BLE001
                    self.send_error(400, str(exc))
                    return
                if self.path == "/ack":
                    reply = {"user": str(data.get("user") or "nurse"), "t": time.time()}
                    if data.get("event_id"):
                        reply["event_id"] = data["event_id"]
                    client._dispatch(client._ack_callbacks, reply, "ack")
                else:
                    client._dispatch(client._select_callbacks, reply, "select")
                body = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="web", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            for q in list(self._subscribers):
                q.put(None)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ------------------------------------------------------------- notifier

    def notify(self, topic: str, payload: dict[str, Any]) -> None:
        item = {"topic": topic, **payload}
        with self._lock:
            self._history.append(item)
            subs = list(self._subscribers)
        for q in subs:
            q.put(item)

    def on_ack(self, callback: AckCallback) -> None:
        self._ack_callbacks.append(callback)

    def on_select(self, callback: AckCallback) -> None:
        """Patient selection from the page: ``{"n": k}`` (k = 0 clears the lock) or ``{"survey": bool}``."""
        self._select_callbacks.append(callback)

    @staticmethod
    def _parse_select(data: dict[str, Any]) -> dict[str, Any]:
        if "survey" in data:
            return {"survey": bool(data["survey"]), "t": time.time()}
        n = data.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0 or n > 99:
            raise ValueError("n must be an integer 0..99 (0 = auto), or pass survey: true/false")
        return {"n": n, "t": time.time()}

    def _dispatch(self, callbacks: list[AckCallback], payload: dict[str, Any], label: str) -> None:
        for cb in list(callbacks):
            try:
                cb(payload)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{label} callback: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ SSE

    def _serve_events(self, handler: BaseHTTPRequestHandler) -> None:
        q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        with self._lock:
            backlog = list(self._history)
            self._subscribers.add(q)
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            handler.end_headers()
            # SSE is one request per socket. send_header("Connection", ...) would
            # overwrite this flag, so set it after the headers are out.
            handler.close_connection = True
            for item in backlog:
                handler.wfile.write(f"data: {json.dumps(item, default=str)}\n\n".encode())
            handler.wfile.flush()
            idle = 0
            while not self._stopping.is_set():
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    idle += 1
                    if idle >= 15:
                        handler.wfile.write(b": ping\n\n")
                        handler.wfile.flush()
                        idle = 0
                    continue
                if item is None:
                    break
                idle = 0
                handler.wfile.write(f"data: {json.dumps(item, default=str)}\n\n".encode())
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self._lock:
                self._subscribers.discard(q)
