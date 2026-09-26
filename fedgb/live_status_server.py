"""
live_status_server.py
----------------------
Tiny local dashboard for watching the multi-seed training run in real time.
Parses outputs/seeds/multiseed_centralized.log + polls nvidia-smi, serves a
single self-contained HTML page that refreshes every second.

Run from fedgb/:  python live_status_server.py
Then open:        http://localhost:8766
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import time
import webbrowser

PORT = 8766
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "outputs", "seeds", "uidatagb_resume.log")

LINE_RE = re.compile(
    r"\[(?P<method>\w+) seed=(?P<seed>\d+)(?: client=(?P<client>\d+))?\] epoch (?P<epoch>\d+)/(?P<epoch_total>\d+):\s+"
    r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<batch>\d+)/(?P<batch_total>\d+)\s*"
    r"\[(?P<elapsed>\d+:\d+)<(?P<remaining>\d+:\d+),\s*(?P<rate>[\d.]+)(?P<rate_unit>s/batch|batch/s)"
)
DONE_RE = re.compile(r"\[done\] total wall-time ([\d.]+)s")
ERROR_RE = re.compile(r"Traceback|Error(?!Action)")


def read_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=3, text=True,
        ).strip()
        util, mem_used, mem_total, temp = [x.strip() for x in out.split(",")]
        return {"util": int(util), "mem_used": int(mem_used),
                "mem_total": int(mem_total), "temp": int(temp)}
    except Exception:
        return None


def build_status():
    if not os.path.exists(LOG_PATH):
        return {"found": False}

    with open(LOG_PATH, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 60000), os.SEEK_SET)
        chunk = f.read().decode("utf-8", errors="ignore")

    # tqdm rewrites the same logical update many times per line-buffer flush;
    # split on the literal marker every update starts with instead of "\n".
    parts = re.split(r"(?=\[\w+ seed=\d+\] epoch)", chunk)
    last_match = None
    completed_epochs = []
    seen_epoch_keys = set()
    for part in parts:
        m = LINE_RE.search(part)
        if m:
            last_match = m
            if m.group("pct") == "100":
                key = (m.group("seed"), m.group("epoch"))
                if key not in seen_epoch_keys:
                    seen_epoch_keys.add(key)
                    completed_epochs.append({
                        "seed": m.group("seed"),
                        "epoch": int(m.group("epoch")),
                        "elapsed": m.group("elapsed"),
                    })

    done = bool(DONE_RE.search(chunk))
    error = bool(ERROR_RE.search(chunk)) and "NativeCommandError" not in chunk

    status = {
        "found": True,
        "done": done,
        "error": error,
        "gpu": read_gpu(),
        "completed_epochs": completed_epochs[-10:],
        "n_completed_epochs": len(completed_epochs),
    }
    if last_match:
        g = last_match.groupdict()
        status["current"] = {
            "method": g["method"], "seed": g["seed"], "client": g.get("client"),
            "epoch": int(g["epoch"]), "epoch_total": int(g["epoch_total"]),
            "pct": int(g["pct"]),
            "batch": int(g["batch"]), "batch_total": int(g["batch_total"]),
            "elapsed": g["elapsed"], "remaining": g["remaining"],
            "rate": g["rate"], "rate_unit": g["rate_unit"],
        }
    return status


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>UIdataGB Training Monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark light; }
  * { box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', -apple-system, Roboto, sans-serif; max-width: 780px;
    margin: 36px auto 64px; padding: 0 18px; color:#e8eaf0;
    background:
      radial-gradient(1100px 480px at 12% -10%, rgba(96,140,255,.16), transparent 60%),
      radial-gradient(900px 420px at 100% 0%, rgba(255,150,80,.10), transparent 55%),
      #0a0c11;
    min-height: 100vh;
  }
  h1 { font-size: 1.5rem; font-weight: 700; margin: 0 0 4px; letter-spacing: -.01em;
       display:flex; align-items:center; gap:10px; }
  h1 .emoji { font-size: 1.35rem; }
  .sub { color:#8891a5; font-size: .84rem; margin-bottom: 26px; }
  .sub b { color:#b7bfd4; font-weight: 600; }

  .pill { display:inline-flex; align-items:center; gap:6px; font-size:.72rem; font-weight:700;
          text-transform:uppercase; letter-spacing:.06em; padding: 4px 11px 4px 8px;
          border-radius: 999px; background:#16351f; color:#5fe08a; border:1px solid #1f5c33; }
  .pill .dot { width:7px; height:7px; border-radius:50%; background:#4caf50;
               box-shadow: 0 0 0 3px rgba(76,175,80,.22); animation: pulse 1.6s ease-in-out infinite; }
  .pill.err { background:#3a1414; color:#ff9494; border-color:#6b2323; }
  .pill.err .dot { background:#e05252; box-shadow: 0 0 0 3px rgba(224,82,82,.22); animation:none; }
  .pill.done { background:#132038; color:#7fb2ff; border-color:#254a82; }
  .pill.done .dot { background:#4f8cff; box-shadow: 0 0 0 3px rgba(79,140,255,.22); animation:none; }
  .pill.wait { background:#241f16; color:#e0b45f; border-color:#5c4a1f; }
  .pill.wait .dot { background:#e0a94c; box-shadow: 0 0 0 3px rgba(224,169,76,.22); }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }

  .card {
    background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,0)), #12151c;
    border:1px solid #23273240; box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 8px 24px -12px rgba(0,0,0,.5);
    border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;
  }
  .card-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }
  .card-title { font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em;
                color:#8891a5; }
  .row { display:flex; justify-content:space-between; align-items:baseline; margin: 9px 0; font-size:.92rem;}
  .label { color:#8891a5; }
  .val { font-variant-numeric: tabular-nums; font-weight:600; color:#eef0f6; }
  .val.big { font-size:1.5rem; font-weight:750; letter-spacing:-.01em; }

  .bar-outer { background:#1c2029; border-radius:8px; height:12px; overflow:hidden; margin:6px 0 4px;
               border: 1px solid #262b36; position:relative; }
  .bar-inner { background:linear-gradient(90deg,#3d7aff,#7ad1ff); height:100%; width:0%; border-radius:8px;
               transition: width .5s cubic-bezier(.4,0,.2,1); position:relative; overflow:hidden; }
  .bar-inner::after { content:''; position:absolute; inset:0;
    background:linear-gradient(110deg, transparent 30%, rgba(255,255,255,.35) 48%, transparent 66%);
    background-size: 200% 100%; animation: shimmer 2.2s linear infinite; }
  .bar-inner.gpu { background: linear-gradient(90deg,#ff8a3d,#ffd24f); }
  .bar-inner.gpu.hot { background: linear-gradient(90deg,#ff5050,#ff9d4f); }
  .bar-inner.small { height:100%; }
  .bar-outer.thin { height:7px; }
  @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -40% 0; } }

  .gpu-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 18px 22px; }
  .gpu-metric .num { font-size:1.3rem; font-weight:750; }
  .gpu-metric .num.warm { color:#ffbd5c; }
  .gpu-metric .num.hot { color:#ff7a6b; }
  .gpu-metric .num.cool { color:#7fe0a0; }

  table { width:100%; border-collapse: collapse; font-size:.85rem; }
  thead th { text-align:left; padding: 6px 8px; color:#8891a5; font-weight:600;
             font-size:.72rem; text-transform:uppercase; letter-spacing:.05em;
             border-bottom:1px solid #262b36; }
  tbody td { text-align:left; padding: 8px 8px; border-bottom:1px solid #1a1d25;
             font-variant-numeric: tabular-nums; }
  tbody tr:hover { background: rgba(255,255,255,.02); }
  tbody tr:last-child td { border-bottom:none; }
  .mono { font-variant-numeric: tabular-nums; }

  #err-banner { display:none; background:#2a1414; border:1px solid #6b2323; color:#ffb4ab;
                padding:12px 16px; border-radius:12px; margin-bottom:16px; font-size:.85rem; }

  footer { text-align:center; color:#565d6e; font-size:.76rem; margin-top:28px; }
</style></head>
<body>
  <h1><span class="emoji">🫀</span>UIdataGB Training Monitor</h1>
  <div class="sub">Auto-refreshing every second &middot; last update <b id="lastupdate">-</b></div>
  <span class="pill wait" id="status-pill"><span class="dot"></span><span id="status-text">Connecting</span></span>

  <div style="height:18px"></div>

  <div id="err-banner">No progress line found in the last minute of log, or an error was detected — check the log directly.</div>

  <div class="card" id="waiting-card">Waiting for training log&hellip;</div>

  <div class="card" id="main-card" style="display:none">
    <div class="card-head">
      <span class="card-title">Current run</span>
      <span class="val" id="method-seed">-</span>
    </div>
    <div class="row"><span class="label">Epoch</span><span class="val big" id="epoch-text">-</span></div>
    <div class="bar-outer"><div class="bar-inner" id="epoch-bar"></div></div>
    <div class="row"><span class="label">Batch</span><span class="val" id="batch-text">-</span></div>
    <div class="bar-outer thin"><div class="bar-inner small" id="batch-bar"></div></div>
    <div class="row"><span class="label">Elapsed / remaining (this epoch)</span><span class="val" id="time-text">-</span></div>
    <div class="row"><span class="label">Rate</span><span class="val" id="rate-text">-</span></div>
  </div>

  <div class="card" id="gpu-card" style="display:none">
    <div class="card-head"><span class="card-title">GPU</span></div>
    <div class="gpu-grid">
      <div class="gpu-metric">
        <div class="row" style="margin:0 0 6px"><span class="label">Utilization</span>
          <span class="num" id="gpu-util-text">-</span></div>
        <div class="bar-outer"><div class="bar-inner gpu" id="gpu-bar"></div></div>
      </div>
      <div class="gpu-metric">
        <div class="row" style="margin:0 0 6px"><span class="label">VRAM</span>
          <span class="num" id="gpu-mem-pct">-</span></div>
        <div class="bar-outer"><div class="bar-inner gpu" id="gpu-mem-bar"></div></div>
      </div>
    </div>
    <div class="row" style="margin-top:16px"><span class="label">VRAM used</span><span class="val" id="gpu-mem-text">-</span></div>
    <div class="row"><span class="label">Temperature</span><span class="val" id="gpu-temp-text">-</span></div>
  </div>

  <div class="card">
    <div class="card-head">
      <span class="card-title">Recent epoch completions</span>
      <span class="val" id="n-epochs">-</span>
    </div>
    <table id="epoch-table"><thead><tr><th>Seed</th><th>Epoch</th><th>Wall time</th></tr></thead>
      <tbody id="epoch-tbody"></tbody></table>
  </div>

  <footer>polling <code>/status</code> every 1000ms &middot; localhost:8766</footer>

<script>
function gpuNumClass(pct, kind) {
  if (kind === 'temp') return pct >= 80 ? 'num hot' : (pct >= 65 ? 'num warm' : 'num cool');
  return pct >= 90 ? 'num hot' : (pct >= 60 ? 'num warm' : 'num cool');
}

async function tick() {
  const pill = document.getElementById('status-pill');
  const pillText = document.getElementById('status-text');
  try {
    const res = await fetch('/status', {cache: 'no-store'});
    const s = await res.json();
    document.getElementById('lastupdate').textContent = new Date().toLocaleTimeString();

    if (!s.found) {
      pill.className = 'pill wait'; pillText.textContent = 'Waiting';
      document.getElementById('waiting-card').style.display = 'block';
      document.getElementById('main-card').style.display = 'none';
      document.getElementById('gpu-card').style.display = 'none';
      return;
    }
    document.getElementById('waiting-card').style.display = 'none';
    document.getElementById('main-card').style.display = 'block';

    document.getElementById('err-banner').style.display = s.error ? 'block' : 'none';
    if (s.done) { pill.className = 'pill done'; pillText.textContent = 'All done'; }
    else if (s.error) { pill.className = 'pill err'; pillText.textContent = 'Error'; }
    else { pill.className = 'pill'; pillText.textContent = 'Running'; }

    if (s.current) {
      const c = s.current;
      document.getElementById('method-seed').textContent =
        c.client !== null && c.client !== undefined
          ? `${c.method} · seed ${c.seed} · client ${c.client}`
          : `${c.method} · seed ${c.seed}`;
      document.getElementById('epoch-text').textContent = `${c.epoch} / ${c.epoch_total}  (${c.pct}%)`;
      document.getElementById('epoch-bar').style.width = c.pct + '%';
      document.getElementById('batch-text').textContent = `${c.batch} / ${c.batch_total}`;
      document.getElementById('batch-bar').style.width =
        Math.round(100 * c.batch / Math.max(1, c.batch_total)) + '%';
      document.getElementById('time-text').textContent = `${c.elapsed} elapsed, ${c.remaining} left`;
      document.getElementById('rate-text').textContent = `${c.rate} ${c.rate_unit}`;
    }

    if (s.gpu) {
      document.getElementById('gpu-card').style.display = 'block';
      document.getElementById('gpu-util-text').textContent = s.gpu.util + '%';
      document.getElementById('gpu-util-text').className = gpuNumClass(s.gpu.util, 'util');
      document.getElementById('gpu-bar').style.width = s.gpu.util + '%';
      document.getElementById('gpu-bar').className = 'bar-inner gpu' + (s.gpu.util >= 90 ? ' hot' : '');

      const memPct = Math.round(100 * s.gpu.mem_used / Math.max(1, s.gpu.mem_total));
      document.getElementById('gpu-mem-pct').textContent = memPct + '%';
      document.getElementById('gpu-mem-pct').className = gpuNumClass(memPct, 'util');
      document.getElementById('gpu-mem-bar').style.width = memPct + '%';
      document.getElementById('gpu-mem-bar').className = 'bar-inner gpu' + (memPct >= 90 ? ' hot' : '');

      document.getElementById('gpu-mem-text').textContent =
        `${s.gpu.mem_used.toLocaleString()} MiB / ${s.gpu.mem_total.toLocaleString()} MiB`;
      const tempEl = document.getElementById('gpu-temp-text');
      tempEl.textContent = s.gpu.temp + ' °C';
      tempEl.className = 'val ' + (s.gpu.temp >= 80 ? '' : '');
    }

    document.getElementById('n-epochs').textContent = s.n_completed_epochs + ' completed';
    const tbody = document.getElementById('epoch-tbody');
    tbody.innerHTML = '';
    for (const e of s.completed_epochs.slice().reverse()) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${e.seed}</td><td>${e.epoch}</td><td class="mono">${e.elapsed}</td>`;
      tbody.appendChild(tr);
    }

    if (s.done) {
      document.getElementById('method-seed').textContent += '  ✓';
    }
  } catch (e) {
    pill.className = 'pill err'; pillText.textContent = 'Disconnected';
  }
}
tick();
setInterval(tick, 1000);
</script>
</body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/status":
            body = json.dumps(build_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    print(f"\n  FedGB Live Training Monitor")
    print(f"  ─────────────────────────────────")
    print(f"  Watching: {LOG_PATH}")
    print(f"  Open:     http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
