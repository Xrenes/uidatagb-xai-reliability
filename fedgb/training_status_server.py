"""
training_status_server.py
---------------------------
Live dashboard for the three background jobs from the current revision pass:
  1. Stage-2 ResNet-50 5-seed training  (outputs/session_extra_seeds_5.log)
  2. Stage-2 DenseNet-121 5-seed training (outputs/session_densenet_seeds.log)
  3. pHash threshold sensitivity sweep  (outputs/phash_threshold_sensitivity.log)
  4. Group-aware bootstrap statistics   (outputs/group_aware_statistics.log)

Parses each log's plain-text progress lines and polls nvidia-smi, serves a
single self-contained auto-refreshing HTML page.

Run from fedgb/:  python training_status_server.py
Then open:        http://localhost:8767
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import time

PORT = 8767
HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = [
    {"key": "resnet50_seeds", "label": "Stage 2 ResNet-50 (5 seeds)",
     "log": os.path.join(HERE, "outputs", "session_extra_seeds_5.log")},
    {"key": "densenet121_seeds", "label": "Stage 2 DenseNet-121 (5 seeds)",
     "log": os.path.join(HERE, "outputs", "session_densenet_seeds.log")},
    {"key": "phash_sweep", "label": "pHash threshold sensitivity sweep",
     "log": os.path.join(HERE, "outputs", "phash_threshold_sensitivity.log")},
    {"key": "group_stats", "label": "Group-aware bootstrap statistics",
     "log": os.path.join(HERE, "outputs", "group_aware_statistics.log")},
]

EPOCH_RE = re.compile(
    r"\[(?P<tag>[\w=. ]+)\]\s+epoch\s+(?P<epoch>\d+)/(?P<epoch_total>\d+)\s+"
    r"loss=(?P<loss>[\d.]+)\s+\((?P<elapsed>\d+)s elapsed\)"
)
RESULT_RE = re.compile(
    r"\[(?P<tag>[\w=. ]+)\]\s+accuracy=(?P<acc>[\d.]+)\s+f1=(?P<f1>[\d.]+)\s+auc=(?P<auc>[\d.]+)"
)
SUMMARY_RE = re.compile(
    r"\[multiseed[^\]]*\]\s+(?P<n>\d+)-seed Stage-2 summary:\s+"
    r"accuracy=(?P<acc>[\d.]+)\+/-(?P<acc_std>[\d.]+)\s+F1=(?P<f1>[\d.]+)\+/-(?P<f1_std>[\d.]+)\s+"
    r"AUC=(?P<auc>[\d.]+)\+/-(?P<auc_std>[\d.]+)"
)
HASHED_RE = re.compile(r"hashed (\d+)/(\d+)")
THRESH_RE = re.compile(r"threshold=(\d+):\s+(\{.*\})")
GROUP_SEED_RE = re.compile(r"seed=(\d+):\s+balanced_accuracy=([\d.]+)\s+macro_pr_auc=([\d.]+)")
ERROR_RE = re.compile(r"Traceback \(most recent call last\)|(?<!\w)Error(?!Action)")


def tail_text(path, n_bytes=40000):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_bytes), os.SEEK_SET)
        return f.read().decode("utf-8", errors="ignore")


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


def parse_training_log(text):
    if text is None:
        return {"found": False}
    epochs = [m.groupdict() for m in EPOCH_RE.finditer(text)]
    results = [m.groupdict() for m in RESULT_RE.finditer(text)]
    summary_m = SUMMARY_RE.search(text)
    has_error = bool(ERROR_RE.search(text))
    last_epoch = epochs[-1] if epochs else None
    return {
        "found": True,
        "last_epoch": last_epoch,
        "n_completed_runs": len(results),
        "completed_runs": results[-8:],
        "summary": summary_m.groupdict() if summary_m else None,
        "has_error": has_error,
        "tail_lines": text.strip().splitlines()[-6:],
    }


def parse_phash_log(text):
    if text is None:
        return {"found": False}
    hashed = HASHED_RE.findall(text)
    last_hashed = hashed[-1] if hashed else None
    thresholds_done = THRESH_RE.findall(text)
    done = "wrote outputs/leakage/phash_threshold_sensitivity_summary.json" in text or "done." in text.lower()
    has_error = bool(ERROR_RE.search(text))
    return {
        "found": True,
        "last_hashed": {"n": last_hashed[0], "total": last_hashed[1]} if last_hashed else None,
        "thresholds_completed": [t[0] for t in thresholds_done],
        "done": done,
        "has_error": has_error,
        "tail_lines": text.strip().splitlines()[-6:],
    }


def parse_group_stats_log(text):
    if text is None:
        return {"found": False}
    seeds_done = GROUP_SEED_RE.findall(text)
    done = "wrote" in text and "group_aware_statistics.json" in text
    has_error = bool(ERROR_RE.search(text))
    return {
        "found": True,
        "seeds_completed": [{"seed": s, "balanced_accuracy": ba, "pr_auc": pa} for s, ba, pa in seeds_done],
        "done": done,
        "has_error": has_error,
        "tail_lines": text.strip().splitlines()[-6:],
    }


def build_status():
    status = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": read_gpu(), "jobs": {}}
    for job in JOBS:
        text = tail_text(job["log"])
        if job["key"] in ("resnet50_seeds", "densenet121_seeds"):
            parsed = parse_training_log(text)
        elif job["key"] == "phash_sweep":
            parsed = parse_phash_log(text)
        else:
            parsed = parse_group_stats_log(text)
        parsed["label"] = job["label"]
        parsed["log_path"] = job["log"]
        status["jobs"][job["key"]] = parsed
    return status


PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>UIdataGB revision - background job status</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; background:#0e1420; color:#e6e9ef; margin:0; padding:24px; }
  h1 { font-size:1.3rem; margin:0 0 4px; }
  .meta { color:#8b93a7; font-size:0.85rem; margin-bottom:20px; display:flex; align-items:center; gap:10px; }
  .live-dot { width:8px; height:8px; border-radius:50%; background:#3ecf8e; display:inline-block; animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.25; } }
  .live-dot.stale { background:#e0a63e; animation:none; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; }
  .card { background:#161d2e; border:1px solid #29314a; border-radius:10px; padding:16px 18px; transition: box-shadow 0.4s ease, border-color 0.4s ease; }
  .card.flash { box-shadow: 0 0 0 2px #3e7cf0; border-color:#3e7cf0; }
  .card h2 { font-size:1.05rem; margin:0 0 10px; display:flex; align-items:center; gap:8px; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .dot.ok { background:#3ecf8e; }
  .dot.warn { background:#e0a63e; }
  .dot.err { background:#e05555; }
  .dot.idle { background:#555; }
  table { width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:6px; }
  td, th { text-align:left; padding:3px 6px; border-bottom:1px solid #22293c; }
  th { color:#8b93a7; font-weight:600; }
  pre { background:#0b101c; border-radius:6px; padding:8px 10px; font-size:0.72rem; overflow-x:auto; white-space:pre-wrap; max-height:140px; }
  .gpu-bar-wrap { background:#0b101c; border-radius:6px; height:18px; overflow:hidden; margin-top:4px; }
  .gpu-bar { background:#3e7cf0; height:100%; transition: width 0.6s ease; }
  .summary { background:#0b101c; border-radius:6px; padding:8px 10px; margin-top:8px; font-size:0.85rem; }
  .badge { font-size:0.72rem; padding:1px 7px; border-radius:10px; background:#29314a; color:#c7cde0; }
</style></head>
<body>
<h1>UIdataGB revision &mdash; background job status</h1>
<div class="meta">
  <span class="live-dot" id="liveDot"></span>
  <span id="liveLabel">live</span>
  &middot; last update <span id="lastUpdate">__GENERATED_AT__</span>
  (<span id="secondsAgo">0</span>s ago)
</div>
<div class="grid" id="grid">
__GPU_CARD__
__JOB_CARDS__
</div>
<script>
let lastFetchTime = Date.now();
let lastHtml = {};

function markFlash(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('flash');
  setTimeout(() => el.classList.remove('flash'), 900);
}

async function poll() {
  try {
    const res = await fetch('/render.json', {cache: 'no-store'});
    if (!res.ok) throw new Error('bad status ' + res.status);
    const data = await res.json();
    lastFetchTime = Date.now();
    document.getElementById('lastUpdate').textContent = data.generated_at;
    document.getElementById('liveDot').classList.remove('stale');
    document.getElementById('liveLabel').textContent = 'live';

    const gpuEl = document.getElementById('card-gpu');
    if (gpuEl && lastHtml.gpu !== data.gpu_card) {
      gpuEl.outerHTML = data.gpu_card;
      markFlash('card-gpu');
      lastHtml.gpu = data.gpu_card;
    }
    for (const key of Object.keys(data.job_cards)) {
      const elId = 'card-' + key;
      const el = document.getElementById(elId);
      const html = data.job_cards[key];
      if (el && lastHtml[key] !== html) {
        el.outerHTML = html;
        markFlash(elId);
        lastHtml[key] = html;
      }
    }
  } catch (e) {
    document.getElementById('liveDot').classList.add('stale');
    document.getElementById('liveLabel').textContent = 'connection lost, retrying...';
  }
}

setInterval(poll, 2000);
setInterval(() => {
  document.getElementById('secondsAgo').textContent = Math.round((Date.now() - lastFetchTime) / 1000);
}, 1000);
poll();
</script>
</body></html>"""


def render_gpu_card(gpu):
    if not gpu:
        return '<div class="card" id="card-gpu"><h2><span class="dot idle"></span>GPU</h2>nvidia-smi unavailable</div>'
    pct = gpu["util"]
    mem_pct = round(100 * gpu["mem_used"] / max(1, gpu["mem_total"]))
    return f"""<div class="card" id="card-gpu">
  <h2><span class="dot {'ok' if pct>0 else 'idle'}"></span>GPU (GeForce GTX 1050)</h2>
  <div>Utilization: {pct}%</div>
  <div class="gpu-bar-wrap"><div class="gpu-bar" style="width:{pct}%"></div></div>
  <div style="margin-top:8px;">Memory: {gpu['mem_used']} / {gpu['mem_total']} MB ({mem_pct}%)</div>
  <div class="gpu-bar-wrap"><div class="gpu-bar" style="width:{mem_pct}%"></div></div>
  <div style="margin-top:8px;">Temp: {gpu['temp']}&deg;C</div>
</div>"""


def render_training_card(job, key):
    label = job["label"]
    if not job.get("found"):
        return f'<div class="card" id="card-{key}"><h2><span class="dot idle"></span>{label}</h2>log not found yet</div>'
    dot = "err" if job.get("has_error") else ("ok" if job.get("last_epoch") else "warn")
    body = ""
    if job.get("last_epoch"):
        le = job["last_epoch"]
        elapsed_min = round(int(le["elapsed"]) / 60, 1)
        body += (f'<div><span class="badge">{le["tag"]}</span> epoch {le["epoch"]}/{le["epoch_total"]} '
                 f'&middot; loss={le["loss"]} &middot; {elapsed_min} min elapsed</div>')
    if job.get("completed_runs"):
        body += '<table><tr><th>run</th><th>acc</th><th>f1</th><th>auc</th></tr>'
        for r in job["completed_runs"]:
            body += f'<tr><td>{r["tag"]}</td><td>{r["acc"]}</td><td>{r["f1"]}</td><td>{r["auc"]}</td></tr>'
        body += '</table>'
    if job.get("summary"):
        s = job["summary"]
        body += (f'<div class="summary"><b>{s["n"]}-seed summary</b><br>'
                  f'accuracy = {s["acc"]} &plusmn; {s["acc_std"]}<br>'
                  f'macro-F1 = {s["f1"]} &plusmn; {s["f1_std"]}<br>'
                  f'macro-AUC = {s["auc"]} &plusmn; {s["auc_std"]}</div>')
    tail = "\n".join(job.get("tail_lines", []))
    body += f'<pre>{tail}</pre>' if tail else ""
    return f'<div class="card" id="card-{key}"><h2><span class="dot {dot}"></span>{label}</h2>{body}</div>'


def render_phash_card(job, key):
    label = job["label"]
    if not job.get("found"):
        return f'<div class="card" id="card-{key}"><h2><span class="dot idle"></span>{label}</h2>log not found yet</div>'
    dot = "err" if job.get("has_error") else ("ok" if job.get("done") else "warn")
    body = ""
    if job.get("last_hashed"):
        h = job["last_hashed"]
        body += f'<div>Hashing: {h["n"]}/{h["total"]} images</div>'
    if job.get("thresholds_completed"):
        body += f'<div>Thresholds completed: {", ".join(job["thresholds_completed"])}</div>'
    if job.get("done"):
        body += '<div class="summary">Sweep complete &mdash; see outputs/leakage/phash_threshold_sensitivity.csv</div>'
    tail = "\n".join(job.get("tail_lines", []))
    body += f'<pre>{tail}</pre>' if tail else ""
    return f'<div class="card" id="card-{key}"><h2><span class="dot {dot}"></span>{label}</h2>{body}</div>'


def render_group_stats_card(job, key):
    label = job["label"]
    if not job.get("found"):
        return f'<div class="card" id="card-{key}"><h2><span class="dot idle"></span>{label}</h2>log not found yet</div>'
    dot = "err" if job.get("has_error") else ("ok" if job.get("done") else "warn")
    body = ""
    if job.get("seeds_completed"):
        body += '<table><tr><th>seed</th><th>balanced acc</th><th>PR-AUC</th></tr>'
        for s in job["seeds_completed"]:
            body += f'<tr><td>{s["seed"]}</td><td>{s["balanced_accuracy"]}</td><td>{s["pr_auc"]}</td></tr>'
        body += '</table>'
    if job.get("done"):
        body += '<div class="summary">Complete &mdash; see outputs/phase0/group_aware_statistics.json</div>'
    tail = "\n".join(job.get("tail_lines", []))
    body += f'<pre>{tail}</pre>' if tail else ""
    return f'<div class="card" id="card-{key}"><h2><span class="dot {dot}"></span>{label}</h2>{body}</div>'


RENDERERS = {
    "resnet50_seeds": render_training_card,
    "densenet121_seeds": render_training_card,
    "phash_sweep": render_phash_card,
    "group_stats": render_group_stats_card,
}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/status.json"):
            self._send_json(build_status())
            return

        if self.path.startswith("/render.json"):
            # pre-rendered HTML fragments, polled client-side every 2s so the
            # page updates in place without a full reload (meta-refresh flashes
            # the whole page and made it hard to tell the data was actually live)
            status = build_status()
            self._send_json({
                "generated_at": status["generated_at"],
                "gpu_card": render_gpu_card(status["gpu"]),
                "job_cards": {k: RENDERERS[k](v, k) for k, v in status["jobs"].items()},
            })
            return

        status = build_status()
        job_cards = "\n".join(RENDERERS[k](v, k) for k, v in status["jobs"].items())
        html = (PAGE_TEMPLATE
                .replace("__GENERATED_AT__", status["generated_at"])
                .replace("__GPU_CARD__", render_gpu_card(status["gpu"]))
                .replace("__JOB_CARDS__", job_cards))
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    # bind 0.0.0.0 (not 127.0.0.1) so it accepts connections regardless of
    # whether the client resolves "localhost" to the IPv4 or IPv6 loopback
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[status] serving http://localhost:{PORT}  (Ctrl+C to stop)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
