#!/usr/bin/env python3
"""
overlap_progress_server.py
---------------------------
Tiny local HTTP server that serves a live-updating dashboard for
compute_overlap_all_methods.py (Grad-CAM / Grad-CAM++ / Saliency
cosine-overlap computation across all classes and clients).

Run from fedgb/:
    python overlap_progress_server.py

Then open:  http://localhost:8766
"""
from __future__ import annotations

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PATH = os.path.join(REPO_ROOT, "outputs", "overlap_progress.json")
RESULT_PATH = os.path.join(REPO_ROOT, "outputs", "overlap_all_methods.json")
PORT = 8766

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>XAI-FedGB &mdash; Overlap Computation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f5f1; --ground2: #ffffff; --ink: #17181c; --muted: #6b6a66;
    --line: #e4e2db; --line-strong: #d2cfc4;
    --accent: #4f6f9e; --amber: #b8752f; --amber-soft: #f6ead9;
    --green: #4c8567; --green-soft: #e2f0e7; --track: #eae8e1;
    --shadow: 0 1px 2px rgba(20,20,15,0.04), 0 8px 24px -12px rgba(20,20,15,0.10);
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1013; --ground2: #17191e; --ink: #eeece5; --muted: #918f89;
      --line: #262922; --line-strong: #34372e; --accent: #8ba7d6;
      --amber: #e0a563; --amber-soft: #2c2116; --green: #7fc79a; --green-soft: #142a1e;
      --track: #1d1f24; --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 30px -14px rgba(0,0,0,0.6);
    }
  }
  * { box-sizing: border-box; }
  html, body { overflow-x: hidden; }
  body {
    margin: 0; padding: 2.5rem 1.25rem 4rem;
    background: var(--bg); color: var(--ink); font-family: var(--sans);
  }
  .page { max-width: 760px; margin: 0 auto; }
  .eyebrow {
    font-family: var(--mono); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 .5rem;
  }
  h1 { font-size: 1.5rem; font-weight: 650; letter-spacing: -0.015em; margin: 0 0 .35rem; }
  .dek { color: var(--muted); font-size: .92rem; line-height: 1.5; margin: 0 0 1.75rem; max-width: 58ch; }
  code { font-family: var(--mono); background: var(--track); padding: .12rem .4rem; border-radius: 5px; font-size: .84em; }

  .statusbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1.25rem; }
  .pill {
    display: inline-flex; align-items: center; gap: .45rem; font-family: var(--mono);
    font-size: .74rem; font-weight: 600; padding: .35rem .75rem; border-radius: 999px;
    background: var(--track); color: var(--muted); border: 1px solid var(--line);
  }
  .pill .dot { width: .45rem; height: .45rem; border-radius: 50%; background: currentColor; }
  .pill.running { color: var(--amber); background: var(--amber-soft); border-color: transparent; }
  .pill.running .dot { animation: pulse 1.6s ease-in-out infinite; }
  .pill.done { color: var(--green); background: var(--green-soft); border-color: transparent; }
  @keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .45; transform: scale(1.25); } }
  @media (prefers-reduced-motion: reduce) { .pill.running .dot { animation: none; } }
  .refresh-hint { font-family: var(--mono); font-size: .72rem; color: var(--muted); }

  .card {
    background: var(--ground2); border: 1px solid var(--line); border-radius: 16px;
    box-shadow: var(--shadow); padding: 1.6rem 1.6rem 1.4rem; margin-bottom: 1.1rem;
  }
  .timer-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .timer-block .num {
    font-family: var(--mono); font-variant-numeric: tabular-nums;
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1;
  }
  .timer-block.pct .num { color: var(--accent); }
  .timer-block .lbl { font-family: var(--mono); font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-top: .5rem; }
  .bar-track { height: 10px; border-radius: 999px; background: var(--track); overflow: hidden; margin-top: 1.3rem; border: 1px solid var(--line); }
  .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--green)); width: 0%; transition: width .5s cubic-bezier(.4,0,.2,1); }
  .bar-caption { display: flex; justify-content: space-between; font-family: var(--mono); font-size: .72rem; color: var(--muted); margin-top: .5rem; }

  .section-label { font-family: var(--mono); font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 1.9rem 0 .7rem; }
  .methods { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: .7rem; }
  .method { position: relative; border: 1px solid var(--line); border-radius: 12px; padding: .85rem .8rem .75rem; background: var(--ground2); }
  .method::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 12px 12px 0 0; background: var(--line); }
  .method.active { border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); }
  .method.active::before { background: var(--accent); }
  .method.complete::before { background: var(--green); }
  .method .name { font-weight: 600; font-size: .86rem; }
  .method .formula { font-family: var(--mono); font-size: .68rem; color: var(--muted); margin-top: .2rem; }
  .method .state { font-family: var(--mono); font-size: .72rem; margin-top: .55rem; display: flex; align-items: center; gap: .35rem; }
  .method.pending .state { color: var(--muted); }
  .method.active .state { color: var(--amber); }
  .method.complete .state { color: var(--green); }

  .imgrow { display: flex; align-items: baseline; justify-content: space-between; margin-top: .3rem; }
  .imgrow .val { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: .95rem; font-weight: 600; }
  .imgrow .cap { font-family: var(--mono); font-size: .72rem; color: var(--muted); }
  .mini-track { height: 5px; border-radius: 999px; background: var(--track); overflow: hidden; margin-top: .5rem; }
  .mini-fill { height: 100%; background: var(--amber); width: 0%; transition: width .4s ease; }

  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; }
  .stat { background: var(--ground2); border: 1px solid var(--line); border-radius: 12px; padding: .8rem .85rem; }
  .stat .val { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.15rem; font-weight: 700; }
  .stat .lbl { font-family: var(--mono); font-size: .66rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-top: .3rem; }

  footer { text-align: center; font-family: var(--mono); font-size: .72rem; color: var(--muted); margin-top: 2rem; }
  .row-gap { display: flex; flex-direction: column; gap: 1.1rem; }
</style>
</head>
<body>
<div class="page">
  <p class="eyebrow">XAI-FedGB / compute_overlap_all_methods.py</p>
  <h1>Cosine-Overlap Computation</h1>
  <p class="dek">
    Tracking local-vs-global heatmap agreement O<sub>k</sub><sup>c</sup> across
    Grad-CAM, Grad-CAM++, Saliency Maps, Eigen-CAM, and Score-CAM &mdash;
    5 disease classes &times; 3 hospital clients &times; 689 validation
    images, evaluated against the round-10 FedProx global model on GPU
    (CUDA). This page polls the server every 2 seconds.
  </p>

  <div class="statusbar">
    <span class="pill" id="statusPill"><span class="dot"></span><span id="statusText">loading&hellip;</span></span>
    <span class="refresh-hint" id="lastUpdated">&mdash;</span>
  </div>

  <div class="row-gap">
    <div class="card">
      <div class="timer-row">
        <div class="timer-block pct">
          <div class="num" id="pctNum">&mdash;</div>
          <div class="lbl">Overall progress</div>
        </div>
        <div class="timer-block" style="text-align:right">
          <div class="num" id="etaNum">&mdash;</div>
          <div class="lbl">Time remaining</div>
        </div>
      </div>
      <div class="bar-track"><div class="bar-fill" id="barFill"></div></div>
      <div class="bar-caption">
        <span id="stepCaption">step 0 / 0</span>
        <span id="elapsedCaption">elapsed &mdash;</span>
      </div>
    </div>

    <div>
      <div class="section-label">Method sequence</div>
      <div class="methods" id="methodsRow">
        <div class="method pending" data-method="gradcam">
          <div class="name">Grad-CAM</div>
          <div class="formula">ReLU(&Sigma; &alpha;<sub>k</sub><sup>c</sup> A<sup>k</sup>)</div>
          <div class="state"><span class="check"></span><span class="txt">pending</span></div>
        </div>
        <div class="method pending" data-method="gradcam_pp">
          <div class="name">Grad-CAM++</div>
          <div class="formula">2nd-order weighted</div>
          <div class="state"><span class="check"></span><span class="txt">pending</span></div>
        </div>
        <div class="method pending" data-method="saliency">
          <div class="name">Saliency Map</div>
          <div class="formula">|&part;y<sup>c</sup>/&part;x|</div>
          <div class="state"><span class="check"></span><span class="txt">pending</span></div>
        </div>
        <div class="method pending" data-method="eigencam">
          <div class="name">Eigen-CAM</div>
          <div class="formula">1st principal component of A<sup>k</sup></div>
          <div class="state"><span class="check"></span><span class="txt">pending</span></div>
        </div>
        <div class="method pending" data-method="scorecam">
          <div class="name">Score-CAM</div>
          <div class="formula">&Sigma; softmax(f(x&odot;M<sub>k</sub>)) A<sup>k</sup></div>
          <div class="state"><span class="check"></span><span class="txt">pending</span></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="section-label" style="margin-top:0">Current method &mdash; image throughput</div>
      <div class="imgrow">
        <span class="val" id="imgVal">0 / 0</span>
        <span class="cap" id="imgMethodLabel">&mdash;</span>
      </div>
      <div class="mini-track"><div class="mini-fill" id="miniFill"></div></div>
    </div>

    <div class="stats">
      <div class="stat"><div class="val" id="statElapsed">&mdash;</div><div class="lbl">Elapsed</div></div>
      <div class="stat"><div class="val" id="statMethodsDone">0 / 5</div><div class="lbl">Methods done</div></div>
      <div class="stat"><div class="val" id="statSteps">0 / 0</div><div class="lbl">Total steps</div></div>
    </div>
  </div>

  <footer>XAI-FedGB &middot; local computation monitor &middot; auto-refreshing every 2s</footer>
</div>

<script>
const METHOD_LABELS = { gradcam: "Grad-CAM", gradcam_pp: "Grad-CAM++", saliency: "Saliency Map", eigencam: "Eigen-CAM", scorecam: "Score-CAM" };
const METHOD_ORDER = ["gradcam", "gradcam_pp", "saliency", "eigencam", "scorecam"];

function fmtSecs(s) {
  if (s === null || s === undefined || isNaN(s)) return "\\u2014";
  s = Math.round(s);
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60), r = s % 60;
  if (m < 60) return m + "m " + r + "s";
  const h = Math.floor(m / 60), rm = m % 60;
  return h + "h " + rm + "m";
}

function render(data) {
  const pill = document.getElementById("statusPill");
  const statusText = document.getElementById("statusText");
  pill.classList.remove("running", "done");
  if (data.status === "done") {
    pill.classList.add("done");
    statusText.textContent = "Complete";
  } else if (data.status === "running" || data.status === "starting") {
    pill.classList.add("running");
    statusText.textContent = data.status === "starting" ? "Starting\\u2026"
      : "Running \\u2014 " + (data.current_method_label || METHOD_LABELS[data.current_method] || "");
  } else {
    statusText.textContent = "Not started";
  }

  document.getElementById("lastUpdated").textContent = data.updated_at ? "as of " + data.updated_at : "\\u2014";

  const pct = data.percent ?? 0;
  document.getElementById("pctNum").textContent = pct + "%";
  document.getElementById("barFill").style.width = pct + "%";
  document.getElementById("etaNum").innerHTML = data.status === "done" ? "done" : fmtSecs(data.eta_sec);
  document.getElementById("stepCaption").textContent = "step " + (data.overall_step ?? 0) + " / " + (data.overall_total ?? 0);
  document.getElementById("elapsedCaption").textContent = "elapsed " + fmtSecs(data.elapsed_sec);

  document.getElementById("statElapsed").textContent = fmtSecs(data.elapsed_sec);
  document.getElementById("statMethodsDone").textContent = (data.methods_done ?? 0) + " / " + (data.methods_total ?? 5);
  document.getElementById("statSteps").textContent = (data.overall_step ?? 0) + " / " + (data.overall_total ?? 0);

  document.getElementById("imgVal").textContent = (data.image_index ?? 0) + " / " + (data.images_total ?? 0);
  document.getElementById("imgMethodLabel").textContent = data.current_method_label || METHOD_LABELS[data.current_method] || "\\u2014";
  const imgPct = data.images_total ? (100 * (data.image_index ?? 0) / data.images_total) : 0;
  document.getElementById("miniFill").style.width = imgPct + "%";

  document.querySelectorAll(".method").forEach((el) => {
    const m = el.dataset.method;
    const idx = METHOD_ORDER.indexOf(m);
    el.classList.remove("pending", "active", "complete");
    const txt = el.querySelector(".state .txt");
    const check = el.querySelector(".state .check");
    if ((data.methods_done ?? 0) > idx) {
      el.classList.add("complete"); txt.textContent = "complete"; check.textContent = "\\u2713";
    } else if (data.current_method === m) {
      el.classList.add("active"); txt.textContent = "running\\u2026"; check.textContent = "\\u25cf";
    } else {
      el.classList.add("pending"); txt.textContent = "pending"; check.textContent = "";
    }
  });
}

async function poll() {
  try {
    const res = await fetch("/status", {cache: "no-store"});
    const data = await res.json();
    render(data);
  } catch (e) {
    document.getElementById("statusText").textContent = "waiting for server\\u2026";
  }
}

poll();
setInterval(poll, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep console quiet

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            if os.path.exists(PROGRESS_PATH):
                with open(PROGRESS_PATH, "r") as f:
                    payload = f.read()
            else:
                payload = json.dumps({"status": "not_started"})
            body = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"[overlap-progress] Serving on http://localhost:{PORT}")
    print(f"[overlap-progress] Watching {PROGRESS_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[overlap-progress] Stopped.")


if __name__ == "__main__":
    main()
