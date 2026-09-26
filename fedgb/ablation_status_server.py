"""
ablation_status_server.py
---------------------------
Tiny local HTTP server that serves a live status dashboard for the
stage2_ablation_fixes.py training run. Parses the run.log file on every
request (no polling loop of its own) and returns JSON at /status plus a
static dashboard HTML page at /.

Run from fedgb/:  python ablation_status_server.py
Then open http://localhost:8811/ in a browser.
"""
import http.server
import json
import os
import re
import socketserver
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "outputs", "session_corrected", "ablation", "run.log")
SUMMARY_PATH = os.path.join(HERE, "outputs", "session_corrected", "ablation", "summary.json")
METRICS_DIR = os.path.join(HERE, "outputs", "session_corrected", "ablation")
PORT = 8811

BASELINE = {"accuracy": 0.3100, "f1_macro": 0.2944, "auc_macro": 0.7351}

EPOCH_RE = re.compile(
    r"\[(?P<tag>[\w_]+)\] epoch (?P<epoch>\d+)/(?P<total>\d+) loss=(?P<loss>[\d.]+) \((?P<elapsed>\d+)s elapsed\)"
)
RESULT_RE = re.compile(
    r"\[(?P<tag>[\w_]+)\] accuracy=(?P<accuracy>[\d.]+) f1=(?P<f1>[\d.]+) auc=(?P<auc>[\d.]+)"
)
EXPERIMENT_START_RE = re.compile(r"\[ablation\] === (?P<tag>[\w_]+) ===")


def parse_log():
    if not os.path.exists(LOG_PATH):
        return {"status": "not_started", "experiments": {}}

    with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    experiments = {}
    order = []
    current_tag = None

    for line in lines:
        m = EXPERIMENT_START_RE.search(line)
        if m:
            current_tag = m.group("tag")
            if current_tag not in experiments:
                order.append(current_tag)
                experiments[current_tag] = {"epochs": [], "result": None, "done": False}
            continue

        m = EPOCH_RE.search(line)
        if m and m.group("tag") == current_tag:
            experiments[current_tag]["epochs"].append({
                "epoch": int(m.group("epoch")),
                "total": int(m.group("total")),
                "loss": float(m.group("loss")),
                "elapsed_s": int(m.group("elapsed")),
            })
            continue

        m = RESULT_RE.search(line)
        if m and m.group("tag") == current_tag:
            acc = float(m.group("accuracy"))
            experiments[current_tag]["result"] = {
                "accuracy": acc,
                "f1_macro": float(m.group("f1")),
                "auc_macro": float(m.group("auc")),
                "delta_accuracy": acc - BASELINE["accuracy"],
            }
            experiments[current_tag]["done"] = True

    overall_done = os.path.exists(SUMMARY_PATH)
    status = "complete" if overall_done else ("running" if order else "starting")

    return {
        "status": status,
        "baseline": BASELINE,
        "order": order,
        "experiments": experiments,
        "raw_tail": "".join(lines[-15:]),
        "gpu": read_gpu_stats(),
    }


def read_gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        util, mem_used, mem_total = [x.strip() for x in out.stdout.strip().split(",")]
        return {
            "available": True,
            "utilization_pct": int(util),
            "memory_used_mib": int(mem_used),
            "memory_total_mib": int(mem_total),
            "memory_pct": round(int(mem_used) / int(mem_total) * 100, 1),
        }
    except Exception:
        return {"available": False}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep console quiet

    def do_GET(self):
        if self.path == "/status":
            data = json.dumps(parse_log()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/" or self.path == "/index.html":
            html_path = os.path.join(HERE, "ablation_status.html")
            with open(html_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"[status-server] serving on http://localhost:{PORT}/")
        httpd.serve_forever()
