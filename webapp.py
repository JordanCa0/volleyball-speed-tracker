"""Web UI: upload a clip, get ball speeds back.

Analysis runs at roughly 1.4 fps on a laptop, so a 60-second clip takes
~20 minutes. That is far too long to hold an HTTP request open, so uploads are
queued to a background thread and the page polls for progress.

Deliberately single-process and in-memory: this is a local tool for one person
analysing their own footage, not a service. Jobs die with the process.

    .venv/bin/python webapp.py
    open http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from flask import (Flask, Response, jsonify, redirect, render_template_string,
                   request, send_file, url_for)
from werkzeug.utils import secure_filename

from analysis import analyse_video, hits_to_csv
from calibration import load_from_config

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
ALLOWED = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024      # 2 GB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@dataclass
class Job:
    id: str
    filename: str
    path: Path
    status: str = "queued"          # queued | running | done | failed
    frames_done: int = 0
    frames_total: int | None = None
    error: str | None = None
    result: object = None
    created: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    max_frames: int | None = None

    def as_dict(self) -> dict:
        data = {
            "id": self.id, "filename": self.filename, "status": self.status,
            "frames_done": self.frames_done, "frames_total": self.frames_total,
            "error": self.error, "created": self.created,
        }
        if self.result is not None:
            hits = [
                {
                    "index": i,
                    "start": round(h.start_time, 2),
                    "kmh": round(h.peak_speed_kmh, 1),
                    "mph": round(h.peak_speed_kmh * 0.621371, 1),
                    "points": h.n_points,
                    "depth": round(h.mean_depth_m, 1),
                    "reliable": h.reliable,
                    "truncated": h.truncated,
                }
                for i, h in enumerate(self.result.hits, 1)
            ]
            fastest = self.result.fastest
            data["result"] = {
                "frames": self.result.frames_processed,
                "with_ball": self.result.frames_with_ball,
                "fps": round(self.result.fps, 1),
                "elapsed": round(self.result.elapsed_s),
                "scale_source": self.result.intrinsics_source,
                "hits": hits,
                "fastest_kmh": round(fastest.peak_speed_kmh, 1) if fastest else None,
                "fastest_mph": round(fastest.peak_speed_kmh * 0.621371, 1)
                if fastest else None,
                "annotated": bool(self.result.annotated_path),
            }
        return data


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def load_intrinsics():
    config_path = BASE / "config.json"
    if not config_path.exists():
        return None
    try:
        return load_from_config(json.loads(config_path.read_text()))
    except (json.JSONDecodeError, KeyError):
        return None


def run_job(job: Job) -> None:
    job.status = "running"
    annotated = OUTPUTS / f"{job.id}.mp4"
    try:
        def progress(done, total):
            job.frames_done, job.frames_total = done, total

        job.result = analyse_video(
            str(job.path),
            intrinsics=load_intrinsics(),
            annotate_path=str(annotated),
            max_frames=job.max_frames,
            progress=progress,
        )
        job.status = "done"
    except Exception as exc:                      # surfaced in the UI
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"


@app.post("/upload")
def upload():
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        return redirect(url_for("index"))

    name = secure_filename(uploaded.filename)
    if Path(name).suffix.lower() not in ALLOWED:
        return render_template_string(
            PAGE, jobs=sorted_jobs(),
            error=f"{Path(name).suffix or 'that file type'} isn't a video "
                  f"({', '.join(sorted(ALLOWED))})"), 400

    UPLOADS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{job_id}_{name}"
    uploaded.save(path)

    limit = request.form.get("max_frames", "").strip()
    job = Job(id=job_id, filename=name, path=path,
              max_frames=int(limit) if limit.isdigit() and int(limit) > 0 else None)
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return redirect(url_for("index"))


@app.get("/jobs")
def jobs_json():
    with JOBS_LOCK:
        return jsonify([job.as_dict() for job in sorted_jobs()])


@app.get("/download/<job_id>.csv")
def download_csv(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job.result is None:
        return "not ready", 404
    return Response(
        hits_to_csv(job.result.hits), mimetype="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="{Path(job.filename).stem}_speeds.csv"'})


@app.get("/video/<job_id>.mp4")
def annotated_video(job_id: str):
    path = OUTPUTS / f"{job_id}.mp4"
    if not path.exists():
        return "not ready", 404
    return send_file(path, mimetype="video/mp4")


def sorted_jobs() -> list[Job]:
    return sorted(JOBS.values(), key=lambda j: j.created, reverse=True)


@app.get("/")
def index():
    return render_template_string(PAGE, jobs=sorted_jobs(), error=None)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volleyball Speed Tracker</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #ffffff; --ink: #14171a; --muted: #5c6672;
    --line: #e3e7ec; --accent: #1f6feb; --good: #1a7f37; --warn: #9a6700;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0d1117; --card: #161b22; --ink: #e6edf3; --muted: #8b949e;
      --line: #30363d; --accent: #4493f8; --good: #3fb950; --warn: #d29922;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem; background: var(--bg); color: var(--ink);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
  .sub { color: var(--muted); margin: 0 0 1.5rem; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 1.25rem; margin-bottom: 1rem;
  }
  label { display: block; font-weight: 600; margin-bottom: .4rem; }
  input[type=file] {
    width: 100%; padding: .8rem; border: 1px dashed var(--line);
    border-radius: 8px; background: transparent; color: var(--ink);
  }
  input[type=number] {
    padding: .5rem; border: 1px solid var(--line); border-radius: 6px;
    background: transparent; color: var(--ink); width: 8rem;
  }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 7px;
    padding: .65rem 1.2rem; font-size: 1rem; font-weight: 600; cursor: pointer;
  }
  button:hover { filter: brightness(1.08); }
  .row { display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap; }
  .hint { color: var(--muted); font-size: .85rem; margin-top: .5rem; }
  table { width: 100%; border-collapse: collapse; margin-top: .75rem; }
  th, td { text-align: left; padding: .45rem .5rem; border-bottom: 1px solid var(--line); }
  th { font-size: .78rem; text-transform: uppercase; color: var(--muted); letter-spacing: .04em; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .big { font-size: 2rem; font-weight: 700; }
  .pill {
    display: inline-block; padding: .1rem .5rem; border-radius: 99px;
    font-size: .75rem; font-weight: 600;
  }
  .pill.ok { background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }
  .pill.warn { background: color-mix(in srgb, var(--warn) 20%, transparent); color: var(--warn); }
  .pill.run { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent); }
  .bar { height: 6px; background: var(--line); border-radius: 99px; overflow: hidden; margin-top: .5rem; }
  .bar > div { height: 100%; background: var(--accent); width: 0; transition: width .4s; }
  .err { color: #cf222e; }
  a { color: var(--accent); }
  .muted { color: var(--muted); }
  .tablewrap { overflow-x: auto; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Volleyball Speed Tracker</h1>
  <p class="sub">Upload a clip filmed from behind the server. Depth comes from
  the ball's apparent size, so speeds work even when the ball flies away from
  the camera.</p>

  {% if error %}<div class="card err">{{ error }}</div>{% endif %}

  <form class="card" method="post" action="/upload" enctype="multipart/form-data">
    <label for="video">Video file</label>
    <input id="video" type="file" name="video" accept="video/*" required>
    <div class="row" style="margin-top:1rem">
      <div>
        <label for="max_frames">Limit frames <span class="muted">(optional)</span></label>
        <input id="max_frames" type="number" name="max_frames" min="1" placeholder="all">
      </div>
      <button type="submit">Analyse</button>
    </div>
    <p class="hint">Analysis runs about 1.4 frames per second, so a 60-second
    30fps clip takes roughly 20 minutes. Limit the frames for a quick look.</p>
  </form>

  <div id="jobs"></div>
</div>

<script>
function pill(job) {
  if (job.status === 'done') return '<span class="pill ok">done</span>';
  if (job.status === 'failed') return '<span class="pill warn">failed</span>';
  if (job.status === 'running') return '<span class="pill run">running</span>';
  return '<span class="pill run">queued</span>';
}

function renderJob(job) {
  let html = '<div class="card"><div class="row" style="justify-content:space-between">';
  html += '<div><strong>' + job.filename + '</strong> ' + pill(job);
  html += '<div class="muted" style="font-size:.85rem">' + job.created + '</div></div>';
  if (job.status === 'done' && job.result) {
    html += '<div style="text-align:right"><div class="big">' +
      (job.result.fastest_kmh !== null ? job.result.fastest_kmh + ' <span style="font-size:1rem">km/h</span>' : '—') +
      '</div><div class="muted">' +
      (job.result.fastest_mph !== null ? job.result.fastest_mph + ' mph fastest' : 'no flights found') +
      '</div></div>';
  }
  html += '</div>';

  if (job.status === 'running' || job.status === 'queued') {
    const pct = job.frames_total ? Math.round(100 * job.frames_done / job.frames_total) : 0;
    html += '<div class="bar"><div style="width:' + pct + '%"></div></div>';
    html += '<div class="hint">' + job.frames_done + ' / ' + (job.frames_total || '?') + ' frames</div>';
  }
  if (job.status === 'failed') {
    html += '<div class="err">' + job.error + '</div>';
  }
  if (job.status === 'done' && job.result) {
    const r = job.result;
    html += '<div class="hint">' + r.frames + ' frames, ball located in ' +
      r.with_ball + ' (' + Math.round(100 * r.with_ball / r.frames) + '%), ' +
      r.elapsed + 's &middot; scale: ' + r.scale_source + '</div>';
    if (r.hits.length) {
      html += '<div class="tablewrap"><table><tr><th>#</th><th>Start</th>' +
        '<th class="num">km/h</th><th class="num">mph</th>' +
        '<th class="num">Points</th><th class="num">Depth</th><th>Quality</th></tr>';
      for (const h of r.hits) {
        const q = h.reliable ? '<span class="pill ok">reliable</span>'
          : (h.truncated ? '<span class="pill warn">truncated</span>'
                         : '<span class="pill warn">short</span>');
        html += '<tr><td>' + h.index + '</td><td>' + h.start + 's</td>' +
          '<td class="num">' + h.kmh + '</td><td class="num">' + h.mph + '</td>' +
          '<td class="num">' + h.points + '</td><td class="num">' + h.depth + ' m</td>' +
          '<td>' + q + '</td></tr>';
      }
      html += '</table></div>';
      html += '<p style="margin-top:.9rem"><a href="/download/' + job.id + '.csv">Download CSV</a>';
      if (r.annotated) html += ' &middot; <a href="/video/' + job.id + '.mp4">Annotated video</a>';
      html += '</p>';
    } else {
      html += '<p class="hint">No flight segments found. The ball may not have been ' +
        'detected, or nothing moved fast enough to count as a hit.</p>';
    }
  }
  return html + '</div>';
}

async function refresh() {
  try {
    const res = await fetch('/jobs');
    const jobs = await res.json();
    document.getElementById('jobs').innerHTML = jobs.map(renderJob).join('');
  } catch (e) { /* keep polling */ }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    UPLOADS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    print("Volleyball Speed Tracker — http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
