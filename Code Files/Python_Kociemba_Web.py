import cv2
import numpy as np
import kociemba
import threading
import time
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

FACE_LABELS = {
    "U": "White / Up",
    "R": "Red / Right",
    "F": "Green / Front",
    "D": "Yellow / Down",
    "L": "Orange / Left",
    "B": "Blue / Back",
}

# CSS colours for each face letter
FACE_CSS = {
    "U": "#ffffff",
    "R": "#c41e3a",
    "F": "#009b48",
    "D": "#ffd500",
    "L": "#ff5800",
    "B": "#0046ad",
}

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "refs": {},                          # face → (h, s, v)  calibration HSV
    "faces": {f: None for f in "URFDLB"},
    "calibrating": True,
    "calib_face": "U",                   # which face is currently being calibrated
    "current_hsv": None,                 # live HSV reading from calibration box
    "solution": None,
    "error": None,
}
state_lock = threading.Lock()

# Latest frames
current_jpeg = None   # bytes – for MJPEG streaming
current_raw = None    # ndarray – for face sampling
frame_lock = threading.Lock()


# ── Colour helpers ────────────────────────────────────────────────────────────
def hue_dist(h1, h2):
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


def color_dist(hsv1, ref):
    h1, s1, v1 = hsv1
    h2, s2, v2 = ref
    dh = hue_dist(h1, h2) * 2
    return dh * dh + (int(s1) - int(s2)) ** 2 + (int(v1) - int(v2)) ** 2


def classify_color(hsv, refs):
    h, s, v = hsv
    if v < 100:
        return "R"
    if s < refs["U"][1] + 30 and v > refs["U"][2] - 40:
        return "U"
    best, best_d = "U", float("inf")
    for face in ("R", "F", "D", "L", "B"):
        d = color_dist(hsv, refs[face])
        if d < best_d:
            best_d, best = d, face
    return best


# ── Frame drawing helpers ─────────────────────────────────────────────────────
def draw_calib_overlay(frame, face):
    """Draw calibration box and return the median HSV of its inner region."""
    fh, fw = frame.shape[:2]
    cx, cy, sz = fw // 2, fh // 2, 60
    x1, y1, x2, y2 = cx - sz, cy - sz, cx + sz, cy + sz
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(frame, f"Calibrate: {FACE_LABELS[face]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, "Center the sticker in the box, then click Capture",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    hsv_f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv_f[y1:y2, x1:x2]
    pad = sz // 2
    inner = roi[pad:-pad, pad:-pad]
    ah = int(np.median(inner[:, :, 0]))
    as_ = int(np.median(inner[:, :, 1]))
    av = int(np.median(inner[:, :, 2]))
    cv2.putText(frame, f"HSV: {ah},{as_},{av}", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
    return (ah, as_, av)


def sample_face(frame, refs, grid_size=None, spacing=None):
    """Draw 3×3 grid overlay and return the 9 detected colour labels."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * 0.85)
    if grid_size is None:
        grid_size = side // 3 - 6
    if spacing is None:
        spacing = 6
    total = 3 * grid_size + 2 * spacing
    sx = w // 2 - total // 2
    sy = h // 2 - total // 2
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    labels = []
    for row in range(3):
        for col in range(3):
            x = sx + col * (grid_size + spacing)
            y = sy + row * (grid_size + spacing)
            cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), (0, 255, 0), 2)
            roi = hsv_frame[y:y + grid_size, x:x + grid_size]
            pad = grid_size // 4
            inner = roi[pad:-pad, pad:-pad]
            avg = (int(np.median(inner[:, :, 0])),
                   int(np.median(inner[:, :, 1])),
                   int(np.median(inner[:, :, 2])))
            label = classify_color(avg, refs) if refs else "?"
            labels.append(label)
            cv2.putText(frame, label, (x + 5, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return labels


# ── Camera background thread ──────────────────────────────────────────────────
def camera_loop():
    global current_jpeg, current_raw
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        with state_lock:
            calibrating = state["calibrating"]
            calib_face = state["calib_face"]
            refs = dict(state["refs"])

        if calibrating and calib_face:
            hsv = draw_calib_overlay(frame, calib_face)
            with state_lock:
                state["current_hsv"] = hsv
        elif refs:
            sample_face(frame, refs)

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ret:
            with frame_lock:
                current_jpeg = jpeg.tobytes()
                current_raw = frame.copy()

        time.sleep(0.033)  # ~30 fps

    cap.release()


def gen_frames():
    while True:
        with frame_lock:
            frame = current_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with state_lock:
        refs_keys = list(state["refs"].keys())
        faces = {k: v for k, v in state["faces"].items()}
        calibrating = state["calibrating"]
        calib_face = state["calib_face"]
        solution = state["solution"]
        error = state["error"]
    return jsonify(refs=refs_keys, faces=faces, calibrating=calibrating,
                   calib_face=calib_face, solution=solution, error=error)


@app.route("/api/calibrate/<face>", methods=["POST"])
def api_calibrate(face):
    face = face.upper()
    if face not in "URFDLB":
        return jsonify(error="Invalid face"), 400
    with state_lock:
        hsv = state["current_hsv"]
        if hsv is None:
            return jsonify(error="No frame yet"), 500
        state["refs"][face] = hsv
        order = list("URFDLB")
        idx = order.index(face)
        if idx + 1 < len(order):
            state["calib_face"] = order[idx + 1]
            state["calibrating"] = True
        else:
            state["calibrating"] = False
            state["calib_face"] = None
    return jsonify(ok=True, face=face, hsv=hsv)


@app.route("/api/capture/<face>", methods=["POST"])
def api_capture(face):
    face = face.upper()
    if face not in "URFDLB":
        return jsonify(error="Invalid face"), 400
    with state_lock:
        refs = dict(state["refs"])
    if len(refs) < 6:
        return jsonify(error="Calibration incomplete — calibrate all 6 faces first"), 400
    with frame_lock:
        raw = current_raw.copy() if current_raw is not None else None
    if raw is None:
        return jsonify(error="No frame available"), 500
    labels = sample_face(raw, refs)
    labels[4] = face  # center is always the face's own colour
    with state_lock:
        state["faces"][face] = labels
    return jsonify(face=face, labels=labels)


def solution_to_arduino(solution_str):
    """Convert kociemba solution like 'R2 B R2 D' F' ...' to Arduino C++ code."""
    moves = solution_str.strip().split()
    lines = ['void solve() {']
    for m in moves:
        face = m[0]  # U, R, F, D, L, B
        if m.endswith("2"):
            func = f"move{face}2();"
            label = f'{face}2'
        elif m.endswith("'"):
            func = f"move{face}(false);"
            label = f"{face}'"
        else:
            func = f"move{face}();"
            label = face
        lines.append(f'  Serial.println("{label}");  {func:20s} delay(MOVE_PAUSE_MS);')
    lines.append('  Serial.println("=== Solve Complete! ===");')
    lines.append('}')
    return "\n".join(lines)


@app.route("/api/solve", methods=["POST"])
def api_solve():
    with state_lock:
        faces = {k: v for k, v in state["faces"].items()}
    if any(v is None for v in faces.values()):
        return jsonify(error="Not all 6 faces have been captured yet"), 400
    cube_str = "".join("".join(faces[f]) for f in "URFDLB")
    try:
        solution = kociemba.solve(cube_str)
        arduino_code = solution_to_arduino(solution)
        with state_lock:
            state["solution"] = solution
            state["error"] = None
        return jsonify(solution=solution, arduino=arduino_code, cube_str=cube_str)
    except Exception as exc:
        with state_lock:
            state["solution"] = None
            state["error"] = str(exc)
        return jsonify(error=str(exc)), 400


@app.route("/api/set_cell/<face>/<int:index>/<color>", methods=["POST"])
def api_set_cell(face, index, color):
    face = face.upper()
    color = color.upper()
    if face not in "URFDLB" or color not in "URFDLB":
        return jsonify(error="Invalid face or color"), 400
    if not (0 <= index <= 8):
        return jsonify(error="Invalid index"), 400
    with state_lock:
        if state["faces"][face] is None:
            state["faces"][face] = [face] * 9  # init with all same color
        state["faces"][face][index] = color
    return jsonify(ok=True, face=face, index=index, color=color)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        state["refs"] = {}
        state["faces"] = {f: None for f in "URFDLB"}
        state["calibrating"] = True
        state["calib_face"] = "U"
        state["solution"] = None
        state["error"] = None
    return jsonify(ok=True)


# ── HTML / JS frontend ────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rubik's Cube Solver</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #14142b;
  color: #e0e0e0;
  font-family: 'Segoe UI', system-ui, sans-serif;
  min-height: 100vh;
}
h1 {
  text-align: center;
  padding: 18px 0 10px;
  font-size: 1.7rem;
  color: #f0c040;
  letter-spacing: 2px;
}
.layout {
  display: flex;
  gap: 28px;
  padding: 0 24px 32px;
  flex-wrap: wrap;
  justify-content: center;
}

/* ── Left panel ── */
.left-panel { display: flex; flex-direction: column; gap: 14px; width: 480px; }
#video-feed {
  width: 100%;
  border-radius: 10px;
  border: 2px solid #333;
  background: #000;
}
.phase-banner {
  background: #1c1c3a;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 9px 14px;
  font-size: 0.93rem;
}
.phase-banner span { color: #f0c040; font-weight: bold; }

.section-title {
  font-size: 0.75rem;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
}
.btn-row { display: flex; flex-wrap: wrap; gap: 8px; }
button {
  padding: 7px 13px;
  border-radius: 6px;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.82rem;
  transition: opacity .15s, transform .1s;
}
button:hover:not(:disabled) { opacity: .85; transform: translateY(-1px); }
button:disabled { opacity: .35; cursor: not-allowed; }

.btn-calib   { background: #00b4d8; color: #000; }
.btn-calib.done { background: #2e7d32; color: #c8e6c9; }
.btn-capture { background: #388e3c; color: #fff; }
.btn-capture.done { background: #1b5e20; }
.btn-solve   { background: #f0c040; color: #000; font-size: .95rem; padding: 9px 22px; }
.btn-reset   { background: #424242; color: #eee; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.chip {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: .75rem;
  background: #2a2a4a;
  color: #888;
}
.chip.done    { background: #1b5e20; color: #a5d6a7; }
.chip.current { background: #0d47a1; color: #90caf9; }

.result-box {
  border-radius: 8px;
  padding: 12px 16px;
  font-size: .92rem;
}
.result-box.ok  { background: #0d3b1f; border: 1px solid #2e7d32; }
.result-box.err { background: #3b0d0d; border: 1px solid #b71c1c; color: #ff8a80; }
.result-box h3  { margin-bottom: 6px; color: #f0c040; }
.moves { font-family: monospace; font-size: 1.1rem; letter-spacing: 3px; word-break: break-all; }

/* ── Right panel: cube net ── */
.right-panel { display: flex; flex-direction: column; gap: 10px; padding-top: 4px; }
.cube-net {
  display: grid;
  /*  columns: gap  U  gap  gap  */
  grid-template-columns: repeat(4, auto);
  grid-template-rows: repeat(3, auto);
  grid-template-areas:
    ".  U  .  ."
    "L  F  R  B"
    ".  D  .  .";
  gap: 6px;
}
[data-area="U"] { grid-area: U; }
[data-area="L"] { grid-area: L; }
[data-area="F"] { grid-area: F; }
[data-area="R"] { grid-area: R; }
[data-area="B"] { grid-area: B; }
[data-area="D"] { grid-area: D; }

.face-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
.face-lbl { font-size: .7rem; color: #888; text-align: center; }
.face-grid {
  display: grid;
  grid-template-columns: repeat(3, 34px);
  grid-template-rows: repeat(3, 34px);
  gap: 2px;
  border-radius: 4px;
  padding: 2px;
  transition: outline .2s;
}
.face-grid.captured { outline: 2px solid #4caf50; }
.face-grid.active   { outline: 2px solid #00b4d8; }
.cell {
  width: 34px; height: 34px;
  border-radius: 4px;
  background: #2a2a4a;
  border: 1px solid #1a1a2e;
  transition: background .25s;
  cursor: pointer;
}
.cell:hover { opacity: .8; }
</style>
</head>
<body>
<h1>🟧 Rubik's Cube Solver</h1>
<div class="layout">

  <!-- ── Left panel ── -->
  <div class="left-panel">
    <img id="video-feed" src="/video_feed" alt="Camera">

    <div class="phase-banner">
      Phase: <span id="phase-text">Calibration</span>
    </div>

    <!-- Calibration -->
    <div>
      <div class="section-title">1 · Calibration — show each center sticker in the box</div>
      <div class="btn-row">
        <button class="btn-calib" id="calib-U" onclick="calibrate('U')">U · White</button>
        <button class="btn-calib" id="calib-R" onclick="calibrate('R')">R · Red</button>
        <button class="btn-calib" id="calib-F" onclick="calibrate('F')">F · Green</button>
        <button class="btn-calib" id="calib-D" onclick="calibrate('D')">D · Yellow</button>
        <button class="btn-calib" id="calib-L" onclick="calibrate('L')">L · Orange</button>
        <button class="btn-calib" id="calib-B" onclick="calibrate('B')">B · Blue</button>
      </div>
    </div>

    <!-- Capture -->
    <div>
      <div class="section-title">2 · Capture each face</div>
      <div class="btn-row">
        <button class="btn-capture" id="cap-U" onclick="capture('U')">U ↑</button>
        <button class="btn-capture" id="cap-R" onclick="capture('R')">R →</button>
        <button class="btn-capture" id="cap-F" onclick="capture('F')">F ●</button>
        <button class="btn-capture" id="cap-D" onclick="capture('D')">D ↓</button>
        <button class="btn-capture" id="cap-L" onclick="capture('L')">L ←</button>
        <button class="btn-capture" id="cap-B" onclick="capture('B')">B ●</button>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn-solve" onclick="solve()">🔍 Solve</button>
      <button class="btn-reset" onclick="resetAll()">↺ Reset</button>
    </div>

    <div id="result-area"></div>
  </div>

  <!-- ── Right panel: cube net ── -->
  <div class="right-panel">
    <div class="section-title">Cube faces</div>
    <div class="cube-net">
      <div class="face-wrap" data-area="U">
        <div class="face-lbl">U · White</div>
        <div class="face-grid" id="face-U"></div>
      </div>
      <div class="face-wrap" data-area="L">
        <div class="face-lbl">L · Orange</div>
        <div class="face-grid" id="face-L"></div>
      </div>
      <div class="face-wrap" data-area="F">
        <div class="face-lbl">F · Green</div>
        <div class="face-grid" id="face-F"></div>
      </div>
      <div class="face-wrap" data-area="R">
        <div class="face-lbl">R · Red</div>
        <div class="face-grid" id="face-R"></div>
      </div>
      <div class="face-wrap" data-area="B">
        <div class="face-lbl">B · Blue</div>
        <div class="face-grid" id="face-B"></div>
      </div>
      <div class="face-wrap" data-area="D">
        <div class="face-lbl">D · Yellow</div>
        <div class="face-grid" id="face-D"></div>
      </div>
    </div>
  </div>

</div>
<script>
const FACE_CSS = {
  U: '#ffffff', R: '#c41e3a', F: '#009b48',
  D: '#ffd500', L: '#ff5800', B: '#0046ad'
};
const COLOR_ORDER = ['U','R','F','D','L','B'];
const faceLabels = {};  // face -> [9 labels]

// Build empty 9-cell grids with click handlers
['U','R','F','D','L','B'].forEach(f => {
  const g = document.getElementById('face-' + f);
  faceLabels[f] = null;
  let html = '';
  for (let i = 0; i < 9; i++) {
    html += `<div class="cell" data-face="${f}" data-idx="${i}"></div>`;
  }
  g.innerHTML = html;
  g.querySelectorAll('.cell').forEach(cell => {
    cell.addEventListener('click', () => cycleCell(cell));
  });
});

function cycleCell(cell) {
  const face = cell.dataset.face;
  const idx = parseInt(cell.dataset.idx);
  if (!faceLabels[face]) return;  // face not captured yet
  const cur = faceLabels[face][idx];
  const next = COLOR_ORDER[(COLOR_ORDER.indexOf(cur) + 1) % COLOR_ORDER.length];
  faceLabels[face][idx] = next;
  cell.style.background = FACE_CSS[next] || '#333';
  cell.title = next;
  fetch(`/api/set_cell/${face}/${idx}/${next}`, { method: 'POST' });
}

function renderFace(face, labels) {
  faceLabels[face] = [...labels];
  const g = document.getElementById('face-' + face);
  const cells = g.querySelectorAll('.cell');
  labels.forEach((lbl, i) => {
    cells[i].style.background = FACE_CSS[lbl] || '#333';
    cells[i].title = lbl;
  });
  g.classList.add('captured');
  g.classList.remove('active');
}

function clearFace(face) {
  faceLabels[face] = null;
  const g = document.getElementById('face-' + face);
  g.querySelectorAll('.cell').forEach(c => { c.style.background = ''; c.title = ''; });
  g.classList.remove('captured', 'active');
}

async function calibrate(face) {
  const res = await fetch('/api/calibrate/' + face, { method: 'POST' });
  const data = await res.json();
  if (data.ok) {
    const btn = document.getElementById('calib-' + face);
    btn.classList.add('done');
    btn.textContent = btn.textContent + ' ✓';
  } else {
    showResult(data.error, 'err');
  }
  await pollState();
}

async function capture(face) {
  const res = await fetch('/api/capture/' + face, { method: 'POST' });
  const data = await res.json();
  if (data.labels) {
    renderFace(face, data.labels);
    document.getElementById('cap-' + face).classList.add('done');
  } else {
    showResult(data.error || 'Capture failed', 'err');
  }
}

async function solve() {
  const res = await fetch('/api/solve', { method: 'POST' });
  const data = await res.json();
  if (data.solution) {
    const escaped = data.arduino.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    showResult(`<h3>✅ Solution</h3><div class="moves">${data.solution}</div><h3 style="margin-top:12px">Arduino Code</h3><pre style="background:#0d0d1a;padding:12px;border-radius:6px;overflow-x:auto;font-size:0.85rem;line-height:1.5">${escaped}</pre>`, 'ok');
  } else {
    showResult(data.error, 'err');
  }
}

async function resetAll() {
  await fetch('/api/reset', { method: 'POST' });
  ['U','R','F','D','L','B'].forEach(f => {
    clearFace(f);
    document.getElementById('cap-' + f).classList.remove('done');
    const cb = document.getElementById('calib-' + f);
    cb.classList.remove('done');
    cb.textContent = cb.textContent.replace(' ✓', '');
  });
  document.getElementById('result-area').innerHTML = '';
  document.getElementById('phase-text').textContent = 'Calibration';
}

function showResult(html, type) {
  document.getElementById('result-area').innerHTML =
    `<div class="result-box ${type}">${html}</div>`;
}

async function pollState() {
  try {
    const res = await fetch('/api/state');
    const s = await res.json();

    // Phase banner
    if (s.calibrating) {
      document.getElementById('phase-text').textContent =
        'Calibration — ' + (s.calib_face || '');
    } else {
      const n = Object.values(s.faces).filter(v => v !== null).length;
      document.getElementById('phase-text').textContent =
        'Scanning (' + n + ' / 6 faces captured)';
    }

    // Mark calibrated buttons
    (s.refs || []).forEach(f => {
      const btn = document.getElementById('calib-' + f);
      if (btn && !btn.classList.contains('done')) {
        btn.classList.add('done');
        if (!btn.textContent.includes('✓')) btn.textContent += ' ✓';
      }
    });

    // Update face grids
    for (const [f, labels] of Object.entries(s.faces)) {
      if (labels) {
        renderFace(f, labels);
        document.getElementById('cap-' + f).classList.add('done');
      }
    }
  } catch(_) {}
}

setInterval(pollState, 1000);
pollState();
</script>
</body>
</html>"""


if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("Server running → http://localhost:3000")
    app.run(host="0.0.0.0", port=3000, threaded=True)

