"use strict";

// ── Constants ────────────────────────────────────────────────────────────────
const FRAME_INTERVAL_MS = 100;   // 10 fps to backend
const RECONNECT_DELAY_MS = 3000;
const CAPTURE_W = 320;
const CAPTURE_H = 240;
const JPEG_QUALITY = 0.65;

const EMOTION_STATES = [
  "Neutral","Happy","Sad","Angry","Stressed","Tired","Surprised","Focused","Distracted"
];
const EMOTION_EMOJIS = {
  Neutral:"😐",Happy:"😊",Sad:"😢",Angry:"😠",
  Stressed:"😤",Tired:"😴",Surprised:"😲",Focused:"🎯",Distracted:"😵"
};

// ── Chart.js global defaults ─────────────────────────────────────────────────
Chart.defaults.color = "#9CA3AF";
Chart.defaults.borderColor = "#1E2A3A";
Chart.defaults.font.family = "'Segoe UI', system-ui, monospace";
Chart.defaults.font.size = 11;

// ── Application ──────────────────────────────────────────────────────────────
const App = (() => {
  let ws = null;
  let cameraActive = false;
  let frameTimer = null;
  let reconnectTimer = null;
  let reconnectAttempt = 0;
  let fpsAccum = [];
  let lastFrameTime = 0;
  let recRecords = [];
  let txCount = 0;
  let rxCount = 0;
  let pendingFrame = false;  // rate-gate: only one frame in-flight at a time

  // Chart instances
  let chartRppg = null;
  let chartTrend = null;
  let chartEmo = null;

  // ── DOM helpers ─────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);
  const setText = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  const setHtml = (id, v) => { const el = $(id); if (el) el.innerHTML = v; };
  const setWidth = (id, pct) => { const el = $(id); if (el) el.style.width = `${Math.max(0, Math.min(100, pct))}%`; };
  const setColor = (id, c) => { const el = $(id); if (el) el.style.color = c; };
  const setBg    = (id, c) => { const el = $(id); if (el) el.style.background = c; };

  function scoreColor(s) {
    return s < 35 ? "#00FF88" : s < 60 ? "#FFD700" : "#FF4B4B";
  }
  function bpmColor(b) {
    if (!b) return "#6B7280";
    return (b >= 50 && b <= 100) ? "#00FF88" : (b >= 40 && b <= 120) ? "#FFD700" : "#FF4B4B";
  }
  function attnColor(s) {
    return s >= 65 ? "#00D4FF" : s >= 44 ? "#00FF88" : s >= 26 ? "#FFD700" : "#FF4B4B";
  }
  function wbColor(s) {
    return s >= 72 ? "#00FF88" : s >= 52 ? "#FFD700" : "#FF4B4B";
  }

  // ── Charts init ──────────────────────────────────────────────────────────────
  function initCharts() {
    const dark = "#0B0F1A";
    const panel = "#141925";

    // rPPG Signal
    const ctxR = $("chart-rppg").getContext("2d");
    chartRppg = new Chart(ctxR, {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          data: [], borderColor: "#00D4FF", borderWidth: 1.5,
          fill: true, backgroundColor: "rgba(0,212,255,0.06)",
          pointRadius: 0, tension: 0.3,
        }]
      },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false, grid: { display: false } },
          y: { display: false, grid: { color: "#1E2A3A" } }
        }
      }
    });

    // Trend (BPM + Fatigue + Stress + Wellbeing)
    const ctxT = $("chart-trend").getContext("2d");
    chartTrend = new Chart(ctxT, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label: "BPM",      data: [], borderColor: "#00D4FF", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
          { label: "Fatigue%", data: [], borderColor: "#FF8C00", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
          { label: "Stress%",  data: [], borderColor: "#FF4B4B", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
          { label: "Wellbeing",data: [], borderColor: "#00FF88", borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false, borderDash: [4,2] },
        ]
      },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { boxWidth: 10, padding: 8 } } },
        scales: {
          x: { display: false, grid: { display: false } },
          y: { grid: { color: "#1E2A3A" }, ticks: { maxTicksLimit: 4 } }
        }
      }
    });

    // Emotion scores (top states)
    const ctxE = $("chart-emo").getContext("2d");
    const EMO_COLORS = {
      Neutral:"#9CA3AF",Happy:"#FFD700",Sad:"#60A5FA",Angry:"#FF4B4B",
      Stressed:"#FF8C00",Tired:"#A78BFA",Surprised:"#FB923C",
      Focused:"#00D4FF",Distracted:"#F472B6"
    };
    chartEmo = new Chart(ctxE, {
      type: "line",
      data: {
        labels: [],
        datasets: EMOTION_STATES.slice(0, 5).map(s => ({
          label: `${EMOTION_EMOJIS[s]} ${s}`,
          data: [],
          borderColor: EMO_COLORS[s],
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        }))
      },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { boxWidth: 10, padding: 6 } } },
        scales: {
          x: { display: false, grid: { display: false } },
          y: { min: 0, max: 100, grid: { color: "#1E2A3A" }, ticks: { maxTicksLimit: 4 } }
        }
      }
    });
  }

  // ── WebSocket ────────────────────────────────────────────────────────────────
  function connect() {
    const url = window.BACKEND_WS;
    if (!url || url.includes("YOUR-SERVICE-NAME")) {
      setBanner("warning", "⚠️ Backend URL not configured — edit frontend/config.js");
      return;
    }
    setBanner("connecting", "⚡ Connecting to backend…");
    ws = new WebSocket(url);

    ws.onopen = () => {
      reconnectAttempt = 0;
      setBanner("connected", "✅ Connected — backend ready");
      setTimeout(() => hideBanner(), 2000);
      $("conn-retry").style.display = "none";
      setWsBadge("OPEN");
    };

    ws.onclose = () => {
      ws = null;
      setWsBadge("CLOSED");
      setBanner("disconnected", `⚠️ Disconnected — reconnecting in ${RECONNECT_DELAY_MS / 1000}s…`);
      $("conn-retry").style.display = "";
      reconnectTimer = setTimeout(() => {
        reconnectAttempt++;
        connect();
      }, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      setWsBadge("ERR");
      setBanner("error", "❌ Connection error — check backend URL in config.js");
    };

    ws.onmessage = evt => {
      try {
        rxCount++;
        setText("badge-rx", `↓${rxCount}`);
        const data = JSON.parse(evt.data);
        handleMessage(data);
      } catch (e) { /* ignore malformed */ }
    };
  }

  function reconnect() {
    clearTimeout(reconnectTimer);
    if (ws) { try { ws.close(); } catch (_) {} ws = null; }
    connect();
  }

  function sendJson(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
      if (obj.frame) {
        txCount++;
        setText("badge-tx", `↑${txCount}`);
      }
    }
  }

  function setWsBadge(state) {
    const el = $("badge-ws");
    if (!el) return;
    const cls = { OPEN: "badge-green", CLOSED: "badge-red", ERR: "badge-red", "…": "badge-warn" };
    el.className = `badge ${cls[state] || "badge-warn"}`;
    el.textContent = `WS:${state}`;
  }

  // ── Camera ────────────────────────────────────────────────────────────────────
  async function startCamera() {
    const video = $("webcam");
    const overlay = $("cam-overlay");
    const annOverlay = $("ann-overlay");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      cameraActive = true;
      overlay.style.display = "none";
      annOverlay.style.display = "none";

      // Begin capture loop
      frameTimer = setInterval(captureAndSend, FRAME_INTERVAL_MS);
    } catch (err) {
      overlay.innerHTML = `<div class="cam-error">Camera denied: ${err.message}</div>`;
    }
  }

  function captureAndSend() {
    if (!cameraActive || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (pendingFrame) return;  // previous frame still processing — skip this tick

    const video = $("webcam");
    if (!video.videoWidth) return;

    const canvas = $("capture-canvas");
    canvas.width  = CAPTURE_W;
    canvas.height = CAPTURE_H;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);

    const b64 = canvas.toDataURL("image/jpeg", JPEG_QUALITY).split(",")[1];
    pendingFrame = true;
    sendJson({ frame: b64 });

    // Client-side FPS measurement
    const now = performance.now();
    if (lastFrameTime) {
      fpsAccum.push(1000 / (now - lastFrameTime));
      if (fpsAccum.length > 20) fpsAccum.shift();
    }
    lastFrameTime = now;
  }

  // ── Message handler ──────────────────────────────────────────────────────────
  function handleMessage(d) {
    pendingFrame = false;  // ungate — ready for next frame
    if (d.type === "rec_state") { updateRecordingBadge(d.recording); return; }
    if (d.type === "rec_cleared") { recRecords = []; renderRecordTable(); return; }
    if (d.type === "csv") { downloadCsvBlob(d.data); return; }
    if (d.type === "error") {
      console.warn("Backend error:", d.message);
      showAnnotatedError(d.message);
      return;
    }
    if (d.type !== "frame") return;

    updateHeader(d);
    updateVitals(d);
    updateEmotion(d);
    updateAI(d);
    updateFacialSignals(d);
    updateAnnotated(d.annotated);
    updateCharts(d);
    if (d.recording) {
      updateRecordingBadge(true);
      setText("rec-count", `${d.rec_count} snapshot${d.rec_count !== 1 ? "s" : ""}`);
    }
  }

  // ── Header badges ────────────────────────────────────────────────────────────
  function updateHeader(d) {
    const fps = fpsAccum.length ? (fpsAccum.reduce((a, b) => a + b) / fpsAccum.length).toFixed(1) : "--";
    const fpsColor = fpsAccum.length && fpsAccum[fpsAccum.length-1] >= 8 ? "#00FF88" : "#FFD700";

    $("badge-face").className = `badge ${d.face_ok ? "badge-green" : "badge-red"}`;
    $("badge-face").textContent = `● ${d.face_ok ? "FACE" : "NO FACE"}`;
    $("badge-ai").className = `badge ${d.deep_ok ? "badge-cyan" : "badge-off"}`;
    $("badge-ai").textContent = `● ${d.deep_ok ? "DEEP AI" : "AI OFFLINE"}`;
    $("badge-gpu").className = `badge ${d.gpu ? "badge-purple" : "badge-off"}`;
    $("badge-gpu").textContent = `● ${d.gpu ? "GPU" : "CPU"}`;
    $("badge-cal").className = `badge ${d.calibrating ? "badge-warn" : "badge-green"}`;
    $("badge-cal").textContent = `● ${d.calibrating ? "CAL…" : "READY"}`;
    $("badge-fps").textContent = `${fps} fps`;
    $("badge-fps").style.color = fpsColor;
    $("badge-tick").textContent = `#${d.tick.toLocaleString()}`;
    $("badge-live").className = `badge ${d.face_ok ? "badge-green" : "badge-off"}`;
    $("badge-live").textContent = `● ${d.face_ok ? "LIVE" : "NO FACE"}`;
  }

  // ── Vital cards ──────────────────────────────────────────────────────────────
  function updateVitals(d) {
    // Heart Rate
    if (d.bpm) {
      const c = bpmColor(d.bpm);
      setText("bpm-val", Math.round(d.bpm));
      setColor("bpm-val", c);
      setText("bpm-sub", `Quality: ${Math.round(d.bpm_conf * 100)}%`);
      const pct = Math.max(0, Math.min(100, (d.bpm - 40) / 100 * 100));
      setWidth("bpm-bar", pct);
      $("bpm-bar").style.background = c;
      $("card-hr").style.borderTopColor = "#FF5E7A";
    } else {
      setText("bpm-val", "--");
      setText("bpm-sub", `Calibrating… ${Math.round(d.bpm_fill * 100)}%`);
      setWidth("bpm-bar", d.bpm_fill * 100);
      $("bpm-bar").style.background = "#6B7280";
    }

    // Fatigue
    const fat = d.fatigue;
    const fc = scoreColor(fat.score);
    setText("fat-val", Math.round(fat.score));
    setColor("fat-val", fc);
    setText("fat-sub", `${fat.state} · EAR ${fat.ear} · PERCLOS ${fat.perclos}%`);
    setWidth("fat-bar", fat.score);
    $("fat-bar").style.background = fc;
    $("card-fat").style.borderTopColor = "#FF8C00";

    // Stress
    const str = d.stress;
    const sc = scoreColor(str.score);
    setText("str-val", Math.round(str.score));
    setColor("str-val", sc);
    setText("str-sub", `${str.state} · HRV ${str.hrv} · Motion ${str.move}`);
    setWidth("str-bar", str.score);
    $("str-bar").style.background = sc;
    $("card-str").style.borderTopColor = "#FF4B4B";

    // Breathing
    const br = d.breathing;
    if (br.rate && br.conf > 0.2) {
      const brc = (br.rate >= 12 && br.rate <= 20) ? "#00FF88" : "#FFD700";
      setText("br-val", Math.round(br.rate));
      setColor("br-val", brc);
      setText("br-sub", `Confidence: ${Math.round(br.conf * 100)}%`);
      setWidth("br-bar", Math.min(100, br.rate / 25 * 100));
      $("br-bar").style.background = brc;
    } else {
      setText("br-val", "--");
      setText("br-sub", "Estimating breathing pattern…");
    }
    $("card-br").style.borderTopColor = "#00FF88";

    // Risk banner
    const risk = d.risk;
    const rcls = ["risk-normal", "risk-warn", "risk-high"];
    const rtitles = ["✅ SYSTEM STATUS: NORMAL", "⚠️ STATUS: WARNING", "🚨 STATUS: HIGH RISK"];
    const rban = $("risk-banner");
    rban.className = `risk-banner ${rcls[risk.code] || "risk-normal"}`;
    setText("risk-title", rtitles[risk.code] || rtitles[0]);
    setText("risk-msg", risk.messages.length ? risk.messages.join(" · ") : "All vitals in healthy range");
  }

  // ── Emotion ──────────────────────────────────────────────────────────────────
  function updateEmotion(d) {
    const emo = d.emotion;
    const color = emo.color || "#9CA3AF";

    setText("emo-emoji", emo.emoji || "😐");
    setText("emo-state", emo.state);
    setColor("emo-state", color);

    if (emo.calibrating) {
      setText("emo-persist", "⏳ Calibrating…");
    } else {
      const p = emo.persistence;
      const mins = Math.floor(p / 60);
      const secs = Math.floor(p % 60);
      const t = mins ? `${mins}m ${secs}s` : `${secs}s`;
      setText("emo-persist", `⏱ ${t} stable · ${Math.round(emo.stability * 100)}% certainty`);
    }

    const tag = $("emo-tag");
    if (emo.calibrating) {
      tag.innerHTML = `<span class="tag-cal">CALIBRATING</span>`;
    } else if (emo.stable) {
      tag.innerHTML = `<span class="tag-stable" style="background:${color}22;color:${color}">STABLE</span>`;
    } else {
      tag.innerHTML = "";
    }

    const conf = Math.round(emo.confidence * 100);
    setWidth("emo-conf-bar", conf);
    setText("emo-conf-txt", `Confidence: ${conf}%`);
    setWidth("emo-stab-bar", Math.round(emo.stability * 100));

    setText("emo-dom-emoji", EMOTION_EMOJIS[emo.dominant] || "😐");
    setText("emo-dom-state", emo.dominant);
    $("emotion-main").style.borderTopColor = color;

    // Emotion bars
    const scores = emo.scores || {};
    const sorted = [...EMOTION_STATES].sort((a, b) => (scores[b] || 0) - (scores[a] || 0));
    const EMO_COLORS = {
      Neutral:"#9CA3AF",Happy:"#FFD700",Sad:"#60A5FA",Angry:"#FF4B4B",
      Stressed:"#FF8C00",Tired:"#A78BFA",Surprised:"#FB923C",
      Focused:"#00D4FF",Distracted:"#F472B6"
    };
    const rows = sorted.map(s => {
      const pct = Math.round((scores[s] || 0) * 100);
      const c = EMO_COLORS[s] || "#9CA3AF";
      const active = s === emo.state;
      return `<div class="emo-bar-row${active ? " emo-bar-active" : ""}">
        <div class="emo-bar-label">
          <span>${EMOTION_EMOJIS[s] || ""} ${s}</span>
          <span style="color:${active ? c : "#4B5563"}">${pct}%</span>
        </div>
        <div class="bar-track" style="height:5px">
          <div class="bar-fill" style="width:${pct}%;background:${c};opacity:${active?1:0.5}"></div>
        </div>
      </div>`;
    }).join("");
    setHtml("emo-bars-list", rows);
  }

  // ── AI Row ───────────────────────────────────────────────────────────────────
  function updateAI(d) {
    const att = d.attention;
    const ac = attnColor(att.score);
    setText("attn-val", Math.round(att.score));
    setColor("attn-val", ac);
    setText("attn-level", `${att.level} · Cog. Load ${Math.round(att.cog)}%`);
    setWidth("attn-bar", att.score);
    $("attn-bar").style.background = ac;
    setWidth("cog-bar", att.cog);
    setWidth("gaze-bar", att.gaze);
    setWidth("stab-bar", att.stab);

    const wb = d.wellbeing;
    const wc = wbColor(wb.score);
    setText("wb-val", Math.round(wb.score));
    setColor("wb-val", wc);
    setText("wb-label", `${wb.label} ${wb.icon} ${wb.trend}`);
    setWidth("wb-bar", wb.score);
    $("wb-bar").style.background = wc;

    // Insights
    const ins = (wb.insights || []).map(l =>
      `<div class="insight-item">▸ ${l}</div>`
    ).join("") || `<div class="insight-item muted">Collecting data…</div>`;
    setHtml("insights-list", ins);

    const recs = wb.recs || [];
    const recsEl = $("recs-section");
    if (recs.length) {
      recsEl.style.display = "";
      setHtml("recs-list", recs.map(r => `<div class="rec-item">→ ${r}</div>`).join(""));
    } else {
      recsEl.style.display = "none";
    }
  }

  // ── Facial Signals ───────────────────────────────────────────────────────────
  function updateFacialSignals(d) {
    const ff = d.face;
    if (!ff || !ff.valid) {
      setHtml("facial-grid", `<div class="muted">No face detected</div>`);
      setText("facial-pose", "");
      return;
    }

    const sigs = [
      ["Smile (AU12)", ff.smile],
      ["Frown (corner)", ff.frown],
      ["Brow Furrow (AU4)", ff.furrow],
      ["Inner Brow (AU1)", ff.ibrow],
      ["Eye Openness (EAR)", ff.ear * 3],
      ["Mouth Open (MAR)", ff.mouth],
      ["Head Movement", ff.energy],
    ];

    const cells = sigs.map(([label, raw]) => {
      const clamp = Math.max(0, Math.min(1, raw));
      const c = clamp < 0.35 ? "#00FF88" : clamp < 0.65 ? "#FFD700" : "#FF4B4B";
      return `<div class="facial-cell">
        <div class="facial-row">
          <span class="facial-lbl">${label}</span>
          <span style="color:${c};font-weight:bold">${raw.toFixed(3)}</span>
        </div>
        <div class="bar-track" style="height:4px;margin-top:2px">
          <div class="bar-fill" style="width:${clamp*100}%;background:${c}"></div>
        </div>
      </div>`;
    }).join("");

    setHtml("facial-grid", cells);
    setText("facial-pose",
      `Yaw ${ff.yaw.toFixed(1)}°  Pitch ${ff.pitch.toFixed(1)}°  Energy ${ff.energy.toFixed(2)}`
    );
  }

  // ── Annotated frame ──────────────────────────────────────────────────────────
  function updateAnnotated(b64) {
    if (!b64) return;
    const img = $("annotated");
    img.src = `data:image/jpeg;base64,${b64}`;
    img.style.display = "";
    // Ensure overlay is hidden once we have real frames
    const ov = $("ann-overlay");
    if (ov) ov.style.display = "none";
  }

  function showAnnotatedError(msg) {
    const ov = $("ann-overlay");
    if (!ov) return;
    ov.style.display = "";
    ov.innerHTML = `<div style="color:#FF4B4B;font-size:12px;padding:8px;text-align:center;max-width:300px">
      ⚠️ Backend error:<br><code style="font-size:10px;word-break:break-all">${msg}</code>
    </div>`;
  }

  // ── Charts ────────────────────────────────────────────────────────────────────
  function updateCharts(d) {
    // rPPG
    if (chartRppg && d.rppg_signal && d.rppg_signal.length > 5) {
      const sig = d.rppg_signal;
      const mn = Math.min(...sig), mx = Math.max(...sig);
      const rng = mx - mn || 1;
      const norm = sig.map(v => (v - mn) / rng * 2 - 1);
      chartRppg.data.labels = norm.map((_, i) => i);
      chartRppg.data.datasets[0].data = norm;
      chartRppg.update("none");
    }

    // Trend
    if (chartTrend) {
      const n = (d.bpm_hist || []).length;
      const labels = Array.from({length: n}, (_, i) => i);
      chartTrend.data.labels = labels;
      chartTrend.data.datasets[0].data = d.bpm_hist || [];
      chartTrend.data.datasets[1].data = d.fat_hist || [];
      chartTrend.data.datasets[2].data = d.str_hist || [];
      chartTrend.data.datasets[3].data = d.wb_hist  || [];
      chartTrend.update("none");
    }

    // Emotion scores
    if (chartEmo && d.score_hist_last) {
      const scores = d.score_hist_last;
      EMOTION_STATES.slice(0, 5).forEach((s, i) => {
        const ds = chartEmo.data.datasets[i];
        ds.data.push((scores[s] || 0) * 100);
        if (ds.data.length > 60) ds.data.shift();
      });
      const n = chartEmo.data.datasets[0].data.length;
      chartEmo.data.labels = Array.from({length: n}, (_, i) => i);
      chartEmo.update("none");
    }
  }

  // ── Recording ────────────────────────────────────────────────────────────────
  function startRecording() {
    sendJson({ cmd: "start_rec" });
    updateRecordingBadge(true);
    $("btn-rec-start").disabled = true;
    $("btn-rec-stop").disabled  = false;
    $("btn-rec-dl").disabled    = true;
  }

  function stopRecording() {
    sendJson({ cmd: "stop_rec" });
    updateRecordingBadge(false);
    $("btn-rec-start").disabled = false;
    $("btn-rec-stop").disabled  = true;
    $("btn-rec-dl").disabled    = false;
  }

  function clearRecording() {
    sendJson({ cmd: "clear_rec" });
    recRecords = [];
    renderRecordTable();
    $("btn-rec-start").disabled = false;
    $("btn-rec-stop").disabled  = true;
    $("btn-rec-dl").disabled    = true;
    updateRecordingBadge(false);
    setText("rec-count", "0 snapshots");
  }

  function downloadCSV() {
    sendJson({ cmd: "get_csv" });
  }

  function downloadCsvBlob(csvText) {
    const blob = new Blob([csvText], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `session_${new Date().toISOString().slice(0,19).replace(/:/g,"-")}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function updateRecordingBadge(active) {
    const badge = $("rec-badge");
    if (active) {
      badge.className = "rec-badge recording";
      badge.textContent = "● REC";
    } else {
      badge.className = "rec-badge stopped";
      badge.textContent = "⏹ STOPPED";
    }
  }

  function renderRecordTable() {
    if (!recRecords.length) {
      setHtml("rec-table-wrap", `<div class="muted">No data yet — press Start Recording to begin.</div>`);
      return;
    }
    const cols = Object.keys(recRecords[0]);
    const th = cols.map(c => `<th>${c}</th>`).join("");
    const rows = recRecords.slice(-10).reverse().map(r =>
      `<tr>${cols.map(c => `<td>${r[c]}</td>`).join("")}</tr>`
    ).join("");
    setHtml("rec-table-wrap",
      `<div class="rec-scroll">
        <table class="rec-table">
          <thead><tr>${th}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
       </div>`
    );
  }

  // ── Connection banner ─────────────────────────────────────────────────────────
  function setBanner(state, text) {
    const banner = $("conn-banner");
    banner.style.display = "";
    banner.className = `conn-banner conn-${state}`;
    setText("conn-text", text);
  }

  function hideBanner() {
    $("conn-banner").style.display = "none";
  }

  // ── Boot ──────────────────────────────────────────────────────────────────────
  function init() {
    initCharts();
    connect();
  }

  document.addEventListener("DOMContentLoaded", init);

  return { startCamera, reconnect, startRecording, stopRecording, clearRecording, downloadCSV };
})();
