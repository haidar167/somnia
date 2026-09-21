/**
 * THE SPECIMEN — Living AI Organism Web Console JS
 */

// Canvas State
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize black canvas
function initCanvas() {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
initCanvas();

// Drawing event listeners (Mouse & Touch)
function startDrawing(e) {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  lastX = clientX - rect.left;
  lastY = clientY - rect.top;
}

function draw(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  const currX = clientX - rect.left;
  const currY = clientY - rect.top;

  ctx.beginPath();
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(currX, currY);
  ctx.stroke();

  lastX = currX;
  lastY = currY;
}

function stopDrawing() {
  isDrawing = false;
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

document.getElementById('btn-clear').addEventListener('click', initCanvas);

// Extract 28x28 normalized float array from 280x280 canvas
function getCanvas28x28Array() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  const imgData = tempCtx.getImageData(0, 0, 28, 28);
  const data = imgData.data;
  const floats = [];
  for (let i = 0; i < data.length; i += 4) {
    // Greyscale pixel value normalized to [0, 1]
    floats.push(data[i] / 255.0);
  }
  return floats;
}

// Draw 784-float array onto main 280x280 canvas
function setCanvasFrom784Array(arr) {
  initCanvas();
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext('2d');
  const imgData = tempCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.floor(arr[i] * 255);
    imgData.data[i * 4 + 0] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  tempCtx.putImageData(imgData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tempCanvas, 0, 0, 280, 280);
}

// Load Presets
async function loadPreset(presetName) {
  try {
    const res = await fetch(`/preset/${presetName}`);
    if (res.ok) {
      const data = await res.json();
      setCanvasFrom784Array(data.image);
    }
  } catch (err) {
    console.error("Failed to load preset:", err);
  }
}

// Feed Button
document.getElementById('btn-feed').addEventListener('click', async () => {
  const floatArr = getCanvas28x28Array();
  try {
    const res = await fetch('/feed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: floatArr })
    });
    const data = await res.json();
    console.log("Feed response:", data);
  } catch (err) {
    console.error("Feed error:", err);
  }
});

// Reveal Truth Button
document.getElementById('btn-reveal').addEventListener('click', async () => {
  const inputEl = document.getElementById('reveal-input');
  const val = parseInt(inputEl.value, 10);
  if (isNaN(val) || val < 0 || val > 9) {
    alert("Please enter a digit from 0 to 9.");
    return;
  }
  try {
    await fetch('/reveal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: val })
    });
    inputEl.value = '';
  } catch (err) {
    console.error("Reveal error:", err);
  }
});

// Sleep Button
document.getElementById('btn-sleep').addEventListener('click', async () => {
  const btn = document.getElementById('btn-sleep');
  btn.disabled = true;
  btn.innerText = "DREAMING...";
  try {
    await fetch('/sleep', { method: 'POST' });
  } catch (err) {
    console.error("Sleep error:", err);
  } finally {
    setTimeout(() => {
      btn.disabled = false;
      btn.innerText = "TRIGGER SLEEP";
    }, 2000);
  }
});

// Brainwaves Rolling Chart Renderer
const bwCanvas = document.getElementById('brainwaves-canvas');
const bwCtx = bwCanvas.getContext('2d');

function drawBrainwaves(history) {
  const w = bwCanvas.width;
  const h = bwCanvas.height;
  bwCtx.clearRect(0, 0, w, h);

  // Background grid
  bwCtx.strokeStyle = '#1E293B';
  bwCtx.lineWidth = 1;
  bwCtx.beginPath();
  for (let y = 0; y <= h; y += 40) {
    bwCtx.moveTo(0, y);
    bwCtx.lineTo(w, y);
  }
  bwCtx.stroke();

  if (!history || history.length < 2) return;

  const n = history.length;
  const step = w / 60;

  // Channels to render
  const channels = [
    { key: 'p_error', color: '#FF3B30', scale: h },
    { key: 'confidence', color: '#38BDF8', scale: h },
    { key: 'mean_act', color: '#00FF9D', scale: h * 2 },
    { key: 'margin', color: '#FACC15', scale: h },
  ];

  channels.forEach(ch => {
    bwCtx.strokeStyle = ch.color;
    bwCtx.lineWidth = 1.8;
    bwCtx.beginPath();
    history.forEach((pt, idx) => {
      const x = (60 - n + idx) * step;
      const val = pt[ch.key] !== undefined ? pt[ch.key] : 0;
      const y = h - Math.min(Math.max(val * ch.scale, 5), h - 5);
      if (idx === 0) bwCtx.moveTo(x, y);
      else bwCtx.lineTo(x, y);
    });
    bwCtx.stroke();
  });
}

// UI State Updater
function updateUI(state) {
  // Header
  document.getElementById('uptime-display').innerText = state.uptime_str || '00:00:00';
  document.getElementById('sleep-count-display').innerText = state.sleep_count || 0;
  document.getElementById('feed-count-display').innerText = state.total_feeds || 0;
  document.getElementById('visitor-count-display').innerText = state.visitor_count || 1;
  document.getElementById('vital-status').innerText = state.is_sleeping ? 'SLEEPING (DREAMING)' : 'ALIVE';
  document.getElementById('vital-status').style.color = state.is_sleeping ? '#A78BFA' : '#00FF9D';

  const v = state.vitals;
  if (!v) return;

  // Vitals
  const pErr = v.p_error || 0.0;
  document.getElementById('p-error-val').innerText = pErr.toFixed(3);
  document.getElementById('p-error-bar').style.width = `${Math.min(pErr * 100, 100)}%`;

  const moodEl = document.getElementById('mood-badge');
  moodEl.innerText = (v.mood || 'CALM').toUpperCase();
  moodEl.className = `mood-badge mood-${(v.mood || 'calm').replace(' ', '_')}`;

  // Prediction & Confidence
  document.getElementById('prediction-val').innerText = v.prediction !== null ? v.prediction : '--';
  document.getElementById('confidence-badge').innerText = `${((v.confidence || 0) * 100).toFixed(1)}% CONF`;

  // Disagreement Banner
  const disBanner = document.getElementById('disagreement-banner');
  if (v.disagreement) {
    disBanner.classList.remove('hidden');
    document.getElementById('disagreement-text').innerText = v.disagreement_note;
  } else {
    disBanner.classList.add('hidden');
  }

  // Feature stats
  if (v.stats) {
    document.getElementById('stat-alive').innerText = `${((v.stats.alive_neurons || 0) * 100).toFixed(1)}%`;
    document.getElementById('stat-top10').innerText = (v.stats.top10_mean || 0).toFixed(3);
    document.getElementById('stat-std').innerText = (v.stats.std_act || 0).toFixed(3);
    document.getElementById('stat-margin').innerText = (v.stats.margin || 0).toFixed(3);
  }

  // Brainwaves Chart
  drawBrainwaves(state.brainwaves);

  // Dream Feed
  const dreamContainer = document.getElementById('dream-feed');
  if (state.recent_dreams && state.recent_dreams.length > 0) {
    dreamContainer.innerHTML = state.recent_dreams.map(d => `
      <div class="dream-card">
        <img src="${d.b64}" alt="Dream">
        <span class="dream-badge ${d.is_nightmare ? 'badge-nightmare' : 'badge-lucid'}">
          ${d.is_nightmare ? 'NIGHTMARE' : 'LUCID'} [${d.target_class}]
        </span>
      </div>
    `).join('');
  }

  // Terminal Log
  const term = document.getElementById('terminal-log');
  if (state.event_log) {
    term.innerHTML = state.event_log.map(e => `
      <div class="log-entry">
        <span class="log-time">[${e.time}]</span>
        <span class="log-msg log-msg-${e.type || 'info'}">${e.message}</span>
      </div>
    `).join('');
  }
}

// WebSocket Live Telemetry Connection
let ws;
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  ws = new WebSocket(wsUrl);

  const pill = document.getElementById('ws-pill');

  ws.onopen = () => {
    pill.innerText = 'LIVE WS';
    pill.classList.add('connected');
  };

  ws.onmessage = (event) => {
    try {
      const state = JSON.parse(event.data);
      updateUI(state);
    } catch (err) {
      console.error("WS parse error:", err);
    }
  };

  ws.onclose = () => {
    pill.innerText = 'RECONNECTING';
    pill.classList.remove('connected');
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

connectWebSocket();
