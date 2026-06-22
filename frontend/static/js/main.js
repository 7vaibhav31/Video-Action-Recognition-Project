// ================================================================
// main.js — Video Action Recognition Dashboard
// Handles: Neural network canvas, drag & drop, upload, results
// ================================================================

// ── Neural Network Canvas Animation ─────────────────────────────
const canvas = document.getElementById('neural-canvas');
const ctx    = canvas.getContext('2d');

let nodes = [];
let W, H;

function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}

function initNodes(count = 60) {
  nodes = [];
  for (let i = 0; i < count; i++) {
    nodes.push({
      x:   Math.random() * W,
      y:   Math.random() * H,
      vx:  (Math.random() - 0.5) * 0.4,
      vy:  (Math.random() - 0.5) * 0.4,
      r:   Math.random() * 2 + 1,
      pulse: Math.random() * Math.PI * 2,
    });
  }
}

function drawNeural() {
  ctx.clearRect(0, 0, W, H);

  const CONNECT_DIST = 160;

  // Draw edges
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx   = nodes[i].x - nodes[j].x;
      const dy   = nodes[i].y - nodes[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < CONNECT_DIST) {
        const alpha = (1 - dist / CONNECT_DIST) * 0.35;
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[j].x, nodes[j].y);
        ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.lineWidth   = 0.6;
        ctx.stroke();
      }
    }
  }

  // Draw nodes
  nodes.forEach(node => {
    node.pulse += 0.025;
    const radius = node.r + Math.sin(node.pulse) * 0.6;
    const alpha  = 0.4 + Math.sin(node.pulse) * 0.2;

    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
    ctx.fill();

    // Move
    node.x += node.vx;
    node.y += node.vy;

    // Bounce off edges
    if (node.x < 0 || node.x > W) node.vx *= -1;
    if (node.y < 0 || node.y > H) node.vy *= -1;
  });

  requestAnimationFrame(drawNeural);
}

window.addEventListener('resize', () => { resize(); initNodes(); });
resize();
initNodes();
drawNeural();


// ── Scroll Reveal ────────────────────────────────────────────────
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => {
        entry.target.classList.add('visible');
      }, entry.target.dataset.delay || 0);
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.reveal').forEach((el, i) => {
  el.dataset.delay = i * 80;
  revealObserver.observe(el);
});


// ── Drag & Drop Upload ───────────────────────────────────────────
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo  = document.getElementById('file-info');
const fileName  = document.getElementById('file-name');
const fileSize  = document.getElementById('file-size');
const predictBtn = document.getElementById('predict-btn');

let selectedFile = null;

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function setFile(file) {
  if (!file) return;
  const allowed = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska'];
  const ext     = file.name.split('.').pop().toLowerCase();
  const allowedExt = ['mp4', 'avi', 'mov', 'mkv', 'webm'];

  if (!allowedExt.includes(ext)) {
    showError(`Unsupported format ".${ext}". Please upload MP4, AVI, MOV, MKV or WebM.`);
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.classList.add('visible');
  predictBtn.disabled = false;
  clearError();
  clearResult();
  // Reset input so the same file can be re-selected later
  fileInput.value = '';
}

// Click to open file picker
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

// Drag events
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});


// ── Inference Request ────────────────────────────────────────────
const loadingState = document.getElementById('loading-state');
const resultContent = document.getElementById('result-content');
const errorState   = document.getElementById('error-state');

// Helper to wait
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

// Polling loop to query task status
async function pollTaskStatus(taskId, apiBase) {
  const maxPolls = 60; // 120 seconds total timeout
  let pollCount = 0;
  let consecFailures = 0;

  while (pollCount < maxPolls) {
    pollCount++;
    // Update progress dots and status text dynamically in the UI
    const dots = '.'.repeat((pollCount % 3) + 1);
    setLoading(
      true, 
      `Analyzing${dots}`, 
      `Running prediction pipeline (step ${pollCount}/${maxPolls})...`
    );

    try {
      const response = await fetch(`${apiBase}/status/${taskId}`);
      
      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      consecFailures = 0; // Reset consecutive failures on success

      if (data.status === 'complete') {
        showResult(data.result);
        return true;
      } else if (data.status === 'failed') {
        showError(data.error || 'Server-side inference failed.');
        return false;
      } else {
        // Status is 'processing', wait and continue
        await delay(2000);
      }
    } catch (err) {
      console.warn('Poll request failed:', err);
      consecFailures++;
      if (consecFailures >= 5) {
        showError(`Lost connection to the backend server. Please make sure it is still running.`);
        return false;
      }
      // Retry after delay
      await delay(2000);
    }
  }

  showError('Analysis timed out. The server took too long to respond. Please try a shorter video.');
  return false;
}

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Capture reference before async (selectedFile may change)
  const videoFile = selectedFile;

  // Show initial loading state (Upload phase)
  setLoading(true, 'Uploading...', 'Uploading video file to server...');
  clearResult();
  clearError();

  // Determine API URL based on host (local vs deployed backend)
  const API_BASE = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' || 
                   window.location.hostname === '' || 
                   window.location.protocol === 'file:'
    ? 'http://127.0.0.1:5000'
    : 'https://video-action-recognition-project.onrender.com';

  const formData = new FormData();
  formData.append('video', videoFile);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errMsg = 'Prediction failed. Please try again.';
      try { const d = await response.json(); if (d.error) errMsg = d.error; } catch (_) {}
      showError(errMsg);
      return;
    }

    const data = await response.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    if (data.status === 'processing' && data.task_id) {
      // Start polling status
      setLoading(true, 'Analyzing...', 'Queueing task on server...');
      await pollTaskStatus(data.task_id, API_BASE);
    } else {
      showError('Unexpected response from server.');
    }
  } catch (err) {
    console.error('Fetch error:', err);
    showError(`Network error. Make sure the backend server is running at ${API_BASE}`);
  } finally {
    setLoading(false);
    // Reset so a new file can be picked immediately
    selectedFile = null;
    fileInfo.classList.remove('visible');
    predictBtn.disabled = true;
    fileInput.value = '';
  }
});


// ── UI State Helpers ─────────────────────────────────────────────
function setLoading(on, text = 'Analyzing...', subtext = 'Extracting frames → CNN features → LSTM prediction') {
  loadingState.classList.toggle('visible', on);
  predictBtn.disabled = on;

  const loaderText = loadingState.querySelector('.loader-text');
  const loaderSub  = loadingState.querySelector('.loader-sub');

  if (loaderText) loaderText.textContent = text;
  if (loaderSub)  loaderSub.textContent  = subtext;

  // Use innerHTML to preserve the SVG icon inside the button
  if (on) {
    predictBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="animation:spin 0.8s linear infinite">
        <path d="M12 2a10 10 0 0 1 10 10" />
      </svg>
      ${text}`;
  } else {
    predictBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polygon points="5 3 19 12 5 21 5 3"/>
      </svg>
      Analyze Video`;
  }
}

function clearResult() {
  resultContent.classList.remove('visible');
}

function clearError() {
  errorState.classList.remove('visible');
  errorState.textContent = '';
}

function showError(msg) {
  errorState.textContent = msg;
  errorState.classList.add('visible');
}

function showResult(data) {
  // Populate main prediction
  document.getElementById('result-action-name').textContent = data.predicted_class;
  document.getElementById('result-confidence-pct').textContent = data.confidence.toFixed(1) + '%';

  // Animate confidence bar
  const bar = document.getElementById('result-conf-bar');
  bar.style.width = '0%';
  setTimeout(() => { bar.style.width = data.confidence + '%'; }, 50);

  // Populate top-3
  const top3Container = document.getElementById('top3-list');
  top3Container.innerHTML = '';

  data.top3.forEach((item, i) => {
    const div = document.createElement('div');
    div.className = 'top3-item';
    div.innerHTML = `
      <span class="top3-rank">#${i + 1}</span>
      <span class="top3-label">${item.label}</span>
      <div class="top3-bar-wrap">
        <div class="top3-bar" data-pct="${item.confidence}"></div>
      </div>
      <span class="top3-pct">${item.confidence.toFixed(1)}%</span>
    `;
    top3Container.appendChild(div);
  });

  // Animate top-3 bars
  setTimeout(() => {
    document.querySelectorAll('.top3-bar').forEach(bar => {
      bar.style.width = bar.dataset.pct + '%';
    });
  }, 200);

  // Processing time
  document.getElementById('proc-time').textContent = data.processing_time + 's';

  // Show result panel
  resultContent.classList.add('visible');

  // Scroll to results
  resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


// ── Smooth scroll for nav links ──────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth' });
  });
});


// ── Counter animation for hero stats ─────────────────────────────
function animateCounter(el, target, duration = 1500) {
  const start = performance.now();
  const isFloat = target % 1 !== 0;

  function update(now) {
    const elapsed  = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    const value    = eased * target;
    el.textContent = isFloat ? value.toFixed(1) : Math.round(value).toLocaleString();
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll('.stat-number').forEach(el => {
        const target = parseFloat(el.dataset.target);
        animateCounter(el, target);
      });
      statsObserver.disconnect();
    }
  });
}, { threshold: 0.5 });

const statsEl = document.querySelector('.hero-stats');
if (statsEl) statsObserver.observe(statsEl);
