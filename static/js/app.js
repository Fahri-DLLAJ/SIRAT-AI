/* ============================================================
   DLLAJ AI — Crosswalk Violation Detection System
   Frontend Application Logic
   ============================================================ */

const API_BASE = '';
let socket = null;
let currentVideoFile = null;
let roiPoints = [];
let crosswalkPoints = [];
let centerlinePoints = [];
let activeZoneType = 'crosswalk'; // 'crosswalk' or 'centerline'
let isDrawingROI = false;
let isDetecting = false;
let violations = [];
let firstFrame = null;
let currentModalViolationId = null;

// ── Canvas references ──
let canvas, ctx;
let canvasImage = new Image();

// ============================================================
//  INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', init);

function init() {
  canvas = document.getElementById('video-canvas');
  ctx = canvas.getContext('2d');

  initDragDrop();
  initWebSocket();
  initConfidenceSlider();

  // File input bridge
  const fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
  });

  // Canvas events (will be active during ROI drawing)
  canvas.addEventListener('click', handleCanvasClick);
  canvas.addEventListener('mousemove', handleCanvasMouseMove);

  // Modal close on backdrop click
  document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeModal();
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });

  updateStatus('ready', 'Ready');
}

// ============================================================
//  DRAG & DROP
// ============================================================

function initDragDrop() {
  const zone = document.getElementById('upload-zone');

  zone.addEventListener('click', () => {
    document.getElementById('file-input').click();
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length) handleFileSelect(files[0]);
  });
}

// ============================================================
//  VIDEO UPLOAD
// ============================================================

function handleFileSelect(file) {
  const allowedExtensions = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
  const ext = file.name.split('.').pop().toLowerCase();

  if (!allowedExtensions.includes(ext)) {
    showToast(`Invalid file type ".${ext}". Supported: ${allowedExtensions.join(', ')}`, 'error');
    return;
  }

  if (file.size > 500 * 1024 * 1024) {
    showToast('File too large. Maximum size is 500MB.', 'error');
    return;
  }

  currentVideoFile = file;

  // Show file info
  document.getElementById('upload-zone').hidden = true;
  document.getElementById('file-info').hidden = false;
  document.getElementById('file-name').textContent = `${file.name} (${formatFileSize(file.size)})`;

  showToast(`Selected: ${file.name}`, 'info');
  uploadVideo();
}

async function uploadVideo() {
  if (!currentVideoFile) {
    showToast('No file selected.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('video', currentVideoFile);

  const progressContainer = document.getElementById('upload-progress');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');

  progressContainer.hidden = false;

  try {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        progressFill.style.width = pct + '%';
        progressText.textContent = pct + '%';
      }
    });

    const response = await new Promise((resolve, reject) => {
      xhr.open('POST', `${API_BASE}/api/upload`);

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.error || `Upload failed (${xhr.status})`));
          } catch {
            reject(new Error(`Upload failed (${xhr.status})`));
          }
        }
      };

      xhr.onerror = () => reject(new Error('Network error during upload'));
      xhr.send(formData);
    });

    progressFill.style.width = '100%';
    progressText.textContent = '100%';

    showToast('Video uploaded successfully!', 'success');
    await extractFirstFrame();

  } catch (err) {
    showToast(`Upload error: ${err.message}`, 'error');
    progressContainer.hidden = true;
  }
}

async function extractFirstFrame() {
  try {
    const res = await fetch(`${API_BASE}/api/first-frame`, { method: 'POST' });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'Failed to extract first frame');
    }

    const data = await res.json();
    firstFrame = data.image; // base64 image

    // Draw first frame on canvas
    canvasImage.onload = () => {
      canvas.width = canvasImage.width;
      canvas.height = canvasImage.height;
      ctx.drawImage(canvasImage, 0, 0);
      hidePlaceholder();
    };
    canvasImage.src = 'data:image/jpeg;base64,' + firstFrame;

    // Advance to ROI step
    showStep(2);
    isDrawingROI = true;
    showToast('Click on the frame to define the crosswalk zone.', 'info');

  } catch (err) {
    showToast(`Frame extraction error: ${err.message}`, 'error');
  }
}

function resetUpload() {
  currentVideoFile = null;
  firstFrame = null;
  roiPoints = [];
  crosswalkPoints = [];
  centerlinePoints = [];
  activeZoneType = 'crosswalk';
  isDrawingROI = false;

  // Reset zone selector UI
  const btnCrosswalk = document.getElementById('btn-zone-crosswalk');
  const btnCenterline = document.getElementById('btn-zone-centerline');
  if (btnCrosswalk) btnCrosswalk.classList.add('active');
  if (btnCenterline) btnCenterline.classList.remove('active');
  
  const hint = document.getElementById('roi-hint');
  if (hint) hint.textContent = 'Click on the video frame to draw a polygon around the crosswalk area.';

  document.getElementById('upload-zone').hidden = false;
  document.getElementById('file-info').hidden = true;
  document.getElementById('upload-progress').hidden = true;
  document.getElementById('progress-fill').style.width = '0%';
  document.getElementById('file-input').value = '';

  showStep(1);
  showPlaceholder();

  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ============================================================
//  ROI DRAWING (Crosswalk Zone)
// ============================================================

function initROICanvas() {
  isDrawingROI = true;
  roiPoints = [];
  updatePointCount();
}

function handleCanvasClick(e) {
  if (!isDrawingROI) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  roiPoints.push({ x: Math.round(x), y: Math.round(y) });
  updatePointCount();
  drawROI();

  // Enable complete button if 3+ points
  const btn = document.getElementById('btn-complete-roi');
  btn.disabled = roiPoints.length < 3;
}

function handleCanvasMouseMove(e) {
  if (!isDrawingROI || roiPoints.length === 0) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  drawROI(mx, my);
}

function selectZoneType(type) {
  if (type === activeZoneType) return;

  // Save current points to the previous zone
  if (activeZoneType === 'crosswalk') {
    crosswalkPoints = [...roiPoints];
  } else {
    centerlinePoints = [...roiPoints];
  }

  // Switch active zone
  activeZoneType = type;
  roiPoints = type === 'crosswalk' ? [...crosswalkPoints] : [...centerlinePoints];

  // Update UI buttons active class
  const btnCrosswalk = document.getElementById('btn-zone-crosswalk');
  const btnCenterline = document.getElementById('btn-zone-centerline');
  if (btnCrosswalk) btnCrosswalk.classList.toggle('active', type === 'crosswalk');
  if (btnCenterline) btnCenterline.classList.toggle('active', type === 'centerline');

  // Update hint text
  const hint = document.getElementById('roi-hint');
  if (hint) {
    if (type === 'crosswalk') {
      hint.textContent = 'Click on the video frame to draw a polygon around the crosswalk area.';
    } else {
      hint.textContent = 'Click on the video frame to draw a polygon representing the center line / lane divider.';
    }
  }

  updatePointCount();
  drawROI();

  // Enable complete button if 3+ points
  document.getElementById('btn-complete-roi').disabled = roiPoints.length < 3;
}

function drawFinalROI() {
  if (canvasImage.complete && canvasImage.naturalWidth > 0) {
    ctx.drawImage(canvasImage, 0, 0, canvas.width, canvas.height);
  }

  // Draw crosswalk points if any
  if (crosswalkPoints.length >= 3) {
    ctx.beginPath();
    ctx.moveTo(crosswalkPoints[0].x, crosswalkPoints[0].y);
    for (let i = 1; i < crosswalkPoints.length; i++) {
      ctx.lineTo(crosswalkPoints[i].x, crosswalkPoints[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 220, 50, 0.2)';
    ctx.fill();
    ctx.strokeStyle = '#FFD700';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  // Draw centerline points if any
  if (centerlinePoints.length >= 3) {
    ctx.beginPath();
    ctx.moveTo(centerlinePoints[0].x, centerlinePoints[0].y);
    for (let i = 1; i < centerlinePoints.length; i++) {
      ctx.lineTo(centerlinePoints[i].x, centerlinePoints[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 0, 127, 0.2)';
    ctx.fill();
    ctx.strokeStyle = '#FF007F';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
}

function drawROI(mouseX, mouseY) {
  // Redraw background
  if (canvasImage.complete && canvasImage.naturalWidth > 0) {
    ctx.drawImage(canvasImage, 0, 0, canvas.width, canvas.height);
  }

  // Draw the INACTIVE zone first (if it has points)
  const inactiveZone = activeZoneType === 'crosswalk' ? 'centerline' : 'crosswalk';
  const inactivePoints = inactiveZone === 'crosswalk' ? crosswalkPoints : centerlinePoints;

  if (inactivePoints && inactivePoints.length >= 3) {
    ctx.beginPath();
    ctx.moveTo(inactivePoints[0].x, inactivePoints[0].y);
    for (let i = 1; i < inactivePoints.length; i++) {
      ctx.lineTo(inactivePoints[i].x, inactivePoints[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = inactiveZone === 'crosswalk' 
      ? 'rgba(255, 220, 50, 0.1)' 
      : 'rgba(255, 0, 127, 0.1)';
    ctx.fill();
    ctx.strokeStyle = inactiveZone === 'crosswalk' ? '#FFD700' : '#FF007F';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  if (roiPoints.length === 0) return;

  // Draw active filled polygon (semi-transparent)
  ctx.beginPath();
  ctx.moveTo(roiPoints[0].x, roiPoints[0].y);
  for (let i = 1; i < roiPoints.length; i++) {
    ctx.lineTo(roiPoints[i].x, roiPoints[i].y);
  }
  if (mouseX !== undefined && mouseY !== undefined) {
    ctx.lineTo(mouseX, mouseY);
  }
  ctx.closePath();
  ctx.fillStyle = activeZoneType === 'crosswalk'
    ? 'rgba(255, 220, 50, 0.15)'
    : 'rgba(255, 0, 127, 0.15)';
  ctx.fill();

  // Draw active lines
  ctx.beginPath();
  ctx.moveTo(roiPoints[0].x, roiPoints[0].y);
  for (let i = 1; i < roiPoints.length; i++) {
    ctx.lineTo(roiPoints[i].x, roiPoints[i].y);
  }

  // Preview line to mouse
  if (mouseX !== undefined && mouseY !== undefined) {
    ctx.lineTo(mouseX, mouseY);
  }

  ctx.closePath();
  ctx.strokeStyle = activeZoneType === 'crosswalk' ? '#FFD700' : '#FF007F';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw active points
  roiPoints.forEach((pt, i) => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? '#00d4ff' : (activeZoneType === 'crosswalk' ? '#FFD700' : '#FF007F');
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Point label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText(`${i + 1}`, pt.x + 10, pt.y - 8);
  });
}

function clearROI() {
  roiPoints = [];
  if (activeZoneType === 'crosswalk') {
    crosswalkPoints = [];
  } else {
    centerlinePoints = [];
  }
  updatePointCount();
  document.getElementById('btn-complete-roi').disabled = true;
  drawFinalROI();
}

async function completeROI() {
  if (roiPoints.length < 3) {
    showToast('At least 3 points required to define a zone.', 'warning');
    return;
  }

  // Save current points to active zone
  if (activeZoneType === 'crosswalk') {
    crosswalkPoints = [...roiPoints];
  } else {
    centerlinePoints = [...roiPoints];
  }

  isDrawingROI = false;
  drawFinalROI();

  await sendROI();

  if (activeZoneType === 'crosswalk') {
    showToast('Zebra cross zone set! You can now define the Center Line zone or start detection.', 'success');
  } else {
    showToast('Center line zone set!', 'success');
  }
}

async function sendROI() {
  try {
    const res = await fetch(`${API_BASE}/api/set-roi`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        zone_type: activeZoneType,
        points: roiPoints.map(p => [p.x, p.y])
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Failed to set ${activeZoneType} zone`);
    }

    showStep(3);

  } catch (err) {
    showToast(`ROI error: ${err.message}`, 'error');
    isDrawingROI = true; // Allow re-drawing
  }
}

function updatePointCount() {
  document.getElementById('point-count').textContent = roiPoints.length;
}

// ============================================================
//  DETECTION
// ============================================================

async function startDetection() {
  if (isDetecting) return;

  const trafficLight = document.getElementById('traffic-light-toggle').checked;
  const confidence = parseFloat(document.getElementById('confidence-threshold').value);

  if (!socket || !socket.connected) {
    showToast('WebSocket not connected. Please wait...', 'error');
    return;
  }

  // Emit start_detection via WebSocket for threaded processing
  socket.emit('start_detection', {
    traffic_light: trafficLight,
    confidence_threshold: confidence
  });

  isDetecting = true;

  document.getElementById('btn-start').hidden = true;
  document.getElementById('btn-stop').hidden = false;
  document.getElementById('detection-progress').hidden = false;
  document.getElementById('stats-panel').hidden = false;

  updateStatus('detecting', 'Detecting...');
  showToast('Detection started!', 'success');
}

function stopDetection() {
  if (socket && socket.connected) {
    socket.emit('stop_detection');
  }

  isDetecting = false;

  document.getElementById('btn-start').hidden = false;
  document.getElementById('btn-stop').hidden = true;

  updateStatus('ready', 'Stopped');
  showToast('Detection stopped.', 'info');
}

// ============================================================
//  WEBSOCKET (Socket.IO)
// ============================================================

function initWebSocket() {
  if (typeof io === 'undefined') {
    console.warn('Socket.IO not loaded. Real-time features disabled.');
    return;
  }

  socket = io(API_BASE || window.location.origin, {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 10
  });

  socket.on('connect', () => {
    updateStatus('connected', 'Connected');
    console.log('[WS] Connected:', socket.id);
  });

  socket.on('disconnect', () => {
    updateStatus('disconnected', 'Disconnected');
    console.log('[WS] Disconnected');
  });

  socket.on('connect_error', (err) => {
    console.warn('[WS] Connection error:', err.message);
    updateStatus('disconnected', 'Connection Error');
  });

  // Receive processed frame
  socket.on('frame', (data) => {
    if (data.image) {
      const img = new Image();
      img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
      };
      img.src = 'data:image/jpeg;base64,' + data.image;
      hidePlaceholder();
    }
  });

  // Receive violation event
  socket.on('violation', (data) => {
    addViolationToLog(data);
    showToast(`⚠️ Violation detected: ${data.violation_name || 'Crosswalk violation'}`, 'warning');
  });

  // Progress updates
  socket.on('progress', (data) => {
    const fill = document.getElementById('detection-progress-fill');
    const text = document.getElementById('detection-progress-text');
    const frameText = document.getElementById('detection-frame-text');

    const pct = data.percent || (data.total > 0 ? Math.round((data.current / data.total) * 100) : 0);
    fill.style.width = pct + '%';
    text.textContent = pct + '%';
    frameText.textContent = `Frame ${data.current} / ${data.total}`;

    // Update stats if provided
    if (data.stats) {
      document.getElementById('stat-total').textContent = data.stats.total_violations || 0;
      document.getElementById('stat-vehicles').textContent = data.stats.total_vehicles || 0;
      document.getElementById('stat-pedestrians').textContent = data.stats.total_pedestrians || 0;
    }
  });

  // Detection complete
  socket.on('detection_complete', (data) => {
    isDetecting = false;

    document.getElementById('btn-start').hidden = false;
    document.getElementById('btn-stop').hidden = true;

    updateStatus('ready', 'Complete');

    const fill = document.getElementById('detection-progress-fill');
    fill.style.width = '100%';

    if (data && data.stats) {
      document.getElementById('stat-total').textContent = data.stats.total_violations || 0;
      document.getElementById('stat-vehicles').textContent = data.stats.total_vehicles || 0;
      document.getElementById('stat-pedestrians').textContent = data.stats.total_pedestrians || 0;
    }

    showToast('Detection completed!', 'success');
    loadViolations();
  });

  // Error
  socket.on('error', (data) => {
    showToast(`Error: ${data.message || 'Unknown error'}`, 'error');
    console.error('[WS] Error:', data);
  });
}

// ============================================================
//  VIOLATION LOG
// ============================================================

function addViolationToLog(violation) {
  violations.push(violation);
  updateViolationCount();

  const grid = document.getElementById('violations-grid');
  const emptyState = document.getElementById('empty-violations');
  if (emptyState) emptyState.hidden = true;

  const card = createViolationCard(violation);
  grid.prepend(card);

  // Update stats
  document.getElementById('stat-total').textContent = violations.length;
}

function createViolationCard(v) {
  const card = document.createElement('div');
  card.className = 'violation-card';
  card.dataset.id = v.id || Date.now();

  const confidenceVal = v.confidence || 0;
  const confidenceClass = confidenceVal >= 0.7 ? 'high' : confidenceVal >= 0.4 ? 'medium' : 'low';
  const confidencePct = (confidenceVal * 100).toFixed(0);

  // Use screenshot_path from backend (served via /violations/screenshots/...)
  const thumbSrc = v.screenshot_path
    ? `/${v.screenshot_path}`
    : '';

  card.innerHTML = `
    ${thumbSrc
      ? `<img class="violation-card-thumb" src="${thumbSrc}" alt="Violation">`
      : `<div class="violation-card-thumb" style="display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.03);">
           <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="color:var(--text-muted)"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
         </div>`
    }
    <div class="violation-card-body">
      <h4>${escapeHtml(v.violation_name || 'Crosswalk Violation')}</h4>
      <div class="violation-card-meta">
        <span>🕐 ${escapeHtml(v.time_in_video || '00:00')}</span>
        <span>🚗 ${escapeHtml(v.vehicle_type || 'Vehicle')}</span>
        <span class="confidence-badge ${confidenceClass}">${confidencePct}%</span>
      </div>
    </div>
  `;

  card.addEventListener('click', () => showViolationDetail(card.dataset.id));
  return card;
}

function showViolationDetail(violationId) {
  const v = violations.find(
    (item) => (item.id || '').toString() === violationId.toString()
  );
  if (!v) return;

  currentModalViolationId = violationId;

  const modal = document.getElementById('modal-overlay');
  const screenshot = document.getElementById('modal-screenshot');

  if (v.screenshot_path) {
    screenshot.src = `/${v.screenshot_path}`;
    screenshot.style.display = 'block';
  } else {
    screenshot.style.display = 'none';
  }

  document.getElementById('modal-title').textContent = v.violation_name || 'Crosswalk Violation';
  document.getElementById('modal-id').textContent = v.id || 'N/A';
  document.getElementById('modal-type').textContent = v.violation_name || 'Crosswalk Violation';
  document.getElementById('modal-time').textContent = v.time_in_video || '00:00';
  document.getElementById('modal-vehicle').textContent = v.vehicle_type || 'Unknown';

  const conf = v.confidence || 0;
  const confClass = conf >= 0.7 ? 'high' : conf >= 0.4 ? 'medium' : 'low';
  document.getElementById('modal-confidence').innerHTML =
    `<span class="confidence-badge ${confClass}">${(conf * 100).toFixed(1)}%</span>`;

  document.getElementById('modal-description').textContent =
    v.description || 'Vehicle detected in crosswalk zone while pedestrian is crossing.';

  document.getElementById('modal-legal').textContent =
    v.legal_reference || 'Pasal 106 ayat (4) huruf b UU No. 22 Tahun 2009 tentang Lalu Lintas dan Angkutan Jalan';

  document.getElementById('modal-penalty').textContent =
    v.penalty || 'Pidana kurungan paling lama 2 bulan atau denda paling banyak Rp500.000';

  modal.hidden = false;
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  document.getElementById('modal-overlay').hidden = true;
  document.body.style.overflow = '';
  currentModalViolationId = null;
}

async function loadViolations() {
  try {
    const res = await fetch(`${API_BASE}/api/violations`);
    if (!res.ok) throw new Error('Failed to load violations');

    const data = await res.json();
    violations = data.violations || data || [];

    const grid = document.getElementById('violations-grid');
    const emptyState = document.getElementById('empty-violations');

    // Clear existing cards (keep empty state)
    const cards = grid.querySelectorAll('.violation-card');
    cards.forEach((c) => c.remove());

    if (violations.length === 0) {
      if (emptyState) emptyState.hidden = false;
    } else {
      if (emptyState) emptyState.hidden = true;
      violations.forEach((v) => {
        grid.appendChild(createViolationCard(v));
      });
    }

    updateViolationCount();
  } catch (err) {
    console.warn('Could not load violations:', err.message);
  }
}

async function deleteViolation(id) {
  try {
    const res = await fetch(`${API_BASE}/api/violations/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete violation');

    violations = violations.filter(
      (v) => (v.id || '').toString() !== id.toString()
    );

    const card = document.querySelector(`.violation-card[data-id="${id}"]`);
    if (card) card.remove();

    updateViolationCount();
    showToast('Violation deleted.', 'info');

    // Show empty state if no violations left
    if (violations.length === 0) {
      const emptyState = document.getElementById('empty-violations');
      if (emptyState) emptyState.hidden = false;
    }

  } catch (err) {
    showToast(`Delete error: ${err.message}`, 'error');
  }
}

function deleteCurrentViolation() {
  if (!currentModalViolationId) return;
  deleteViolation(currentModalViolationId);
  closeModal();
}

function exportViolations() {
  if (violations.length === 0) {
    showToast('No violations to export.', 'warning');
    return;
  }

  const dataStr = JSON.stringify(violations, null, 2);
  const blob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `violations_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast('Violations exported!', 'success');
}

function updateViolationCount() {
  document.getElementById('violation-count').textContent = violations.length;
}

// ============================================================
//  UI HELPERS
// ============================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');

  const iconSVGs = {
    success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>'
  };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <div class="toast-icon">${iconSVGs[type] || iconSVGs.info}</div>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  // Auto remove
  setTimeout(() => {
    toast.classList.add('removing');
    toast.addEventListener('animationend', () => toast.remove());
  }, 4000);
}

function updateStatus(status, text) {
  const dot = document.querySelector('.status-dot');
  const label = document.querySelector('.status-text');

  // Clear all status classes
  dot.className = 'status-dot';
  dot.classList.add(status);
  label.textContent = text;
}

function formatTime(seconds) {
  if (typeof seconds !== 'number' || isNaN(seconds)) return '00:00.000';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function showStep(stepNumber) {
  const roiPanel = document.getElementById('roi-panel');
  const detectionPanel = document.getElementById('detection-panel');

  if (stepNumber >= 2) {
    roiPanel.hidden = false;
    isDrawingROI = true;
  }

  if (stepNumber >= 3) {
    detectionPanel.hidden = false;
    isDrawingROI = false;
  }
}

function switchTab(tabName) {
  // Update tab buttons
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  });

  // Update tab content
  document.querySelectorAll('.tab-content').forEach((content) => {
    content.classList.remove('active');
    content.hidden = true;
  });

  const target = document.getElementById(`tab-${tabName}`);
  if (target) {
    target.hidden = false;
    target.classList.add('active');
  }

  // Load violations when switching to violations tab
  if (tabName === 'violations') {
    loadViolations();
  }
}

function hidePlaceholder() {
  const placeholder = document.getElementById('canvas-placeholder');
  if (placeholder) placeholder.hidden = true;
}

function showPlaceholder() {
  const placeholder = document.getElementById('canvas-placeholder');
  if (placeholder) placeholder.hidden = false;
}

function initConfidenceSlider() {
  const slider = document.getElementById('confidence-threshold');
  const display = document.getElementById('confidence-value');

  slider.addEventListener('input', () => {
    display.textContent = parseFloat(slider.value).toFixed(2);
  });
}
