/**
 * app.js
 * ──────
 * Main Frontend Application Controller for Autonomous Driving Multi-Task Cockpit.
 */

let currentData = null;
let currentView = 'split'; // 'split' | 'seg' | 'depth' | 'hazard' | 'rgb'
let viewer3D = null;
let activePresetId = null;

// DOM Elements
const imgPrimary = document.getElementById('view-primary');
const imgSecondary = document.getElementById('view-secondary');
const splitDivider = document.getElementById('split-divider');
const sliderSplit = document.getElementById('slider-split-pos');
const lblSplit = document.getElementById('lbl-split-pos');
const stageLoader = document.getElementById('stage-loader');
const viewportContainer = document.getElementById('viewport-container');

const probeTooltip = document.getElementById('probe-tooltip');
const probeColorDot = document.getElementById('probe-color-dot');
const probeClassName = document.getElementById('probe-class-name');
const probeCategory = document.getElementById('probe-category');
const probeDistance = document.getElementById('probe-distance');
const probeSafety = document.getElementById('probe-safety');

const hudDevice = document.getElementById('hud-device');
const hudLatency = document.getElementById('hud-latency');
const hudFps = document.getElementById('hud-fps');
const hudParams = document.getElementById('hud-params');
const selectResolution = document.getElementById('select-resolution');

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize 3D WebGL Viewer
    viewer3D = new PointCloudViewer('container-3d');

    // 2. Setup Event Listeners
    setupEventListeners();

    // 3. Load System Status & Preset Scenarios
    await loadSystemStatus();
    await loadPresets();

    // 4. Trigger default prediction on Scene 1
    loadPresetPrediction('scene1_urban');
});

function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('#view-tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('#view-tabs .tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentView = btn.dataset.view;
            updatePerceptionView();
        });
    });

    // Split Slider Control
    sliderSplit.addEventListener('input', (e) => {
        const val = e.target.value;
        lblSplit.textContent = `${val}%`;
        setSplitPosition(val);
    });

    // Cursor Hover Probe on 2D Image Stage
    viewportContainer.addEventListener('mousemove', handleViewportHover);
    viewportContainer.addEventListener('mouseleave', () => {
        probeTooltip.style.display = 'none';
    });

    // File Upload
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadCustomImage(e.target.files[0]);
        }
    });

    // Drag and Drop
    const dropzone = document.getElementById('upload-dropzone');
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = '#06b6d4'; });
    dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = ''; });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '';
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            uploadCustomImage(e.dataTransfer.files[0]);
        }
    });

    // 3D Controls
    document.getElementById('select-3d-colormap').addEventListener('change', (e) => {
        if (viewer3D) viewer3D.setColorMode(e.target.value);
    });

    document.getElementById('slider-point-size').addEventListener('input', (e) => {
        const size = parseFloat(e.target.value);
        document.getElementById('lbl-point-size').textContent = `${size.toFixed(1)}px`;
        if (viewer3D) viewer3D.setPointSize(size);
    });

    document.getElementById('btn-reset-3d').addEventListener('click', () => {
        if (viewer3D) viewer3D.resetCamera();
    });

    document.getElementById('btn-bev-view').addEventListener('click', () => {
        if (viewer3D) viewer3D.setBEVView();
    });

    selectResolution.addEventListener('change', () => {
        if (activePresetId) {
            loadPresetPrediction(activePresetId);
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// API Communication & Prediction
// ─────────────────────────────────────────────────────────────────────────────
async function loadSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        hudDevice.textContent = `${data.device} (${data.gpu_name})`;
        if (!data.is_cuda) {
            hudDevice.classList.remove('badge-gpu');
            hudDevice.style.background = 'rgba(59, 130, 246, 0.2)';
            hudDevice.style.color = '#38bdf8';
        }
        hudParams.textContent = `${data.total_parameters_M}M (-35%)`;
    } catch (err) {
        console.error('Failed to load status:', err);
    }
}

async function loadPresets() {
    try {
        const res = await fetch('/api/presets');
        const presets = await res.json();
        const container = document.getElementById('presets-list');
        container.innerHTML = '';

        presets.forEach(p => {
            const card = document.createElement('div');
            card.className = `preset-card ${p.id === 'scene1_urban' ? 'active' : ''}`;
            card.dataset.id = p.id;
            card.innerHTML = `
                <img src="${p.thumb}" alt="${p.title}" class="preset-thumb">
                <div class="preset-info">
                    <span class="preset-name">${p.title}</span>
                    <span class="preset-desc">${p.city}</span>
                </div>
            `;
            card.addEventListener('click', () => {
                document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                loadPresetPrediction(p.id);
            });
            container.appendChild(card);
        });
    } catch (err) {
        console.error('Failed to load presets:', err);
    }
}

async function loadPresetPrediction(presetId) {
    activePresetId = presetId;
    showLoader(true);
    const resolution = selectResolution.value;

    const formData = new FormData();
    formData.append('preset_id', presetId);
    formData.append('resolution', resolution);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            handlePredictionResult(data);
        }
    } catch (err) {
        console.error('Prediction failed:', err);
    } finally {
        showLoader(false);
    }
}

async function uploadCustomImage(file) {
    activePresetId = null;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    showLoader(true);
    const resolution = selectResolution.value;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('resolution', resolution);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            handlePredictionResult(data);
        }
    } catch (err) {
        console.error('Upload prediction failed:', err);
    } finally {
        showLoader(false);
    }
}

function handlePredictionResult(data) {
    currentData = data;

    // 1. Update Telemetry HUD
    hudLatency.textContent = `${data.inference_time_ms} ms`;
    hudFps.textContent = `${data.fps} FPS`;

    // 2. Update Perception View
    updatePerceptionView();

    // 3. Update 3D WebGL Point Cloud
    if (viewer3D && data.point_cloud) {
        viewer3D.updateData(data.point_cloud);
    }

    // 4. Update Analytics & Hazard Rangefinder
    updateAnalytics(data.stats);
}

// ─────────────────────────────────────────────────────────────────────────────
// Perception View & Split Slider Management
// ─────────────────────────────────────────────────────────────────────────────
function updatePerceptionView() {
    if (!currentData || !currentData.images) return;

    const splitControls = document.getElementById('split-controls');

    if (currentView === 'split') {
        splitControls.style.display = 'flex';
        imgPrimary.src = currentData.images.rgb;
        imgSecondary.src = currentData.images.blend_seg;
        imgSecondary.style.display = 'block';
        splitDivider.style.display = 'block';
        setSplitPosition(sliderSplit.value);
    } else {
        splitControls.style.display = 'none';
        imgSecondary.style.display = 'none';
        splitDivider.style.display = 'none';

        if (currentView === 'seg') imgPrimary.src = currentData.images.segmentation;
        else if (currentView === 'depth') imgPrimary.src = currentData.images.depth;
        else if (currentView === 'hazard') imgPrimary.src = currentData.images.hazard;
        else if (currentView === 'rgb') imgPrimary.src = currentData.images.rgb;
    }
}

function setSplitPosition(percent) {
    imgSecondary.style.clipPath = `polygon(0 0, ${percent}% 0, ${percent}% 100%, 0 100%)`;
    splitDivider.style.left = `${percent}%`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive Hover Distance & Semantic Prober
// ─────────────────────────────────────────────────────────────────────────────
function handleViewportHover(e) {
    if (!currentData || !currentData.probe) return;

    const rect = viewportContainer.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, mouseX / rect.width));
    const normY = Math.max(0, Math.min(1, mouseY / rect.height));

    const gridW = currentData.probe.grid_w;
    const gridH = currentData.probe.grid_h;

    const cellX = Math.floor(normX * (gridW - 1));
    const cellY = Math.floor(normY * (gridH - 1));

    const classId = currentData.probe.classes[cellY][cellX];
    const distanceM = currentData.probe.depth[cellY][cellX];

    const meta = getClassInfo(classId);
    const safety = getSafetyStatus(distanceM, meta.isDynamic);

    // Update tooltip content
    probeColorDot.style.backgroundColor = `rgb(${meta.color.join(',')})`;
    probeClassName.textContent = meta.name;
    probeCategory.textContent = meta.category;
    probeDistance.textContent = `${distanceM.toFixed(1)} m`;
    probeSafety.textContent = safety.text;
    probeSafety.className = `val ${safety.class}`;

    // Position tooltip
    probeTooltip.style.display = 'block';
    const tipW = 180;
    const tipH = 90;
    let left = mouseX + 15;
    let top = mouseY + 15;

    if (left + tipW > rect.width) left = mouseX - tipW - 15;
    if (top + tipH > rect.height) top = mouseY - tipH - 15;

    probeTooltip.style.left = `${left}px`;
    probeTooltip.style.top = `${top}px`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics & Radar Dashboard
// ─────────────────────────────────────────────────────────────────────────────
function updateAnalytics(stats) {
    if (!stats) return;

    // Rangefinder Stats
    document.getElementById('stat-min-dist').textContent = `${stats.depth.min_distance_m} m`;
    document.getElementById('stat-road-dist').textContent = `${stats.depth.road_distance_m} m`;
    document.getElementById('stat-max-dist').textContent = `${stats.depth.max_distance_m} m`;
    document.getElementById('stat-hazards').textContent = `${stats.depth.obstacles_under_10m} items`;

    // Hazard Progress Bar
    const hazardScore = Math.min(100, stats.depth.obstacles_under_10m * 25);
    document.getElementById('hazard-bar-fill').style.width = `${hazardScore}%`;

    // 19-Class Semantic Composition Bars
    const listContainer = document.getElementById('class-dist-list');
    listContainer.innerHTML = '';

    stats.classes.slice(0, 6).forEach(c => {
        const row = document.createElement('div');
        row.className = 'class-row';
        row.innerHTML = `
            <span class="c-dot" style="background-color: ${c.color}"></span>
            <span class="c-name" title="${c.name}">${c.name}</span>
            <div class="c-bar-wrapper">
                <div class="c-bar-fill" style="width: ${c.percentage}%; background-color: ${c.color}"></div>
            </div>
            <span class="c-pct">${c.percentage.toFixed(1)}%</span>
        `;
        listContainer.appendChild(row);
    });
}

function showLoader(show) {
    if (show) stageLoader.classList.remove('hidden');
    else stageLoader.classList.add('hidden');
}
