// Contextor Dashboard JavaScript

const API_BASE = '';
let refreshInterval = null;
let isConnected = false;

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    refreshAll();
    startAutoRefresh();
});

function initEventListeners() {
    // Buttons
    document.getElementById('btn-generate').addEventListener('click', generateContext);
    document.getElementById('btn-refresh').addEventListener('click', refreshAll);
    document.getElementById('btn-cleanup').addEventListener('click', runCleanup);

    // Modal
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('image-modal').addEventListener('click', (e) => {
        if (e.target.id === 'image-modal') closeModal();
    });

    // Keyboard
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

function startAutoRefresh() {
    // Refresh every 5 seconds
    refreshInterval = setInterval(refreshAll, 5000);
}

// ==================== API Calls ====================

async function apiCall(endpoint, method = 'GET') {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, { method });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// ==================== Refresh Functions ====================

async function refreshAll() {
    try {
        await Promise.all([
            refreshStatus(),
            refreshLatestContext(),
            refreshImages(),
            refreshTranscripts(),
            refreshHistory(),
            refreshStorage()
        ]);
        setConnectionStatus(true);
    } catch (error) {
        setConnectionStatus(false);
    }
}

async function refreshStatus() {
    try {
        const data = await apiCall('/api/status');
        updateStatusPanel(data);
    } catch (error) {
        updateStatusPanel(null);
    }
}

async function refreshLatestContext() {
    try {
        const response = await apiCall('/api/context/latest');
        if (response.success && response.data) {
            updateContextPanel(response.data);
        }
    } catch (error) {
        console.error('Failed to load context:', error);
    }
}

async function refreshImages() {
    try {
        const response = await apiCall('/api/images/recent?limit=12');
        if (response.success) {
            updateImagesPanel(response.data);
        }
    } catch (error) {
        console.error('Failed to load images:', error);
    }
}

async function refreshTranscripts() {
    try {
        const response = await apiCall('/api/transcripts/recent');
        if (response.success) {
            updateTranscriptsPanel(response.data);
        }
    } catch (error) {
        console.error('Failed to load transcripts:', error);
    }
}

async function refreshHistory() {
    try {
        const response = await apiCall('/api/context/history?limit=10');
        if (response.success) {
            updateHistoryPanel(response.data);
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

async function refreshStorage() {
    try {
        const response = await apiCall('/api/storage/stats');
        if (response.success) {
            updateStoragePanel(response.data);
        }
    } catch (error) {
        console.error('Failed to load storage stats:', error);
    }
}

// ==================== Update Functions ====================

function setConnectionStatus(connected) {
    isConnected = connected;
    const indicator = document.getElementById('status-indicator');
    const dot = indicator.querySelector('.status-dot');
    const text = indicator.querySelector('.status-text');

    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Connected';
    } else {
        dot.className = 'status-dot error';
        text.textContent = 'Disconnected';
    }
}

function updateStatusPanel(data) {
    if (!data) {
        document.getElementById('uptime').textContent = '--';
        document.getElementById('audio-status').textContent = '--';
        document.getElementById('camera-status').textContent = '--';
        document.getElementById('storage-status').textContent = '--';
        return;
    }

    // Uptime
    const uptime = formatUptime(data.uptime_seconds);
    document.getElementById('uptime').textContent = uptime;

    // Audio
    const audioEl = document.getElementById('audio-status');
    if (data.audio.capture_running) {
        audioEl.textContent = data.audio.is_speaking ? 'Speaking' : 'Listening';
        audioEl.className = 'status-value good';
    } else {
        audioEl.textContent = 'Stopped';
        audioEl.className = 'status-value error';
    }

    // Camera
    const cameraEl = document.getElementById('camera-status');
    if (data.vision.camera_running) {
        cameraEl.textContent = `${data.vision.frame_count} frames`;
        cameraEl.className = 'status-value good';
    } else {
        cameraEl.textContent = 'Stopped';
        cameraEl.className = 'status-value error';
    }

    // Storage
    const storageEl = document.getElementById('storage-status');
    const usage = data.storage.usage_percent;
    storageEl.textContent = `${usage}%`;
    storageEl.className = usage > 80 ? 'status-value warning' : 'status-value good';
}

function updateContextPanel(context) {
    // Meta
    document.getElementById('context-time').textContent = formatTime(context.generated_at);
    document.getElementById('context-period').textContent =
        `${Math.round(context.period.duration_seconds / 60)} min`;

    // Transcript
    const transcriptBox = document.getElementById('transcript-box');
    if (context.audio.full_transcript) {
        transcriptBox.innerHTML = `<p>${escapeHtml(context.audio.full_transcript)}</p>`;
    } else {
        transcriptBox.innerHTML = '<p class="placeholder">No speech detected...</p>';
    }

    // Objects
    const objectsGrid = document.getElementById('objects-grid');
    const objects = context.vision.objects_detected;
    if (Object.keys(objects).length > 0) {
        objectsGrid.innerHTML = Object.entries(objects)
            .sort((a, b) => b[1].count - a[1].count)
            .map(([label, data]) => `
                <div class="object-tag">
                    ${escapeHtml(label)}
                    <span class="object-count">${data.count}</span>
                </div>
            `).join('');
    } else {
        objectsGrid.innerHTML = '<p class="placeholder">No objects detected...</p>';
    }

    // Summary
    const summaryBox = document.getElementById('summary-box');
    summaryBox.innerHTML = `<p>${escapeHtml(context.summary)}</p>`;
}

function updateImagesPanel(images) {
    const grid = document.getElementById('images-grid');

    if (images.length === 0) {
        grid.innerHTML = '<p class="placeholder">No images captured yet...</p>';
        return;
    }

    grid.innerHTML = images.map(img => `
        <div class="image-thumb" onclick="openImageModal('${img.url}', '${escapeHtml(img.filename)}')">
            <img src="${img.url}" alt="${escapeHtml(img.filename)}" loading="lazy">
            <div class="image-label">${formatTime(img.modified)}</div>
        </div>
    `).join('');
}

function updateTranscriptsPanel(transcripts) {
    const list = document.getElementById('transcripts-list');

    if (transcripts.length === 0) {
        list.innerHTML = '<p class="placeholder">Waiting for speech...</p>';
        return;
    }

    list.innerHTML = transcripts.slice().reverse().map(t => `
        <div class="transcript-item">
            <div class="text">${escapeHtml(t.text)}</div>
            <div class="meta">
                <span>${formatTime(t.timestamp)}</span>
                <span>${Math.round(t.confidence * 100)}% conf</span>
            </div>
        </div>
    `).join('');
}

function updateHistoryPanel(history) {
    const list = document.getElementById('history-list');

    if (history.length === 0) {
        list.innerHTML = '<p class="placeholder">No history yet...</p>';
        return;
    }

    list.innerHTML = history.map(item => `
        <div class="history-item" onclick="loadContextFile('${escapeHtml(item.filename)}')">
            <span class="filename">${escapeHtml(item.filename)}</span>
            <span class="time">${formatTime(item.modified)}</span>
        </div>
    `).join('');
}

function updateStoragePanel(stats) {
    // Bar
    const bar = document.getElementById('storage-bar-used');
    bar.style.width = `${stats.usage_percent}%`;

    // Details
    const details = document.getElementById('storage-details');
    details.innerHTML = `
        <p>Context files: ${stats.context_files} (${stats.context_size_mb} MB)</p>
        <p>Images: ${stats.image_files} (${stats.images_size_mb} MB)</p>
        <p>Total: ${stats.total_size_mb} MB / ${stats.max_storage_gb * 1024} MB</p>
    `;
}

// ==================== Actions ====================

async function generateContext() {
    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    btn.textContent = 'Generating...';

    try {
        const response = await apiCall('/api/context/generate', 'POST');
        if (response.success) {
            showNotification('Context generated successfully');
            await refreshAll();
        }
    } catch (error) {
        showNotification('Failed to generate context', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Context Now';
    }
}

async function runCleanup() {
    const btn = document.getElementById('btn-cleanup');
    btn.disabled = true;
    btn.textContent = 'Cleaning...';

    try {
        const response = await apiCall('/api/storage/cleanup', 'POST');
        if (response.success) {
            showNotification(`Cleaned up ${response.data.deleted_count} files`);
            await refreshStorage();
        }
    } catch (error) {
        showNotification('Cleanup failed', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Cleanup';
    }
}

async function loadContextFile(filename) {
    try {
        const response = await apiCall(`/api/context/file/${filename}`);
        if (response.success) {
            updateContextPanel(response.data);
            showNotification(`Loaded: ${filename}`);
        }
    } catch (error) {
        showNotification('Failed to load file', 'error');
    }
}

// ==================== Modal ====================

function openImageModal(url, filename) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-image');
    const info = document.getElementById('modal-info');

    img.src = url;
    info.textContent = filename;
    modal.classList.add('active');
}

function closeModal() {
    document.getElementById('image-modal').classList.remove('active');
}

// ==================== Utilities ====================

function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
}

function formatTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message, type = 'success') {
    // Simple console notification for now
    console.log(`[${type.toUpperCase()}] ${message}`);

    // Could add toast notification here
}
