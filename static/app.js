// =====================================================
// API BASE URL CONFIGURATION
// - Local dev: FastAPI serves both frontend and API on same origin â†’ use relative paths ""
// - Production (Vercel frontend + Render backend): use absolute Render URL
// After deploying to Render, replace the URL below with your actual Render URL
// =====================================================
const RENDER_BACKEND_URL = "https://soil-test-api.onrender.com"; // â† UPDATE THIS after Render deploy
const IS_LOCAL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const API_BASE = IS_LOCAL ? "" : RENDER_BACKEND_URL;

// Global state variables
let selectedFile = null;
let activeCMData = null;
let trainingPollInterval = null;
let rocChart = null;
let lossChart = null;
let accChart = null;
let liveChart = null;
let availableClasses = [];
let uploadedClassesInSession = [];
let sessionUploadSummary = {};
let lastSpeechText = "";

// Init on load
document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    checkModelStatus();
    loadClasses();
    loadSplitStatus();
    checkActiveTraining();
    
    // Clear file inputs on page load to prevent browser caching/persistence
    const trainFiles = document.getElementById('train-files-input');
    if (trainFiles) trainFiles.value = '';
    const mainFile = document.getElementById('file-input');
    if (mainFile) mainFile.value = '';
    
    // Load theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        updateThemeUI(true);
    }
});

// Toast Helper
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMsg = document.getElementById('toast-message');
    
    toastMsg.textContent = message;
    toast.className = `toast show ${type}`;
    
    setTimeout(() => {
        toast.className = 'toast';
    }, 4000);
}

// Tab Navigation
function switchTab(tabId) {
    document.querySelectorAll('.tab-pane').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    
    document.getElementById(`tab-${tabId}`).classList.add('active');
    document.getElementById(`nav-${tabId}`).classList.add('active');
}

function switchAdminSubtab(subtabId) {
    document.querySelectorAll('.admin-subtab-pane').forEach(pane => pane.classList.remove('active'));
    document.querySelectorAll('.admin-tab').forEach(tab => tab.classList.remove('active'));
    
    document.getElementById(`admin-subtab-${subtabId}`).classList.add('active');
    
    if (subtabId === 'config') document.getElementById('btn-admin-config').classList.add('active');
    if (subtabId === 'train') document.getElementById('btn-admin-train').classList.add('active');
    if (subtabId === 'metrics') {
        document.getElementById('btn-admin-metrics').classList.add('active');
        loadMetricsReports();
    }
}

// Drag & Drop Setup
function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    
    // Prevent defaults
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, e => e.preventDefault(), false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('hover'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('hover'), false);
    });
    
    dropzone.addEventListener('drop', e => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleFile(files[0]);
    });
    
    fileInput.addEventListener('change', e => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });
    
    dropzone.addEventListener('click', e => {
        // Only click if prompt is visible
        if (e.target.closest('.dropzone-prompt') || e.target === dropzone) {
            fileInput.click();
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file.', 'error');
        return;
    }
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('dropzone-preview').style.display = 'block';
        document.querySelector('.dropzone-prompt').style.display = 'none';
        document.getElementById('btn-analyze').removeAttribute('disabled');
    };
    reader.readAsDataURL(file);
}

function clearUpload(e) {
    if (e) e.stopPropagation();
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('image-preview').src = '#';
    document.getElementById('dropzone-preview').style.display = 'none';
    document.querySelector('.dropzone-prompt').style.display = 'flex';
    document.getElementById('btn-analyze').setAttribute('disabled', 'true');
    
    // Clear predictions view
    document.getElementById('result-empty-state').style.display = 'flex';
    document.getElementById('result-content').style.display = 'none';
}

// API: Check model status
async function checkModelStatus() {
    try {
        const res = await fetch(API_BASE + '/api/metrics');
        const statusDot = document.querySelector('.status-indicator .status-dot');
        const statusText = document.getElementById('model-status-text');
        
        if (res.ok) {
            statusDot.className = 'status-dot green';
            statusText.textContent = 'Active Model Loaded';
        } else {
            statusDot.className = 'status-dot orange';
            statusText.textContent = 'No Model Trained';
        }
    } catch {
        // Server could be starting
    }
}

// API: Load Classes
async function loadClasses() {
    try {
        const res = await fetch(API_BASE + '/api/classes');
        const data = await res.json();
        availableClasses = data.classes;
        
        const container = document.getElementById('class-checkboxes-container');
        container.innerHTML = '';
        
        availableClasses.forEach(cls => {
            const item = document.createElement('label');
            item.className = 'class-checkbox-item';
            
            const chk = document.createElement('input');
            chk.type = 'checkbox';
            chk.value = cls;
            chk.checked = true;
            chk.className = 'class-select-chk';
            
            const span = document.createElement('span');
            span.textContent = cls;
            
            item.appendChild(chk);
            item.appendChild(span);
            container.appendChild(item);
        });

        // Populate the upload dropdown selector
        const select = document.getElementById('upload-class-select');
        if (select) {
            select.innerHTML = '<option value="" disabled selected>-- Select a category --</option>';
            
            // Filter classes already uploaded in this session
            const remainingClasses = availableClasses.filter(c => !uploadedClassesInSession.includes(c));
            
            remainingClasses.forEach(cls => {
                const opt = document.createElement('option');
                opt.value = cls;
                opt.textContent = cls;
                select.appendChild(opt);
            });
            const optNew = document.createElement('option');
            optNew.value = '__new__';
            optNew.textContent = '[+] Register New Soil Category...';
            select.appendChild(optNew);
            
            if (remainingClasses.length === 0) {
                select.innerHTML = '<option value="" disabled selected>All categories updated in this session</option>';
            }
        }
    } catch (err) {
        showToast('Failed to load soil classes from dataset.', 'error');
    }
}

// API: Load Dataset Split Status
async function loadSplitStatus() {
    try {
        const res = await fetch(API_BASE + '/api/split-status');
        const data = await res.json();
        
        const tableBody = document.getElementById('split-table-body');
        
        if (!data.split_done) {
            tableBody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No data split calculated yet. Execute partitioning above.</td></tr>`;
            document.getElementById('split-total-classes').textContent = '0';
            document.getElementById('split-total-images').textContent = '0';
            return;
        }
        
        tableBody.innerHTML = '';
        let totalClasses = 0;
        let totalImages = 0;
        
        for (const [cls, counts] of Object.entries(data.counts)) {
            totalClasses++;
            const classTotal = counts.train + counts.val + counts.test;
            totalImages += classTotal;
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${cls}</strong></td>
                <td>${classTotal}</td>
                <td>${counts.train}</td>
                <td>${counts.val}</td>
                <td>${counts.test}</td>
            `;
            tableBody.appendChild(tr);
        }
        
        document.getElementById('split-total-classes').textContent = totalClasses;
        document.getElementById('split-total-images').textContent = totalImages;
    } catch {
        showToast('Error loading split metrics.', 'error');
    }
}

// API: Run Splitting
async function runDatasetSplit() {
    try {
        showToast('Splitting dataset... Please wait.', 'success');
        const res = await fetch(API_BASE + '/api/split', { method: 'POST' });
        const data = await res.json();
        
        if (data.status === 'success') {
            showToast('Dataset successfully split into Train/Val/Test directories!', 'success');
            loadSplitStatus();
        } else {
            showToast('Split failed: ' + data.detail, 'error');
        }
    } catch {
        showToast('Error connecting to split API.', 'error');
    }
}

// API: Trigger Classification Analysis
async function analyzeSoil() {
    if (!selectedFile) return;
    
    const btn = document.getElementById('btn-analyze');
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span class="status-dot green animate-pulse"></span> Analyzing...`;
    btn.setAttribute('disabled', 'true');
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const res = await fetch(API_BASE + '/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) {
            const data = await res.json();
            showToast(data.detail || 'Prediction failed.', 'error');
            return;
        }
        
        const data = await res.json();
        
        // Render Output Panel
        document.getElementById('result-empty-state').style.display = 'none';
        document.getElementById('result-content').style.display = 'block';
        
        // Load details
        document.getElementById('result-class-name').textContent = data.predicted_class;
        document.getElementById('result-confidence-text').textContent = `${data.confidence.toFixed(1)}%`;
        
        // Set radial progress dasharray
        const ring = document.getElementById('result-ring');
        ring.setAttribute('stroke-dasharray', `${data.confidence.toFixed(0)}, 100`);
        
        document.getElementById('result-desc').textContent = data.info.description;
        document.getElementById('result-props').textContent = data.info.properties;
        document.getElementById('result-suitability').textContent = data.info.suitability;
        
        // Sort and load probabilities bars
        const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
        const barsContainer = document.getElementById('probabilities-bars-container');
        barsContainer.innerHTML = '';
        
        sortedProbs.forEach(([cls, pct]) => {
            const isBest = cls === data.predicted_class;
            const barItem = document.createElement('div');
            barItem.className = `probability-bar-item ${isBest ? 'best' : ''}`;
            barItem.innerHTML = `
                <div class="probability-bar-labels">
                    <span class="class-name">${cls}</span>
                    <span class="val-pct">${pct.toFixed(1)}%</span>
                </div>
                <div class="probability-track">
                    <div class="probability-fill" style="width: ${pct.toFixed(1)}%"></div>
                </div>
            `;
            barsContainer.appendChild(barItem);
        });
        
        showToast('Classification analysis complete!', 'success');
        
        // Speak result aloud
        const speechText = `The analysis indicates this sample is ${data.predicted_class} with a confidence score of ${data.confidence.toFixed(0)} percent. This soil is suitable for ${data.info.suitability}`;
        lastSpeechText = speechText;
        speakResult(speechText);
    } catch {
        showToast('Server connection error during prediction.', 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.removeAttribute('disabled');
    }
}

// Text-To-Speech helper
function speakResult(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // cancel any active speech
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        
        // Find English voice
        const voices = window.speechSynthesis.getVoices();
        const englishVoice = voices.find(v => v.lang.startsWith('en'));
        if (englishVoice) {
            utterance.voice = englishVoice;
        }
        window.speechSynthesis.speak(utterance);
    }
}

// Replay classification announcement
function replaySpeech() {
    if (lastSpeechText) {
        speakResult(lastSpeechText);
    } else {
        showToast('No classification results to read aloud yet.', 'info');
    }
}

// API: Check for running training at startup
async function checkActiveTraining() {
    try {
        const res = await fetch(API_BASE + '/api/train-status');
        const data = await res.json();
        if (data.status === 'training') {
            startPollingTraining();
        }
    } catch {}
}

// API: Start Training
async function startModelTraining() {
    const selectedChks = document.querySelectorAll('.class-select-chk:checked');
    if (selectedChks.length < 2) {
        showToast('Please select at least 2 soil classes for training.', 'error');
        return;
    }
    
    const classes = Array.from(selectedChks).map(chk => chk.value);
    const epochs = parseInt(document.getElementById('param-epochs').value) || 10;
    const batchSize = parseInt(document.getElementById('param-batch-size').value) || 32;
    const lr = parseFloat(document.getElementById('param-lr').value) || 0.001;
    
    const btn = document.getElementById('btn-start-training');
    btn.setAttribute('disabled', 'true');
    btn.textContent = 'Starting training...';
    
    const terminal = document.getElementById('training-terminal-logs');
    if (terminal) terminal.innerHTML = '<div class="log-line system">[SYSTEM] Connecting to training pipeline...</div>';
    
    try {
        const res = await fetch(API_BASE + '/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ classes, epochs, batch_size: batchSize, learning_rate: lr })
        });
        
        if (!res.ok) {
            const data = await res.json();
            showToast(data.detail || 'Failed to start training.', 'error');
            btn.removeAttribute('disabled');
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg> Initialize Model Training`;
            return;
        }
        
        showToast('Model training initialized in the background.', 'success');
        
        // Reset session upload states on successful training initiation
        uploadedClassesInSession = [];
        sessionUploadSummary = {};
        resetUploadForm();
        
        startPollingTraining();
    } catch {
        showToast('Error launching training thread.', 'error');
        btn.removeAttribute('disabled');
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg> Initialize Model Training`;
    }
}

// Start polling training status
function startPollingTraining() {
    document.getElementById('training-monitor-idle').style.display = 'none';
    document.getElementById('training-monitor-active').style.display = 'block';
    document.getElementById('btn-start-training').setAttribute('disabled', 'true');
    
    // Clear live chart if exists
    if (liveChart) {
        liveChart.destroy();
        liveChart = null;
    }
    
    pollTrainingStatus();
    trainingPollInterval = setInterval(pollTrainingStatus, 1500);
}

// Poll status
async function pollTrainingStatus() {
    try {
        const res = await fetch(API_BASE + '/api/train-status');
        const data = await res.json();
        
        if (data.status === 'idle') {
            clearInterval(trainingPollInterval);
            resetTrainingForm();
            return;
        }
        
        if (data.status === 'failed') {
            clearInterval(trainingPollInterval);
            showToast('Training failed: ' + data.error, 'error');
            resetTrainingForm();
            return;
        }
        
        if (data.status === 'completed') {
            clearInterval(trainingPollInterval);
            showToast('Training successfully completed! Model updated.', 'success');
            checkModelStatus();
            resetTrainingForm();
            
            // Auto switch to reports
            switchAdminSubtab('metrics');
            return;
        }
        
        // Active status training details update
        document.getElementById('training-progress-pct').textContent = `${data.progress}%`;
        document.getElementById('training-progress-fill').style.width = `${data.progress}%`;
        
        const metrics = data.metrics || {};
        document.getElementById('monitor-kpi-epoch').textContent = `${data.current_epoch}/${data.total_epochs}`;
        document.getElementById('monitor-kpi-loss').textContent = metrics.loss ? metrics.loss.toFixed(4) : '-';
        document.getElementById('monitor-kpi-acc').textContent = metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : '-';
        document.getElementById('monitor-kpi-vloss').textContent = metrics.val_loss ? metrics.val_loss.toFixed(4) : '-';
        document.getElementById('monitor-kpi-vacc').textContent = metrics.val_accuracy ? `${(metrics.val_accuracy * 100).toFixed(1)}%` : '-';
        
        // Update terminal logs
        if (data.logs && data.logs.length > 0) {
            updateTerminalLogs(data.logs);
        }
        
        // Update live charts
        if (data.history && data.history.length > 0) {
            updateLiveChart(data.history);
        }
    } catch {
        // Fail silently temporarily
    }
}

function resetTrainingForm() {
    const btn = document.getElementById('btn-start-training');
    btn.removeAttribute('disabled');
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg> Initialize Model Training`;
    
    document.getElementById('training-monitor-idle').style.display = 'block';
    document.getElementById('training-monitor-active').style.display = 'none';
}

function updateLiveChart(history) {
    const epochs = history.map(h => h.epoch);
    const loss = history.map(h => h.loss);
    const accuracy = history.map(h => h.accuracy * 100);
    const val_loss = history.map(h => h.val_loss);
    const val_accuracy = history.map(h => h.val_accuracy * 100);
    
    if (liveChart) {
        liveChart.data.labels = epochs;
        liveChart.data.datasets[0].data = accuracy;
        liveChart.data.datasets[1].data = val_accuracy;
        liveChart.update();
    } else {
        const ctx = document.getElementById('live-training-chart').getContext('2d');
        liveChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Train Acc (%)',
                        data: accuracy,
                        borderColor: '#10b981',
                        borderWidth: 2,
                        tension: 0.2,
                        fill: false
                    },
                    {
                        label: 'Val Acc (%)',
                        data: val_accuracy,
                        borderColor: '#d97706',
                        borderWidth: 2,
                        tension: 0.2,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    }
}

// API: Load Performance Reports Tab
async function loadMetricsReports() {
    try {
        const res = await fetch(API_BASE + '/api/metrics');
        if (!res.ok) {
            // No model metrics found, show empty state or hide report blocks
            showToast('Trained model evaluation metrics not available yet.', 'error');
            return;
        }
        
        const data = await res.json();
        activeCMData = data.confusion_matrix;
        
        // Fill class badges
        const badgesContainer = document.getElementById('metrics-classes-pills');
        badgesContainer.innerHTML = '';
        data.confusion_matrix.classes.forEach(cls => {
            const span = document.createElement('span');
            span.className = 'class-pill';
            span.textContent = cls;
            badgesContainer.appendChild(span);
        });
        
        // KPIs
        document.getElementById('kpi-accuracy').textContent = `${(data.overall.accuracy * 100).toFixed(2)}%`;
        document.getElementById('kpi-precision').textContent = `${(data.overall.precision * 100).toFixed(2)}%`;
        document.getElementById('kpi-recall').textContent = `${(data.overall.recall * 100).toFixed(2)}%`;
        document.getElementById('kpi-f1').textContent = `${(data.overall.f1_score * 100).toFixed(2)}%`;
        
        // Render Heatmap Matrix
        renderConfusionMatrix(data.confusion_matrix);
        
        // Render ROC Curve Chart
        renderROCChart(data.roc_auc);
        
        // Render History Charts
        renderHistoryCharts(data.training_history);
    } catch {
        showToast('Error importing evaluation reports.', 'error');
    }
}

// Render Confusion Matrix
function renderConfusionMatrix(cm) {
    const container = document.getElementById('confusion-matrix-heatmap');
    container.innerHTML = '';
    
    const classes = cm.classes;
    const matrix = cm.matrix;
    const size = classes.length;
    
    // Set grid columns template dynamically
    container.style.gridTemplateColumns = `repeat(${size + 1}, 1fr)`;
    
    // 1. Corner cell
    const corner = document.createElement('div');
    corner.className = 'cm-corner';
    corner.innerHTML = '<span>True \\ Pred</span>';
    container.appendChild(corner);
    
    // 2. Col headers
    classes.forEach(cls => {
        const colHeader = document.createElement('div');
        colHeader.className = 'cm-header-label';
        colHeader.textContent = getShortName(cls);
        colHeader.title = cls;
        container.appendChild(colHeader);
    });
    
    // 3. Rows
    for (let r = 0; r < size; r++) {
        // Row label
        const rowHeader = document.createElement('div');
        rowHeader.className = 'cm-header-label';
        rowHeader.textContent = getShortName(classes[r]);
        rowHeader.title = classes[r];
        container.appendChild(rowHeader);
        
        // Row values
        const rowSum = matrix[r].reduce((a, b) => a + b, 0);
        
        for (let c = 0; c < size; c++) {
            const val = matrix[r][c];
            const pct = rowSum > 0 ? val / rowSum : 0;
            const isDiagonal = r === c;
            
            const cell = document.createElement('div');
            cell.className = 'cm-cell';
            
            // Heatmap color scaling
            if (isDiagonal) {
                // Diagonal: Matches (emerald green)
                cell.style.backgroundColor = `rgba(16, 185, 129, ${Math.max(0.1, pct * 0.9)})`;
            } else {
                // Off-diagonal: Errors (coral red)
                cell.style.backgroundColor = val > 0 ? `rgba(239, 68, 68, ${Math.max(0.15, pct * 1.0)})` : 'rgba(255, 255, 255, 0.01)';
            }
            
            cell.innerHTML = `
                <span class="cell-value">${val}</span>
                <span class="cell-pct">${(pct * 100).toFixed(0)}%</span>
            `;
            cell.title = `True: ${classes[r]}, Pred: ${classes[c]} (${val} samples)`;
            container.appendChild(cell);
        }
    }
}

// Shorten class name for confusion matrix labels
function getShortName(name) {
    return name.replace(' Soil', '');
}

// Render ROC Chart
function renderROCChart(rocData) {
    if (rocChart) rocChart.destroy();
    
    const datasets = [];
    const colors = ['#10b981', '#d97706', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'];
    let idx = 0;
    
    for (const [cls, curve] of Object.entries(rocData)) {
        // Map downsampled points to xy points
        const chartPoints = curve.fpr.map((fpr, i) => ({ x: fpr, y: curve.tpr[i] }));
        
        datasets.push({
            label: `${cls} (AUC: ${curve.auc.toFixed(3)})`,
            data: chartPoints,
            borderColor: colors[idx % colors.length],
            borderWidth: 2,
            tension: 0.1,
            fill: false,
            pointRadius: 1,
            pointHoverRadius: 4
        });
        idx++;
    }
    
    // Add reference line
    datasets.push({
        label: 'Random Guess (AUC: 0.500)',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: '#4b5563',
        borderWidth: 1.5,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0
    });
    
    const ctx = document.getElementById('roc-chart').getContext('2d');
    rocChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'False Positive Rate (FPR)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    title: { display: true, text: 'True Positive Rate (TPR)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Render Historical Curves
function renderHistoryCharts(history) {
    if (lossChart) lossChart.destroy();
    if (accChart) accChart.destroy();
    
    const epochs = history.epochs;
    
    // Loss Chart
    const ctxLoss = document.getElementById('history-loss-chart').getContext('2d');
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: history.loss,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: false
                },
                {
                    label: 'Validation Loss',
                    data: history.val_loss,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
            }
        }
    });
    
    // Accuracy Chart
    const ctxAcc = document.getElementById('history-acc-chart').getContext('2d');
    accChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: history.accuracy.map(a => a * 100),
                    borderColor: '#10b981',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: false
                },
                {
                    label: 'Validation Accuracy',
                    data: history.val_accuracy.map(a => a * 100),
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

// Image Download Helpers
function downloadChart(canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const ctx = tempCanvas.getContext('2d');
    
    // Fill slate background matching panel
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    ctx.drawImage(canvas, 0, 0);
    
    const link = document.createElement('a');
    link.download = filename;
    link.href = tempCanvas.toDataURL('image/png');
    link.click();
}

function downloadConfusionMatrix() {
    if (!activeCMData) {
        showToast("No confusion matrix data loaded.", "error");
        return;
    }
    
    const classes = activeCMData.classes;
    const matrix = activeCMData.matrix;
    const size = classes.length;
    
    const cellSize = 65;
    const padding = 100;
    const width = size * cellSize + padding + 40;
    const height = size * cellSize + padding + 40;
    
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    // Panel background
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, width, height);
    
    // Title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 15px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Confusion Matrix Heatmap', width / 2, 30);
    
    ctx.font = '10px Outfit, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('True Class (Rows) vs Predicted Class (Cols)', width / 2, 50);
    
    const offset = padding;
    
    // Headers
    ctx.font = 'bold 10px Outfit, sans-serif';
    ctx.textAlign = 'center';
    
    for (let i = 0; i < size; i++) {
        const shortName = getShortName(classes[i]);
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(shortName, offset + (i * cellSize) + (cellSize / 2), offset - 20);
        
        ctx.textAlign = 'right';
        ctx.fillText(shortName, offset - 20, offset + (i * cellSize) + (cellSize / 2));
        ctx.textAlign = 'center';
    }
    
    // Grid cells
    for (let r = 0; r < size; r++) {
        const rowSum = matrix[r].reduce((a, b) => a + b, 0);
        for (let c = 0; c < size; c++) {
            const val = matrix[r][c];
            const pct = rowSum > 0 ? val / rowSum : 0;
            const isDiagonal = r === c;
            
            const x = offset + c * cellSize;
            const y = offset + r * cellSize;
            
            if (isDiagonal) {
                const intensity = Math.max(0.1, pct * 0.9);
                ctx.fillStyle = `rgba(16, 185, 129, ${intensity})`;
            } else {
                const intensity = Math.max(0.15, pct * 1.0);
                ctx.fillStyle = val > 0 ? `rgba(239, 68, 68, ${intensity})` : 'rgba(255, 255, 255, 0.02)';
            }
            
            ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 13px Outfit, sans-serif';
            ctx.fillText(val.toString(), x + cellSize / 2, y + cellSize / 2 - 6);
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
            ctx.font = '9px Outfit, sans-serif';
            ctx.fillText(`${(pct * 100).toFixed(0)}%`, x + cellSize / 2, y + cellSize / 2 + 8);
        }
    }
    
    const link = document.createElement('a');
    link.download = 'confusion_matrix.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
}

function updateTerminalLogs(logs) {
    const terminal = document.getElementById('training-terminal-logs');
    if (!terminal) return;
    
    const currentLinesCount = terminal.querySelectorAll('.log-line').length;
    if (currentLinesCount === logs.length) return; // Prevent unnecessary DOM reflows
    
    terminal.innerHTML = '';
    logs.forEach(line => {
        const div = document.createElement('div');
        div.className = 'log-line';
        
        if (line.startsWith('[INFO]')) div.className += ' info';
        else if (line.startsWith('[SUCCESS]')) div.className += ' success';
        else if (line.startsWith('[EPOCH]')) div.className += ' epoch';
        else if (line.startsWith('[METRIC]')) div.className += ' metric';
        else if (line.startsWith('[ERROR]')) div.className += ' error';
        else if (line.startsWith('[SYSTEM]')) div.className += ' system';
        
        div.textContent = line;
        terminal.appendChild(div);
    });
    
    // Auto-scroll to bottom
    terminal.scrollTop = terminal.scrollHeight;
}

// Theme Toggle Logic
function toggleTheme() {
    const body = document.body;
    body.classList.toggle('light-theme');
    
    const isLight = body.classList.contains('light-theme');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    
    updateThemeUI(isLight);
}

function updateThemeUI(isLight) {
    const themeText = document.getElementById('theme-text');
    const toggleBtn = document.getElementById('theme-toggle');
    if (!toggleBtn) return;
    
    if (isLight) {
        if (themeText) themeText.textContent = 'Dark Mode';
        toggleBtn.querySelector('svg').innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    } else {
        if (themeText) themeText.textContent = 'Light Mode';
        toggleBtn.querySelector('svg').innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
    }
    
    updateChartColors(isLight);
}

function updateChartColors(isLight) {
    const textColor = isLight ? '#1f2937' : '#94a3b8';
    const gridColor = isLight ? 'rgba(31, 41, 55, 0.08)' : 'rgba(255, 255, 255, 0.05)';
    
    const charts = [liveChart, rocChart, lossChart, accChart];
    charts.forEach(chart => {
        if (chart) {
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.y.grid.color = gridColor;
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.y.ticks.color = textColor;
            if (chart.options.plugins.legend.labels) {
                chart.options.plugins.legend.labels.color = textColor;
            }
            chart.update();
        }
    });
}

// Training Image Upload Handlers
function toggleCustomClassInput(select) {
    const customGroup = document.getElementById('custom-class-group');
    if (select.value === '__new__') {
        customGroup.style.display = 'block';
        document.getElementById('upload-custom-class').focus();
    } else {
        customGroup.style.display = 'none';
        document.getElementById('upload-custom-class').value = '';
    }
}

async function uploadTrainingImages() {
    const select = document.getElementById('upload-class-select');
    let className = select.value;
    
    if (className === '__new__') {
        className = document.getElementById('upload-custom-class').value.trim();
    }
    
    if (!className) {
        showToast('Please select or specify a soil category.', 'error');
        return;
    }
    
    const fileInput = document.getElementById('train-files-input');
    const files = Array.from(fileInput.files);
    if (files.length === 0) {
        showToast('Please select at least one image file to upload.', 'error');
        return;
    }
    
    const btn = document.getElementById('btn-upload-train-data');
    const originalText = btn.innerHTML;
    btn.setAttribute('disabled', 'true');
    
    const batchSize = 20;
    let uploadedCount = 0;
    
    try {
        for (let i = 0; i < files.length; i += batchSize) {
            const batch = files.slice(i, i + batchSize);
            btn.textContent = `Uploading: ${uploadedCount} / ${files.length} files...`;
            
            const formData = new FormData();
            formData.append('class_name', className);
            batch.forEach(file => {
                formData.append('files', file);
            });
            
            const res = await fetch(API_BASE + '/api/upload-dataset', {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) {
                const err = await res.json();
                showToast(`Batch upload failed: ${err.detail || 'Error'}`, 'error');
                return;
            }
            
            const data = await res.json();
            uploadedCount += data.uploaded_count;
        }
        
        showToast(`Successfully uploaded ${uploadedCount} images to "${className}"!`, 'success');
        
        // Track session uploads
        if (!uploadedClassesInSession.includes(className)) {
            uploadedClassesInSession.push(className);
        }
        sessionUploadSummary[className] = (sessionUploadSummary[className] || 0) + uploadedCount;
        
        // Render cumulative upload summary list
        renderSessionUploadSummary();
        
        // Clear input values to allow next upload
        fileInput.value = '';
        select.value = '';
        const customClassInput = document.getElementById('upload-custom-class');
        if (customClassInput) customClassInput.value = '';
        const customClassGroup = document.getElementById('custom-class-group');
        if (customClassGroup) customClassGroup.style.display = 'none';
        
        // Reload details (which will exclude already uploaded classes from dropdown)
        loadClasses();
        loadSplitStatus();
    } catch {
        showToast('Network error uploading training images.', 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.removeAttribute('disabled');
    }
}

function renderSessionUploadSummary() {
    const summaryContainer = document.getElementById('upload-session-summary');
    const summaryText = document.getElementById('upload-summary-text');
    
    if (!summaryText) return;
    
    const activeClasses = Object.keys(sessionUploadSummary);
    
    if (activeClasses.length === 0) {
        summaryContainer.style.display = 'none';
        summaryText.innerHTML = '';
        return;
    }
    
    summaryContainer.style.display = 'flex';
    summaryText.innerHTML = '<div style="font-weight: 700; margin-bottom: 0.5rem; color: var(--color-text-main); text-align: left; font-size: 0.85rem;">Staged for next training:</div>';
    
    activeClasses.forEach(cls => {
        const count = sessionUploadSummary[cls];
        const row = document.createElement('div');
        row.style.cssText = 'font-size: 0.85rem; font-weight: 500; color: var(--color-accent); display: flex; align-items: center; justify-content: space-between; gap: 0.35rem; margin-bottom: 0.25rem; width: 100%;';
        
        row.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.35rem;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" width="10" height="10" style="color: #10b981;">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
                <span>${cls}: <strong>+${count}</strong> files</span>
            </div>
            <button class="btn btn-danger btn-sm" onclick="removeStagedClass('${cls}')" style="padding: 2px 6px; font-size: 0.65rem; border-radius: 4px; line-height: 1; border: none; background-color: rgba(239, 68, 68, 0.2); color: #f87171; cursor: pointer;">
                Remove
            </button>
        `;
        summaryText.appendChild(row);
    });
}

function removeStagedClass(className) {
    if (sessionUploadSummary[className] !== undefined) {
        delete sessionUploadSummary[className];
    }
    uploadedClassesInSession = uploadedClassesInSession.filter(c => c !== className);
    
    renderSessionUploadSummary();
    loadClasses();
    showToast(`Removed "${className}" from staged upload session.`, 'success');
}

function resetUploadForm() {
    document.getElementById('train-files-input').value = '';
    document.getElementById('upload-class-select').value = '';
    document.getElementById('upload-custom-class').value = '';
    document.getElementById('custom-class-group').style.display = 'none';
    
    document.getElementById('upload-session-summary').style.display = 'none';
    document.getElementById('upload-actions-row').style.display = 'flex';
    
    // Repopulate with remaining classes excluded
    loadClasses();
}

function clearUploadSession() {
    uploadedClassesInSession = [];
    sessionUploadSummary = {};
    renderSessionUploadSummary();
    resetUploadForm();
    showToast('Session upload staging cleared.', 'success');
}


