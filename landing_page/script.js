/**
 * MATE Identifier Landing Page - JavaScript
 * Particle animation and form handling
 */

// === Particle System for Background (Spacetime Fabric) ===
class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.points = [];
        this.springs = [];
        this.mouse = { x: -1000, y: -1000, radius: 190 }; // Reduced influence radius (50%)
        this.time = 0; // Time counter for ambient animation

        // Configuration
        this.config = {
            spacing: 60,        // Grid spacing
            stiffness: 0.04,   // Spring stiffness (lower = wobblier)
            friction: 0.92,    // Damping (lower = more sliding)
            mouseForce: 45,    // Push strength
            returnForce: 0.03, // Strength to return to origin
            // Ambient animation settings
            ambientAmplitude: 6.0,   // Max displacement in pixels (increased for visibility)
            ambientSpeed: 0.003,     // Animation speed (faster)
            ambientWaveLength: 150,  // Wavelength of the sine pattern
            pulseSpeed: 0.05         // Speed of the mouse interaction pulse
        };

        // Colors
        this.colors = {
            grid: 'rgba(26, 26, 26, 0.08)',
            active: 'rgba(217, 119, 87, 0.6)'
        };

        this.resize();
        this.init();
        this.bindEvents();
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    init() {
        this.points = [];
        this.springs = [];

        // 1. Create Grid Points
        const cols = Math.ceil(this.canvas.width / this.config.spacing) + 1;
        const rows = Math.ceil(this.canvas.height / this.config.spacing) + 1;

        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                this.points.push({
                    x: x * this.config.spacing,
                    y: y * this.config.spacing,
                    ox: x * this.config.spacing, // Original X
                    oy: y * this.config.spacing, // Original Y
                    vx: 0, // Velocity X
                    vy: 0, // Velocity Y
                    idx: x + y * cols, // Index
                    col: x,
                    row: y,
                    fixed: false // Edges could be fixed if desired
                });
            }
        }
    }

    bindEvents() {
        window.addEventListener('resize', () => {
            this.resize();
            this.init();
        });

        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.x;
            this.mouse.y = e.y;
        });

        // Touch support
        window.addEventListener('touchmove', (e) => {
            if (e.touches[0]) {
                this.mouse.x = e.touches[0].clientX;
                this.mouse.y = e.touches[0].clientY;
            }
        });

        window.addEventListener('mouseout', () => {
            this.mouse.x = -1000;
            this.mouse.y = -1000;
        });
    }

    update() {
        // Increment time for ambient animation
        this.time += 1;

        // Update physics for each point
        this.points.forEach(p => {
            // Calculate ambient wave offset (creates organic, flowing movement)
            // Reason: Using sine waves with different frequencies creates natural-looking motion
            const waveX = Math.sin(
                this.time * this.config.ambientSpeed +
                p.oy / this.config.ambientWaveLength
            ) * this.config.ambientAmplitude;

            const waveY = Math.cos(
                this.time * this.config.ambientSpeed * 0.7 +
                p.ox / this.config.ambientWaveLength
            ) * this.config.ambientAmplitude * 0.8;

            // 1. Hooke's Law: Return to origin + ambient offset
            const targetX = p.ox + waveX;
            const targetY = p.oy + waveY;
            const dx = targetX - p.x;
            const dy = targetY - p.y;

            p.vx += dx * this.config.returnForce;
            p.vy += dy * this.config.returnForce;

            // 2. Mouse Repulsion (Antigravity Force)
            const dmx = p.x - this.mouse.x;
            const dmy = p.y - this.mouse.y;
            const dist = Math.sqrt(dmx * dmx + dmy * dmy);

            if (dist < this.mouse.radius) {
                // Add a "breathing" effect to the repulsion force
                const pulse = 1 + Math.sin(this.time * this.config.pulseSpeed) * 0.1;
                const force = (1 - dist / this.mouse.radius) * this.config.mouseForce * pulse;
                const angle = Math.atan2(dmy, dmx);

                p.vx += Math.cos(angle) * force;
                p.vy += Math.sin(angle) * force;
            }

            // 3. Apply Velocity & Friction
            p.vx *= this.config.friction;
            p.vy *= this.config.friction;

            p.x += p.vx;
            p.y += p.vy;
        });
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw Grid Lines (Springs visual)
        this.ctx.beginPath();
        const cols = Math.ceil(this.canvas.width / this.config.spacing) + 1;

        this.points.forEach(p => {
            // Connect to right neighbor
            if (p.col < cols - 1) {
                const next = this.points[p.idx + 1];
                if (next) {
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(next.x, next.y);
                }
            }
            // Connect to bottom neighbor
            const bottomIdx = p.idx + cols;
            if (bottomIdx < this.points.length && p.row < Math.ceil(this.canvas.height / this.config.spacing)) {
                const next = this.points[bottomIdx];
                if (next) {
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(next.x, next.y);
                }
            }
        });

        // Stroke style for the grid
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 1;
        this.ctx.stroke();

        // Draw Intersections ( Dots )
        // Only draw dots that have moved significantly to reduce noise
        const gradientRadius = this.mouse.radius * 4; // Enlarged gradient area
        this.points.forEach(p => {
            const disp = Math.abs(p.x - p.ox) + Math.abs(p.y - p.oy);
            // Draw interactive dots near mouse
            const dmx = p.x - this.mouse.x;
            const dmy = p.y - this.mouse.y;
            const dist = Math.sqrt(dmx * dmx + dmy * dmy);

            // Always draw with dynamic alpha to avoid flickering at zero-crossing
            // Dynamic alpha with "twinkle" effect
            const twinkle = Math.sin(this.time * 0.05 + p.idx) * 0.15;
            const baseAlpha = (disp / 20) + Math.max(0, (1 - dist / gradientRadius) * 0.5);
            const alpha = Math.min(0.8, Math.max(0.1, baseAlpha + twinkle));

            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(26, 26, 26, ${alpha})`;
            this.ctx.fill();
        });
    }

    animate() {
        this.update();
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// === File Upload Handler ===
class FileUploadHandler {
    constructor() {
        this.uploadZone = document.getElementById('upload-zone');
        this.fileInput = document.getElementById('file-input');
        this.resultsContainer = document.getElementById('results-container');
        this.resultsBody = document.getElementById('results-body');

        if (!this.uploadZone) return;

        this.particleClasses = ['Proton', 'Deuteron', 'Triton', 'Helium-3', 'Alpha', 'Carbon-12', 'Carbon-13', 'Carbon-14'];
        this.bindEvents();
    }

    bindEvents() {
        // Click to upload
        this.uploadZone.addEventListener('click', () => this.fileInput.click());

        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('dragover');
        });

        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('dragover');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });

        // Sample buttons
        document.querySelectorAll('.btn-sample').forEach(btn => {
            btn.addEventListener('click', () => {
                const sampleType = btn.dataset.sample;
                this.loadSampleData(sampleType);
            });
        });

        // Download buttons
        document.getElementById('download-csv')?.addEventListener('click', () => this.downloadResults('csv'));
        document.getElementById('download-json')?.addEventListener('click', () => this.downloadResults('json'));
    }

    async handleFiles(files) {
        if (!files.length) return;

        this.showNotification(`Processing ${files.length} file(s)...`, 'info');

        // Simulate processing delay
        await this.simulateProcessing();

        // Generate mock results
        const results = Array.from(files).map(file => this.generateMockResult(file.name));

        this.displayResults(results);
        this.showNotification('Classification complete!', 'success');
    }

    async loadSampleData(sampleType) {
        const sampleFiles = {
            proton: ['run_0137_proton.h5', 'run_0138_proton.root', 'run_0139_proton.h5'],
            alpha: ['run_0201_alpha.root', 'run_0202_alpha.h5', 'run_0203_alpha.root'],
            mixed: ['experiment_batch_01.h5', 'experiment_batch_02.root', 'calibration_run.h5', 'test_beam_data.root', 'simulation_output.h5']
        };

        const files = sampleFiles[sampleType] || sampleFiles.mixed;
        this.showNotification(`Loading ${sampleType} sample data...`, 'info');

        await this.simulateProcessing();

        const results = files.map(name => this.generateMockResult(name, sampleType));
        this.displayResults(results);
        this.showNotification('Sample data classified!', 'success');
    }

    generateMockResult(fileName, bias = null) {
        // Generate realistic confidence scores
        const scores = this.particleClasses.map(() => Math.random());
        const sum = scores.reduce((a, b) => a + b, 0);
        const normalized = scores.map(s => s / sum);

        // Apply bias for demo samples
        if (bias) {
            const biasIndex = {
                proton: 0,
                alpha: 4,
                mixed: Math.floor(Math.random() * 5)
            }[bias];
            normalized[biasIndex] = 0.85 + Math.random() * 0.12;
            const remaining = 1 - normalized[biasIndex];
            const otherSum = normalized.reduce((a, b, i) => i !== biasIndex ? a + b : a, 0);
            normalized.forEach((_, i) => {
                if (i !== biasIndex) normalized[i] = (normalized[i] / otherSum) * remaining;
            });
        }

        // Sort to get top predictions
        const indexed = normalized.map((conf, i) => ({ class: this.particleClasses[i], conf }));
        indexed.sort((a, b) => b.conf - a.conf);

        return {
            fileName,
            prediction: indexed[0].class,
            confidence: indexed[0].conf,
            top3: indexed.slice(0, 3)
        };
    }

    displayResults(results) {
        this.resultsContainer.style.display = 'block';
        this.resultsBody.innerHTML = '';

        results.forEach(result => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><code>${result.fileName}</code></td>
                <td><strong style="color: var(--color-accent-primary)">${result.prediction}</strong></td>
                <td><span style="font-family: var(--font-mono)">${(result.confidence * 100).toFixed(1)}%</span></td>
                <td>
                    ${result.top3.map(t => `
                        <span style="display: inline-block; margin-right: 8px; font-size: 0.85rem">
                            ${t.class}: <span style="color: var(--color-accent-primary); font-family: var(--font-mono)">${(t.conf * 100).toFixed(1)}%</span>
                        </span>
                    `).join('')}
                </td>
            `;
            this.resultsBody.appendChild(row);
        });

        // Scroll to results
        this.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Store results for download
        this.currentResults = results;
    }

    downloadResults(format) {
        if (!this.currentResults) return;

        let content, mimeType, extension;

        if (format === 'csv') {
            const headers = ['File', 'Prediction', 'Confidence', 'Class1', 'Conf1', 'Class2', 'Conf2', 'Class3', 'Conf3'];
            const rows = this.currentResults.map(r => [
                r.fileName,
                r.prediction,
                (r.confidence * 100).toFixed(2),
                ...r.top3.flatMap(t => [t.class, (t.conf * 100).toFixed(2)])
            ]);
            content = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
            mimeType = 'text/csv';
            extension = 'csv';
        } else {
            content = JSON.stringify(this.currentResults, null, 2);
            mimeType = 'application/json';
            extension = 'json';
        }

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mate_results_${Date.now()}.${extension}`;
        a.click();
        URL.revokeObjectURL(url);
    }

    simulateProcessing() {
        return new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 700));
    }

    showNotification(message, type) {
        const existing = document.querySelector('.notification');
        if (existing) existing.remove();

        const colors = {
            success: '#2d6a4f', // Deep green
            error: '#c1121f',   // Deep red
            info: '#1a1a1a'     // Dark charcoal (Neutral)
        };

        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.style.cssText = `
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 12px 20px;
            background: ${colors[type] || colors.info};
            color: #ffffff;
            border-radius: 6px;
            font-family: var(--font-body);
            font-size: 0.9rem;
            font-weight: 500;
            z-index: 1000;
            animation: slide-in 0.3s ease-out;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        `;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slide-out 0.3s ease-out forwards';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// === Smooth Scroll ===
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const href = this.getAttribute('href');

            // Handle home/top link
            if (href === '#') {
                window.scrollTo({ top: 0, behavior: 'smooth' });
                return;
            }

            const target = document.querySelector(href);
            if (target) {
                // If targeting the demo options, center it in the viewport
                if (target.id === 'demo-options') {
                    const elementRect = target.getBoundingClientRect();
                    const absoluteElementTop = elementRect.top + window.pageYOffset;
                    const middle = absoluteElementTop - (window.innerHeight / 2) + (elementRect.height / 2);
                    window.scrollTo({ top: middle, behavior: 'smooth' });
                } else {
                    // Default behavior with header offset
                    const offset = 80;
                    const top = target.getBoundingClientRect().top + window.pageYOffset - offset;
                    window.scrollTo({ top, behavior: 'smooth' });
                }
            }
        });
    });
}

// === Scroll Reveal Animation ===
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.feature-card, .step, .particle-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        observer.observe(el);
    });
}

// Add revealed class styles
const style = document.createElement('style');
style.textContent = `
    .revealed { opacity: 1 !important; transform: translateY(0) !important; }
    @keyframes slide-in { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    @keyframes slide-out { from { transform: translateY(0); opacity: 1; } to { transform: translateY(20px); opacity: 0; } }
    .loading-spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite; display: inline-block; vertical-align: middle; margin-right: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    code { font-family: var(--font-mono); font-size: 0.85em; background: rgba(0,0,0,0.06); padding: 2px 6px; border-radius: 4px; color: var(--color-text-primary); }
`;
document.head.appendChild(style);

// === Initialize Everything ===
document.addEventListener('DOMContentLoaded', () => {
    // Initialize particle system
    const canvas = document.getElementById('particle-canvas');
    if (canvas) {
        new ParticleSystem(canvas);
    }

    // Initialize file upload handler
    new FileUploadHandler();

    // Initialize smooth scroll
    initSmoothScroll();

    // Initialize scroll reveal
    initScrollReveal();

    console.log('MATE Identifier Web Application initialized');
});
