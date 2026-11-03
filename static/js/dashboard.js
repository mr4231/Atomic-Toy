/**
 * Scientific Simulation Dashboard Frontend (Three.js 3D + Plotly.js + Flask API)
 * Atomic Toy Model — IIT Hyderabad Scientific Computing Project
 */

// Global State
let simData = null;           // Current trajectory object from backend
let currentFrame = 0;
let isPlaying = false;
let playbackSpeed = 1.0;
let animationReqId = null;
let lastFrameTimestamp = 0;

// Three.js Scene Variables
let scene, camera, renderer, controls;
let boxWireframe = null;
let particleMeshes = [];
let boxLength = 10.0;

document.addEventListener("DOMContentLoaded", () => {
    initThreeJS();
    initEventListeners();
    loadDefaultConfigAndRun();
    loadAllExperimentResults();
});

/* ==========================================================================
   1. THREE.JS 3D VIEWPORT & ANIMATION
   ========================================================================== */

function initThreeJS() {
    const container = document.getElementById("sim-viewport");
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050811);

    // Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(18, 18, 24);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Orbit Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 0.8);
    dirLight1.position.set(20, 30, 20);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xc084fc, 0.4);
    dirLight2.position.set(-20, -10, -20);
    scene.add(dirLight2);

    // Initial Box
    createBoxWireframe(10.0);

    // Handle Resize
    window.addEventListener("resize", onWindowResize);

    // Render loop
    requestAnimationFrame(renderLoop);
}

function createBoxWireframe(length) {
    if (boxWireframe) scene.remove(boxWireframe);
    boxLength = length;

    const geometry = new THREE.BoxGeometry(length, length, length);
    const edges = new THREE.EdgesGeometry(geometry);
    const material = new THREE.LineBasicMaterial({ color: 0x38bdf8, opacity: 0.4, transparent: true });
    boxWireframe = new THREE.LineSegments(edges, material);
    boxWireframe.position.set(length / 2, length / 2, length / 2);
    scene.add(boxWireframe);

    // Adjust camera target to box center
    controls.target.set(length / 2, length / 2, length / 2);
    controls.update();
}

function updateParticleMeshes(numParticles) {
    // Remove existing meshes
    particleMeshes.forEach(m => scene.remove(m));
    particleMeshes = [];

    const sphereGeom = new THREE.SphereGeometry(0.25, 16, 16);
    for (let i = 0; i < numParticles; i++) {
        // Color palette for atoms
        const color = new THREE.Color().setHSL((i * 0.13) % 1.0, 0.8, 0.55);
        const material = new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.3,
            metalness: 0.2,
        });
        const mesh = new THREE.Mesh(sphereGeom, material);
        scene.add(mesh);
        particleMeshes.push(mesh);
    }
}

function setAnimationFrame(frameIdx) {
    if (!simData || !simData.positions || frameIdx >= simData.positions.length) return;

    currentFrame = frameIdx;
    const positions = simData.positions[frameIdx]; // shape: (N, 3)

    for (let i = 0; i < positions.length && i < particleMeshes.length; i++) {
        particleMeshes[i].position.set(positions[i][0], positions[i][1], positions[i][2]);
    }

    // Update overlay UI
    const frameTimes = simData.frame_times;
    document.getElementById("lbl-sim-time").innerText = (frameTimes[frameIdx] || 0.0).toFixed(3);
    document.getElementById("lbl-sim-step").innerText = frameIdx;
    document.getElementById("lbl-sim-maxsteps").innerText = (frameTimes.length - 1);
}

function renderLoop(timestamp) {
    requestAnimationFrame(renderLoop);
    controls.update();
    renderer.render(scene, camera);

    if (isPlaying && simData) {
        if (timestamp - lastFrameTimestamp > (40 / playbackSpeed)) {
            lastFrameTimestamp = timestamp;
            let nextFrame = currentFrame + 1;
            if (nextFrame >= simData.positions.length) {
                nextFrame = 0; // Loop trajectory
            }
            setAnimationFrame(nextFrame);
        }
    }
}

function onWindowResize() {
    const container = document.getElementById("sim-viewport");
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

/* ==========================================================================
   2. EVENT LISTENERS & CONTROL PANEL
   ========================================================================== */

function initEventListeners() {
    document.getElementById("btn-play-pause").addEventListener("click", togglePlayPause);
    document.getElementById("btn-restart").addEventListener("click", () => setAnimationFrame(0));
    document.getElementById("btn-run-sim").addEventListener("click", submitSimulation);

    const speedSlider = document.getElementById("slider-speed");
    speedSlider.addEventListener("input", (e) => {
        playbackSpeed = parseFloat(e.target.value);
        document.getElementById("lbl-speed-val").innerText = playbackSpeed.toFixed(1) + "x";
    });

    // Form change warning
    const formInputs = document.querySelectorAll("#form-config input, #form-config select");
    formInputs.forEach(input => {
        input.addEventListener("change", () => {
            document.getElementById("stale-warning").style.display = "block";
        });
    });
}

function togglePlayPause() {
    isPlaying = !isPlaying;
    const btn = document.getElementById("btn-play-pause");
    btn.innerText = isPlaying ? "⏸ Pause" : "▶ Play";
}

/* ==========================================================================
   3. SIMULATION EXECUTION (API CALLS)
   ========================================================================== */

async function loadDefaultConfigAndRun() {
    try {
        const res = await fetch("/api/config/default");
        const cfg = await res.json();
        
        // Fill form
        document.getElementById("cfg-particles").value = cfg.number_of_particles || 25;
        document.getElementById("cfg-box").value = cfg.box_size || 10.0;
        document.getElementById("cfg-cutoff").value = cfg.cutoff || 0.8;
        document.getElementById("cfg-r0").value = cfg.equilibrium_distance || 0.8;
        document.getElementById("cfg-spring").value = cfg.spring_constant || 15.0;
        document.getElementById("cfg-vscale").value = cfg.initial_velocity_scale || 8.0;
        document.getElementById("cfg-dt").value = cfg.dt || 0.01;
        document.getElementById("cfg-steps").value = cfg.steps || 500;
        document.getElementById("cfg-gamma").value = cfg.gamma || 0.0;
        document.getElementById("cfg-seed").value = cfg.seed || 0;
        document.getElementById("cfg-integrator").value = cfg.integrator || "velocity_verlet";
        document.getElementById("cfg-dtype").value = cfg.dtype || "float64";

        submitSimulation();
    } catch (err) {
        console.error("Failed to fetch default config", err);
    }
}

async function submitSimulation() {
    const statusBadge = document.getElementById("status-indicator");
    statusBadge.className = "status-badge running";
    document.getElementById("lbl-status").innerText = "Simulating...";

    const payload = {
        number_of_particles: parseInt(document.getElementById("cfg-particles").value),
        box_size: parseFloat(document.getElementById("cfg-box").value),
        cutoff: parseFloat(document.getElementById("cfg-cutoff").value),
        equilibrium_distance: parseFloat(document.getElementById("cfg-r0").value),
        spring_constant: parseFloat(document.getElementById("cfg-spring").value),
        initial_velocity_scale: parseFloat(document.getElementById("cfg-vscale").value),
        dt: parseFloat(document.getElementById("cfg-dt").value),
        steps: parseInt(document.getElementById("cfg-steps").value),
        gamma: parseFloat(document.getElementById("cfg-gamma").value),
        seed: parseInt(document.getElementById("cfg-seed").value),
        integrator: document.getElementById("cfg-integrator").value,
        dtype: document.getElementById("cfg-dtype").value,
    };

    try {
        const response = await fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errJson = await response.json();
            throw new Error(errJson.error || "Simulation failed");
        }

        simData = await response.json();
        document.getElementById("stale-warning").style.display = "none";

        // Update 3D Scene
        createBoxWireframe(payload.box_size);
        updateParticleMeshes(payload.number_of_particles);
        setAnimationFrame(0);

        // Update Labels & Overview
        document.getElementById("lbl-sim-n").innerText = payload.number_of_particles;
        document.getElementById("stat-particles").innerText = payload.number_of_particles;
        document.getElementById("stat-drift").innerText = simData.max_relative_energy_drift.toExponential(4);

        // Update Energy Section Active Run Meta
        if (document.getElementById("meta-n")) {
            document.getElementById("meta-n").innerText = payload.number_of_particles;
            document.getElementById("meta-box").innerText = payload.box_size.toFixed(1);
            document.getElementById("meta-rcut").innerText = payload.cutoff.toFixed(2);
            document.getElementById("meta-dt").innerText = payload.dt;
            document.getElementById("meta-steps").innerText = payload.steps;
            document.getElementById("meta-seed").innerText = payload.seed;
            document.getElementById("meta-drift").innerText = simData.max_relative_energy_drift.toExponential(4);
        }
        
        const initPairs = simData.initial_diagnostics.initial_interacting_pair_count || 0;
        const finalPairs = simData.final_diagnostics.final_interacting_pairs || 0;
        document.getElementById("stat-pairs").innerText = `${initPairs} → ${finalPairs}`;
        document.getElementById("diag-min-sep").innerText = (simData.initial_diagnostics.minimum_initial_particle_separation || 0).toFixed(3);

        // Update Plotly Energy Chart
        renderEnergyChart(simData.times, simData.kinetic, simData.potential, simData.total_energy);

        statusBadge.className = "status-badge ready";
        document.getElementById("lbl-status").innerText = "Ready";

    } catch (err) {
        alert("Error running simulation: " + err.message);
        statusBadge.className = "status-badge warning";
        document.getElementById("lbl-status").innerText = "Error";
    }
}

/* ==========================================================================
   4. PLOTLY CHARTS & EXPERIMENT RESULTS
   ========================================================================== */

function renderEnergyChart(times, kinetic, potential, total) {
    const traceK = { x: times, y: kinetic, mode: 'lines', name: 'Kinetic K', line: { color: '#38bdf8' } };
    const traceU = { x: times, y: potential, mode: 'lines', name: 'Potential U', line: { color: '#fbbf24' } };
    const traceE = { x: times, y: total, mode: 'lines', name: 'Total E', line: { color: '#4ade80', width: 2.5 } };

    const layout = {
        paper_bgcolor: '#0b0f19',
        plot_bgcolor: '#0b0f19',
        margin: { l: 50, r: 20, t: 30, b: 40 },
        xaxis: { title: 'Time (s)', color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: 'Energy', color: '#94a3b8', gridcolor: '#1e293b' },
        legend: { font: { color: '#e2e8f0' }, orientation: 'h', y: 1.1 },
    };

    Plotly.newPlot('chart-energy', [traceK, traceU, traceE], layout, { responsive: true });
}

async function loadAllExperimentResults() {
    loadStudyResults('timestep', renderTimestepSection);
    loadStudyResults('dissipation', renderDissipationSection);
    loadStudyResults('precision', renderPrecisionSection);
    loadStudyResults('performance', renderPerformanceSection);
    loadStudyResults('memory', renderMemorySection);
}

async function loadStudyResults(studyName, renderCallback) {
    try {
        const res = await fetch(`/api/results/${studyName}`);
        const json = await res.json();
        if (json.exists) {
            renderCallback(json.data);
        }
    } catch (err) {
        console.warn(`Could not load ${studyName} results`, err);
    }
}

async function runExperimentSuite(experimentName) {
    try {
        const res = await fetch("/api/experiments/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ experiment: experimentName }),
        });
        const data = await res.json();
        if (data.status === "success") {
            alert(`Experiment suite '${experimentName}' completed successfully!`);
            loadAllExperimentResults();
        } else {
            alert(`Experiment failed: ${data.error}`);
        }
    } catch (err) {
        alert("Error invoking experiment suite: " + err.message);
    }
}

/* Renderers for Experiment Sections */

function renderTimestepSection(rows) {
    const container = document.getElementById("container-timestep-table");
    let html = `<table class="data-table"><thead><tr><th>dt</th><th>Steps</th><th>Integrator</th><th>Max Rel Energy Drift</th></tr></thead><tbody>`;
    rows.forEach(r => {
        html += `<tr><td>${r.dt}</td><td>${r.steps}</td><td>${r.integrator}</td><td>${parseFloat(r.max_relative_energy_drift).toExponential(4)}</td></tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;

    // Plot dt vs drift
    const dts = rows.map(r => parseFloat(r.dt));
    const drifts = rows.map(r => parseFloat(r.max_relative_energy_drift));

    const trace = { x: dts, y: drifts, mode: 'lines+markers', marker: { color: '#38bdf8', size: 8 } };
    const layout = {
        paper_bgcolor: '#0b0f19', plot_bgcolor: '#0b0f19',
        margin: { l: 60, r: 20, t: 30, b: 40 },
        xaxis: { title: 'Timestep dt', color: '#94a3b8', gridcolor: '#1e293b', autorange: 'reversed' },
        yaxis: { title: 'Max Relative Drift', color: '#94a3b8', gridcolor: '#1e293b', type: 'log' },
    };
    Plotly.newPlot('chart-timestep', [trace], layout, { responsive: true });
}

function renderDissipationSection(rows) {
    const container = document.getElementById("container-dissipation-table");
    let html = `<table class="data-table"><thead><tr><th>Gamma γ</th><th>Steps</th><th>Final Total E</th><th>Final Mean Speed</th><th>Relaxation Time</th></tr></thead><tbody>`;
    rows.forEach(r => {
        html += `<tr><td>${r.gamma}</td><td>${r.steps}</td><td>${parseFloat(r.final_total_energy).toFixed(4)}</td><td>${parseFloat(r.final_mean_speed).toFixed(4)}</td><td>${r.relaxation_time}</td></tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;
}

function renderPrecisionSection(rows) {
    const container = document.getElementById("container-precision-table");
    let html = `<table class="data-table"><thead><tr><th>dtype</th><th>Pos Bytes</th><th>Pair Storage Bytes</th><th>Max Energy Drift</th><th>Final Pos Diff vs float64</th></tr></thead><tbody>`;
    rows.forEach(r => {
        html += `<tr><td><strong>${r.dtype}</strong></td><td>${r.position_bytes}</td><td>${r.pair_displacement_bytes}</td><td>${parseFloat(r.max_relative_energy_drift).toExponential(4)}</td><td>${parseFloat(r.final_position_difference_vs_float64).toExponential(4)}</td></tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;
}

function renderPerformanceSection(rows) {
    const container = document.getElementById("container-performance-table");
    let html = `<table class="data-table"><thead><tr><th>Implementation</th><th>N</th><th>Repeats</th><th>Median Sec</th><th>Speedup</th><th>Parity Norm</th></tr></thead><tbody>`;
    rows.forEach(r => {
        html += `<tr><td>${r.implementation}</td><td>${r.n_particles}</td><td>${r.repeats}</td><td>${parseFloat(r.median_seconds).toFixed(6)}</td><td>${parseFloat(r.speedup_vs_python).toFixed(2)}x</td><td>${parseFloat(r.force_parity_norm).toExponential(2)}</td></tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;

    // Plot N vs median seconds
    const seriesMap = {};
    rows.forEach(r => {
        if (!seriesMap[r.implementation]) seriesMap[r.implementation] = { x: [], y: [] };
        seriesMap[r.implementation].x.push(parseInt(r.n_particles));
        seriesMap[r.implementation].y.push(parseFloat(r.median_seconds));
    });

    const traces = Object.keys(seriesMap).map(name => ({
        x: seriesMap[name].x,
        y: seriesMap[name].y,
        mode: 'lines+markers',
        name: name,
    }));

    const layout = {
        paper_bgcolor: '#0b0f19', plot_bgcolor: '#0b0f19',
        margin: { l: 60, r: 20, t: 30, b: 40 },
        xaxis: { title: 'Particle Count N', color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: 'Median Wall Time (s)', color: '#94a3b8', gridcolor: '#1e293b', type: 'log' },
        legend: { font: { color: '#e2e8f0' }, orientation: 'h', y: 1.1 },
    };
    Plotly.newPlot('chart-performance', traces, layout, { responsive: true });
}

function renderMemorySection(rows) {
    const container = document.getElementById("container-memory-table");
    let html = `<table class="data-table"><thead><tr><th>N Particles</th><th>dtype</th><th>Pair Displacement Bytes</th><th>MiB</th></tr></thead><tbody>`;
    rows.forEach(r => {
        html += `<tr><td>${r.n_particles}</td><td>${r.dtype}</td><td>${r.pair_displacement_bytes}</td><td>${parseFloat(r.pair_displacement_mebibytes).toFixed(2)} MiB</td></tr>`;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;
}
