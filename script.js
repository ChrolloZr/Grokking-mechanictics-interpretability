/*
 * Timeless: The Physics of Modular Grokking & Mechanistic Interpretability
 * Interactive Core Javascript File
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- Navigation Controls ---
  const navItems = document.querySelectorAll('.nav-item');
  const sections = document.querySelectorAll('.section');

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = item.getAttribute('data-target');
      
      // Update nav active state
      navItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      
      // Update section visibility
      sections.forEach(sec => {
        sec.classList.remove('active');
        if (sec.id === targetId) {
          sec.classList.add('active');
        }
      });

      // Initialize visualizations on tab change after section becomes active (visible)
      if (targetId === 'physics') {
        resizeSandboxCanvas();
        if (!physicsSandboxRunning) {
          initPhysicsSandbox();
        } else {
          resetPhysicsSandbox();
        }
      }
      if (targetId === 'dashboard') {
        drawTrigVisualizer();
      }
    });
  });

  // --- Copy to Clipboard Utility ---
  window.copyCode = function(button) {
    const codeContainer = button.closest('.code-container');
    const codeBlock = codeContainer.querySelector('code');
    const text = codeBlock.innerText;

    navigator.clipboard.writeText(text).then(() => {
      const originalText = button.innerHTML;
      button.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-check"><path d="M20 6 9 17l-5-5"/></svg>
        Copied!
      `;
      button.style.color = 'var(--accent-green)';
      setTimeout(() => {
        button.innerHTML = originalText;
        button.style.color = '';
      }, 2000);
    });
  };

  // ==========================================
  // --- PHYSICS OF GROKKING SANDBOX ---
  // ==========================================
  const sandboxCanvas = document.getElementById('physics-canvas');
  let ctxSandbox = null;
  let particles = [];
  let physicsSandboxRunning = false;
  let animationFrameId = null;

  // Physics Simulation Variables
  let gdSpeed = 0.5; // Gradient Descent Step size
  let weightDecay = 0.3; // L2 contraction force
  let phaseTransitionActive = false;
  let systemEntropy = 1.0;
  let systemEnergy = 1.0;

  const sliderGd = document.getElementById('slider-gd');
  const sliderWd = document.getElementById('slider-wd');
  const tempSpan = document.getElementById('entropy-val');
  const energySpan = document.getElementById('energy-val');
  const transitionBtn = document.getElementById('btn-transition');
  const resetBtn = document.getElementById('btn-reset-physics');

  if (sliderGd) {
    sliderGd.addEventListener('input', (e) => {
      gdSpeed = parseFloat(e.target.value);
    });
  }
  if (sliderWd) {
    sliderWd.addEventListener('input', (e) => {
      weightDecay = parseFloat(e.target.value);
    });
  }
  if (transitionBtn) {
    transitionBtn.addEventListener('click', () => {
      phaseTransitionActive = true;
      transitionBtn.classList.add('btn-primary');
      transitionBtn.innerText = "Phase Transition Active...";
    });
  }
  if (resetBtn) {
    resetBtn.addEventListener('click', resetPhysicsSandbox);
  }

  function resizeSandboxCanvas() {
    if (!sandboxCanvas) return;
    // Query clientWidth of parent to avoid offsetWidth = 0 when parent is invisible
    const parentWidth = sandboxCanvas.parentElement.clientWidth || 800;
    sandboxCanvas.width = parentWidth;
    sandboxCanvas.height = 350;
  }

  function initPhysicsSandbox() {
    if (!sandboxCanvas) return;
    ctxSandbox = sandboxCanvas.getContext('2d');
    physicsSandboxRunning = true;
    resizeSandboxCanvas();
    resetPhysicsSandbox();
    animatePhysicsSandbox();
  }

  function resetPhysicsSandbox() {
    if (!sandboxCanvas) return;
    resizeSandboxCanvas();
    particles = [];
    phaseTransitionActive = false;
    systemEntropy = 1.0;
    systemEnergy = 1.0;
    if (transitionBtn) {
      transitionBtn.classList.remove('btn-primary');
      transitionBtn.innerText = "Trigger Phase Transition";
    }

    const width = sandboxCanvas.width;
    const height = sandboxCanvas.height;

    // Create 160 particles dispersed randomly representing disorganized "memorized noise" state
    for (let i = 0; i < 160; i++) {
      particles.push({
        x: Math.random() * (width - 40) + 20,
        y: Math.random() * (height - 40) + 20,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        radius: 3.5,
        color: `hsla(${220 + Math.random() * 40}, 85%, 65%, 0.85)`,
        gridX: null,
        gridY: null
      });
    }
  }

  function animatePhysicsSandbox() {
    if (!physicsSandboxRunning || !ctxSandbox) return;
    
    const width = sandboxCanvas.width;
    const height = sandboxCanvas.height;
    
    // Clear canvas
    ctxSandbox.fillStyle = '#090a0f';
    ctxSandbox.fillRect(0, 0, width, height);

    // Draw Grid Lines Background
    const gridSize = 40;
    ctxSandbox.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctxSandbox.lineWidth = 1;
    for (let x = 0; x < width; x += gridSize) {
      ctxSandbox.beginPath();
      ctxSandbox.moveTo(x, 0);
      ctxSandbox.lineTo(x, height);
      ctxSandbox.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
      ctxSandbox.beginPath();
      ctxSandbox.moveTo(0, y);
      ctxSandbox.lineTo(width, y);
      ctxSandbox.stroke();
    }

    // Draw attractive potential wells (Representing generalizable mechanical structures learned by model)
    const wells = [
      {x: width * 0.25, y: height * 0.3, radius: 45},
      {x: width * 0.75, y: height * 0.3, radius: 45},
      {x: width * 0.5, y: height * 0.7, radius: 45},
      {x: width * 0.25, y: height * 0.7, radius: 45},
      {x: width * 0.75, y: height * 0.7, radius: 45}
    ];

    wells.forEach(w => {
      const grad = ctxSandbox.createRadialGradient(w.x, w.y, 2, w.x, w.y, w.radius);
      grad.addColorStop(0, 'rgba(6, 182, 212, 0.06)');
      grad.addColorStop(1, 'rgba(99, 102, 241, 0)');
      ctxSandbox.fillStyle = grad;
      ctxSandbox.beginPath();
      ctxSandbox.arc(w.x, w.y, w.radius, 0, Math.PI * 2);
      ctxSandbox.fill();
    });

    // Update entropy and energy HUD statistics
    if (phaseTransitionActive) {
      systemEntropy = Math.max(0.02, systemEntropy - 0.015);
      systemEnergy = Math.max(0.12, systemEnergy - 0.012);
    } else {
      systemEntropy = 1.0;
      systemEnergy = 1.0;
    }

    if (tempSpan) tempSpan.innerText = systemEntropy.toFixed(2);
    if (energySpan) energySpan.innerText = systemEnergy.toFixed(2);

    // Update and draw particles
    particles.forEach(p => {
      let fx = 0;
      let fy = 0;
      
      const centerX = width / 2;
      const centerY = height / 2;
      
      // Pull particles inward (L2 Weight Decay springs)
      fx += (centerX - p.x) * (weightDecay * 0.005);
      fy += (centerY - p.y) * (weightDecay * 0.005);

      // Modular mechanical landscape pulls (Kinematic trajectories)
      wells.forEach(w => {
        const dx = w.x - p.x;
        const dy = w.y - p.y;
        const distSq = dx*dx + dy*dy;
        const dist = Math.sqrt(distSq);
        
        if (dist < 120) {
          const force = (gdSpeed * 0.8) / (distSq / 15 + 1);
          fx += (dx / dist) * force;
          fy += (dy / dist) * force;
        }
      });

      //snapping coordinates to perfect alignment
      if (phaseTransitionActive) {
        const snapX = Math.round(p.x / 40) * 40;
        const snapY = Math.round(p.y / 40) * 40;
        
        fx += (snapX - p.x) * 0.15;
        fy += (snapY - p.y) * 0.15;
        
        p.color = `hsla(180, 90%, ${45 + Math.random()*20}%, 0.95)`;
      } else {
        // Entropic thermal noise
        p.vx += (Math.random() - 0.5) * systemEntropy * 0.65;
        p.vy += (Math.random() - 0.5) * systemEntropy * 0.65;
      }

      // Physics integration
      p.vx += fx;
      p.vy += fy;
      p.vx *= 0.85;
      p.vy *= 0.85;

      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 5 || p.x > width - 5) p.vx *= -1;
      if (p.y < 5 || p.y > height - 5) p.vy *= -1;

      // Draw particle
      ctxSandbox.fillStyle = p.color;
      ctxSandbox.shadowBlur = phaseTransitionActive ? 8 : 0;
      ctxSandbox.shadowColor = 'var(--accent-cyan)';
      ctxSandbox.beginPath();
      ctxSandbox.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctxSandbox.fill();
      ctxSandbox.shadowBlur = 0;
    });

    // Draw connection networks on synchronization
    if (phaseTransitionActive) {
      ctxSandbox.strokeStyle = 'rgba(6, 182, 212, 0.08)';
      ctxSandbox.lineWidth = 1;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = dx*dx + dy*dy;
          if (dist < 900) {
            ctxSandbox.beginPath();
            ctxSandbox.moveTo(particles[i].x, particles[i].y);
            ctxSandbox.lineTo(particles[j].x, particles[j].y);
            ctxSandbox.stroke();
          }
        }
      }
    }

    animationFrameId = requestAnimationFrame(animatePhysicsSandbox);
  }

  // ==========================================
  // --- INTERACTIVE FOURIER DFT DASHBOARD ---
  // ==========================================
  const fourierCanvas = document.getElementById('fourier-dft-canvas');
  const sliderFreq = document.getElementById('slider-freq-dft');
  const currentFreqSpan = document.getElementById('current-freq-display');

  if (sliderFreq) {
    sliderFreq.addEventListener('input', (e) => {
      if (currentFreqSpan) currentFreqSpan.innerText = e.target.value;
      drawTrigVisualizer();
    });
  }

  function drawTrigVisualizer() {
    if (!fourierCanvas) return;
    const ctx = fourierCanvas.getContext('2d');
    const parentWidth = fourierCanvas.parentElement.clientWidth || 800;
    fourierCanvas.width = parentWidth;
    fourierCanvas.height = 350;

    const width = fourierCanvas.width;
    const height = fourierCanvas.height;
    const freq = sliderFreq ? parseInt(sliderFreq.value) : 14;

    ctx.fillStyle = '#090a0f';
    ctx.fillRect(0, 0, width, height);

    const halfWidth = width / 2;
    
    // Draw boundary line
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(halfWidth, 0);
    ctx.lineTo(halfWidth, height);
    ctx.stroke();

    // 1. DRAW SPATIAL EMBEDDING WAVE: cos(2*pi*x*w/p) using p=131
    const p = 131; // Integrated modular prime
    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.fillRect(15, 45, halfWidth - 30, height - 90);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.strokeRect(15, 45, halfWidth - 30, height - 90);

    ctx.fillStyle = 'var(--text-secondary)';
    ctx.font = '12px var(--font-sans)';
    ctx.fillText(`Spatial Embedding cos(2\u03C0 \u00B7 x \u00B7 w / ${p})`, 25, 30);
    ctx.fillText(`Modulus x \u2208 [0, 130]`, 25, height - 20);

    // Axis line
    ctx.strokeStyle = '#232533';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(30, height / 2);
    ctx.lineTo(halfWidth - 30, height / 2);
    ctx.stroke();

    // Plot spatial cosine values
    ctx.strokeStyle = 'var(--accent-cyan)';
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 5;
    ctx.shadowColor = 'var(--accent-cyan-glow)';
    ctx.beginPath();

    const startX = 35;
    const endX = halfWidth - 35;
    const plotWidth = endX - startX;
    const centerY = height / 2;
    const amplitude = height * 0.28;

    for (let x = 0; x <= p; x++) {
      const px = startX + (x / p) * plotWidth;
      const angle = (2 * Math.PI * x * freq) / p;
      const py = centerY - Math.cos(angle) * amplitude;
      
      if (x === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Plot discrete coordinates
    for (let x = 0; x <= p; x += 6) {
      const px = startX + (x / p) * plotWidth;
      const angle = (2 * Math.PI * x * freq) / p;
      const py = centerY - Math.cos(angle) * amplitude;
      
      ctx.fillStyle = 'var(--text-primary)';
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // 2. DRAW FOURIER SPECTRUM (DFT) ON RIGHT HALF
    ctx.fillStyle = 'var(--text-secondary)';
    ctx.fillText(`1D Discrete Fourier Transform |S(k)|`, halfWidth + 25, 30);
    ctx.fillText(`Sparsity Frequency component k`, halfWidth + 25, height - 20);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.fillRect(halfWidth + 15, 45, halfWidth - 30, height - 90);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.strokeRect(halfWidth + 15, 45, halfWidth - 30, height - 90);

    // Spectrum Base Axis
    ctx.strokeStyle = '#232533';
    ctx.beginPath();
    ctx.moveTo(halfWidth + 30, height - 60);
    ctx.lineTo(width - 30, height - 60);
    ctx.stroke();

    const maxFreqIndex = p / 2;
    const startXRight = halfWidth + 35;
    const endXRight = width - 35;
    const plotWidthRight = endXRight - startXRight;

    // Draw bars representing sparse structural peak frequencies
    for (let k = 1; k < maxFreqIndex; k += 2) {
      const px = startXRight + (k / maxFreqIndex) * plotWidthRight;
      let magnitude = Math.random() * 4 + 2; // baseline noise
      
      if (k === freq || k === (p - freq)) {
        magnitude = height * 0.65; // Massive clean spike!
      } else if (k === 14 || k === 28 || k === 42) {
        magnitude = height * 0.12; 
      }

      ctx.fillStyle = k === freq ? 'var(--accent-purple)' : 'rgba(255, 255, 255, 0.15)';
      ctx.shadowBlur = k === freq ? 10 : 0;
      ctx.shadowColor = 'var(--accent-purple-glow)';
      
      ctx.fillRect(px - 2, height - 60 - magnitude, 4, magnitude);
      ctx.shadowBlur = 0;

      if (k === freq || k === 14 || k === 28 || k === 42) {
        ctx.fillStyle = k === freq ? 'var(--accent-purple)' : 'var(--text-muted)';
        ctx.fillText(`k=${k}`, px - 10, height - 65 - magnitude);
      }
    }
  }
});
