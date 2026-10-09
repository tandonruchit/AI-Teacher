/**
 * ArgonLearn Interactive Frontend Controller (app.js)
 *
 * Powers:
 * 1. Live ADK Multi-Agent API synthesis (`POST /api/learn`).
 * 2. Interactive 8x6 Token Grid & Attention Heatmap Canvas (`Masking Ratio: 80% -> 20%`).
 * 3. Live 3D Manim Cosine-Similarity Vector Clustering Renderer & AST Viewer.
 * 4. Persistent Firestore Knowledge Graph DAG SVG Renderer with glowing edges.
 * 5. Cloud Run + OpenTelemetry Trace Waterfall + structlog Intent-vs-Outcome HUD.
 * 6. Hands-free 30-90s Silent Video Demo Reel player for the Gemini 4 Challenge.
 */

(function () {
  "use strict";

  const state = {
    learnerId: "learner_demo_01",
    sessionId: "session_demo_01",
    conceptId: "discrete_parallel_diffusion",
    conceptTitle: "Discrete Parallel Diffusion",
    masteryMode: "Novice",
    maskingRatio: 80,
    explanations: {
      ELI5: "",
      Novice: "",
      Practitioner: "",
      Hero: "",
    },
    dag: null,
    manimCode: "",
    manimAngle: 0.0,
    cosineTarget: 0.62,
    cosineCurrent: 0.62,
    traceSpans: [],
    auditLogs: [],
    demoTimers: [],
  };

  // DOM References
  const promptInput = document.getElementById("promptInput");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnHeroMode = document.getElementById("btnHeroMode");
  const btnAutoDemo = document.getElementById("btnAutoDemo");
  const btnProdProof = document.getElementById("btnProdProof");
  const btnCloseProof = document.getElementById("btnCloseProof");
  const btnEndingBanner = document.getElementById("btnEndingBanner");
  const btnDismissEnding = document.getElementById("btnDismissEnding");
  const btnHitlGate = document.getElementById("btnHitlGate");
  const btnCancelHitl = document.getElementById("btnCancelHitl");
  const btnApproveHitl = document.getElementById("btnApproveHitl");

  const hudConceptText = document.getElementById("hudConceptText");
  const eli5Text = document.getElementById("eli5Text");
  const activeTierTitle = document.getElementById("activeTierTitle");
  const activeTierText = document.getElementById("activeTierText");
  const governingEquationCode = document.getElementById("governingEquationCode");
  const skillHudToast = document.getElementById("skillHudToast");
  const dagSvg = document.getElementById("dagSvg");
  const dagStatsTag = document.getElementById("dagStatsTag");

  const maskingSlider = document.getElementById("maskingSlider");
  const maskingReadout = document.getElementById("maskingReadout");
  const metricResolved = document.getElementById("metricResolved");
  const metricEntropy = document.getElementById("metricEntropy");
  const metricSpeedup = document.getElementById("metricSpeedup");
  const simWidgetTitle = document.getElementById("simWidgetTitle");

  const tokenCanvas = document.getElementById("tokenGridCanvas");
  const tokenCtx = tokenCanvas.getContext("2d");

  const manimCanvas = document.getElementById("manim3dCanvas");
  const manimCtx = manimCanvas.getContext("2d");
  const manimCosineMeter = document.getElementById("manimCosineMeter");
  const manimSourceCode = document.getElementById("manimSourceCode");
  const tabManimVisual = document.getElementById("tabManimVisual");
  const tabManimCode = document.getElementById("tabManimCode");
  const manimVisualPane = document.getElementById("manimVisualPane");
  const manimCodePane = document.getElementById("manimCodePane");

  const prodProofDrawer = document.getElementById("prodProofDrawer");
  const otelTraceWaterfall = document.getElementById("otelTraceWaterfall");
  const structlogAuditBox = document.getElementById("structlogAuditBox");
  const hitlModalBackdrop = document.getElementById("hitlModalBackdrop");
  const endingCalloutOverlay = document.getElementById("endingCalloutOverlay");

  // ---------------------------------------------------------------------------
  // 1. Interactive Token Grid & Parallel Attention Heatmap Renderer (2D Canvas)
  // ---------------------------------------------------------------------------
  const TOKEN_LABELS = [
    "VQ_04", "PATCH", "MOTION", "DEPTH", "EDGE", "CHROMA", "FLOW_X", "FLOW_Y",
    "NORM", "LIGHT", "POSE_A", "POSE_B", "SPEC", "SHADE", "TEX_1", "TEX_2",
    "OBJ_0", "OBJ_1", "BG_SKY", "HORIZ", "CAM_Z", "VEL_U", "VEL_V", "ANCHOR",
    "KEY_0", "KEY_1", "ATTN_Q", "ATTN_K", "ATTN_V", "LAT_0", "LAT_1", "LAT_2",
    "FREQ_L", "FREQ_H", "PHASE", "SYNC", "FRAME", "STRIDE", "TOKEN", "EMBED",
    "MASK_0", "MASK_1", "GRAD", "COS_S", "NORM_2", "PROJ", "HEAD_4", "OUT_Z"
  ];

  function renderTokenGrid() {
    const W = tokenCanvas.width;
    const H = tokenCanvas.height;
    tokenCtx.clearRect(0, 0, W, H);

    const cols = 8;
    const rows = 6;
    const padX = 12;
    const padY = 10;
    const gridW = W - padX * 2;
    const gridH = H - padY * 2;
    const cellW = gridW / cols;
    const cellH = gridH / rows;

    const maskPct = state.maskingRatio;
    const threshold = maskPct / 100.0;
    let resolvedCount = 0;
    const resolvedCenters = [];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        // Deterministic quasi-random confidence rank per token cell
        const rank = ((idx * 41 + r * 17 + c * 29) % 48) / 48.0;
        const isResolved = rank >= threshold;
        if (isResolved) {
          resolvedCount++;
        }

        const x = padX + c * cellW + 3;
        const y = padY + r * cellH + 3;
        const w = cellW - 6;
        const h = cellH - 6;

        const sharpness = Math.min(1.0, Math.max(0.15, (1.0 - threshold) * 0.85 + rank * 0.25));

        if (isResolved) {
          resolvedCenters.push({ x: x + w / 2, y: y + h / 2, sharpness });
          const grad = tokenCtx.createLinearGradient(x, y, x + w, y + h);
          grad.addColorStop(0, `rgba(16, 185, 129, ${0.28 + sharpness * 0.52})`);
          grad.addColorStop(1, `rgba(56, 189, 248, ${0.22 + sharpness * 0.48})`);
          tokenCtx.fillStyle = grad;
          tokenCtx.strokeStyle = `rgba(56, 189, 248, ${0.55 + sharpness * 0.45})`;
          tokenCtx.lineWidth = 1.6;
        } else {
          tokenCtx.fillStyle = "rgba(30, 27, 75, 0.55)";
          tokenCtx.strokeStyle = "rgba(168, 85, 247, 0.28)";
          tokenCtx.lineWidth = 1.0;
        }

        tokenCtx.beginPath();
        tokenCtx.roundRect(x, y, w, h, 5);
        tokenCtx.fill();
        tokenCtx.stroke();

        tokenCtx.font = isResolved ? "bold 10px monospace" : "10px monospace";
        tokenCtx.fillStyle = isResolved ? "#f8fafc" : "#94a3b8";
        const label = isResolved ? TOKEN_LABELS[idx % TOKEN_LABELS.length] : "[MASK]";
        tokenCtx.fillText(label, x + 7, y + h / 2 + 3);
      }
    }

    // Draw live parallel self-attention heatmap arcs between resolved anchor tokens
    for (let i = 0; i < resolvedCenters.length; i += 2) {
      const a = resolvedCenters[i];
      const b = resolvedCenters[(i + 3) % resolvedCenters.length];
      if (!a || !b || a === b) continue;
      tokenCtx.beginPath();
      tokenCtx.moveTo(a.x, a.y);
      tokenCtx.quadraticCurveTo((a.x + b.x) / 2, Math.min(a.y, b.y) - 14, b.x, b.y);
      tokenCtx.strokeStyle = `rgba(56, 189, 248, ${0.14 + (1.0 - threshold) * 0.22})`;
      tokenCtx.lineWidth = 1.1;
      tokenCtx.stroke();
    }

    const stepNum = Math.max(1, Math.min(16, Math.round(((100 - maskPct) / 85) * 16)));
    const cosinePeak = (0.32 + (1.0 - threshold) * 0.64).toFixed(2);
    maskingReadout.textContent = `Current Masking: ${maskPct}% (Parallel Step t = ${stepNum}/16)`;
    metricResolved.textContent = `${resolvedCount} / 48 (${Math.round((resolvedCount / 48) * 100)}%)`;
    metricEntropy.textContent = `${cosinePeak} Cosine Peak`;
    metricSpeedup.textContent = `${(12.0 + (1.0 - threshold) * 8.4).toFixed(1)}× vs Autoregressive`;

    state.cosineTarget = parseFloat(cosinePeak);
  }

  // ---------------------------------------------------------------------------
  // 2. 3D Manim Cosine Similarity Vector Clustering Animation (60fps Canvas)
  // ---------------------------------------------------------------------------
  const CLUSTERS = [
    { name: "Visual Token Cluster A", color: "#38bdf8", base: [1.5, 1.1, 0.9] },
    { name: "Motion Trajectory Cluster B", color: "#a855f7", base: [-1.4, 1.2, -0.7] },
    { name: "Semantic Anchor Cluster C", color: "#10b981", base: [0.3, -1.5, 1.2] },
  ];

  function project3D(x, y, z, angleY, angleX, W, H) {
    // Rotate around Y
    const cosY = Math.cos(angleY);
    const sinY = Math.sin(angleY);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    // Rotate around X
    const cosX = Math.cos(angleX);
    const sinX = Math.sin(angleX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    const scale = 210 / (4.2 + z2 * 0.35);
    return {
      px: W / 2 + x1 * scale,
      py: H / 2 - y2 * scale,
      depth: z2,
    };
  }

  function animateManim3D() {
    state.manimAngle += 0.012;
    state.cosineCurrent += (state.cosineTarget - state.cosineCurrent) * 0.08;

    const W = manimCanvas.width;
    const H = manimCanvas.height;
    manimCtx.clearRect(0, 0, W, H);

    const origin = project3D(0, 0, 0, state.manimAngle, 0.38, W, H);

    // Draw 3D Coordinate Axes
    const axes = [
      { end: [2.2, 0, 0], label: "e_x", color: "rgba(148, 163, 184, 0.35)" },
      { end: [0, 2.0, 0], label: "e_y", color: "rgba(148, 163, 184, 0.35)" },
      { end: [0, 0, 2.2], label: "e_z", color: "rgba(148, 163, 184, 0.35)" },
    ];
    axes.forEach((ax) => {
      const p = project3D(ax.end[0], ax.end[1], ax.end[2], state.manimAngle, 0.38, W, H);
      manimCtx.beginPath();
      manimCtx.moveTo(origin.px, origin.py);
      manimCtx.lineTo(p.px, p.py);
      manimCtx.strokeStyle = ax.color;
      manimCtx.lineWidth = 1;
      manimCtx.stroke();
      manimCtx.fillStyle = "#64748b";
      manimCtx.font = "10px monospace";
      manimCtx.fillText(ax.label, p.px + 4, p.py);
    });

    // Tightness of clustering increases as masking ratio decreases (or in Hero mode)
    const tightness = state.masteryMode === "Hero" ? 0.92 : Math.max(0.35, 1.0 - state.maskingRatio / 100.0);

    CLUSTERS.forEach((cluster, cIdx) => {
      const [bx, by, bz] = cluster.base;
      const centroid = project3D(bx, by, bz, state.manimAngle, 0.38, W, H);

      // Draw primary centroid vector arrow from origin
      manimCtx.beginPath();
      manimCtx.moveTo(origin.px, origin.py);
      manimCtx.lineTo(centroid.px, centroid.py);
      manimCtx.strokeStyle = cluster.color;
      manimCtx.lineWidth = 2.4;
      manimCtx.stroke();

      // Draw 6 satellite token vectors converging toward the cluster centroid
      for (let k = 0; k < 6; k++) {
        const spread = (1.0 - tightness) * 0.95 + 0.08;
        const offsetX = Math.sin(k * 2.1 + cIdx) * spread;
        const offsetY = Math.cos(k * 1.7 + cIdx) * spread;
        const offsetZ = Math.sin(k * 1.3 - cIdx) * spread;

        const pt = project3D(
          bx + offsetX,
          by + offsetY,
          bz + offsetZ,
          state.manimAngle,
          0.38,
          W,
          H
        );

        manimCtx.beginPath();
        manimCtx.moveTo(origin.px, origin.py);
        manimCtx.lineTo(pt.px, pt.py);
        manimCtx.strokeStyle = `${cluster.color}66`;
        manimCtx.lineWidth = 1.1;
        manimCtx.stroke();

        manimCtx.beginPath();
        manimCtx.arc(pt.px, pt.py, 3.5, 0, Math.PI * 2);
        manimCtx.fillStyle = cluster.color;
        manimCtx.fill();
      }

      // Label centroid
      manimCtx.fillStyle = "#f8fafc";
      manimCtx.font = "bold 11px sans-serif";
      manimCtx.fillText(cluster.name, centroid.px + 8, centroid.py - 6);
    });

    const displayedCos = state.masteryMode === "Hero" ? "0.94" : state.cosineCurrent.toFixed(2);
    manimCosineMeter.textContent = `Mean Cluster Alignment: ${displayedCos}`;
    requestAnimationFrame(animateManim3D);
  }

  // ---------------------------------------------------------------------------
  // 3. Skill-Tree Directed Acyclic Graph (DAG) SVG Renderer
  // ---------------------------------------------------------------------------
  function renderKnowledgeGraphDAG(dagData) {
    if (!dagData || !dagData.nodes) return;
    state.dag = dagData;

    const nodesMap = dagData.nodes;
    const nodeIds = Object.keys(nodesMap);
    dagStatsTag.textContent = `${nodeIds.length} Nodes • Acyclic Verified (${dagData.is_valid_dag ? "DAG=True" : "Check"}) • Async Indexed`;

    // Fixed clean layout positions for foundational + active frontier node
    const positions = {
      linear_algebra_cosine_similarity: { x: 95, y: 105 },
      self_attention_mechanism: { x: 255, y: 60 },
      vector_quantization_vqvae: { x: 255, y: 155 },
      transformer_architecture: { x: 430, y: 60 },
    };

    let dynamicIdx = 0;
    nodeIds.forEach((cid) => {
      if (!positions[cid]) {
        positions[cid] = { x: 585, y: 108 + (dynamicIdx * 55) };
        dynamicIdx++;
      }
    });

    let svgMarkup = `
      <defs>
        <linearGradient id="edgeGlow" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.65" />
          <stop offset="100%" stop-color="#10b981" stop-opacity="0.95" />
        </linearGradient>
      </defs>
    `;

    // Draw directed edges
    (dagData.edges || []).forEach((edge) => {
      const src = positions[edge.source_id];
      const dst = positions[edge.target_id];
      if (!src || !dst) return;
      const isHighlight = edge.target_id === state.conceptId;
      const midX = (src.x + dst.x) / 2;
      svgMarkup += `
        <path
          d="M ${src.x} ${src.y} C ${midX} ${src.y}, ${midX} ${dst.y}, ${dst.x} ${dst.y}"
          fill="none"
          stroke="${isHighlight ? "url(#edgeGlow)" : "rgba(148, 163, 184, 0.35)"}"
          stroke-width="${isHighlight ? "3.2" : "1.6"}"
          stroke-dasharray="${isHighlight ? "6 3" : "none"}"
        />
      `;
    });

    // Draw vertices
    nodeIds.forEach((cid) => {
      const node = nodesMap[cid];
      const pos = positions[cid];
      if (!pos) return;
      const isTarget = cid === state.conceptId;
      const scorePct = Math.round((node.mastery_score || 0.5) * 100);
      const fill = isTarget ? "#10b981" : "#38bdf8";
      const shortLabel = (node.title || cid).slice(0, 24);

      svgMarkup += `
        <g transform="translate(${pos.x}, ${pos.y})">
          <circle
            r="${isTarget ? 18 : 14}"
            fill="rgba(15, 23, 42, 0.95)"
            stroke="${fill}"
            stroke-width="${isTarget ? 3.2 : 2}"
          />
          <text
            y="4"
            text-anchor="middle"
            fill="#f8fafc"
            font-size="9"
            font-family="monospace"
            font-weight="bold"
          >${scorePct}%</text>
          <text
            y="${isTarget ? 32 : 28}"
            text-anchor="middle"
            fill="${isTarget ? "#6ee7b7" : "#cbd5e1"}"
            font-size="10"
            font-weight="${isTarget ? "700" : "500"}"
          >${shortLabel}</text>
        </g>
      `;
    });

    dagSvg.innerHTML = svgMarkup;
  }

  // ---------------------------------------------------------------------------
  // 4. OpenTelemetry Trace Waterfall & structlog JSON Renderer
  // ---------------------------------------------------------------------------
  function renderObservabilityProof() {
    const spans = state.traceSpans || [];
    if (spans.length === 0) {
      otelTraceWaterfall.innerHTML = "<div class='trace-row'>No spans recorded yet.</div>";
    } else {
      const maxDur = Math.max(1.0, ...spans.map((s) => s.duration_ms || 1.0));
      otelTraceWaterfall.innerHTML = spans
        .map((s) => {
          const widthPct = Math.max(12, Math.min(100, Math.round(((s.duration_ms || 1) / maxDur) * 100)));
          return `
            <div class="trace-row">
              <div style="display:flex;justify-content:space-between;">
                <span><strong>${s.agent_name}</strong> → ${s.name}</span>
                <span style="color:#38bdf8;">${s.duration_ms} ms [${s.model_id}]</span>
              </div>
              <div class="trace-bar-bg">
                <div class="trace-bar-fill" style="width:${widthPct}%"></div>
              </div>
            </div>
          `;
        })
        .join("");
    }

    structlogAuditBox.textContent = JSON.stringify(state.auditLogs.slice(-6), null, 2);
  }

  // ---------------------------------------------------------------------------
  // 5. Multi-Agent Pipeline Invocation (`POST /api/learn`)
  // ---------------------------------------------------------------------------
  function updateActiveTierView(tier) {
    state.masteryMode = tier;
    document.querySelectorAll(".tier-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tier === tier);
    });

    const titles = {
      ELI5: "ELI5 Everyday Analogy",
      Novice: "Novice Architectural Breakdown",
      Practitioner: "Practitioner Algorithmic Mechanics",
      Hero: "Hero Mode: Rigorous Mathematical Proof & Bounds",
    };
    activeTierTitle.textContent = titles[tier] || "Architectural Breakdown";
    activeTierText.textContent = state.explanations[tier] || state.explanations.Novice || "";

    document.getElementById("pillTriage").classList.add("active-route");
    document.getElementById("pillPedagogy").classList.toggle("active-route", tier !== "Hero");
    document.getElementById("pillVisualizer").classList.toggle("active-route", tier === "Hero");
  }

  async function executeLearningPipeline(customMode) {
    const mode = customMode || state.masteryMode;
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    hudConceptText.textContent = "Synthesizing across Triage (Flash) → Pedagogy (Pro) → Visualizer...";

    try {
      const response = await fetch("/api/learn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          learner_id: state.learnerId,
          session_id: state.sessionId,
          mastery_mode: mode,
          masking_ratio: state.maskingRatio / 100.0,
        }),
      });
      const data = await response.json();
      if (data.status !== "success") return;

      const primary = data.triage.primary_concept;
      state.conceptId = primary.concept_id;
      state.conceptTitle = primary.concept_title.split("(")[0].trim();
      state.explanations = data.pedagogy.explanations;

      hudConceptText.textContent = `Level: ${mode} | Concept: ${state.conceptTitle}`;
      eli5Text.textContent = state.explanations.ELI5;
      governingEquationCode.textContent = data.pedagogy.governing_equation;
      simWidgetTitle.textContent = `${state.conceptTitle} — Interactive Token Grid`;

      updateActiveTierView(mode);

      if (data.pedagogy.mastery_update) {
        const badgeText = data.pedagogy.mastery_update.hud_badge || `[+1 Mastered: ${state.conceptTitle}]`;
        skillHudToast.textContent = badgeText;
        skillHudToast.classList.add("pop-anim");
        setTimeout(() => skillHudToast.classList.remove("pop-anim"), 900);
        renderKnowledgeGraphDAG(data.pedagogy.mastery_update.dag_snapshot);
      }

      if (data.visualizer && data.visualizer.manim_animation) {
        state.manimCode = data.visualizer.manim_animation.manim_python_code || "";
        manimSourceCode.textContent = state.manimCode;
      }

      if (data.compaction_telemetry) {
        document.getElementById("proofCompactionStats").textContent =
          `Turns: ${data.compaction_telemetry.retained_turn_count} | Tokens: ${data.compaction_telemetry.compacted_tokens}`;
      }

      if (data.observability) {
        state.traceSpans = data.observability.trace_spans || [];
        state.auditLogs = data.observability.recent_audit_logs || [];
        renderObservabilityProof();
      }

      renderTokenGrid();
    } catch (err) {
      console.error("ArgonLearn pipeline error:", err);
    }
  }

  // ---------------------------------------------------------------------------
  // 6. Event Listeners & 30-90s Silent Video Demo Storyboard Player
  // ---------------------------------------------------------------------------
  function setStoryboardStep(stepNum) {
    document.querySelectorAll(".sb-step").forEach((el) => {
      el.classList.toggle("active", parseInt(el.dataset.step, 10) === stepNum);
    });
  }

  btnAnalyze.addEventListener("click", () => {
    setStoryboardStep(1);
    executeLearningPipeline("Novice");
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      promptInput.value = chip.dataset.prompt;
      setStoryboardStep(1);
      executeLearningPipeline("Novice");
    });
  });

  maskingSlider.addEventListener("input", (e) => {
    state.maskingRatio = parseInt(e.target.value, 10);
    setStoryboardStep(2);
    renderTokenGrid();
  });

  btnHeroMode.addEventListener("click", () => {
    setStoryboardStep(3);
    executeLearningPipeline("Hero");
  });

  document.querySelectorAll(".tier-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const tier = tab.dataset.tier;
      if (tier === "Hero") {
        setStoryboardStep(3);
        executeLearningPipeline("Hero");
      } else {
        updateActiveTierView(tier);
      }
    });
  });

  tabManimVisual.addEventListener("click", () => {
    tabManimVisual.classList.add("active");
    tabManimCode.classList.remove("active");
    manimVisualPane.classList.remove("hidden");
    manimCodePane.classList.add("hidden");
  });

  tabManimCode.addEventListener("click", () => {
    tabManimCode.classList.add("active");
    tabManimVisual.classList.remove("active");
    manimCodePane.classList.remove("hidden");
    manimVisualPane.classList.add("hidden");
  });

  btnProdProof.addEventListener("click", async () => {
    setStoryboardStep(4);
    prodProofDrawer.classList.toggle("hidden");
    try {
      const res = await fetch("/api/observability/telemetry");
      const data = await res.json();
      state.traceSpans = data.trace_spans || state.traceSpans;
      state.auditLogs = data.audit_logs || state.auditLogs;
      renderObservabilityProof();
    } catch (_) {}
  });

  btnCloseProof.addEventListener("click", () => {
    prodProofDrawer.classList.add("hidden");
  });

  btnHitlGate.addEventListener("click", () => {
    hitlModalBackdrop.classList.remove("hidden");
  });

  btnCancelHitl.addEventListener("click", () => {
    hitlModalBackdrop.classList.add("hidden");
  });

  btnApproveHitl.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/hitl/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          concept_id: state.conceptId,
          scene_class_name: "CosineSimilarityClusteringScene",
          execute_render: true,
          publish_to_shared_repo: true,
          human_approved: true,
          session_id: state.sessionId,
        }),
      });
      const data = await res.json();
      if (data.status === "success") {
        btnHitlGate.textContent = "✔ HITL Approved & Compiled";
        btnHitlGate.style.borderColor = "#10b981";
        btnHitlGate.style.color = "#34d399";
      }
    } catch (_) {}
    hitlModalBackdrop.classList.add("hidden");
  });

  btnEndingBanner.addEventListener("click", () => {
    endingCalloutOverlay.classList.remove("hidden");
  });

  btnDismissEnding.addEventListener("click", () => {
    endingCalloutOverlay.classList.add("hidden");
  });

  // Hands-Free Auto-Play of the 30-90s Silent Video Demo Storyboard
  btnAutoDemo.addEventListener("click", () => {
    state.demoTimers.forEach((t) => clearTimeout(t));
    state.demoTimers = [];

    endingCalloutOverlay.classList.add("hidden");
    prodProofDrawer.classList.add("hidden");

    // 0:00 - The Hook: Meta Muse
    setStoryboardStep(1);
    promptInput.value = "Meta just released Muse: Masked Generative Video Transformer.";
    state.maskingRatio = 80;
    maskingSlider.value = "80";
    executeLearningPipeline("Novice");

    // 0:03 - Interactive Toybox: Smoothly sweep Masking Ratio slider 80% -> 20%
    state.demoTimers.push(
      setTimeout(() => {
        setStoryboardStep(2);
        let currentRatio = 80;
        const interval = setInterval(() => {
          currentRatio -= 2;
          if (currentRatio <= 20) {
            currentRatio = 20;
            clearInterval(interval);
          }
          state.maskingRatio = currentRatio;
          maskingSlider.value = String(currentRatio);
          renderTokenGrid();
        }, 80);
      }, 2500)
    );

    // 0:07 - 0-to-Hero Graph & Manim Reel
    state.demoTimers.push(
      setTimeout(() => {
        setStoryboardStep(3);
        executeLearningPipeline("Hero");
      }, 6500)
    );

    // 0:11 - Prod & Architecture Proof (Cloud Run + OpenTelemetry Trace Waterfall)
    state.demoTimers.push(
      setTimeout(() => {
        setStoryboardStep(4);
        prodProofDrawer.classList.remove("hidden");
        renderObservabilityProof();
      }, 10500)
    );
  });

  // Initial Boot
  renderTokenGrid();
  requestAnimationFrame(animateManim3D);
  executeLearningPipeline("Novice");
})();
