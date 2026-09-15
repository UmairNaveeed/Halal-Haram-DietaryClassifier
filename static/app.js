/**
 * HALAL-BERT Classifier - Interactive Frontend Application
 * Designed & Engineered for Umair Naveed's FYP Demonstration
 */

document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------------
  // DOM Elements
  // -------------------------------------------------------------
  const modelStatusPill = document.getElementById('model-status-pill');
  const modelStatusText = document.getElementById('model-status-text');

  // Tabs
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  // Mode 1: Single
  const inputSingle = document.getElementById('input-single-text');
  const btnPredictSingle = document.getElementById('btn-predict-single');
  const presetChips = document.querySelectorAll('.preset-chip');

  // Mode 2: Batch
  const inputBatch = document.getElementById('input-batch-text');
  const btnPredictBatch = document.getElementById('btn-predict-batch');
  const btnSampleMixed = document.getElementById('btn-load-sample-mixed');
  const btnSampleClean = document.getElementById('btn-load-sample-clean');

  // Mode 3: Counterfactual
  const inputCfA = document.getElementById('input-cf-a');
  const inputCfB = document.getElementById('input-cf-b');
  const btnPredictCf = document.getElementById('btn-predict-cf');

  // Results Deck & Sub-Cards
  const resultsDeck = document.getElementById('results-deck');
  const singleResultCard = document.getElementById('single-result-card');
  const batchResultCard = document.getElementById('batch-result-card');
  const cfResultCard = document.getElementById('cf-result-card');

  // Single Result Elements
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const gaugeConfidence = document.getElementById('gauge-confidence');
  const gaugeLatency = document.getElementById('gauge-latency');
  const probHalalText = document.getElementById('prob-halal-text');
  const probHalalBar = document.getElementById('prob-halal-bar');
  const probHaramText = document.getElementById('prob-haram-text');
  const probHaramBar = document.getElementById('prob-haram-bar');
  const diagInputText = document.getElementById('diag-input-text');
  const diagEngine = document.getElementById('diag-engine');
  const diagFlagRow = document.getElementById('diag-flag-row');
  const diagFlaggedKeywords = document.getElementById('diag-flagged-keywords');
  const btnCopyReport = document.getElementById('btn-copy-report');
  const btnCopyText = document.getElementById('btn-copy-text');
  const btnToggleJson = document.getElementById('btn-toggle-json');
  const jsonDrawer = document.getElementById('json-drawer');
  const jsonCodeBlock = document.getElementById('json-code-block');

  // Batch Result Elements
  const batchOverallBadge = document.getElementById('batch-overall-badge');
  const batchOverallText = document.getElementById('batch-overall-text');
  const batchTotalCount = document.getElementById('batch-total-count');
  const batchHaramCount = document.getElementById('batch-haram-count');
  const batchTableBody = document.getElementById('batch-table-body');

  // Counterfactual Result Elements
  const cfTextA = document.getElementById('cf-text-a');
  const cfScoreA = document.getElementById('cf-score-a');
  const cfVerdictA = document.getElementById('cf-verdict-a');
  const cfTextB = document.getElementById('cf-text-b');
  const cfScoreB = document.getElementById('cf-score-b');
  const cfVerdictB = document.getElementById('cf-verdict-b');

  // Citation Copy
  const btnCopyBibtex = document.getElementById('btn-copy-bibtex');
  const bibtexCode = document.getElementById('bibtex-code');

  let lastSingleResult = null;

  // -------------------------------------------------------------
  // Check Backend & Model Status
  // -------------------------------------------------------------
  async function checkModelStatus() {
    try {
      const res = await fetch('/api/model-info');
      if (res.ok) {
        const data = await res.json();
        const loaded = data.live_status && data.live_status.is_model_loaded;
        const device = data.live_status && data.live_status.device ? data.live_status.device.toUpperCase() : 'CPU';
        modelStatusText.textContent = loaded ? `BERT-LoRA Active (${device})` : 'BERT-LoRA Standby (Fast Mode)';
        modelStatusPill.classList.toggle('active-ready', loaded);
      }
    } catch (e) {
      console.warn('Backend telemetry unreachable:', e);
      modelStatusText.textContent = 'API Offline';
    }
  }
  checkModelStatus();

  // -------------------------------------------------------------
  // Tabs Navigation
  // -------------------------------------------------------------
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mode');
      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      document.getElementById(`view-${mode}`).classList.add('active');
    });
  });

  // -------------------------------------------------------------
  // Preset Chips
  // -------------------------------------------------------------
  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const sample = chip.getAttribute('data-text');
      inputSingle.value = sample;
      runSinglePrediction(sample);
    });
  });

  // -------------------------------------------------------------
  // Sample Label Shortcuts
  // -------------------------------------------------------------
  if (btnSampleMixed) {
    btnSampleMixed.addEventListener('click', () => {
      inputBatch.value = 'Enriched wheat flour, vegetable shortening, sugar, water, white wine vinegar, salt, pork gelatin (2%), natural vanilla flavoring, carmine coloring.';
    });
  }

  if (btnSampleClean) {
    btnSampleClean.addEventListener('click', () => {
      inputBatch.value = 'Organic cold-pressed olive oil, water, white wine vinegar, sea salt, dried oregano, black pepper, garlic extract, lemon juice.';
    });
  }

  // -------------------------------------------------------------
  // Single Item Prediction Handler
  // -------------------------------------------------------------
  btnPredictSingle.addEventListener('click', () => {
    const text = inputSingle.value.trim();
    if (!text) {
      inputSingle.focus();
      return;
    }
    runSinglePrediction(text);
  });

  inputSingle.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const text = inputSingle.value.trim();
      if (text) runSinglePrediction(text);
    }
  });

  async function runSinglePrediction(text) {
    setButtonLoading(btnPredictSingle, true);
    hideAllResults();

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed.');
      }

      const data = await res.json();
      lastSingleResult = data;
      renderSingleResult(data);
    } catch (err) {
      alert(`Classification error: ${err.message}`);
    } finally {
      setButtonLoading(btnPredictSingle, false);
    }
  }

  function renderSingleResult(data) {
    const isHalal = data.prediction === 'HALAL';

    // Update Result Card class
    singleResultCard.className = `result-card ${isHalal ? 'verdict-halal' : 'verdict-haram'}`;

    // Verdict Icon
    verdictIcon.className = `verdict-badge-icon ${isHalal ? 'icon-halal' : 'icon-haram'}`;
    verdictIcon.innerHTML = isHalal
      ? `<svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>`
      : `<svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;

    // Verdict Title
    verdictTitle.textContent = data.prediction;
    verdictTitle.className = `verdict-title ${isHalal ? 'halal' : 'haram'}`;

    // Confidence & Latency
    gaugeConfidence.textContent = `${data.confidence}%`;
    gaugeLatency.textContent = `⚡ ${data.latency_ms}ms`;

    // Probability Bars
    const halalP = data.probabilities.halal;
    const haramP = data.probabilities.haram;
    probHalalText.textContent = `${halalP}%`;
    probHalalBar.style.width = `${halalP}%`;
    probHaramText.textContent = `${haramP}%`;
    probHaramBar.style.width = `${haramP}%`;

    // Diagnostics
    diagInputText.textContent = `"${data.text}"`;
    diagEngine.textContent = data.engine || 'BERT-LoRA Adapter';

    if (data.flagged_keywords && data.flagged_keywords.length > 0) {
      diagFlagRow.style.display = 'flex';
      diagFlaggedKeywords.textContent = data.flagged_keywords.join(', ');
    } else {
      diagFlagRow.style.display = 'none';
    }

    // Raw JSON Block
    jsonCodeBlock.textContent = JSON.stringify(data, null, 2);

    // Reveal Deck & Card
    resultsDeck.style.display = 'block';
    singleResultCard.style.display = 'block';
    resultsDeck.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // -------------------------------------------------------------
  // Batch Ingredient Audit Handler
  // -------------------------------------------------------------
  btnPredictBatch.addEventListener('click', async () => {
    const raw = inputBatch.value.trim();
    if (!raw) {
      inputBatch.focus();
      return;
    }

    setButtonLoading(btnPredictBatch, true);
    hideAllResults();

    try {
      const res = await fetch('/api/batch-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: raw }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Batch analysis failed.');
      }

      const data = await res.json();
      renderBatchResult(data);
    } catch (err) {
      alert(`Batch analysis error: ${err.message}`);
    } finally {
      setButtonLoading(btnPredictBatch, false);
    }
  });

  function renderBatchResult(data) {
    const isHaram = data.overall_prediction === 'HARAM';

    batchOverallBadge.className = `batch-badge ${isHaram ? 'badge-haram' : 'badge-halal'}`;
    batchOverallBadge.textContent = isHaram ? 'PRODUCT STATUS: HARAM (FORBIDDEN)' : 'PRODUCT STATUS: 100% HALAL COMPLIANT';

    batchOverallText.textContent = isHaram
      ? `Flagged ${data.haram_ingredients_detected} forbidden ingredient trigger(s). Product is unsuitable for Halal dietary compliance.`
      : 'All parsed ingredients passed semantic verification with zero prohibited substances detected.';

    batchTotalCount.textContent = data.total_ingredients_scanned;
    batchHaramCount.textContent = data.haram_ingredients_detected;

    // Build Table Rows
    batchTableBody.innerHTML = '';
    data.ingredient_breakdown.forEach((item, idx) => {
      const row = document.createElement('tr');
      const itemIsHaram = item.prediction === 'HARAM';
      row.innerHTML = `
        <td style="color: var(--text-muted);">${idx + 1}</td>
        <td><strong>${escapeHtml(item.name)}</strong></td>
        <td>
          <span class="status-tag ${itemIsHaram ? 'haram' : 'halal'}">
            ${item.prediction}
          </span>
        </td>
        <td style="font-family: var(--font-mono);">${item.confidence}%</td>
        <td>
          <span style="font-size: 0.8rem; color: ${itemIsHaram ? 'var(--color-haram-light)' : 'var(--color-emerald-light)'};">
            ${itemIsHaram ? '⚠️ Prohibited Flag' : '✓ Permissible'}
          </span>
        </td>
      `;
      batchTableBody.appendChild(row);
    });

    // Reveal Deck & Card
    resultsDeck.style.display = 'block';
    batchResultCard.style.display = 'block';
    resultsDeck.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // -------------------------------------------------------------
  // Counterfactual Differential Comparison Handler
  // -------------------------------------------------------------
  btnPredictCf.addEventListener('click', async () => {
    const textA = inputCfA.value.trim();
    const textB = inputCfB.value.trim();

    if (!textA || !textB) {
      alert('Please fill both Candidate A and Candidate B fields.');
      return;
    }

    setButtonLoading(btnPredictCf, true);
    hideAllResults();

    try {
      const [resA, resB] = await Promise.all([
        fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textA }),
        }).then(r => r.json()),
        fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textB }),
        }).then(r => r.json()),
      ]);

      renderCounterfactualResult(resA, resB);
    } catch (err) {
      alert(`Differential comparison error: ${err.message}`);
    } finally {
      setButtonLoading(btnPredictCf, false);
    }
  });

  function renderCounterfactualResult(resA, resB) {
    cfTextA.textContent = `"${resA.text}"`;
    cfScoreA.textContent = `${resA.confidence}% Model Confidence`;
    cfVerdictA.innerHTML = `<div class="cf-pill ${resA.prediction.toLowerCase()}">${resA.prediction}</div>`;

    cfTextB.textContent = `"${resB.text}"`;
    cfScoreB.textContent = `${resB.confidence}% Model Confidence`;
    cfVerdictB.innerHTML = `<div class="cf-pill ${resB.prediction.toLowerCase()}">${resB.prediction}</div>`;

    // Reveal Deck & Card
    resultsDeck.style.display = 'block';
    cfResultCard.style.display = 'block';
    resultsDeck.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // -------------------------------------------------------------
  // Utilities & Helpers
  // -------------------------------------------------------------
  function hideAllResults() {
    singleResultCard.style.display = 'none';
    batchResultCard.style.display = 'none';
    cfResultCard.style.display = 'none';
    jsonDrawer.style.display = 'none';
  }

  function setButtonLoading(btn, isLoading) {
    if (isLoading) {
      btn.disabled = true;
      btn.dataset.prevText = btn.innerHTML;
      btn.innerHTML = `
        <svg class="spinner" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" style="animation: spin 0.8s linear infinite;">
          <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="12"/>
        </svg>
        <span>Running Inference...</span>
      `;
    } else {
      btn.disabled = false;
      if (btn.dataset.prevText) btn.innerHTML = btn.dataset.prevText;
    }
  }

  // Toggle JSON Inspector
  btnToggleJson.addEventListener('click', () => {
    const isVisible = jsonDrawer.style.display === 'block';
    jsonDrawer.style.display = isVisible ? 'none' : 'block';
  });

  // Copy Verification Report
  btnCopyReport.addEventListener('click', () => {
    if (!lastSingleResult) return;
    const report = [
      `=== HALAL-BERT DIETARY VERIFICATION REPORT ===`,
      `Item: "${lastSingleResult.text}"`,
      `Verdict: ${lastSingleResult.prediction}`,
      `Confidence: ${lastSingleResult.confidence}%`,
      `Halal Probability: ${lastSingleResult.probabilities.halal}%`,
      `Haram Probability: ${lastSingleResult.probabilities.haram}%`,
      `Inference Latency: ${lastSingleResult.latency_ms}ms`,
      `Engine: ${lastSingleResult.engine}`,
      `Model: Umair1710/Bert-Lora-Finedtuned-Hala_Haram_Detection`,
      `Verified with BERT-LoRA Dietary Classifier by Umair Naveed`,
    ].join('\n');

    navigator.clipboard.writeText(report).then(() => {
      btnCopyText.textContent = '✓ Summary Copied!';
      setTimeout(() => { btnCopyText.textContent = 'Copy Verification Summary'; }, 2200);
    });
  });

  // Copy BibTeX
  if (btnCopyBibtex) {
    btnCopyBibtex.addEventListener('click', () => {
      navigator.clipboard.writeText(bibtexCode.textContent).then(() => {
        btnCopyBibtex.textContent = '✓ Copied!';
        setTimeout(() => { btnCopyBibtex.textContent = 'Copy BibTeX'; }, 2200);
      });
    });
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
});
