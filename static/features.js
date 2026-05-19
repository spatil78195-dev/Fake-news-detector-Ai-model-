/**
 * features.js — Advanced Feature Panels
 * Handles: URL check, Image check, Voice check, Multi-language check
 * Isolated from script.js — no shared state modifications
 */

'use strict';

// ─────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────
document.querySelectorAll('.feat-tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.feat-tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.feat-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`feat-${target}`)?.classList.add('active');
  });
});

// ─────────────────────────────────────────────
// Shared utilities
// ─────────────────────────────────────────────
function showFeatError(errorEl, msg) {
  errorEl.textContent = msg;
  errorEl.classList.add('visible');
}

function hideFeatError(errorEl) {
  errorEl.classList.remove('visible');
}

function setFeatLoading(loadingEl, on) {
  loadingEl.classList.toggle('visible', on);
}

function verdictClass(result) {
  return result === 'Fake' ? 'fake' : 'real';
}

function verdictEmoji(result) {
  return result === 'Fake' ? '🚨' : '✅';
}

function probBarsHTML(fakeProb, realProb) {
  return `
    <div style="margin-top:12px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="font-size:0.75rem;font-weight:600;color:var(--fake-color);width:34px">FAKE</span>
        <div style="flex:1;height:7px;background:rgba(255,255,255,0.07);border-radius:100px;overflow:hidden">
          <div style="height:100%;border-radius:100px;background:var(--fake-color);width:${fakeProb}%;transition:width 1s ease"></div>
        </div>
        <span style="font-size:0.75rem;font-weight:600;color:var(--fake-color);width:40px;text-align:right">${fakeProb}%</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-size:0.75rem;font-weight:600;color:var(--real-color);width:34px">REAL</span>
        <div style="flex:1;height:7px;background:rgba(255,255,255,0.07);border-radius:100px;overflow:hidden">
          <div style="height:100%;border-radius:100px;background:var(--real-color);width:${realProb}%;transition:width 1s ease"></div>
        </div>
        <span style="font-size:0.75rem;font-weight:600;color:var(--real-color);width:40px;text-align:right">${realProb}%</span>
      </div>
    </div>`;
}

// ═══════════════════════════════════════════════════════════
// FEATURE 1 — URL CHECKER
// ═══════════════════════════════════════════════════════════
(function initUrlChecker() {
  const input      = document.getElementById('url-input');
  const btn        = document.getElementById('url-analyze-btn');
  const loadingEl  = document.getElementById('url-loading');
  const errorEl    = document.getElementById('url-error');
  const resultArea = document.getElementById('url-result');

  if (!btn) return;

  btn.addEventListener('click', runUrlCheck);
  input?.addEventListener('keydown', e => { if (e.key === 'Enter') runUrlCheck(); });

  async function runUrlCheck() {
    const url = input?.value.trim();
    if (!url) { showFeatError(errorEl, 'Please enter a URL.'); return; }
    if (!url.startsWith('http')) { showFeatError(errorEl, 'URL must start with http:// or https://'); return; }

    hideFeatError(errorEl);
    setFeatLoading(loadingEl, true);
    resultArea.innerHTML = '';

    try {
      const res  = await fetch('/api/url-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      renderUrlResult(data);
    } catch (e) {
      showFeatError(errorEl, `Error: ${e.message}`);
    } finally {
      setFeatLoading(loadingEl, false);
    }
  }

  function renderUrlResult(d) {
    const vc   = verdictClass(d.result);
    const cred = d.credibility || {};
    const credColor = cred.score >= 60 ? 'real' : cred.score >= 35 ? 'warn' : 'fake';

    const signals = (cred.signals || []).map(s =>
      `<div class="cred-signal">
         <div class="cred-dot ${s.positive ? 'positive' : 'negative'}"></div>
         <span>${s.text}</span>
       </div>`
    ).join('');

    resultArea.innerHTML = `
      <div class="feat-verdict-banner ${vc}">
        <span class="feat-verdict-emoji">${verdictEmoji(d.result)}</span>
        <div>
          <div class="feat-verdict-label">${d.result}</div>
          <div class="feat-verdict-sub">${d.confidence}% confidence · ${d.word_count} words extracted</div>
        </div>
      </div>

      ${d.title ? `<div class="url-title-text" style="margin-top:16px">${escH(d.title)}</div>` : ''}
      ${d.excerpt ? `<div class="url-excerpt-box">${escH(d.excerpt)}</div>` : ''}

      ${probBarsHTML(d.fake_prob, d.real_prob)}

      <div class="feat-result-grid visible" style="margin-top:18px">
        <div class="feat-result-metric">
          <div class="feat-metric-label">Source Credibility</div>
          <div class="feat-metric-value ${credColor}">${cred.score ?? '--'}/100</div>
          <div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px">${cred.label || ''}</div>
        </div>
        <div class="feat-result-metric">
          <div class="feat-metric-label">Extraction Method</div>
          <div class="feat-metric-value" style="font-size:0.95rem">${d.method || 'N/A'}</div>
        </div>
      </div>

      ${signals ? `<div class="cred-signals" style="margin-top:4px">${signals}</div>` : ''}
    `;
  }
})();

// ═══════════════════════════════════════════════════════════
// FEATURE 2 — IMAGE CHECKER
// ═══════════════════════════════════════════════════════════
(function initImageChecker() {
  const dropzone   = document.getElementById('img-dropzone');
  const fileInput  = document.getElementById('img-file-input');
  const filenameEl = document.getElementById('img-filename');
  const btn        = document.getElementById('img-analyze-btn');
  const loadingEl  = document.getElementById('img-loading');
  const errorEl    = document.getElementById('img-error');
  const resultArea = document.getElementById('img-result');
  const preview    = document.getElementById('img-preview-orig');

  if (!btn) return;

  // Drag-over styling
  dropzone?.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone?.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) loadImageFile(file);
  });

  fileInput?.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) loadImageFile(file);
  });

  function loadImageFile(file) {
    filenameEl.textContent = file.name;
    const reader = new FileReader();
    reader.onload = e => { preview.src = e.target.result; preview.style.display = 'block'; };
    reader.readAsDataURL(file);
  }

  btn.addEventListener('click', async () => {
    const file = fileInput?.files[0];
    if (!file) { showFeatError(errorEl, 'Please select an image file.'); return; }

    hideFeatError(errorEl);
    setFeatLoading(loadingEl, true);
    resultArea.innerHTML = '';

    const form = new FormData();
    form.append('image', file);

    try {
      const res  = await fetch('/api/image-check', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      renderImageResult(data);
    } catch (e) {
      showFeatError(errorEl, `Error: ${e.message}`);
    } finally {
      setFeatLoading(loadingEl, false);
    }
  });

  function renderImageResult(d) {
    const score = d.manipulation_score ?? 50;
    const vc    = score >= 60 ? 'fake' : score >= 35 ? 'warn' : 'real';
    const emoji = score >= 60 ? '🚨' : score >= 35 ? '⚠️' : '✅';

    const issuesList = (d.issues || []).map(i =>
      `<li style="margin-bottom:4px;font-size:0.82rem;color:var(--text-secondary)">${escH(i)}</li>`
    ).join('');

    resultArea.innerHTML = `
      <div class="feat-verdict-banner ${vc}">
        <span class="feat-verdict-emoji">${emoji}</span>
        <div>
          <div class="feat-verdict-label">${d.verdict || 'Unknown'}</div>
          <div class="feat-verdict-sub">Manipulation score: ${score}/100 · ${d.image_size || ''}</div>
        </div>
      </div>

      <div class="feat-result-grid visible" style="margin-top:18px">
        <div class="feat-result-metric">
          <div class="feat-metric-label">ELA Score</div>
          <div class="feat-metric-value ${d.ela_score > 65 ? 'fake' : 'real'}">${d.ela_score ?? '--'}</div>
          <div style="font-size:0.72rem;color:var(--text-muted);margin-top:3px">Error Level Analysis</div>
        </div>
        <div class="feat-result-metric">
          <div class="feat-metric-label">Metadata Score</div>
          <div class="feat-metric-value ${d.metadata_score > 55 ? 'fake' : 'real'}">${d.metadata_score ?? '--'}</div>
          <div style="font-size:0.72rem;color:var(--text-muted);margin-top:3px">EXIF Inspection</div>
        </div>
        <div class="feat-result-metric">
          <div class="feat-metric-label">Noise Score</div>
          <div class="feat-metric-value ${d.opencv_score > 55 ? 'fake' : 'real'}">${d.opencv_score ?? '--'}</div>
          <div style="font-size:0.72rem;color:var(--text-muted);margin-top:3px">Block Analysis</div>
        </div>
        <div class="feat-result-metric">
          <div class="feat-metric-label">Format</div>
          <div class="feat-metric-value" style="font-size:0.95rem">${d.image_format || 'N/A'}</div>
        </div>
      </div>

      ${issuesList ? `<ul style="margin-top:14px;padding-left:16px">${issuesList}</ul>` : ''}

      ${d.ela_preview ? `
        <div class="img-preview-row">
          <div class="img-preview-box">
            <img id="img-preview-orig-2" src="${preview.src}" alt="Original">
            <div class="img-preview-label">Original Image</div>
          </div>
          <div class="img-preview-box">
            <img src="${d.ela_preview}" alt="ELA Analysis">
            <div class="img-preview-label">ELA Heatmap</div>
          </div>
        </div>` : ''}
    `;
  }
})();

// ═══════════════════════════════════════════════════════════
// FEATURE 3 — VOICE CHECKER
// ═══════════════════════════════════════════════════════════
(function initVoiceChecker() {
  const dropzone   = document.getElementById('voice-dropzone');
  const fileInput  = document.getElementById('voice-file-input');
  const filenameEl = document.getElementById('voice-filename');
  const btn        = document.getElementById('voice-analyze-btn');
  const loadingEl  = document.getElementById('voice-loading');
  const errorEl    = document.getElementById('voice-error');
  const resultArea = document.getElementById('voice-result');

  if (!btn) return;

  dropzone?.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone?.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) { fileInput.files = e.dataTransfer.files; filenameEl.textContent = file.name; }
  });

  fileInput?.addEventListener('change', () => {
    if (fileInput.files[0]) filenameEl.textContent = fileInput.files[0].name;
  });

  btn.addEventListener('click', async () => {
    const file = fileInput?.files[0];
    if (!file) { showFeatError(errorEl, 'Please select an audio file (.wav recommended).'); return; }

    hideFeatError(errorEl);
    setFeatLoading(loadingEl, true);
    resultArea.innerHTML = '';

    const form = new FormData();
    form.append('audio', file);

    try {
      const res  = await fetch('/api/voice-check', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      renderVoiceResult(data);
    } catch (e) {
      showFeatError(errorEl, `Error: ${e.message}`);
    } finally {
      setFeatLoading(loadingEl, false);
    }
  });

  function renderVoiceResult(d) {
    const vc = verdictClass(d.result);
    resultArea.innerHTML = `
      <div class="feat-verdict-banner ${vc}">
        <span class="feat-verdict-emoji">${verdictEmoji(d.result)}</span>
        <div>
          <div class="feat-verdict-label">${d.result}</div>
          <div class="feat-verdict-sub">${d.confidence}% confidence · ${d.word_count} words transcribed</div>
        </div>
      </div>

      <div class="voice-transcript-box">"${escH(d.transcript)}"</div>

      ${probBarsHTML(d.fake_prob, d.real_prob)}
    `;
  }
})();

// ═══════════════════════════════════════════════════════════
// FEATURE 4 — MULTI-LANGUAGE
// ═══════════════════════════════════════════════════════════
(function initLangChecker() {
  const textarea   = document.getElementById('lang-input');
  const btn        = document.getElementById('lang-analyze-btn');
  const loadingEl  = document.getElementById('lang-loading');
  const errorEl    = document.getElementById('lang-error');
  const resultArea = document.getElementById('lang-result');

  if (!btn) return;

  btn.addEventListener('click', runLangCheck);

  async function runLangCheck() {
    const text = textarea?.value.trim();
    if (!text || text.length < 20) {
      showFeatError(errorEl, 'Please enter at least 20 characters.');
      return;
    }

    hideFeatError(errorEl);
    setFeatLoading(loadingEl, true);
    resultArea.innerHTML = '';

    try {
      const res  = await fetch('/api/lang-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      renderLangResult(data);
    } catch (e) {
      showFeatError(errorEl, `Error: ${e.message}`);
    } finally {
      setFeatLoading(loadingEl, false);
    }
  }

  function renderLangResult(d) {
    const vc = verdictClass(d.result);
    const isTranslated = d.original_text !== d.translated_text;

    resultArea.innerHTML = `
      <div class="lang-badge">
        🌐 Detected: ${escH(d.detected_lang_name)} (${d.detected_lang_code})
      </div>

      <div class="feat-verdict-banner ${vc}">
        <span class="feat-verdict-emoji">${verdictEmoji(d.result)}</span>
        <div>
          <div class="feat-verdict-label">${d.result}</div>
          <div class="feat-verdict-sub">${d.confidence}% confidence · ${d.word_count} words</div>
        </div>
      </div>

      ${probBarsHTML(d.fake_prob, d.real_prob)}

      ${isTranslated ? `
        <div class="lang-comparison" style="margin-top:16px">
          <div class="lang-box">
            <div class="lang-box-label">Original (${escH(d.detected_lang_name)})</div>
            <div class="lang-box-text">${escH(d.original_text.slice(0, 300))}${d.original_text.length > 300 ? '...' : ''}</div>
          </div>
          <div class="lang-box">
            <div class="lang-box-label">Translated (English)</div>
            <div class="lang-box-text">${escH(d.translated_text.slice(0, 300))}${d.translated_text.length > 300 ? '...' : ''}</div>
          </div>
        </div>` : ''}

      ${d.translation_note ? `<div style="font-size:0.78rem;color:var(--text-muted);margin-top:10px;padding:8px 12px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid var(--border)">${escH(d.translation_note)}</div>` : ''}
      ${d.translation_warning ? `<div style="font-size:0.78rem;color:#f59e0b;margin-top:6px">${escH(d.translation_warning)}</div>` : ''}
    `;
  }
})();

// ─────────────────────────────────────────────
// Shared HTML escape helper
// ─────────────────────────────────────────────
function escH(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
