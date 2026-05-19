/**
 * script.js — Fake News Detector
 * Handles: typing animation, API calls, result display,
 *          Chart.js stats, history management, PDF report, sample articles.
 *
 * Bug fixes applied:
 *   - Renamed `history` to `predHistory` (avoids shadowing window.history)
 *   - Sample text stored in a Map by index, not in data-text attribute
 *     (prevents HTML entity corruption in textarea)
 *   - Download button toggled via .hidden CSS class, not inline style
 *   - PDF: auto page-break when article text exceeds page height
 *   - Result card animation re-triggered correctly on each new prediction
 *   - Defensive null-checks on all DOM refs
 */

'use strict';

// ─────────────────────────────────────────────
// DOM References
// ─────────────────────────────────────────────
const newsInput      = document.getElementById('news-input');
const charCounter    = document.getElementById('char-counter');
const analyzeBtn     = document.getElementById('analyze-btn');
const clearBtn       = document.getElementById('clear-btn');
const btnSpinner     = document.getElementById('btn-spinner');
const btnText        = document.getElementById('btn-text');
const errorMsg       = document.getElementById('error-msg');
const errorText      = document.getElementById('error-text');
const resultCard     = document.getElementById('result-card');
const resultLabel    = document.getElementById('result-label');
const resultEmoji    = document.getElementById('result-emoji');
const ringFill       = document.getElementById('ring-fill');
const ringPct        = document.getElementById('ring-pct');
const fakeProbBar    = document.getElementById('fake-prob-bar');
const realProbBar    = document.getElementById('real-prob-bar');
const fakeProbPct    = document.getElementById('fake-prob-pct');
const realProbPct    = document.getElementById('real-prob-pct');
const resultWordCount= document.getElementById('result-word-count');
const resultTime     = document.getElementById('result-time');
const downloadBtn    = document.getElementById('download-btn');
const historyClear   = document.getElementById('history-clear');
const historyBody    = document.getElementById('history-body');
const samplesGrid    = document.getElementById('samples-grid');
const totalAnalyzed  = document.getElementById('total-analyzed');
const totalFake      = document.getElementById('total-fake');
const totalReal      = document.getElementById('total-real');
const sessionAccuracy= document.getElementById('session-accuracy');

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
// FIX: renamed from `history` to avoid shadowing window.history
let predHistory = JSON.parse(sessionStorage.getItem('fnHistory') || '[]');
let chartInst   = null;
let lastResult  = null;   // used for PDF download

// FIX: sample texts stored in a Map (index → raw text) to avoid
//      HTML entity corruption when writing into textarea via data-text
const sampleTextMap = new Map();

// Session stats counters
const stats = JSON.parse(sessionStorage.getItem('fnStats') || '{"total":0,"fake":0,"real":0}');

// ─────────────────────────────────────────────
// Typing animation
// ─────────────────────────────────────────────
(function typingAnimation() {
  const el      = document.getElementById('typing-target');
  if (!el) return;
  const phrases = [
    'Detect Fake News',
    'Fight Misinformation',
    'Verify Your Sources',
    'Protect the Truth',
  ];
  let pIdx = 0, cIdx = 0, deleting = false;

  function tick() {
    const phrase = phrases[pIdx];
    if (!deleting) {
      el.textContent = phrase.slice(0, ++cIdx);
      if (cIdx === phrase.length) {
        deleting = true;
        setTimeout(tick, 2000);
        return;
      }
      setTimeout(tick, 90);
    } else {
      el.textContent = phrase.slice(0, --cIdx);
      if (cIdx === 0) {
        deleting = false;
        pIdx = (pIdx + 1) % phrases.length;
        setTimeout(tick, 400);
        return;
      }
      setTimeout(tick, 45);
    }
  }
  tick();
})();

// ─────────────────────────────────────────────
// Character counter
// ─────────────────────────────────────────────
newsInput.addEventListener('input', () => {
  const len = newsInput.value.length;
  charCounter.textContent = `${len} characters`;
  charCounter.style.color = len < 20 ? '#ef4444' : 'var(--text-muted)';
});

// ─────────────────────────────────────────────
// Clear button
// ─────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  newsInput.value = '';
  charCounter.textContent = '0 characters';
  charCounter.style.color = 'var(--text-muted)';
  hideError();
  resultCard.style.display = 'none';
  // FIX: use .hidden class instead of inline style (avoids CSS display:flex conflict)
  downloadBtn.classList.add('hidden');
});

// ─────────────────────────────────────────────
// Show / hide error
// ─────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorMsg.classList.add('visible');
}

function hideError() {
  errorMsg.classList.remove('visible');
}

// ─────────────────────────────────────────────
// Analyze button — main prediction flow
// ─────────────────────────────────────────────
analyzeBtn.addEventListener('click', analyze);

// Allow Ctrl+Enter to submit
newsInput.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') analyze();
});

async function analyze() {
  const text = newsInput.value.trim();

  // Client-side validation
  if (!text) {
    showError('Please paste a news article before analyzing.');
    return;
  }
  if (text.length < 20) {
    showError('Please enter at least 20 characters for accurate analysis.');
    return;
  }

  hideError();
  setLoading(true);

  // FIX: properly reset animation so it fires again on each new prediction
  resultCard.style.display = 'none';
  resultCard.style.animation = 'none';
  void resultCard.offsetWidth; // trigger reflow

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Server error. Please try again.');
    }

    displayResult(data, text);
    updateStats(data.result);
    addToHistory(text, data);
    renderChart();
    renderHistory();

  } catch (err) {
    showError(`Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ─────────────────────────────────────────────
// Loading state
// ─────────────────────────────────────────────
function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnSpinner.style.display = loading ? 'block' : 'none';
  btnText.textContent = loading ? 'Analyzing...' : 'Analyze Article';
}

// ─────────────────────────────────────────────
// Display result card
// ─────────────────────────────────────────────
function displayResult(data, text) {
  lastResult = { data, text, timestamp: new Date().toLocaleString() };

  const isFake = data.result === 'Fake';

  // Card class + re-enable animation
  resultCard.className = `result-card ${isFake ? 'fake' : 'real'}`;
  resultCard.style.animation = '';
  resultCard.style.display = 'block';

  // Label & emoji
  resultLabel.textContent = data.result;
  resultEmoji.textContent = isFake ? '🚨' : '✅';

  // Confidence ring — reset then animate
  const circumference = 2 * Math.PI * 34; // r=34
  const offset = circumference - (data.confidence / 100) * circumference;
  ringFill.style.transition = 'none';
  ringFill.style.strokeDasharray  = circumference;
  ringFill.style.strokeDashoffset = circumference; // start empty
  void ringFill.getBoundingClientRect();            // force reflow
  ringFill.style.transition = 'stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1)';
  ringFill.style.strokeDashoffset = offset;
  ringPct.textContent = `${data.confidence}%`;

  // Probability bars — reset then animate
  fakeProbBar.style.width = '0%';
  realProbBar.style.width = '0%';
  setTimeout(() => {
    fakeProbBar.style.width = `${data.fake_prob}%`;
    realProbBar.style.width = `${data.real_prob}%`;
  }, 80);
  fakeProbPct.textContent = `${data.fake_prob}%`;
  realProbPct.textContent = `${data.real_prob}%`;

  // Meta
  resultWordCount.textContent = `${data.word_count} words`;
  resultTime.textContent = new Date().toLocaleTimeString();

  // FIX: show download button via .hidden class toggle
  downloadBtn.classList.remove('hidden');
}

// ─────────────────────────────────────────────
// Stats update
// ─────────────────────────────────────────────
function updateStats(result) {
  stats.total += 1;
  if (result === 'Fake') stats.fake += 1;
  else stats.real += 1;

  sessionStorage.setItem('fnStats', JSON.stringify(stats));

  totalAnalyzed.textContent = stats.total;
  totalFake.textContent     = stats.fake;
  totalReal.textContent     = stats.real;

  const acc = stats.total > 0
    ? `${Math.round((stats.real / stats.total) * 100)}% real`
    : '--';
  sessionAccuracy.textContent = acc;
}

// ─────────────────────────────────────────────
// Chart.js doughnut
// ─────────────────────────────────────────────
function renderChart() {
  const canvas = document.getElementById('stats-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  document.getElementById('chart-center-num').textContent = stats.total;

  if (chartInst) chartInst.destroy();

  chartInst = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Fake', 'Real'],
      datasets: [{
        data: [stats.fake || 0.001, stats.real || 0.001],
        backgroundColor: ['rgba(239,68,68,0.85)', 'rgba(34,197,94,0.85)'],
        borderColor:     ['#ef4444', '#22c55e'],
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw === 0.001 ? 0 : ctx.raw}`,
          },
        },
      },
      animation: { animateRotate: true, duration: 800 },
    },
  });
}

// ─────────────────────────────────────────────
// History management
// FIX: renamed `history` → `predHistory` to avoid shadowing window.history
// ─────────────────────────────────────────────
function addToHistory(text, data) {
  const entry = {
    snippet   : text.slice(0, 80) + (text.length > 80 ? '...' : ''),
    result    : data.result,
    confidence: data.confidence,
    time      : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };
  predHistory.unshift(entry);
  if (predHistory.length > 10) predHistory.pop();
  sessionStorage.setItem('fnHistory', JSON.stringify(predHistory));
}

function renderHistory() {
  if (predHistory.length === 0) {
    historyBody.innerHTML = `<tr><td colspan="4" class="history-empty">No predictions yet. Analyze an article above!</td></tr>`;
    return;
  }
  historyBody.innerHTML = predHistory.map(h => `
    <tr>
      <td class="history-snippet">${escHtml(h.snippet)}</td>
      <td><span class="history-badge ${h.result.toLowerCase()}">${h.result}</span></td>
      <td style="color:var(--text-primary);font-weight:600">${h.confidence}%</td>
      <td>${h.time}</td>
    </tr>
  `).join('');
}

historyClear.addEventListener('click', () => {
  predHistory = [];
  sessionStorage.removeItem('fnHistory');
  renderHistory();
});

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ─────────────────────────────────────────────
// Load sample articles from backend
// FIX: raw text stored in Map (sampleTextMap), not data-text attribute,
//      so HTML entities never corrupt the textarea content
// ─────────────────────────────────────────────
async function loadSamples() {
  try {
    const res = await fetch('/api/sample');
    if (!res.ok) throw new Error('Failed to load samples');
    const samples = await res.json();
    renderSamples(samples);
  } catch {
    samplesGrid.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;text-align:center;padding:16px 0">Sample articles unavailable.</p>';
  }
}

function renderSamples(samples) {
  sampleTextMap.clear();

  samplesGrid.innerHTML = samples.map((s, i) => `
    <button class="sample-btn" data-idx="${i}" title="${escHtml(s.title)}">
      <span class="sample-tag ${s.label}">${s.label}</span>
      <span>${escHtml(s.title)}</span>
    </button>
  `).join('');

  // Store raw text in Map
  samples.forEach((s, i) => sampleTextMap.set(i, s.text));

  samplesGrid.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.idx, 10);
      const rawText = sampleTextMap.get(idx) || '';
      newsInput.value = rawText;
      newsInput.dispatchEvent(new Event('input'));
      newsInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      newsInput.focus();
    });
  });
}

// ─────────────────────────────────────────────
// PDF Report (jsPDF)
// FIX: added page-break handling so long articles don't overflow
// ─────────────────────────────────────────────
downloadBtn.addEventListener('click', () => {
  if (!lastResult) return;
  const { data, text, timestamp } = lastResult;

  if (!window.jspdf) {
    alert('PDF library not loaded. Please check your internet connection and refresh.');
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });

  const isFake = data.result === 'Fake';
  const accent = isFake ? [239, 68, 68] : [34, 197, 94];
  const pageH  = doc.internal.pageSize.getHeight(); // ~297mm
  const margin = 14;
  const maxW   = 182; // 210 - 2*14

  // ── Header bar ──
  doc.setFillColor(...accent);
  doc.rect(0, 0, 210, 20, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.text('FAKEGUARD AI  —  ANALYSIS REPORT', margin, 13);

  // ── Verdict ──
  doc.setTextColor(20, 20, 20);
  doc.setFontSize(22);
  doc.text(`Verdict: ${data.result.toUpperCase()}`, margin, 34);

  // ── Stats ──
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(80, 80, 80);
  doc.text(`Confidence: ${data.confidence}%`, margin, 44);
  doc.text(`Fake probability: ${data.fake_prob}%   |   Real probability: ${data.real_prob}%`, margin, 51);
  doc.text(`Word count: ${data.word_count}   |   Analyzed at: ${timestamp}`, margin, 58);

  // ── Divider ──
  doc.setDrawColor(...accent);
  doc.setLineWidth(0.7);
  doc.line(margin, 63, 196, 63);

  // ── Article text with page-break handling ──
  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(40, 40, 40);
  doc.text('Analyzed Article:', margin, 71);

  doc.setFont('helvetica', 'normal');
  doc.setTextColor(60, 60, 60);
  const lines  = doc.splitTextToSize(text, maxW);
  const lineH  = 5;  // mm per line
  let curY     = 78;

  lines.forEach(line => {
    if (curY + lineH > pageH - 14) {
      doc.addPage();
      curY = 20;
    }
    doc.text(line, margin, curY);
    curY += lineH;
  });

  // ── Footer ──
  const pageCount = doc.internal.getNumberOfPages();
  for (let p = 1; p <= pageCount; p++) {
    doc.setPage(p);
    doc.setFontSize(7);
    doc.setTextColor(160, 160, 160);
    doc.text(
      `Generated by FakeGuard AI  |  Page ${p} of ${pageCount}`,
      margin,
      pageH - 6
    );
  }

  doc.save(`fakeguard-report-${Date.now()}.pdf`);
});

// ─────────────────────────────────────────────
// Initialise page
// ─────────────────────────────────────────────
function init() {
  // Restore session stats display
  totalAnalyzed.textContent = stats.total;
  totalFake.textContent     = stats.fake;
  totalReal.textContent     = stats.real;
  sessionAccuracy.textContent = stats.total > 0
    ? `${Math.round((stats.real / stats.total) * 100)}% real`
    : '--';

  // Restore history table
  renderHistory();

  // Restore chart if there are previous predictions
  if (stats.total > 0) renderChart();

  // Load sample articles from the server
  loadSamples();
}

init();
