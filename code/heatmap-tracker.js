/**
 * heatmap-tracker.js
 * Клиентский модуль для LMS Informatika.
 * Собирает:
 *  - координаты mousemove (сэмплированные каждые 200 мс)
 *  - клики (event.target, координаты)
 *  - scroll depth (максимальная глубина прокрутки страницы)
 *  - dwell time (время простоя мыши в одной области > 800 мс)
 *  - навигацию между страницами
 * Отправляет батчами каждые 10 секунд на /api/heatmap
 *
 * Требует согласия пользователя (opt-in cookie: lms_heatmap_consent=yes).
 */
(function () {
  'use strict';

  const CONFIG = {
    endpoint: '/api/heatmap',
    batchIntervalMs: 10000,
    mouseSampleMs: 200,
    dwellThresholdMs: 800,
    consentCookie: 'lms_heatmap_consent',
  };

  // Consent check
  function hasConsent() {
    return document.cookie.split(';').some(c => c.trim().startsWith(CONFIG.consentCookie + '=yes'));
  }
  if (!hasConsent()) {
    console.info('[heatmap-tracker] Consent not granted. Skipping.');
    return;
  }

  const state = {
    studentId: window.STUDENT_ID || 'anonymous',
    sessionId: generateSessionId(),
    events: [],
    lastMouseX: 0,
    lastMouseY: 0,
    lastMouseMoveAt: 0,
    dwellStartAt: 0,
    maxScrollDepth: 0,
  };

  function generateSessionId() {
    return 's_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
  }

  function normalize(x, viewport) {
    return Math.round((x / viewport) * 1000);
  }

  function record(event_type, extra = {}) {
    state.events.push({
      student_id: state.studentId,
      session_id: state.sessionId,
      page_url: location.pathname + location.search,
      event_type,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
      timestamp_ms: Date.now(),
      ...extra,
    });
  }

  // Mouse move — throttled sampling
  document.addEventListener('mousemove', (e) => {
    const now = Date.now();
    if (now - state.lastMouseMoveAt < CONFIG.mouseSampleMs) return;
    const x = normalize(e.clientX, window.innerWidth);
    const y = normalize(e.clientY, window.innerHeight);
    record('mousemove', { x, y });
    state.lastMouseX = x;
    state.lastMouseY = y;
    state.lastMouseMoveAt = now;
    state.dwellStartAt = now;
  }, { passive: true });

  // Clicks
  document.addEventListener('click', (e) => {
    const x = normalize(e.clientX, window.innerWidth);
    const y = normalize(e.clientY, window.innerHeight);
    const tag = (e.target.tagName || 'UNKNOWN').toLowerCase();
    const id = e.target.id || null;
    const cls = (e.target.className || '').toString().slice(0, 60);
    record('click', { x, y, tag, id, class: cls });
  });

  // Scroll — track max depth
  window.addEventListener('scroll', () => {
    const depth = (window.scrollY + window.innerHeight) / document.body.scrollHeight;
    if (depth > state.maxScrollDepth) {
      state.maxScrollDepth = depth;
      record('scroll', { scroll_depth: Math.round(depth * 1000) / 1000 });
    }
  }, { passive: true });

  // Dwell detection
  setInterval(() => {
    const now = Date.now();
    if (state.dwellStartAt && now - state.lastMouseMoveAt > CONFIG.dwellThresholdMs) {
      record('dwell', {
        x: state.lastMouseX,
        y: state.lastMouseY,
        dwell_ms: now - state.lastMouseMoveAt,
      });
      state.dwellStartAt = now;
    }
  }, 500);

  // Page visibility
  document.addEventListener('visibilitychange', () => {
    record(document.hidden ? 'blur' : 'focus');
  });

  // Before unload — flush
  window.addEventListener('beforeunload', () => flush(true));

  // Batched send
  async function flush(sync = false) {
    if (!state.events.length) return;
    const payload = { batch: state.events.splice(0) };
    const body = JSON.stringify(payload);
    try {
      if (sync && navigator.sendBeacon) {
        navigator.sendBeacon(CONFIG.endpoint, new Blob([body], { type: 'application/json' }));
      } else {
        await fetch(CONFIG.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
          keepalive: true,
        });
      }
    } catch (err) {
      // On failure — put back for retry
      state.events.unshift(...payload.batch);
      console.warn('[heatmap-tracker] flush failed, will retry', err);
    }
  }
  setInterval(() => flush(false), CONFIG.batchIntervalMs);

  console.info('[heatmap-tracker] initialized', state.sessionId);
})();
