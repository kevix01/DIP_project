/* REST client helpers. */
const API = {
  async json(url, options) {
    const res = await fetch(url, options);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = { detail: text }; }
    if (!res.ok) {
      const detail = (data && (data.detail || data.message)) || res.statusText;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return data;
  },
  get(url) { return this.json(url); },
  post(url, body) {
    return this.json(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
  },
  postForm(url, form) {
    return this.json(url, { method: 'POST', body: form });
  },
  del(url) { return this.json(url, { method: 'DELETE' }); },
  upload(url, file, fields) {
    const fd = new FormData();
    if (file) fd.append('file', file, file.name);
    for (const [k, v] of Object.entries(fields || {})) fd.append(k, v);
    return this.json(url, { method: 'POST', body: fd });
  }
};

function $(id) { return document.getElementById(id); }
function val(id) { const el = $(id); return el ? el.value : ''; }
function num(id, fallback) {
  const v = parseFloat(val(id));
  return Number.isFinite(v) ? v : fallback;
}
function checked(id) { const el = $(id); return el ? el.checked : false; }
function setStatus(id, text, kind) {
  const el = $(id);
  if (!el) return;
  el.textContent = text || '';
  el.className = 'muted' + (kind ? ' ' + kind : '');
}
function appendLog(id, text) {
  const el = $(id);
  if (!el) return;
  el.textContent += (el.textContent ? '\n' : '') + text;
  el.scrollTop = el.scrollHeight;
}
function badge(text, kind) {
  const b = document.createElement('span');
  b.className = 'badge' + (kind ? ' ' + kind : '');
  b.textContent = text;
  return b;
}
function fmtBytes(n) {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(i ? 1 : 0) + ' ' + units[i];
}
