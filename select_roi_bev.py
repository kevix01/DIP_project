#!/usr/bin/env python3
"""ROI BEV Selector - interfaccia browser.

Permette di selezionare una ROI su un frame (foto o video) e di salvarla
in coordinate BEV tramite omografia. Tutto e' caricabile da UI:
sorgente, omografia, bev_unit e destinazione del salvataggio.
"""
import argparse
import http.server
import io
import json
import shutil
import socketserver
import sys
import tempfile
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

_tmp_paths = []

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ROI BEV Selector</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  background: #1e1e1e; color: #ddd; height: 100vh; display: flex; overflow: hidden;
}
.sidebar {
  width: 360px; min-width: 360px; background: #252526;
  border-right: 1px solid #3a3a3a; padding: 14px; overflow-y: auto;
  flex-shrink: 0; display: flex; flex-direction: column; gap: 12px;
}
.sidebar::-webkit-scrollbar { width: 8px; }
.sidebar::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
.section {
  background: #2d2d2d; border: 1px solid #383838; border-radius: 6px; padding: 12px;
}
.section h2 {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.6px;
  color: #9a9a9a; margin-bottom: 10px; font-weight: 600;
}
.header-section { background: transparent; border: none; padding: 4px 0 0; }
h1 { font-size: 18px; color: #4fc1ff; margin-bottom: 2px; font-weight: 600; }
.subtitle { font-size: 11px; color: #888; }
.btn-row { display: grid; gap: 6px; }
.btn-row.cols-2 { grid-template-columns: 1fr 1fr; }
.btn-row.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
.btn-row + .btn-row { margin-top: 6px; }
.btn {
  background: #0e639c; color: #fff; border: none; border-radius: 4px;
  cursor: pointer; padding: 8px 10px; font-size: 12.5px;
  transition: background .12s; text-align: center;
}
.btn:hover { background: #1177bb; }
.btn:disabled { background: #3a3a3a; color: #777; cursor: not-allowed; }
.btn-danger { background: #c0392b; } .btn-danger:hover { background: #a93226; }
.btn-success { background: #2d7d46; } .btn-success:hover { background: #246439; }
.btn-warn { background: #b8860b; } .btn-warn:hover { background: #8c6508; }
.upload-zone {
  border: 2px dashed #555; border-radius: 6px; padding: 14px 10px;
  text-align: center; cursor: pointer; font-size: 12px; color: #bbb;
  transition: border-color .12s, background .12s, color .12s;
}
.upload-zone:hover { border-color: #4fc1ff; background: #2a2a2a; color: #eee; }
.upload-zone.loaded { border-color: #2d7d46; border-style: solid; color: #8fc; }
.upload-zone .uz-icon { font-size: 22px; margin-bottom: 4px; opacity: .8; }
.upload-zone .uz-sub { font-size: 10px; color: #777; margin-top: 3px; }
label { font-size: 11px; color: #9a9a9a; display: block; margin-bottom: 4px; }
input[type=number] {
  width: 64px; padding: 4px 6px; background: #1e1e1e; color: #ddd;
  border: 1px solid #444; border-radius: 3px; font-size: 12px;
}
input[type=range] { width: 100%; accent-color: #0e639c; }
.status-msg {
  font-size: 12px; padding: 7px 9px; border-radius: 4px; min-height: 20px;
  word-break: break-word;
}
.status-msg:empty { padding: 0; min-height: 0; }
.status-msg.ok { background: #143a14; color: #7f7; }
.status-msg.err { background: #3a1414; color: #f88; }
.status-msg.info { background: #14223a; color: #7cf; }
.bev-info {
  font-size: 11px; color: #ccc; background: #1a1a1a; padding: 8px;
  border-radius: 4px; max-height: 140px; overflow-y: auto;
  font-family: 'SF Mono', Consolas, 'Liberation Mono', monospace;
}
.bev-info::-webkit-scrollbar { width: 6px; }
.bev-info::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }
.bev-row { display: flex; justify-content: space-between; padding: 1px 0; }
.bev-row b { color: #6f6; min-width: 22px; }
.instructions { font-size: 11px; color: #9a9a9a; line-height: 1.65; }
.main {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  justify-content: center; background: #121212; position: relative; min-width: 0;
}
.canvas-container {
  border: 1px solid #3a3a3a; background: #1a1a1a; position: relative;
  flex: 1; width: 100%; display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}
canvas { display: block; cursor: crosshair; }
.canvas-hint {
  position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
  font-size: 12px; color: #ccc; background: rgba(0,0,0,.55); padding: 5px 12px;
  border-radius: 4px; pointer-events: none; white-space: nowrap;
}
.empty-state {
  text-align: center; color: #666; font-size: 14px; padding: 20px;
}
.empty-state .icon { font-size: 56px; margin-bottom: 12px; opacity: .35; }
.frame-row { display: flex; gap: 6px; align-items: center; }
.frame-row input[type=range] { flex: 1; }
#frameIdxDisplay { font-size: 11px; color: #888; margin-top: 4px; text-align: center; }
</style>
</head>
<body>

<div class="sidebar">
  <div class="section header-section">
    <h1>ROI BEV Selector</h1>
    <div class="subtitle">Seleziona ROI e salva in coordinate BEV</div>
  </div>

  <div class="section">
    <h2>Sorgente (video o immagine)</h2>
    <div class="upload-zone" id="sourceZone" onclick="document.getElementById('sourceFile').click()">
      <div class="uz-icon">&#128249;</div>
      <div id="sourceLabel">Carica video o immagine</div>
      <div class="uz-sub">MP4, AVI, MOV, JPG, PNG, BMP...</div>
    </div>
    <input type="file" id="sourceFile" style="display:none" accept="video/*,image/*">
    <div id="frameControls" style="display:none; margin-top:10px;">
      <label>Frame</label>
      <div class="frame-row">
        <input type="range" id="frameSlider" min="1" max="1" value="1">
        <input type="number" id="frameInput" min="1" max="1" value="1">
      </div>
      <div id="frameIdxDisplay">Frame 1</div>
    </div>
  </div>

  <div class="section">
    <h2>Calibrazione</h2>
    <div class="upload-zone" id="homZone" onclick="document.getElementById('homFile').click()">
      <div id="homLabel">Carica omografia (3x3)</div>
      <div class="uz-sub" id="homStatus">Non caricata · .npy / .txt</div>
    </div>
    <input type="file" id="homFile" style="display:none" accept=".npy,.txt,.csv">
    <div class="upload-zone" id="bevZone" style="margin-top:8px;" onclick="document.getElementById('bevFile').click()">
      <div id="bevLabel">Carica bev_unit (scalare)</div>
      <div class="uz-sub" id="bevStatus">Non caricato · .txt</div>
    </div>
    <input type="file" id="bevFile" style="display:none" accept=".txt,.csv">
  </div>

  <div class="section">
    <h2>Vista</h2>
    <div class="btn-row cols-3">
      <button class="btn" onclick="zoomIn()">Zoom +</button>
      <button class="btn" onclick="zoomOut()">Zoom &#8722;</button>
      <button class="btn" onclick="resetView()">Reset</button>
    </div>
    <div class="btn-row">
      <button class="btn" onclick="fitToScreen()">Adatta allo schermo</button>
    </div>
  </div>

  <div class="section">
    <h2>ROI</h2>
    <div class="btn-row cols-2">
      <button class="btn btn-warn" onclick="resetROI()">Rettangolo</button>
      <button class="btn btn-danger" onclick="clearROI()">Cancella punti</button>
    </div>
    <div class="btn-row">
      <button class="btn" onclick="document.getElementById('loadRoiFile').click()">Carica ROI JSON esistente</button>
    </div>
    <input type="file" id="loadRoiFile" style="display:none" accept=".json">
  </div>

  <div class="section">
    <h2>Salvataggio</h2>
    <div class="btn-row">
      <button class="btn btn-success" onclick="saveROI()" id="saveBtn">Salva ROI... (scegli nome)</button>
    </div>
    <div class="btn-row">
      <button class="btn btn-danger" onclick="quitApp()">Esci</button>
    </div>
    <div id="status" class="status-msg" style="margin-top:8px;"></div>
  </div>

  <div class="section">
    <h2>Punti BEV</h2>
    <div class="bev-info" id="bevInfo">Nessun punto</div>
  </div>

  <div class="section instructions">
    <h2>Mouse</h2>
    &#8226; Click sullo sfondo: aggiungi vertice<br>
    &#8226; Click su maniglia + drag: sposta vertice<br>
    &#8226; Right-click su maniglia: rimuovi vertice<br>
    &#8226; Click su un lato: inserisci vertice<br>
    &#8226; Rotella mouse: zoom<br>
    &#8226; Alt + drag sinistro: sposta vista (pan)<br>
    &#8226; Ctrl + drag sinistro: sposta tutta la ROI
  </div>
  <div class="section instructions">
    <h2>Tastiera</h2>
    &#8226; <b>+</b> / <b>=</b>: zoom in<br>
    &#8226; <b>&#8722;</b> / <b>_</b>: zoom out<br>
    &#8226; <b>0</b>: reset zoom e vista<br>
    &#8226; <b>Canc</b>: rimuovi vertice selezionato<br>
    &#8226; <b>S</b>: salva ROI
  </div>
</div>

<div class="main">
  <div class="canvas-container">
    <canvas id="roiCanvas" width="960" height="540" style="display:none"></canvas>
    <div class="empty-state" id="emptyState">
      <div class="icon">&#128444;</div>
      <div>Carica un'immagine o un video per iniziare</div>
    </div>
  </div>
  <div class="canvas-hint" id="canvasHint" style="display:none"></div>
</div>

<script>
const INITIAL = %%INITIAL%%;

let HOMOGRAFIA = INITIAL.homography;
let BEV_UNIT = INITIAL.bev_unit;
let HOMOGRAPHY_FILE = INITIAL.homography_file;
let BEV_UNIT_FILE = INITIAL.bev_unit_file;
let SOURCE_FRAME = INITIAL.source;
let OUTPUT_FILE = INITIAL.output_file || 'roi_bev.json';
let IS_VIDEO = INITIAL.is_video;
let TOTAL_FRAMES = INITIAL.total_frames;
// CURRENT_FRAME_INDEX è 1-based (il "numero frame" che vede l'utente)
let CURRENT_FRAME_INDEX = (INITIAL.frame_index || 0) + 1;
let EXISTING_POINTS = INITIAL.existing_points;

let img = null;
let imgWidth = 0, imgHeight = 0;
let scale = 1, transX = 0, transY = 0;
let points = [];
let selectedIdx = null;
let dragging = false, isPanning = false, isMovingROI = false;
let isAltDown = false, isCtrlDown = false;
let roiStart = null;
let panStart = {x:0, y:0, transX:0, transY:0};

const canvas = document.getElementById('roiCanvas');
const ctx = canvas.getContext('2d');

function applyHomography(pt, H) {
  const x = pt[0], y = pt[1];
  const w = H[2][0]*x + H[2][1]*y + H[2][2];
  const xb = (H[0][0]*x + H[0][1]*y + H[0][2]) / w;
  const yb = (H[1][0]*x + H[1][1]*y + H[1][2]) / w;
  return [xb, yb];
}
function toCanvas(x, y) {
  const cx = canvas.width / 2, cy = canvas.height / 2;
  return { x: cx + (x - imgWidth/2)*scale + transX, y: cy + (y - imgHeight/2)*scale + transY };
}
function toImage(cx, cy) {
  const cx0 = canvas.width / 2, cy0 = canvas.height / 2;
  return { x: imgWidth/2 + (cx - cx0 - transX)/scale, y: imgHeight/2 + (cy - cy0 - transY)/scale };
}
function dist(p1, p2) { return Math.hypot(p1.x - p2.x, p1.y - p2.y); }

function getHandleAt(cx, cy) {
  for (let i = 0; i < points.length; i++) {
    const p = toCanvas(points[i].x, points[i].y);
    if (dist({x:cx,y:cy}, p) < 14) return i;
  }
  return null;
}
function getSegmentAt(cx, cy) {
  if (points.length < 2) return null;
  let best = null, bestD = 16;
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length;
    const a = toCanvas(points[i].x, points[i].y);
    const b = toCanvas(points[j].x, points[j].y);
    const vx = b.x - a.x, vy = b.y - a.y;
    const wx = cx - a.x, wy = cy - a.y;
    const c1 = vx*wx + vy*wy;
    if (c1 <= 0) { const d = dist({x:cx,y:cy}, a); if (d < bestD) { bestD = d; best = i; } continue; }
    const c2 = vx*vx + vy*vy;
    if (c2 <= c1) { const d = dist({x:cx,y:cy}, b); if (d < bestD) { bestD = d; best = j; } continue; }
    const t = c1 / c2;
    const px = a.x + t*vx, py = a.y + t*vy;
    const d = dist({x:cx,y:cy}, {x:px,y:py});
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
function initRectangle() {
  if (!imgWidth || !imgHeight) return;
  const m = 0.1;
  points = [
    {x: imgWidth*m,      y: imgHeight*m},
    {x: imgWidth*(1-m),  y: imgHeight*m},
    {x: imgWidth*(1-m),  y: imgHeight*(1-m)},
    {x: imgWidth*m,      y: imgHeight*(1-m)}
  ];
}
function fitToScreen() {
  if (!img) return;
  const cont = document.querySelector('.canvas-container');
  const cw = cont.clientWidth, ch = cont.clientHeight;
  const ar = imgWidth / imgHeight;
  let dw, dh;
  if (cw / ch > ar) { dh = ch * 0.92; dw = dh * ar; }
  else { dw = cw * 0.92; dh = dw / ar; }
  canvas.style.width = dw + 'px';
  canvas.style.height = dh + 'px';
  scale = 1; transX = 0; transY = 0;
  draw(); updateBEVDisplay();
}
function loadImage(url) {
  img = new Image();
  img.onload = () => {
    imgWidth = img.naturalWidth;
    imgHeight = img.naturalHeight;
    canvas.width = Math.min(1400, imgWidth);
    canvas.height = canvas.width * (imgHeight / imgWidth);
    scale = 1; transX = 0; transY = 0;
    canvas.style.width = ''; canvas.style.height = '';
    if (EXISTING_POINTS && EXISTING_POINTS.length) {
      points = EXISTING_POINTS.map(p => ({x: p[0], y: p[1]}));
      EXISTING_POINTS = null;
    } else {
      initRectangle();
    }
    selectedIdx = null;
    draw(); updateBEVDisplay();
    document.getElementById('canvasHint').textContent =
      imgWidth + ' x ' + imgHeight +
      (IS_VIDEO ? '  -  Frame ' + CURRENT_FRAME_INDEX + '/' + TOTAL_FRAMES : '');
    document.getElementById('canvasHint').style.display = 'block';
    fitToScreen();
  };
  img.onerror = () => setStatus('Errore caricamento frame', 'err');
  img.src = url;
}
function draw() {
  if (!img) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const dw = imgWidth * scale, dh = imgHeight * scale;
  ctx.drawImage(img, cx - dw/2 + transX, cy - dh/2 + transY, dw, dh);
  if (points.length >= 2) {
    ctx.beginPath();
    const p0 = toCanvas(points[0].x, points[0].y);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < points.length; i++) {
      const p = toCanvas(points[i].x, points[i].y);
      ctx.lineTo(p.x, p.y);
    }
    if (points.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 255, 0, 0.15)';
      ctx.fill();
    }
    ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 2; ctx.stroke();
  }
  for (let i = 0; i < points.length; i++) {
    const p = toCanvas(points[i].x, points[i].y);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI*2);
    ctx.fillStyle = selectedIdx === i ? '#ff0000' : '#ff4444';
    ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = '11px sans-serif';
    ctx.fillText(String(i), p.x + 9, p.y - 9);
  }
}
function updateBEVDisplay() {
  const div = document.getElementById('bevInfo');
  if (!points.length) { div.textContent = 'Nessun punto'; return; }
  if (!HOMOGRAFIA) {
    div.innerHTML = '<i style="color:#a80;">Carica un&#39;omografia per vedere le coordinate BEV</i>';
    return;
  }
  let html = '';
  const maxShow = Math.min(points.length, 12);
  for (let i = 0; i < maxShow; i++) {
    const bv = applyHomography([points[i].x, points[i].y], HOMOGRAFIA);
    let x = bv[0], y = bv[1];
    if (BEV_UNIT) { x /= BEV_UNIT; y /= BEV_UNIT; }
    html += '<div class="bev-row"><b>' + i + '</b><span>(' + x.toFixed(2) + ', ' + y.toFixed(2) + ')</span></div>';
  }
  if (points.length > maxShow)
    html += '<div style="font-size:10px;color:#666;">... e altri ' + (points.length - maxShow) + '</div>';
  div.innerHTML = html;
}
function zoomIn()  { if (img) { scale *= 1.25; draw(); } }
function zoomOut() { if (img) { scale *= 0.8;  draw(); } }
function resetView() { if (img) { scale = 1; transX = 0; transY = 0; draw(); } }

function setStatus(msg, kind) {
  const el = document.getElementById('status');
  el.textContent = msg || '';
  el.className = 'status-msg' + (kind ? ' ' + kind : '');
}

window.addEventListener('keydown', e => {
  if (e.code === 'AltLeft' || e.code === 'AltRight') isAltDown = true;
  if (e.code === 'ControlLeft' || e.code === 'ControlRight') isCtrlDown = true;
  if (e.target.tagName === 'INPUT') return;
  if (e.key === '+' || e.key === '=') zoomIn();
  if (e.key === '-' || e.key === '_') zoomOut();
  if (e.key === '0') resetView();
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (selectedIdx !== null && points.length > 4) {
      points.splice(selectedIdx, 1); selectedIdx = null;
      draw(); updateBEVDisplay();
    }
  }
  if (e.key === 's' || e.key === 'S') saveROI();
});
window.addEventListener('keyup', e => {
  if (e.code === 'AltLeft' || e.code === 'AltRight') isAltDown = false;
  if (e.code === 'ControlLeft' || e.code === 'ControlRight') isCtrlDown = false;
});

function canvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  return { x: sx * (canvas.width / rect.width), y: sy * (canvas.height / rect.height) };
}
canvas.addEventListener('mousedown', e => {
  if (!img) return;
  const cc = canvasCoords(e), cx = cc.x, cy = cc.y;
  if (e.button === 2) {
    e.preventDefault();
    const idx = getHandleAt(cx, cy);
    if (idx !== null && points.length > 4) {
      points.splice(idx, 1);
      if (selectedIdx === idx) selectedIdx = null;
      else if (selectedIdx !== null && selectedIdx > idx) selectedIdx--;
      draw(); updateBEVDisplay();
    }
    return;
  }
  const handleIdx = getHandleAt(cx, cy);
  if (handleIdx !== null) { selectedIdx = handleIdx; dragging = true; return; }
  if (e.button === 0) {
    if (isAltDown) { isPanning = true; panStart = {x:cx, y:cy, transX, transY}; }
    else if (isCtrlDown) { isMovingROI = true; panStart = {x:cx, y:cy}; roiStart = points.map(p => ({x:p.x,y:p.y})); }
    else {
      const seg = getSegmentAt(cx, cy);
      if (seg !== null) {
        const ip = toImage(cx, cy);
        points.splice(seg + 1, 0, ip);
        draw(); updateBEVDisplay();
      } else {
        const ip = toImage(cx, cy);
        points.push(ip);
        draw(); updateBEVDisplay();
      }
    }
  }
});
window.addEventListener('mousemove', e => {
  if (!img) return;
  const cc = canvasCoords(e), cx = cc.x, cy = cc.y;
  if (dragging && selectedIdx !== null) {
    points[selectedIdx] = toImage(cx, cy);
    draw(); updateBEVDisplay();
  } else if (isMovingROI && roiStart) {
    const dx = (cx - panStart.x) / scale, dy = (cy - panStart.y) / scale;
    for (let i = 0; i < points.length; i++) { points[i].x = roiStart[i].x + dx; points[i].y = roiStart[i].y + dy; }
    draw(); updateBEVDisplay();
  } else if (isPanning) {
    transX = panStart.transX + (cx - panStart.x);
    transY = panStart.transY + (cy - panStart.y);
    draw();
  }
});
window.addEventListener('mouseup', () => {
  dragging = false; isPanning = false; isMovingROI = false;
  selectedIdx = null; roiStart = null;
});
canvas.addEventListener('wheel', e => {
  if (!img) return;
  e.preventDefault();
  scale *= e.deltaY > 0 ? 0.9 : 1.1;
  draw();
}, {passive:false});
canvas.addEventListener('contextmenu', e => e.preventDefault());

function resetROI() { if (!img) return; initRectangle(); draw(); updateBEVDisplay(); setStatus('ROI reimpostata', 'info'); }
function clearROI() { points = []; selectedIdx = null; draw(); updateBEVDisplay(); }

async function loadFrame(idx) {
  // idx è 1-based (numero frame utente); l'API backend è 0-based
  CURRENT_FRAME_INDEX = idx;
  loadImage('/frame?index=' + (idx - 1) + '&t=' + Date.now());
}

async function uploadSource() {
  const file = document.getElementById('sourceFile').files[0];
  if (!file) return;
  setStatus('Caricamento sorgente...', 'info');
  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await fetch('/upload-source', {method:'POST', body: fd});
    const info = await res.json();
    if (info.error) { setStatus(info.error, 'err'); return; }
    IS_VIDEO = info.is_video;
    TOTAL_FRAMES = info.total_frames;
    CURRENT_FRAME_INDEX = 1;
    imgWidth = info.img_width;
    imgHeight = info.img_height;
    SOURCE_FRAME = info.source;
    document.getElementById('sourceLabel').textContent = info.filename || file.name;
    document.getElementById('sourceZone').classList.add('loaded');
    document.getElementById('emptyState').style.display = 'none';
    canvas.style.display = 'block';
    EXISTING_POINTS = null;
    if (IS_VIDEO) {
      const fc = document.getElementById('frameControls');
      fc.style.display = 'block';
      const s = document.getElementById('frameSlider');
      s.min = 1; s.max = TOTAL_FRAMES; s.value = 1;
      const fi = document.getElementById('frameInput');
      fi.min = 1; fi.max = TOTAL_FRAMES; fi.value = 1;
      document.getElementById('frameIdxDisplay').textContent = 'Frame 1 / ' + TOTAL_FRAMES;
    } else {
      document.getElementById('frameControls').style.display = 'none';
    }
    setStatus('Sorgente: ' + imgWidth + 'x' + imgHeight, 'ok');
    loadFrame(1);
  } catch (e) { setStatus('Errore: ' + e.message, 'err'); }
}
async function uploadHomography() {
  const file = document.getElementById('homFile').files[0];
  if (!file) return;
  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await fetch('/upload-homography', {method:'POST', body: fd});
    const info = await res.json();
    if (info.error) { setStatus(info.error, 'err'); return; }
    HOMOGRAFIA = info.matrix;
    HOMOGRAPHY_FILE = info.filename || file.name;
    document.getElementById('homLabel').textContent = HOMOGRAPHY_FILE;
    document.getElementById('homStatus').textContent = '3x3 caricata';
    document.getElementById('homZone').classList.add('loaded');
    setStatus('Omografia caricata', 'ok');
    updateBEVDisplay();
  } catch (e) { setStatus('Errore: ' + e.message, 'err'); }
}
async function uploadBevUnit() {
  const file = document.getElementById('bevFile').files[0];
  if (!file) return;
  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await fetch('/upload-bev-unit', {method:'POST', body: fd});
    const info = await res.json();
    if (info.error) { setStatus(info.error, 'err'); return; }
    BEV_UNIT = info.bev_unit;
    BEV_UNIT_FILE = info.filename || file.name;
    document.getElementById('bevLabel').textContent = BEV_UNIT_FILE;
    document.getElementById('bevStatus').textContent = 'bev_unit = ' + BEV_UNIT;
    document.getElementById('bevZone').classList.add('loaded');
    setStatus('bev_unit caricato', 'ok');
    updateBEVDisplay();
  } catch (e) { setStatus('Errore: ' + e.message, 'err'); }
}

async function saveROI() {
  if (points.length < 4) { setStatus('Servono almeno 4 punti.', 'err'); return; }
  if (!HOMOGRAFIA) { setStatus('Carica prima un&#39;omografia.', 'err'); return; }
  const bevPoints = points.map(p => {
    const bv = applyHomography([p.x, p.y], HOMOGRAFIA);
    let x = bv[0], y = bv[1];
    if (BEV_UNIT) { x /= BEV_UNIT; y /= BEV_UNIT; }
    return [x, y];
  });
  const payload = {
    homography_file: HOMOGRAPHY_FILE,
    bev_unit_file: BEV_UNIT_FILE,
    bev_unit: BEV_UNIT,
    image_points: points.map(p => [p.x, p.y]),
    bev_points: bevPoints
  };
  if (IS_VIDEO) {
    payload.source_video = SOURCE_FRAME;
    payload.source_video_frame_id = CURRENT_FRAME_INDEX;
  } else {
    payload.source_frame = SOURCE_FRAME;
  }
  const jsonStr = JSON.stringify(payload, null, 2);

  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: OUTPUT_FILE,
        types: [{ description: 'JSON', accept: {'application/json': ['.json']} }]
      });
      const w = await handle.createWritable();
      await w.write(jsonStr);
      await w.close();
      setStatus('Salvato: ' + handle.name, 'ok');
    } catch (e) {
      if (e && e.name !== 'AbortError') setStatus('Errore salvataggio: ' + e.message, 'err');
    }
  } else {
    const name = prompt('Nome file di output:', OUTPUT_FILE);
    if (!name) return;
    const blob = new Blob([jsonStr], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
    setStatus('Download: ' + name + ' (usa Chrome/Edge per scegliere la cartella)', 'ok');
  }
}
function quitApp() {
  if (!confirm('Uscire dall&#39;applicazione?')) return;
  fetch('/quit', {method:'POST'}).catch(()=>{}).then(() => {
    document.body.innerHTML =
      '<div style="display:flex;align-items:center;justify-content:center;height:100vh;' +
      'color:#888;font-family:sans-serif;font-size:14px;">Chiusura in corso...</div>';
    setTimeout(() => window.close(), 600);
  });
}

document.getElementById('sourceFile').addEventListener('change', uploadSource);
document.getElementById('homFile').addEventListener('change', uploadHomography);
document.getElementById('bevFile').addEventListener('change', uploadBevUnit);
document.getElementById('loadRoiFile').addEventListener('change', async e => {
  const file = e.target.files[0]; if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (!data.image_points) throw new Error('formato errato');
    points = data.image_points.map(p => ({x: p[0], y: p[1]}));
    selectedIdx = null;
    draw(); updateBEVDisplay();
    setStatus('ROI caricata: ' + points.length + ' punti', 'ok');
  } catch (err) { setStatus('File JSON non valido: ' + err.message, 'err'); }
});

const slider = document.getElementById('frameSlider');
slider.addEventListener('input', async () => {
  if (!IS_VIDEO) return;
  const idx = parseInt(slider.value);
  document.getElementById('frameInput').value = idx;
  document.getElementById('frameIdxDisplay').textContent = 'Frame ' + idx + ' / ' + TOTAL_FRAMES;
  await loadFrame(idx);
});
const frameInput = document.getElementById('frameInput');
frameInput.addEventListener('change', async () => {
  if (!IS_VIDEO) return;
  let idx = parseInt(frameInput.value);
  if (isNaN(idx) || idx < 1) idx = 1;
  if (idx > TOTAL_FRAMES) idx = TOTAL_FRAMES;
  frameInput.value = idx; slider.value = idx;
  document.getElementById('frameIdxDisplay').textContent = 'Frame ' + idx + ' / ' + TOTAL_FRAMES;
  await loadFrame(idx);
});

// init da stato pre-caricato da CLI
if (INITIAL.has_homography) {
  document.getElementById('homLabel').textContent = HOMOGRAPHY_FILE;
  document.getElementById('homStatus').textContent = '3x3 caricata';
  document.getElementById('homZone').classList.add('loaded');
}
if (INITIAL.has_bev_unit) {
  document.getElementById('bevLabel').textContent = BEV_UNIT_FILE;
  document.getElementById('bevStatus').textContent = 'bev_unit = ' + BEV_UNIT;
  document.getElementById('bevZone').classList.add('loaded');
}
if (INITIAL.has_source) {
  document.getElementById('sourceLabel').textContent = INITIAL.filename;
  document.getElementById('sourceZone').classList.add('loaded');
  document.getElementById('emptyState').style.display = 'none';
  canvas.style.display = 'block';
  if (IS_VIDEO) {
    document.getElementById('frameControls').style.display = 'block';
    const s = document.getElementById('frameSlider');
    s.min = 1; s.max = TOTAL_FRAMES; s.value = CURRENT_FRAME_INDEX;
    const fi = document.getElementById('frameInput');
    fi.min = 1; fi.max = TOTAL_FRAMES; fi.value = CURRENT_FRAME_INDEX;
    document.getElementById('frameIdxDisplay').textContent =
      'Frame ' + CURRENT_FRAME_INDEX + ' / ' + TOTAL_FRAMES;
  }
  loadFrame(CURRENT_FRAME_INDEX);
}
</script>
</body>
</html>"""


def parse_multipart(body, content_type):
    """Parser multipart/form-data minimale, senza dipendenze."""
    boundary = None
    for part in content_type.split(';'):
        part = part.strip()
        if part.startswith('boundary='):
            boundary = part[len('boundary='):]
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            break
    if not boundary:
        return {}
    delim = b'--' + boundary.encode()
    result = {}
    for chunk in body.split(delim):
        chunk = chunk.strip(b'\r\n')
        if not chunk or chunk == b'--':
            continue
        try:
            header_block, content = chunk.split(b'\r\n\r\n', 1)
        except ValueError:
            continue
        headers = {}
        for line in header_block.split(b'\r\n'):
            if b':' in line:
                k, v = line.split(b':', 1)
                headers[k.strip().lower()] = v.strip()
        cd = headers.get(b'content-disposition', b'').decode('utf-8', errors='replace')
        name = filename = None
        for seg in cd.split(';'):
            seg = seg.strip()
            if seg.startswith('name='):
                name = seg[5:].strip('"')
            elif seg.startswith('filename='):
                filename = seg[9:].strip('"')
        if name is None:
            continue
        if content.endswith(b'\r\n'):
            content = content[:-2]
        result[name] = {'data': content, 'filename': filename}
    return result


def load_homography_from_bytes(data, filename):
    fn = (filename or '').lower()
    if fn.endswith('.npy'):
        return np.load(io.BytesIO(data)).astype(np.float64)
    text = data.decode('utf-8', errors='replace')
    return np.loadtxt(io.StringIO(text)).astype(np.float64)


def save_temp_upload(data, filename):
    suffix = Path(filename or 'upload').suffix
    tmpdir = tempfile.mkdtemp(prefix='roi_bev_')
    path = Path(tmpdir) / ('upload' + suffix)
    path.write_bytes(data)
    _tmp_paths.append(tmpdir)
    return path


def probe_source(path):
    path = Path(path)
    if path.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise ValueError('video non leggibile')
        h, w = frame.shape[:2]
        return {'is_video': True, 'total_frames': total, 'img_width': w, 'img_height': h}
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError('immagine non leggibile')
    h, w = img.shape[:2]
    return {'is_video': False, 'total_frames': 1, 'img_width': w, 'img_height': h}


class ROIHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    should_stop = False
    server_instance = None


class ROIHandler(http.server.BaseHTTPRequestHandler):
    html_content = ''
    is_video = False
    video_path = None
    image_path = None
    total_frames = 1
    frame_index = 0
    img_width = 0
    img_height = 0
    homography = None
    bev_unit = None
    homography_file = ''
    bev_unit_file = ''
    source_frame = ''
    output_path = 'roi_bev.json'

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, code=200):
        payload = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.html_content.encode('utf-8'))
        elif parsed.path == '/frame':
            params = urllib.parse.parse_qs(parsed.query)
            idx = int(params.get('index', ['0'])[0])
            try:
                if self.is_video and self.video_path:
                    cap = cv2.VideoCapture(str(self.video_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret or frame is None:
                        self._send_json({'error': f'frame {idx} non disponibile'}, 400)
                        return
                elif self.image_path:
                    frame = cv2.imread(str(self.image_path))
                    if frame is None:
                        self._send_json({'error': 'immagine non disponibile'}, 400)
                        return
                else:
                    self._send_json({'error': 'nessuna sorgente caricata'}, 400)
                    return
                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                data = buf.tobytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length > 0 else b''
        ct = self.headers.get('Content-Type', '')
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/upload-source':
            try:
                fields = parse_multipart(body, ct)
                if 'file' not in fields:
                    self._send_json({'error': 'nessun file ricevuto'}, 400)
                    return
                f = fields['file']
                tmp_path = save_temp_upload(f['data'], f.get('filename'))
                info = probe_source(tmp_path)
                # NB: assegna a livello di CLASSE, non di istanza:
                # ogni richiesta HTTP crea un nuovo handler, e gli attributi
                # di istanza verrebbero persi tra una richiesta e l'altra.
                if info['is_video']:
                    ROIHandler.video_path = tmp_path
                    ROIHandler.image_path = None
                else:
                    ROIHandler.image_path = tmp_path
                    ROIHandler.video_path = None
                ROIHandler.is_video = info['is_video']
                ROIHandler.total_frames = info['total_frames']
                ROIHandler.frame_index = 0
                ROIHandler.img_width = info['img_width']
                ROIHandler.img_height = info['img_height']
                ROIHandler.source_frame = str(tmp_path)
                resp = dict(info)
                resp['source'] = str(tmp_path)
                resp['filename'] = f.get('filename') or 'upload'
                self._send_json(resp)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)

        elif parsed.path == '/upload-homography':
            try:
                fields = parse_multipart(body, ct)
                if 'file' not in fields:
                    self._send_json({'error': 'nessun file'}, 400)
                    return
                f = fields['file']
                H = load_homography_from_bytes(f['data'], f.get('filename'))
                if H.shape != (3, 3):
                    self._send_json({'error': f'shape non 3x3: {H.shape}'}, 400)
                    return
                ROIHandler.homography = H.tolist()
                ROIHandler.homography_file = f.get('filename') or ''
                self._send_json({'matrix': ROIHandler.homography, 'filename': ROIHandler.homography_file})
            except Exception as e:
                self._send_json({'error': str(e)}, 500)

        elif parsed.path == '/upload-bev-unit':
            try:
                fields = parse_multipart(body, ct)
                if 'file' not in fields:
                    self._send_json({'error': 'nessun file'}, 400)
                    return
                f = fields['file']
                val = float(f['data'].decode('utf-8', errors='replace').strip())
                ROIHandler.bev_unit = val
                ROIHandler.bev_unit_file = f.get('filename') or ''
                self._send_json({'bev_unit': val, 'filename': ROIHandler.bev_unit_file})
            except Exception as e:
                self._send_json({'error': str(e)}, 500)

        elif parsed.path == '/quit':
            ROIHTTPServer.should_stop = True
            self.send_response(200)
            self.end_headers()
            threading.Timer(0.1, lambda: (
                ROIHTTPServer.server_instance.shutdown()
                if ROIHTTPServer.server_instance else None
            )).start()
        else:
            self.send_error(404)


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description='Seleziona una ROI su un frame/foto e la salva in coordinate BEV (interfaccia browser).')
    ap.add_argument('--frame', default=None,
                    help='File iniziale (opzionale, caricabile anche da UI)')
    ap.add_argument('--homography', default=None,
                    help='File omografia 3x3 (.npy/.txt, opzionale)')
    ap.add_argument('--bev-unit', default=None,
                    help='File bev_unit (opzionale)')
    ap.add_argument('--roi', default=None,
                    help='File ROI JSON da pre-caricare (opzionale)')
    ap.add_argument('--output', default='roi_bev.json',
                    help='Nome suggerito per il file di output (default: roi_bev.json)')
    ap.add_argument('--frame-index', type=int, default=0,
                    help='Indice frame iniziale per video (default: 0)')
    ap.add_argument('--port', type=int, default=8080,
                    help='Porta del server HTTP (default: 8080)')
    ap.add_argument('--no-browser', action='store_true',
                    help='Non aprire il browser automaticamente')
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    initial = {
        'has_source': False, 'is_video': False, 'total_frames': 1,
        'frame_index': 0, 'img_width': 0, 'img_height': 0,
        'source': '', 'filename': '',
        'has_homography': False, 'homography': None, 'homography_file': '',
        'has_bev_unit': False, 'bev_unit': None, 'bev_unit_file': '',
        'existing_points': None, 'output_file': args.output,
    }

    # omografia (opzionale)
    if args.homography and Path(args.homography).exists():
        try:
            H = load_homography_from_bytes(Path(args.homography).read_bytes(), args.homography)
            if H.shape == (3, 3):
                ROIHandler.homography = H.tolist()
                ROIHandler.homography_file = Path(args.homography).name
                initial['has_homography'] = True
                initial['homography'] = H.tolist()
                initial['homography_file'] = Path(args.homography).name
                print(f'[info] omografia caricata: {args.homography}')
        except Exception as e:
            print(f'[warn] omografia non caricata: {e}')

    # bev_unit (opzionale)
    if args.bev_unit and Path(args.bev_unit).exists():
        try:
            val = float(Path(args.bev_unit).read_text().strip())
            ROIHandler.bev_unit = val
            ROIHandler.bev_unit_file = Path(args.bev_unit).name
            initial['has_bev_unit'] = True
            initial['bev_unit'] = val
            initial['bev_unit_file'] = Path(args.bev_unit).name
            print(f'[info] bev_unit: {val}')
        except Exception as e:
            print(f'[warn] bev_unit non caricato: {e}')

    # sorgente (opzionale)
    if args.frame and Path(args.frame).exists():
        try:
            info = probe_source(args.frame)
            if info['is_video']:
                ROIHandler.video_path = Path(args.frame)
                ROIHandler.image_path = None
            else:
                ROIHandler.image_path = Path(args.frame)
                ROIHandler.video_path = None
            ROIHandler.is_video = info['is_video']
            ROIHandler.total_frames = info['total_frames']
            ROIHandler.frame_index = max(0, min(args.frame_index, info['total_frames'] - 1))
            ROIHandler.img_width = info['img_width']
            ROIHandler.img_height = info['img_height']
            ROIHandler.source_frame = str(Path(args.frame).resolve())
            initial.update({
                'has_source': True, 'is_video': info['is_video'],
                'total_frames': info['total_frames'],
                'frame_index': ROIHandler.frame_index,
                'img_width': info['img_width'], 'img_height': info['img_height'],
                'source': ROIHandler.source_frame, 'filename': Path(args.frame).name,
            })
            print(f'[info] sorgente: {args.frame} ({info["img_width"]}x{info["img_height"]})')
        except Exception as e:
            print(f'[warn] sorgente non caricata: {e}')

    # ROI esistente (opzionale)
    if args.roi and Path(args.roi).exists():
        try:
            data = json.loads(Path(args.roi).read_text(encoding='utf-8'))
            pts = data.get('image_points', [])
            initial['existing_points'] = pts
            print(f'[info] ROI pre-caricata: {len(pts)} vertici')
        except Exception as e:
            print(f'[warn] ROI non caricata: {e}')

    ROIHandler.output_path = args.output

    html = HTML_TEMPLATE.replace('%%INITIAL%%', json.dumps(initial))
    ROIHandler.html_content = html

    server = ROIHTTPServer(('127.0.0.1', args.port), ROIHandler)
    ROIHTTPServer.server_instance = server
    url = f'http://127.0.0.1:{args.port}/'
    print(f'[info] interfaccia disponibile su: {url}')
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        for t in _tmp_paths:
            shutil.rmtree(t, ignore_errors=True)

    print('[done] chiusura.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
