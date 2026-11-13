/* App orchestration: tabs, engine export, calibration, ROI and debug wiring. */
(function () {
  const state = { health: null, lastCalibration: null };

  function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
    if (name === 'export') loadArtifacts();
  }
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));

  async function loadHealth() {
    try {
      const h = await API.get('/api/health');
      state.health = h;
      const dot = $('healthDot');
      dot.className = 'dot ok';
      const gpu = h.gpu && h.gpu.devices && h.gpu.devices.length
        ? `GPU rilevate: ${h.gpu.devices.map(d => `${d.id} - ${d.name} (sm${d.capability})`).join(', ')}`
        : 'GPU rilevate: nessuna (CPU)';
      $('healthInfo').textContent = gpu;
      const d = h.defaults;
      $('ex-weights').value = d.weights;
      $('ex-device').value = d.device;
      $('cal-model').value = d.engine;
      $('cal-device').value = d.device;
      $('dbg-engine').value = d.engine;
      $('dbg-homography').value = d.homography;
      $('dbg-bevunit').value = d.bev_unit_file || '';
      $('dbg-roi').value = d.roi;
    } catch (err) {
      $('healthDot').className = 'dot err';
      $('healthInfo').textContent = 'backend non raggiungibile';
    }
  }

  /* ---------------- engine export ---------------- */
  async function loadArtifacts() {
    const box = $('ex-artifacts');
    try {
      const weights = val('ex-weights');
      const url = weights
        ? '/api/engine/artifacts?weights=' + encodeURIComponent(weights)
        : '/api/engine/artifacts';
      const items = await API.get(url);
      if (!items.length) { box.textContent = 'nessun artifact'; return; }
      box.textContent = '';
      items.forEach(it => {
        const row = document.createElement('div');
        row.className = 'item';
        const name = document.createElement('span');
        name.className = 'growx';
        name.textContent = it.name;
        const size = document.createElement('span');
        size.className = 'muted';
        size.textContent = fmtBytes(it.size);
        row.appendChild(name);
        row.appendChild(badge(it.kind, 'ok'));
        row.appendChild(size);
        box.appendChild(row);
      });
    } catch (err) {
      box.textContent = err.message;
    }
  }

  async function pollEngineJob(jobId) {
    const tick = async () => {
      try {
        const job = await API.get('/api/engine/jobs/' + jobId);
        setStatus('ex-status', job.status);
        $('ex-log').textContent = (job.logs || []).join('\n');
        $('ex-log').scrollTop = $('ex-log').scrollHeight;
        if (job.status === 'running' || job.status === 'queued') {
          setTimeout(tick, 700);
        } else {
          if (job.status === 'completed') {
            setStatus('ex-status', 'completato: ' + (job.result.artifact_name || ''), 'ok');
            await loadArtifacts();
          } else {
            setStatus('ex-status', job.error || 'fallito', 'bad');
          }
        }
      } catch (err) {
        setStatus('ex-status', err.message, 'bad');
      }
    };
    tick();
  }

  $('ex-run').addEventListener('click', async () => {
    setStatus('ex-status', 'avvio…');
    $('ex-log').textContent = '';
    try {
      const fd = new FormData();
      fd.append('weights', val('ex-weights'));
      fd.append('imgsz', val('ex-imgsz'));
      fd.append('fmt', val('ex-format'));
      fd.append('precision', val('ex-precision'));
      fd.append('device', val('ex-device'));
      fd.append('dynamic', String(checked('ex-dynamic')));
      fd.append('force', String(checked('ex-force')));
      const job = await API.postForm('/api/engine/export', fd);
      pollEngineJob(job.id);
    } catch (err) {
      setStatus('ex-status', err.message, 'bad');
    }
  });

  /* ---------------- calibration ---------------- */
  async function pollCalibrationJob(jobId) {
    const tick = async () => {
      try {
        const job = await API.get('/api/calibration/jobs/' + jobId);
        setStatus('cal-status', job.status);
        $('cal-log').textContent = (job.logs || []).join('\n');
        $('cal-log').scrollTop = $('cal-log').scrollHeight;
        if (job.status === 'running' || job.status === 'queued') {
          setTimeout(tick, 900);
          return;
        }
        if (job.status !== 'completed') {
          setStatus('cal-status', job.error || 'fallito', 'bad');
          return;
        }
        setStatus('cal-status', 'completato', 'ok');
        const r = job.result;
        state.lastCalibration = r;
        const box = $('cal-result');
        box.textContent = '';
        const info = document.createElement('div');
        info.className = 'item';
        info.innerHTML = '<span class="growx">bev_unit</span><b>' + (r.bev_unit != null ? r.bev_unit : '—') + '</b>';
        box.appendChild(info);
        const mat = document.createElement('div');
        mat.className = 'item';
        mat.innerHTML = '<span class="growx">H</span><span class="muted small">' +
          (r.homography_matrix || []).map(row => row.map(v => (+v).toFixed(4)).join(', ')).join(' | ') + '</span>';
        box.appendChild(mat);
        const dh = $('cal-dl-h'); dh.href = r.homography_url; dh.style.display = 'inline-block';
        const dt = $('cal-dl-t'); dt.href = '/api/files/' + (r.text_path || '').split(/[\\/]/).pop(); dt.style.display = 'inline-block';
        const db = $('cal-dl-b'); db.href = r.bev_unit_url; db.style.display = 'inline-block';
      } catch (err) {
        setStatus('cal-status', err.message, 'bad');
      }
    };
    tick();
  }

  $('cal-run').addEventListener('click', async () => {
    setStatus('cal-status', 'avvio…');
    $('cal-log').textContent = '';
    try {
      const fd = new FormData();
      fd.append('input_path', val('cal-input'));
      fd.append('kp_model', val('cal-model'));
      fd.append('output_name', val('cal-output'));
      fd.append('device', val('cal-device'));
      fd.append('imgsz', '640');
      fd.append('max_frames', val('cal-maxframes'));
      fd.append('refine_iters', val('cal-refine'));
      fd.append('rigid_pct', val('cal-rigid'));
      fd.append('roi_file', val('dbg-roi'));
      fd.append('reproject_roi', String(checked('cal-reproject')));
      const job = await API.postForm('/api/calibration', fd);
      pollCalibrationJob(job.id);
    } catch (err) {
      setStatus('cal-status', err.message, 'bad');
    }
  });

  $('cal-use-roi').addEventListener('click', async () => {
    if (!state.lastCalibration) { setStatus('cal-status', 'nessuna calibrazione pronta', 'bad'); return; }
    const r = state.lastCalibration;
    try {
      $('dbg-homography').value = r.homography_path;
      if (r.bev_unit_path) $('dbg-bevunit').value = r.bev_unit_path;
      await startSelector({
        frame: val('roi-path'),
        homography: r.homography_path,
        bev_unit_file: r.bev_unit_path,
        restart: true,
      });
      switchTab('roi');
      setStatus('cal-status', 'omografia e bev_unit impostati per la tab ROI', 'ok');
    } catch (err) {
      setStatus('cal-status', err.message, 'bad');
    }
  });

  /* ---------------- ROI (delegated to select_roi_bev.py) ---------------- */
  function renderSelectorStatus(status) {
    const info = $('roi-source-info');
    if (!status.running) {
      info.textContent = 'selettore non in esecuzione';
      return;
    }
    info.textContent = 'selettore attivo · porta ' + status.port;
  }

  async function refreshSelectorStatus() {
    try {
      const status = await API.get('/api/roi/selector/status');
      renderSelectorStatus(status);
    } catch (err) {
      $('roi-source-info').textContent = err.message;
    }
  }

  async function startSelector(opts) {
    setStatus('roi-status', 'avvio selettore…');
    const payload = Object.assign({
      homography: val('dbg-homography'),
      bev_unit_file: val('dbg-bevunit'),
    }, opts || {});
    try {
      const status = await API.post('/api/roi/selector/start', payload);
      renderSelectorStatus(status);
      $('roi-frame-el').src = '/roi/app/?t=' + Date.now();
      setStatus('roi-status', 'selettore avviato: seleziona i vertici e premi "Salva ROI"', 'ok');
    } catch (err) {
      setStatus('roi-status', err.message, 'bad');
    }
  }

  $('roi-pick-file').addEventListener('click', () => $('roi-file').click());
  $('roi-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStatus('roi-status', 'caricamento…');
    try {
      const src = await API.upload('/api/roi/sources', file);
      await startSelector({ frame: src.path });
    } catch (err) {
      setStatus('roi-status', err.message, 'bad');
    }
  });

  $('roi-load-path').addEventListener('click', async () => {
    const path = val('roi-path');
    if (!path) { setStatus('roi-status', 'indicare un percorso', 'bad'); return; }
    await startSelector({ frame: path });
  });

  $('roi-stop').addEventListener('click', async () => {
    try {
      await API.post('/api/roi/selector/stop', {});
      renderSelectorStatus({ running: false });
      setStatus('roi-status', 'selettore fermato');
    } catch (err) {
      setStatus('roi-status', err.message, 'bad');
    }
  });

  async function pollSavedRoi() {
    try {
      const roi = await API.get('/api/roi/selector/saved');
      const link = $('roi-download');
      link.href = `/api/roi/${roi.id}/download`;
      link.style.display = 'inline-block';
      $('roi-cal-info').textContent =
        `ROI salvata (${roi.id}) · ${roi.image_points.length} vertici` +
        (roi.bev_unit != null ? ` · bev_unit ${roi.bev_unit}` : '');
      window.lastRoi = roi;
      return roi;
    } catch (err) {
      return null;
    }
  }

  window.addEventListener('message', (ev) => {
    if (ev.data && ev.data.type === 'roi-saved') pollSavedRoi();
  });

  /* ---------------- debug ---------------- */
  $('dbg-pick-video').addEventListener('click', () => $('dbg-file').click());
  $('dbg-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const src = await API.upload('/api/roi/sources', file);
      $('dbg-input').value = src.path;
      setStatus('dbg-status', 'video caricato: ' + src.filename, 'ok');
    } catch (err) {
      setStatus('dbg-status', err.message, 'bad');
    }
  });

  $('dbg-use-roi').addEventListener('click', async () => {
    try {
      const roi = await pollSavedRoi();
      if (!roi) { setStatus('dbg-status', 'nessuna ROI salvata', 'bad'); return; }
      $('dbg-roi').value = roi.path;
      if (roi.homography_file) $('dbg-homography').value = roi.homography_file;
      if (roi.bev_unit_file) $('dbg-bevunit').value = roi.bev_unit_file;
      setStatus('dbg-status', 'ROI caricata: ' + roi.id, 'ok');
    } catch (err) {
      setStatus('dbg-status', err.message, 'bad');
    }
  });

  $('dbg-run').addEventListener('click', async () => {
    setStatus('dbg-status', 'avvio…');
    $('dbg-log').textContent = '';
    $('dbg-events').textContent = '—';
    $('st-state').textContent = 'starting';
    try {
      const payload = {
        input_path: val('dbg-input'),
        engine_path: val('dbg-engine'),
        roi_path: val('dbg-roi'),
        homography_path: val('dbg-homography'),
        bev_unit_file: val('dbg-bevunit') || null,
        device: val('ex-device') || 'cuda:0',
        source_type: val('dbg-sourcetype'),
        kp_imgsz: num('dbg-imgsz', 640),
        kp_conf: num('dbg-kpconf', 0.1),
        kp_vis: num('dbg-kpvis', 0.6),
        class_conf: val('dbg-classconf'),
        object_px: num('dbg-objectpx', 80),
        smooth: num('dbg-smooth', 11),
        track_ocm_weight: num('dbg-ocm', 2),
        track_high_conf: num('dbg-highconf', 0.45),
        ttc_thresh: num('dbg-ttc', 2.5),
        confirm_frames: num('dbg-confirm', 6),
        table_height: num('dbg-tableh', 280),
        max_frames: num('dbg-maxframes', 0),
        save_violations: checked('dbg-saveviol'),
        no_clip: !checked('dbg-saveviol'),
        save_video: checked('dbg-savevideo')
      };
      const session = await API.post('/api/debug/sessions', payload);
      setStatus('dbg-status', 'sessione ' + session.id, 'ok');
      $('dbg-run').disabled = true;
      $('dbg-stop').disabled = false;
      Stream.connect(session.id);
    } catch (err) {
      setStatus('dbg-status', err.message, 'bad');
      $('dbg-run').disabled = false;
      $('dbg-stop').disabled = true;
    }
  });

  $('dbg-stop').addEventListener('click', () => {
    Stream.stop();
    setStatus('dbg-status', 'arresto in corso…');
  });

  /* ---------------- debug output paths ---------------- */
  let debugOutputs = null;

  async function loadDebugOutputs() {
    try {
      debugOutputs = await API.get('/api/debug/outputs');
    } catch (err) {
      debugOutputs = null;
    }
    renderDebugOutputs();
  }

  function renderDebugOutputs() {
    const box = $('dbg-outputs');
    if (!box) return;
    box.textContent = '';
    if (!debugOutputs) return;
    const saveViol = checked('dbg-saveviol');
    const saveVideo = checked('dbg-savevideo');
    const rows = [
      { label: 'JSON infrazioni', path: debugOutputs.outputs[0].path, on: saveViol },
      { label: 'clip video infrazioni', path: debugOutputs.outputs[1].path, on: saveViol },
      { label: 'video annotato', path: debugOutputs.outputs[2].path, on: saveVideo }
    ];
    const head = document.createElement('div');
    head.className = 'out-label';
    head.textContent = 'File salvati in (sessione corrente):';
    box.appendChild(head);
    rows.forEach(r => {
      const row = document.createElement('div');
      row.className = 'out-row' + (r.on ? '' : ' out-off');
      const lab = document.createElement('span');
      lab.className = 'out-label';
      lab.textContent = r.label + (r.on ? '' : ' (non attivo)');
      const p = document.createElement('span');
      p.className = 'out-path';
      p.textContent = r.path;
      row.appendChild(lab);
      row.appendChild(p);
      box.appendChild(row);
    });
  }

  ['dbg-saveviol', 'dbg-savevideo'].forEach(id => {
    const el = $(id);
    if (el) el.addEventListener('change', renderDebugOutputs);
  });
  loadDebugOutputs();

  $('ex-weights').addEventListener('change', loadArtifacts);

  /* ---------------- init ---------------- */
  loadHealth();
  loadArtifacts();
  refreshSelectorStatus();
  setInterval(refreshSelectorStatus, 4000);
})();
