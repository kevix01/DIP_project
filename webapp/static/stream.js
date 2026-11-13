/* WebSocket stream consumer: frames, events, logs, stats. */
const Stream = {
  ws: null,
  sessionId: null,
  violations: 0,
  alertTimer: null,

  connect(sessionId) {
    this.disconnect();
    this.sessionId = sessionId;
    this.violations = 0;
    $('st-violations').textContent = '0';
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/debug/${sessionId}`);
    this.ws = ws;

    ws.onmessage = (ev) => this.onMessage(ev);
    ws.onopen = () => setStatus('dbg-status', 'connesso', 'ok');
    ws.onclose = () => {
      setStatus('dbg-status', 'disconnesso');
      $('st-state').textContent = 'disconnected';
      $('dbg-run').disabled = false;
      $('dbg-stop').disabled = true;
    };
    ws.onerror = () => setStatus('dbg-status', 'errore websocket', 'bad');
  },

  disconnect() {
    if (this.ws) {
      try { this.ws.close(); } catch (e) { /* ignore */ }
      this.ws = null;
    }
  },

  stop() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'stop' }));
    }
  },

  onMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }

    switch (msg.type) {
      case 'frame': {
        const img = $('dbg-image');
        img.src = 'data:image/jpeg;base64,' + msg.jpeg_base64;
        img.style.display = 'block';
        $('dbg-placeholder').style.display = 'none';
        $('st-fps').textContent = msg.fps != null ? msg.fps.toFixed(1) : '—';
        $('st-frame').textContent = msg.frame_idx;
        $('st-rendered').textContent = msg.frames_rendered;
        $('st-state').textContent = 'running';
        break;
      }
      case 'event': {
        this.onEvent(msg.event);
        break;
      }
      case 'log': {
        appendLog('dbg-log', msg.message);
        break;
      }
      case 'status': {
        $('st-state').textContent = msg.status;
        if (msg.fps) $('st-fps').textContent = msg.fps.toFixed(1);
        if (msg.frames_rendered != null) $('st-rendered').textContent = msg.frames_rendered;
        if (msg.status === 'completed' || msg.status === 'failed' || msg.status === 'stopped') {
          $('dbg-run').disabled = false;
          $('dbg-stop').disabled = true;
        }
        break;
      }
      case 'error': {
        appendLog('dbg-log', 'ERRORE: ' + msg.message);
        setStatus('dbg-status', msg.message, 'bad');
        break;
      }
    }
  },

  onEvent(event) {
    const list = $('dbg-events');
    if (list.textContent === '—') list.textContent = '';

    if (event.type === 'violation') {
      this.violations += 1;
      $('st-violations').textContent = this.violations;
      const div = document.createElement('div');
      div.className = 'event';
      div.innerHTML = `<b>Veicolo ${event.vehicle_id}</b> vs <b>Pedone ${event.pedestrian_id}</b>` +
        ` · TTC ${event.ttc_seconds}s · frame ${event.frame_idx} · ${event.timestamp || ''}`;
      list.prepend(div);
      this.flashAlert();
      appendLog('dbg-log', `INFRAZIONE: V${event.vehicle_id} vs P${event.pedestrian_id} TTC ${event.ttc_seconds}s`);
    } else if (event.type === 'violation_closed') {
      const div = document.createElement('div');
      div.className = 'event closed';
      div.textContent = `Chiusa: veicolo ${event.vehicle_id} · ${event.reason} · frame ${event.frame_idx}`;
      list.prepend(div);
      appendLog('dbg-log', `INFRAZIONE CHIUSA: V${event.vehicle_id} (${event.reason})`);
    }
  },

  flashAlert() {
    const alert = $('dbg-alert');
    alert.style.display = 'block';
    if (this.alertTimer) clearTimeout(this.alertTimer);
    this.alertTimer = setTimeout(() => { alert.style.display = 'none'; }, 2500);
  }
};
