// fjn-splat-viewer: interactive 3DGS scenes with per-instance query highlighting.
// Built on three.js + Spark (both MIT). Markup contract:
//   <div data-splat-viewer data-scenes='[{"id":"a","label":"A","url":"assets/scenes/a/","fallback":"assets/clips/a.mp4","poster":"assets/clips/a.jpg"}]'></div>
// Each scene folder holds scene.json (see fjn-projectpage project-format.md), the .spz and instances.u16.
// three.js and Spark are imported lazily (importmap: three, three/addons/, @sparkjsdev/spark) when the viewer comes within
// data-margin px (default 200) of the screen, so the scene does not count against the page's initial weight.
let THREE, OrbitControls, SplatMesh, SparkRenderer;
async function libs() {
  if (!THREE) {
    const [t, oc, sp] = await Promise.all([import('three'), import('three/addons/controls/OrbitControls.js'), import('@sparkjsdev/spark')]);
    THREE = t; OrbitControls = oc.OrbitControls; SplatMesh = sp.SplatMesh; SparkRenderer = sp.SparkRenderer;
  }
}

const EASE_MS = 380;
const ease = t => 1 - Math.pow(1 - t, 3);

function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function webgl2() {
  try { return !!document.createElement('canvas').getContext('webgl2'); } catch { return false; }
}

class Viewer {
  constructor(el) {
    this.el = el;
    this.scenes = JSON.parse(el.dataset.scenes || '[]');
    this.dim = Number(el.dataset.dim || 0.34);
    this.state('idle');
    this.build();
    this.first = this.scenes.find(s => s.id === el.dataset.initial) || this.scenes[0];
    this.renderTabs(this.first.id);
    const io = new IntersectionObserver(es => {
      if (es.some(e => e.isIntersecting)) { io.disconnect(); this.start(); }
    }, {rootMargin: `${Number(el.dataset.margin ?? 200)}px 0px`});  // data-margin: start this many px before it is on screen
    io.observe(el);
  }

  start() {
    if (this.started) return this.started;
    return this.started = (async () => {
      if (!webgl2()) return this.fallback('WebGL 2 is not available in this browser.', this.first);
      this.state('loading');
      try { await libs(); } catch (err) { return this.fallback(`Could not load the 3D viewer (${err.message}).`, this.first); }
      this.initGL();
      await this.load(this.first);
    })();
  }

  state(s) { this.el.dataset.state = s; }

  build() {
    const el = this.el;
    el.classList.add('fjn-viewer');
    this.stage = h('div', 'fv-stage');
    this.overlay = h('div', 'fv-overlay');
    this.bar = h('div', 'fv-progress');
    this.bar.append(h('span'));
    this.status = h('p', 'fv-status', 'Loading 3D scene…');
    this.overlay.append(this.status, this.bar);
    this.hint = h('p', 'fv-hint', 'Drag to orbit · scroll or pinch to zoom · right-drag to pan');
    this.reset = h('button', 'fv-reset', 'Reset view');
    this.reset.type = 'button';
    this.reset.addEventListener('click', () => this.resetView());
    this.stage.append(this.overlay, this.hint, this.reset);
    this.panel = h('div', 'fv-panel');
    this.tabs = h('div', 'fv-tabs');
    this.tabs.setAttribute('role', 'tablist');
    this.tabs.setAttribute('aria-label', 'Scenes');
    for (const s of this.scenes) {
      const b = h('button', 'fv-tab', s.label);
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.dataset.id = s.id;
      b.addEventListener('click', () => {
        if (this.el.dataset.state === 'fallback') return this.fallback(this.reason, s);
        if (this.started) return this.started.then(() => this.load(s));
        this.first = s; this.renderTabs(s.id); this.start();
      });
      this.tabs.append(b);
    }
    this.meta = h('p', 'fv-meta');
    this.queries = h('div', 'fv-queries');
    this.queries.setAttribute('role', 'listbox');
    this.queries.setAttribute('aria-label', 'Text queries');
    this.note = h('p', 'fv-note');
    const inner = h('div', 'fv-panel-inner');
    inner.append(this.tabs, this.meta, this.queries, this.note);
    this.panel.append(inner);
    el.append(this.stage, this.panel);
  }

  initGL() {
    const r = this.renderer = new THREE.WebGLRenderer({antialias: false, alpha: false, powerPreference: 'high-performance'});
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    r.setClearColor(new THREE.Color(this.el.dataset.background || '#05080b'));
    this.stage.prepend(r.domElement);
    this.scene = new THREE.Scene();
    this.spark = new SparkRenderer({renderer: r});
    this.scene.add(this.spark);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
    this.makeControls();
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.visible = true;
    this.loop = () => this.frame();
    new IntersectionObserver(es => { this.visible = es[0].isIntersecting; }).observe(this.stage);
    r.setAnimationLoop(this.loop);
  }

  makeControls() {
    this.controls?.dispose();
    const c = this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(c, {enableDamping: true, dampingFactor: 0.08, autoRotateSpeed: 0.6,
      autoRotate: !matchMedia('(prefers-reduced-motion: reduce)').matches});
    c.addEventListener('start', () => { c.autoRotate = false; this.hint.classList.add('fv-faded'); });
  }

  resize() {
    const w = this.stage.clientWidth, hgt = this.stage.clientHeight;
    if (!w || !hgt) return;
    this.renderer.setSize(w, hgt, false);
    this.camera.aspect = w / hgt;
    this.camera.updateProjectionMatrix();
  }

  async load(s) {
    if (!s || this.current === s.id) return;
    this.current = s.id;
    this.renderTabs(s.id);
    this.state('loading');
    this.overlay.hidden = false;
    this.status.textContent = `Loading ${s.label}…`;
    this.bar.firstChild.style.width = '0%';
    const token = this.token = {};
    try {
      const base = new URL(s.url, document.baseURI);
      const meta = await (await fetch(new URL('scene.json', base))).json();
      const instBuf = await (await fetch(new URL(meta.instances, base))).arrayBuffer();
      const mesh = new SplatMesh({
        url: new URL(meta.spz, base).href,
        onProgress: e => { if (e.total) this.bar.firstChild.style.width = `${(100 * e.loaded / e.total).toFixed(0)}%`; },
      });
      await mesh.initialized;
      if (token !== this.token) { mesh.dispose?.(); return; }
      if (this.mesh) { this.scene.remove(this.mesh); this.mesh.dispose?.(); }
      this.mesh = mesh;
      this.scene.add(mesh);
      this.meta_ = meta;
      this.inst = new Uint16Array(instBuf);
      const ps = mesh.packedSplats;
      if (ps.numSplats !== this.inst.length) throw new Error(`instance labels (${this.inst.length}) do not match splats (${ps.numSplats})`);
      this.words = ps.packedArray;
      const n = ps.numSplats;
      this.orig = new Uint32Array(n);
      this.cur = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const w = this.orig[i] = this.words[i * 4];
        this.cur[i * 3] = w & 255; this.cur[i * 3 + 1] = (w >>> 8) & 255; this.cur[i * 3 + 2] = (w >>> 16) & 255;
      }
      this.home = meta.camera;
      this.resetView();
      this.renderQueries(meta);
      this.select(meta.queries.find(q => q.text === meta.default_query) || meta.queries[0], true);
      this.status.textContent = `Preparing ${s.label}…`;
      await this.settle(n);
      if (token !== this.token) return;
      this.overlay.hidden = true;
      this.state('ready');
      this.el.dispatchEvent(new CustomEvent('fjn-viewer-ready', {detail: {scene: s.id}}));
    } catch (err) {
      console.warn('fjn-splat-viewer:', err);
      if (token === this.token) this.fallback(`Could not load the 3D scene (${err.message}).`, s);
    }
  }

  // Spark sorts and accumulates splats asynchronously; wait until the new mesh is what is actually drawn.
  settle(n, timeoutMs = 15000) {
    const t0 = performance.now();
    this.settling = true;
    return new Promise(res => {
      let frames = 0;
      const tick = () => {
        // The camera may keep moving (auto-rotate) so sorting never stops; what matters is that the drawn set is the new mesh.
        const sp = this.spark, done = sp.activeSplats === n && sp.display?.numSplats === n;
        frames = done ? frames + 1 : 0;
        if (frames >= 4 || performance.now() - t0 > timeoutMs) { this.settling = false; res(); }
        else requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  renderTabs(id) {
    for (const b of this.tabs.children) b.setAttribute('aria-selected', String(b.dataset.id === id));
  }

  resetView() {
    const c = this.home;
    if (!c) return;
    this.glide = null;
    this.camera.up.set(...c.up);
    this.camera.position.set(...c.position);
    this.camera.fov = c.fov || 50;
    this.camera.updateProjectionMatrix();
    this.makeControls();
    this.controls.target.set(...c.target);
    if (c.min_distance) this.controls.minDistance = c.min_distance;
    if (c.max_distance) this.controls.maxDistance = c.max_distance;
    this.controls.update();
    this.hint.classList.remove('fv-faded');
  }

  renderQueries(meta, readOnly = false) {
    this.meta.textContent = meta.caption || '';
    this.note.textContent = meta.note || '';
    this.queries.replaceChildren();
    for (const q of meta.queries) {
      const b = h('button', 'fv-query');
      b.type = 'button';
      b.setAttribute('role', 'option');
      b.append(h('span', 'fv-q', q.text));
      if (q.iou != null) {
        const v = h('span', 'fv-iou', (100 * q.iou).toFixed(0));
        v.dataset.noClaim = '';
        if (q.iou < 0.25) b.classList.add('fv-low');
        b.append(v);
        b.title = `IoU ${(100 * q.iou).toFixed(1)}% on the benchmark frames · ${q.selected.length} object${q.selected.length === 1 ? '' : 's'} selected`;
      }
      if (readOnly) b.disabled = true;
      else b.addEventListener('click', () => this.select(q));
      q.button = b;
      this.queries.append(b);
    }
  }

  select(q, instant = false) {
    if (!q || !this.mesh) return;
    for (const b of this.queries.children) b.setAttribute('aria-selected', 'false');
    q.button?.setAttribute('aria-selected', 'true');
    if (q.button) this.reveal(q.button);
    this.el.dataset.query = q.text;
    const sel = new Set(q.selected);
    const n = this.orig.length, target = new Float32Array(n * 3), d = this.dim;
    for (let i = 0; i < n; i++) {
      const w = this.orig[i], r = w & 255, g = (w >>> 8) & 255, b = (w >>> 16) & 255, o = i * 3;
      if (sel.has(this.inst[i])) {
        target[o] = Math.min(255, r * 1.06 + 4); target[o + 1] = Math.min(255, g * 1.06 + 4); target[o + 2] = Math.min(255, b * 1.06 + 4);
      } else {
        const l = (0.3 * r + 0.59 * g + 0.11 * b) * d;
        target[o] = l * 0.86; target[o + 1] = l * 0.95; target[o + 2] = l * 1.08;
      }
    }
    if (instant) { this.cur.set(target); this.applyColors(this.cur); this.anim = null; }
    else { this.anim = {from: this.cur.slice(), to: target, t0: performance.now(), dur: EASE_MS}; this.focus(q); }
  }

  // Scroll the active chip into view inside the panel only (scrollIntoView would also move the page).
  reveal(b) {
    for (let p = b.parentElement; p && p !== this.el.parentElement; p = p.parentElement) {
      if (p.scrollHeight <= p.clientHeight + 1 || !/auto|scroll/.test(getComputedStyle(p).overflowY)) continue;
      const pr = p.getBoundingClientRect(), br = b.getBoundingClientRect();
      if (br.top < pr.top || br.bottom > pr.bottom) p.scrollTop += br.top - pr.top - (p.clientHeight - br.height) / 2;
      return;
    }
  }

  // Pan (target and camera together) so the selected objects sit at the orbit centre.
  focus(q) {
    const cs = q.selected.map(i => this.meta_.centroids?.[i]).filter(Boolean);
    if (!cs.length) return;
    const to = new THREE.Vector3();
    for (const c of cs) to.add(new THREE.Vector3(...c));
    to.divideScalar(cs.length);
    this.glide = {from: this.controls.target.clone(), to, t0: performance.now(), dur: 700};
  }

  applyColors(arr) {
    const w = this.words, n = this.orig.length;
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      w[i * 4] = ((arr[o] & 255) | ((arr[o + 1] & 255) << 8) | ((arr[o + 2] & 255) << 16) | (this.orig[i] & 0xff000000)) >>> 0;
    }
    this.mesh.packedSplats.needsUpdate = true;
  }

  frame() {
    if (this.anim && this.mesh) {
      const a = this.anim, k = a.dur ? Math.min(1, (performance.now() - a.t0) / a.dur) : 1, e = ease(k);
      for (let i = 0; i < this.cur.length; i++) this.cur[i] = a.from[i] + (a.to[i] - a.from[i]) * e;
      this.applyColors(this.cur);
      if (k >= 1) this.anim = null;
    }
    if (this.glide) {
      const g = this.glide, k = Math.min(1, (performance.now() - g.t0) / g.dur);
      const next = g.from.clone().lerp(g.to, ease(k)), delta = next.clone().sub(this.controls.target);
      this.controls.target.add(delta); this.camera.position.add(delta);
      if (k >= 1) this.glide = null;
    }
    if (!this.visible && !this.anim && !this.settling && !this.glide) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  snapshot() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  // Stop the render loop and show a still frame (used by page checks: screenshots wait for idle frames).
  freeze() {
    if (!this.renderer || this.frozen) return;
    if (this.anim) { this.cur.set(this.anim.to); this.applyColors(this.cur); this.anim = null; }
    const url = this.snapshot();
    this.renderer.setAnimationLoop(null);
    this.frozen = h('img', 'fv-fallback');
    this.frozen.alt = '';
    this.frozen.src = url;
    this.stage.prepend(this.frozen);
  }

  thaw() {
    if (!this.frozen) return;
    this.frozen.remove(); this.frozen = null;
    this.renderer.setAnimationLoop(this.loop);
  }

  fallback(message, s) {
    s = s || this.first || this.scenes[0] || {};
    this.state('fallback');
    this.reason = this.el.dataset.reason = message;
    this.renderTabs(s.id);
    fetch(new URL('scene.json', new URL(s.url, document.baseURI))).then(r => r.json()).then(m => this.renderQueries(m, true)).catch(() => {});
    const src = s.fallback;
    this.bar.hidden = true;
    this.reset.hidden = true;
    this.status.textContent = message + (src ? ' Showing a pre-rendered orbit instead.' : '');
    if (src) {
      this.overlay.hidden = true;
      this.hint.textContent = message + ' Showing a pre-rendered orbit.';
      this.hint.classList.remove('fv-faded');
      this.stage.querySelector('.fv-fallback')?.remove();
      const v = h('video', 'fv-fallback');
      Object.assign(v, {src, muted: true, loop: true, autoplay: true, playsInline: true});
      if (s.poster) v.poster = s.poster;
      v.setAttribute('aria-label', 'Pre-rendered orbit of the scene');
      this.stage.prepend(v);
    } else this.overlay.hidden = false;
  }
}

export function mountSplatViewer(el) {
  const v = new Viewer(el);
  el.fjnViewer = {
    get ready() { return el.dataset.state === 'ready'; },
    start: () => v.start(),
    freeze: () => v.freeze(),
    thaw: () => v.thaw(),
    select: text => v.select(v.meta_?.queries.find(q => q.text === text), true),
    load: id => v.load(v.scenes.find(s => s.id === id)),
    snapshot: () => v.snapshot(),
  };
  return el.fjnViewer;
}

for (const el of document.querySelectorAll('[data-splat-viewer]')) mountSplatViewer(el);
