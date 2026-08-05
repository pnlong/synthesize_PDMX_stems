/**
 * webMUSHRA-style shared loop player: one waveform, synced playhead across
 * reference/conditions. Switching play sources continues from the same time.
 */

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const s = Math.floor(seconds % 60);
  const m = Math.floor(seconds / 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function computePeaks(channelData, buckets = 400) {
  const peaks = new Float32Array(buckets);
  const block = Math.max(1, Math.floor(channelData.length / buckets));
  for (let i = 0; i < buckets; i += 1) {
    const start = i * block;
    const end = Math.min(channelData.length, start + block);
    let peak = 0;
    for (let j = start; j < end; j += 1) {
      const v = Math.abs(channelData[j]);
      if (v > peak) peak = v;
    }
    peaks[i] = peak;
  }
  return peaks;
}

export class SharedLoopPlayer {
  /**
   * @param {{
   *   canvas: HTMLCanvasElement,
   *   timeEl?: HTMLElement | null,
   *   onActiveChange?: (key: string | null, playing: boolean) => void,
   * }} options
   */
  constructor(options) {
    this.canvas = options.canvas;
    this.timeEl = options.timeEl || null;
    this.onActiveChange = options.onActiveChange || null;
    this.ctx = this.canvas.getContext("2d");
    this.audioCtx = null;
    this.sources = new Map(); // key -> { url, element, peaks }
    this.activeKey = null;
    this.sharedTime = 0;
    this.waveformKey = null;
    this._raf = null;
    this._destroyed = false;

    this.canvas.addEventListener("click", (event) => this._seekFromClick(event));
    window.addEventListener("resize", () => this._draw());
  }

  _ensureAudioCtx() {
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioCtx.state === "suspended") {
      this.audioCtx.resume().catch(() => {});
    }
    return this.audioCtx;
  }

  _notify() {
    this.onActiveChange?.(this.activeKey, this.isPlaying());
  }

  isPlaying() {
    if (!this.activeKey) return false;
    const entry = this.sources.get(this.activeKey);
    return Boolean(entry && !entry.element.paused && !entry.element.ended);
  }

  currentTime() {
    if (this.activeKey) {
      const entry = this.sources.get(this.activeKey);
      if (entry && Number.isFinite(entry.element.currentTime)) {
        return entry.element.currentTime;
      }
    }
    return this.sharedTime;
  }

  async load(sources, { waveformKey = null } = {}) {
    this.stop();
    this._teardownSources();
    this.waveformKey = waveformKey;
    this.sharedTime = 0;

    const entries = Object.entries(sources || {}).filter(([, url]) => Boolean(url));
    await Promise.all(
      entries.map(async ([key, url]) => {
        const element = new Audio();
        element.preload = "auto";
        element.loop = true;
        element.src = url;
        element.addEventListener("timeupdate", () => {
          if (this.activeKey === key) {
            this.sharedTime = element.currentTime;
          }
        });
        const peaks = await this._loadPeaks(url);
        this.sources.set(key, { url, element, peaks });
      }),
    );

    if (!this.waveformKey || !this.sources.has(this.waveformKey)) {
      this.waveformKey = entries[0]?.[0] || null;
    }
    this._draw();
    this._updateTimeLabel();
    this._notify();
  }

  async _loadPeaks(url) {
    try {
      const ctx = this._ensureAudioCtx();
      const response = await fetch(url);
      if (!response.ok) return null;
      const buffer = await response.arrayBuffer();
      const decoded = await ctx.decodeAudioData(buffer.slice(0));
      const channel = decoded.getChannelData(0);
      return computePeaks(channel, Math.max(200, this.canvas.clientWidth || 400));
    } catch {
      return null;
    }
  }

  _teardownSources() {
    for (const entry of this.sources.values()) {
      entry.element.pause();
      entry.element.removeAttribute("src");
      entry.element.load();
    }
    this.sources.clear();
    this.activeKey = null;
  }

  stop() {
    for (const entry of this.sources.values()) {
      entry.element.pause();
    }
    this._stopRaf();
    this._notify();
    this._draw();
  }

  pause() {
    if (!this.activeKey) return;
    const entry = this.sources.get(this.activeKey);
    if (!entry) return;
    this.sharedTime = entry.element.currentTime;
    entry.element.pause();
    this._stopRaf();
    this._notify();
    this._draw();
  }

  /**
   * Play ``key``, continuing from the shared playhead position.
   * If ``key`` is already playing, pause it.
   */
  async toggle(key) {
    if (!this.sources.has(key)) return;
    if (this.activeKey === key && this.isPlaying()) {
      this.pause();
      return;
    }
    await this.play(key);
  }

  async play(key) {
    const entry = this.sources.get(key);
    if (!entry) return;

    this._ensureAudioCtx();
    const resumeAt = this.currentTime();

    for (const [otherKey, other] of this.sources) {
      if (otherKey !== key) {
        other.element.pause();
      }
    }

    const applyAndPlay = async () => {
      const duration = entry.element.duration;
      const t = Number.isFinite(duration) && duration > 0
        ? resumeAt % duration
        : resumeAt;
      try {
        entry.element.currentTime = t;
      } catch {
        // ignore seek errors before ready
      }
      this.sharedTime = entry.element.currentTime || t;
      this.activeKey = key;
      await entry.element.play();
      this._startRaf();
      this._notify();
      this._draw();
    };

    if (entry.element.readyState >= 1) {
      await applyAndPlay();
      return;
    }

    await new Promise((resolve, reject) => {
      const onReady = () => {
        entry.element.removeEventListener("error", onError);
        resolve();
      };
      const onError = () => {
        entry.element.removeEventListener("loadedmetadata", onReady);
        reject(new Error(`Failed to load audio for ${key}`));
      };
      entry.element.addEventListener("loadedmetadata", onReady, { once: true });
      entry.element.addEventListener("error", onError, { once: true });
      entry.element.load();
    });
    await applyAndPlay();
  }

  _seekFromClick(event) {
    const entry = this.sources.get(this.activeKey || this.waveformKey);
    if (!entry) return;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0) return;
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const duration = entry.element.duration;
    if (!Number.isFinite(duration) || duration <= 0) return;
    const t = ratio * duration;
    this.sharedTime = t;
    for (const src of this.sources.values()) {
      if (Number.isFinite(src.element.duration) && src.element.duration > 0) {
        try {
          src.element.currentTime = t % src.element.duration;
        } catch {
          // ignore
        }
      }
    }
    this._updateTimeLabel();
    this._draw();
  }

  _startRaf() {
    this._stopRaf();
    const tick = () => {
      this._updateTimeLabel();
      this._draw();
      if (this.isPlaying()) {
        this._raf = requestAnimationFrame(tick);
      } else {
        this._raf = null;
      }
    };
    this._raf = requestAnimationFrame(tick);
  }

  _stopRaf() {
    if (this._raf != null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  _updateTimeLabel() {
    if (!this.timeEl) return;
    const entry = this.sources.get(this.activeKey || this.waveformKey);
    const duration = entry?.element.duration;
    this.timeEl.textContent = `${formatTime(this.currentTime())} / ${formatTime(duration || 10)}`;
  }

  _draw() {
    const canvas = this.canvas;
    const ctx = this.ctx;
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(canvas.clientWidth || canvas.width || 600));
    const height = Math.max(1, Math.floor(canvas.clientHeight || 72));
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#12151c";
    ctx.fillRect(0, 0, width, height);

    const waveEntry = this.sources.get(this.waveformKey || this.activeKey);
    const peaks = waveEntry?.peaks;
    const mid = height / 2;

    if (peaks && peaks.length) {
      const n = peaks.length;
      ctx.fillStyle = "#5b8def";
      for (let i = 0; i < n; i += 1) {
        const x = (i / n) * width;
        const mag = peaks[i] * (height * 0.42);
        const barW = Math.max(1, width / n - 0.5);
        ctx.fillRect(x, mid - mag, barW, mag * 2);
      }
    } else {
      ctx.strokeStyle = "#2a2f3a";
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(width, mid);
      ctx.stroke();
    }

    const entry = this.sources.get(this.activeKey || this.waveformKey);
    const duration = entry?.element.duration;
    if (Number.isFinite(duration) && duration > 0) {
      const t = this.currentTime() % duration;
      const x = (t / duration) * width;
      ctx.strokeStyle = "#e8eaed";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }

  destroy() {
    this._destroyed = true;
    this.stop();
    this._teardownSources();
    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {});
      this.audioCtx = null;
    }
  }
}

export function createPlayButton({ label = "Play", onClick }) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "play-btn";
  btn.textContent = "▶ Play";
  btn.dataset.label = label;
  btn.addEventListener("click", onClick);
  return btn;
}

export function setPlayButtonState(btn, { active, playing }) {
  if (!btn) return;
  btn.classList.toggle("is-active", Boolean(active));
  btn.classList.toggle("is-playing", Boolean(playing));
  if (playing && active) {
    btn.textContent = "⏸ Pause";
  } else {
    btn.textContent = "▶ Play";
  }
}
