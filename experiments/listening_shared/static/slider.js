const BAND_MIDPOINTS = [10, 30, 50, 70, 90];
const TICK_VALUES = [0, 20, 40, 60, 80, 100];

const CONTENT_BAND_LABELS = [
  "Very different",
  "Different",
  "Mostly same",
  "Same",
  "Identical",
];

const REALISM_BAND_LABELS = [
  "Very synthetic",
  "Synthetic",
  "Mixed",
  "Realistic",
  "Very realistic",
];

export function clampScore(value) {
  const n = Number.parseInt(String(value), 10);
  if (Number.isNaN(n)) {
    return null;
  }
  return Math.max(0, Math.min(100, n));
}

export function bandIndex(score) {
  const s = clampScore(score);
  if (s === null) {
    return null;
  }
  if (s >= 80) return 4;
  if (s >= 60) return 3;
  if (s >= 40) return 2;
  if (s >= 20) return 1;
  return 0;
}

export function bandMidpoint(index) {
  return BAND_MIDPOINTS[index];
}

export function bandLabel(score, rubric) {
  const idx = bandIndex(score);
  if (idx === null) {
    return "—";
  }
  const labels = rubric === "realism" ? REALISM_BAND_LABELS : CONTENT_BAND_LABELS;
  return labels[idx];
}

export function isScoreRated(value) {
  return clampScore(value) !== null;
}

/**
 * Create a 0–100 score slider with band snap-on-click and precise drag/type.
 *
 * @param {HTMLElement} container
 * @param {{ rubric: 'content'|'realism', label: string, help?: string, value?: number|null, onChange?: (value: number) => void }} options
 */
export function createScoreSlider(container, options) {
  const { rubric, label, help = "", value = null, onChange } = options;

  const row = document.createElement("div");
  row.className = "score-slider-row";

  const labelEl = document.createElement("span");
  labelEl.className = "score-slider-label";
  labelEl.textContent = label;
  if (help) {
    labelEl.title = help;
  }

  const wrap = document.createElement("div");
  wrap.className = "score-slider-wrap";

  const ticks = document.createElement("div");
  ticks.className = "score-slider-ticks";
  for (const tick of TICK_VALUES) {
    const span = document.createElement("span");
    span.textContent = String(tick);
    ticks.append(span);
  }

  const bands = document.createElement("div");
  bands.className = "score-slider-bands";
  bands.setAttribute("role", "group");
  bands.setAttribute("aria-label", `${label} band quick select`);

  for (let i = 0; i < BAND_MIDPOINTS.length; i += 1) {
    const band = document.createElement("button");
    band.type = "button";
    band.className = "score-slider-band";
    band.title = `Snap to ${BAND_MIDPOINTS[i]}`;
    band.addEventListener("click", () => commit(BAND_MIDPOINTS[i]));
    bands.append(band);
  }

  const range = document.createElement("input");
  range.type = "range";
  range.min = "0";
  range.max = "100";
  range.step = "1";
  range.className = "score-slider-input";
  range.setAttribute("aria-label", label);

  const bandLabelEl = document.createElement("div");
  bandLabelEl.className = "score-slider-band-label";

  const number = document.createElement("input");
  number.type = "number";
  number.min = "0";
  number.max = "100";
  number.step = "1";
  number.className = "score-slider-value";
  number.setAttribute("aria-label", `${label} exact value`);

  let current = clampScore(value);

  function paint() {
    if (current === null) {
      range.value = "50";
      number.value = "";
      bandLabelEl.textContent = "Not rated";
      return;
    }
    range.value = String(current);
    number.value = String(current);
    bandLabelEl.textContent = bandLabel(current, rubric);
  }

  function commit(next) {
    current = clampScore(next);
    if (current === null) {
      return;
    }
    paint();
    onChange?.(current);
  }

  range.addEventListener("input", () => commit(range.value));
  number.addEventListener("change", () => commit(number.value));

  wrap.append(ticks, bands, range, bandLabelEl);
  row.append(labelEl, wrap, number);
  container.append(row);
  paint();

  return {
    getValue() {
      return current;
    },
    setValue(next) {
      current = clampScore(next);
      paint();
    },
    isRated() {
      return isScoreRated(current);
    },
  };
}
