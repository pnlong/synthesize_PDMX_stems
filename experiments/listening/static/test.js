const params = new URLSearchParams(window.location.search);
const SWEEP_TYPE = params.get("type") || "preset";
const SESSION_SEED = Number(params.get("seed") || "42");

const state = {
  meta: null,
  stemOrder: [],
  stemIndex: 0,
  stemDetail: null,
  ratings: {},
  storageKey: null,
};

const loadingEl = document.getElementById("loading");
const completeEl = document.getElementById("complete");
const testPanelEl = document.getElementById("test-panel");
const titleEl = document.getElementById("title");
const progressEl = document.getElementById("progress");
const saveStatusEl = document.getElementById("save-status");
const stemTitleEl = document.getElementById("stem-title");
const stemMetaEl = document.getElementById("stem-meta");
const stemNoteEl = document.getElementById("stem-note");
const referenceLabelEl = document.getElementById("reference-label");
const referenceAudioEl = document.getElementById("reference-audio");
const rubricHintEl = document.getElementById("rubric-hint");
const samplesEl = document.getElementById("samples");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const exportBtn = document.getElementById("export-btn");
const saveBtn = document.getElementById("save-btn");
const completeMessageEl = document.getElementById("complete-message");
const completePathEl = document.getElementById("complete-path");
const completeDownloadBtn = document.getElementById("complete-download-btn");

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Request failed: ${url}`);
  }
  return response.json();
}

function loadSavedRatings() {
  if (!state.storageKey) {
    return {};
  }
  try {
    const raw = localStorage.getItem(state.storageKey);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function stemSampleCount(entry) {
  return Object.values(entry?.samples || {}).filter(
    (sample) => isRated(sample.content) && isRated(sample.realism)
  ).length;
}

function mergeRatings(local, serverRatings) {
  const merged = { ...local };
  for (const [stemId, entry] of Object.entries(serverRatings || {})) {
    const localEntry = merged[stemId];
    if (!localEntry || stemSampleCount(entry) > stemSampleCount(localEntry)) {
      merged[stemId] = entry;
    }
  }
  return merged;
}

function ratingsFromServerPayload(payload) {
  const ratings = {};
  for (const entry of payload?.ratings || []) {
    if (!entry?.stem_id) {
      continue;
    }
    ratings[entry.stem_id] = {
      category: entry.category,
      samples: Object.fromEntries(
        (entry.samples || []).map((sample) => [sample.variant_id, sample])
      ),
    };
  }
  return ratings;
}

function showSaveStatus(message, isError = false) {
  if (!saveStatusEl) {
    return;
  }
  saveStatusEl.textContent = message;
  saveStatusEl.classList.toggle("error", isError);
  saveStatusEl.classList.remove("hidden");
  window.clearTimeout(showSaveStatus._timer);
  showSaveStatus._timer = window.setTimeout(() => {
    saveStatusEl.classList.add("hidden");
  }, isError ? 6000 : 2500);
}

async function loadServerRatings() {
  try {
    return await fetchJson(`/api/${SWEEP_TYPE}/responses/session`);
  } catch {
    return null;
  }
}

function saveRatings() {
  if (!state.storageKey) {
    return;
  }
  localStorage.setItem(state.storageKey, JSON.stringify(state.ratings));
}

function currentStemId() {
  return state.stemOrder[state.stemIndex];
}

function stemRatings(stemId) {
  if (!state.ratings[stemId]) {
    state.ratings[stemId] = { samples: {} };
  }
  return state.ratings[stemId];
}

function isStemComplete(stemId) {
  if (!state.stemDetail || state.stemDetail.id !== stemId) {
    const saved = state.ratings[stemId];
    if (!saved || !state.meta) {
      return false;
    }
    const nVariants = state.meta.variants.length;
    const rated = Object.values(saved.samples || {}).filter(
      (s) => isRated(s.content) && isRated(s.realism)
    ).length;
    return rated >= nVariants;
  }
  const samples = state.stemDetail.samples || [];
  return samples.every((sample) => {
    const r = stemRatings(stemId).samples[sample.variant_id];
    return r && isRated(r.content) && isRated(r.realism);
  });
}

function updateProgress() {
  const total = state.stemOrder.length;
  const done = state.stemOrder.filter((id) => isStemComplete(id)).length;
  progressEl.textContent = `Stem ${state.stemIndex + 1} of ${total} · ${done}/${total} complete`;
  exportBtn.disabled = done === 0;
  saveBtn.disabled = done === 0;
}

function renderAudio(container, cell) {
  container.innerHTML = "";
  if (!cell || !cell.available || !cell.url) {
    const badge = document.createElement("span");
    badge.className = "unavailable-badge";
    badge.textContent = "Audio not available";
    container.append(badge);
    return;
  }
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = cell.url;
  container.append(audio);
}

function isRated(value) {
  return Number.isInteger(value) && value >= 1 && value <= 5;
}

function renderStarRating(stemId, sample, field, label, help) {
  const row = document.createElement("div");
  row.className = "rating-row";

  const labelEl = document.createElement("span");
  labelEl.className = "rating-label";
  labelEl.textContent = label;
  labelEl.title = help;

  const group = document.createElement("div");
  group.className = "star-rating";
  group.setAttribute("role", "radiogroup");
  group.setAttribute("aria-label", `${label}. ${help}`);

  const valueEl = document.createElement("span");
  valueEl.className = "rating-value";

  const saved = stemRatings(stemId).samples[sample.variant_id] || {};
  let selected = isRated(saved[field]) ? saved[field] : 0;
  let hover = 0;
  const stars = [];

  function paint() {
    const active = hover || selected;
    valueEl.textContent = selected ? `${selected}/5` : "—";
    stars.forEach((btn, index) => {
      const starValue = index + 1;
      btn.classList.toggle("filled", active >= starValue);
      btn.classList.toggle("selected", selected === starValue);
      btn.setAttribute("aria-checked", selected === starValue ? "true" : "false");
    });
  }

  function commit(value) {
    selected = value;
    hover = 0;
    const ratings = stemRatings(stemId);
    if (!ratings.samples[sample.variant_id]) {
      ratings.samples[sample.variant_id] = {
        variant_id: sample.variant_id,
        blind_label: sample.blind_label,
      };
    }
    ratings.samples[sample.variant_id][field] = value;
    ratings.category = state.stemDetail.category;
    saveRatings();
    updateProgress();
    row.closest(".sample-card")?.classList.toggle(
      "incomplete",
      !isSampleComplete(stemId, sample.variant_id)
    );
    paint();
  }

  for (let starValue = 1; starValue <= 5; starValue += 1) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "star-btn";
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-label", `${starValue} out of 5`);
    btn.textContent = "★";
    btn.addEventListener("click", () => commit(starValue));
    btn.addEventListener("mouseenter", () => {
      hover = starValue;
      paint();
    });
    stars.push(btn);
    group.append(btn);
  }

  group.addEventListener("mouseleave", () => {
    hover = 0;
    paint();
  });

  paint();
  row.append(labelEl, group, valueEl);
  return row;
}

function isSampleComplete(stemId, variantId) {
  const r = stemRatings(stemId).samples[variantId];
  return r && isRated(r.content) && isRated(r.realism);
}

function renderStem() {
  const detail = state.stemDetail;
  if (!detail) {
    return;
  }

  stemTitleEl.textContent = detail.id;
  stemMetaEl.textContent = [
    detail.category,
    `track ${detail.track}`,
  ].filter(Boolean).join(" · ");
  stemNoteEl.textContent = detail.note || "";

  const rubric = state.meta.rubric;
  referenceLabelEl.textContent = rubric.reference_label;
  rubricHintEl.textContent = `${rubric.content_label}: ${rubric.content_help} · ${rubric.realism_label}: ${rubric.realism_help}`;

  renderAudio(referenceAudioEl, detail.reference);

  samplesEl.innerHTML = "";
  for (const sample of detail.samples) {
    const card = document.createElement("div");
    card.className = "sample-card";

    const header = document.createElement("div");
    header.className = "sample-header";
    const label = document.createElement("span");
    label.className = "sample-label";
    label.textContent = `Sample ${sample.blind_label}`;
    header.append(label);
    card.append(header);

    const audioSlot = document.createElement("div");
    audioSlot.className = "audio-slot";
    renderAudio(audioSlot, sample.audio);
    card.append(audioSlot);

    card.append(
      renderStarRating(detail.id, sample, "content", rubric.content_label, rubric.content_help)
    );
    card.append(
      renderStarRating(detail.id, sample, "realism", rubric.realism_label, rubric.realism_help)
    );

    card.classList.toggle("incomplete", !isSampleComplete(detail.id, sample.variant_id));
    samplesEl.append(card);
  }

  prevBtn.disabled = state.stemIndex <= 0;
  nextBtn.textContent =
    state.stemIndex >= state.stemOrder.length - 1 ? "Finish" : "Next stem →";
  updateProgress();
}

async function loadStem(index) {
  state.stemIndex = index;
  const stemId = currentStemId();
  state.stemDetail = await fetchJson(
    `/api/${SWEEP_TYPE}/stems/${encodeURIComponent(stemId)}?session_seed=${SESSION_SEED}`
  );
  if (!stemRatings(stemId).samples) {
    stemRatings(stemId).samples = {};
  }
  renderStem();
}

function buildExportPayload() {
  const ratings = [];
  for (const stemId of state.stemOrder) {
    if (!isStemComplete(stemId)) {
      continue;
    }
    const entry = state.ratings[stemId];
    if (!entry) {
      continue;
    }
    const samples = Object.values(entry.samples || {})
      .filter((sample) => isRated(sample.content) && isRated(sample.realism))
      .map((sample) => ({
        variant_id: sample.variant_id,
        blind_label: sample.blind_label,
        content: Number(sample.content),
        realism: Number(sample.realism),
      }));
    if (samples.length === 0) {
      continue;
    }
    ratings.push({
      stem_id: stemId,
      category: entry.category,
      samples,
    });
  }
  return {
    sweep_type: SWEEP_TYPE,
    session_seed: SESSION_SEED,
    manifest_id: state.meta.manifest_id,
    exported_at: new Date().toISOString(),
    ratings,
  };
}

function exportResponses() {
  const payload = buildExportPayload();
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `responses_${SWEEP_TYPE}_${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function saveToServer({ checkpoint = false, silent = false } = {}) {
  const payload = {
    ...buildExportPayload(),
    checkpoint,
  };
  const response = await fetch(`/api/${SWEEP_TYPE}/responses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || "Save failed");
  }
  const result = await response.json();
  if (silent) {
    showSaveStatus("Progress saved");
  } else {
    alert(`Saved to ${result.saved}`);
  }
  return result;
}

async function saveCompletedStem({ silent = true } = {}) {
  const stemId = currentStemId();
  if (!isStemComplete(stemId)) {
    return false;
  }
  saveRatings();
  await saveToServer({ checkpoint: true, silent });
  return true;
}

async function init() {
  titleEl.textContent = `${SWEEP_TYPE === "preset" ? "Preset" : "Patch"} sweep listening test`;

  state.meta = await fetchJson(
    `/api/${SWEEP_TYPE}/meta?session_seed=${SESSION_SEED}`
  );
  state.storageKey = state.meta.storage_key;
  state.stemOrder = state.meta.stem_order;
  const localRatings = loadSavedRatings();
  const serverPayload = await loadServerRatings();
  state.ratings = mergeRatings(
    localRatings,
    ratingsFromServerPayload(serverPayload)
  );
  saveRatings();

  if (state.stemOrder.length === 0) {
    loadingEl.textContent = "No probe stems in manifest.";
    return;
  }

  const resumeIndex = state.stemOrder.findIndex((stemId) => !isStemComplete(stemId));
  const startIndex = resumeIndex === -1 ? 0 : resumeIndex;

  loadingEl.classList.add("hidden");
  testPanelEl.classList.remove("hidden");
  await loadStem(startIndex);
}

prevBtn.addEventListener("click", async () => {
  if (state.stemIndex > 0) {
    await loadStem(state.stemIndex - 1);
  }
});

nextBtn.addEventListener("click", async () => {
  const stemId = currentStemId();
  if (!isStemComplete(stemId)) {
    alert("Rate all samples on this stem before continuing.");
    return;
  }
  try {
    await saveCompletedStem({ silent: true });
  } catch (err) {
    showSaveStatus(`Save failed: ${err.message}`, true);
    const proceed = window.confirm(
      `Could not save progress to the server:\n${err.message}\n\nContinue anyway? (Ratings are still in your browser.)`
    );
    if (!proceed) {
      return;
    }
  }
  if (state.stemIndex < state.stemOrder.length - 1) {
    await loadStem(state.stemIndex + 1);
    return;
  }
  let savedPath = null;
  try {
    const result = await saveToServer({ checkpoint: false, silent: true });
    savedPath = result.saved;
  } catch (err) {
    showSaveStatus(`Final save failed: ${err.message}`, true);
    completeMessageEl.textContent =
      "Could not save to the server. Download a backup copy below, then retry Save to server.";
    completePathEl.textContent = "";
    testPanelEl.classList.add("hidden");
    completeEl.classList.remove("hidden");
    return;
  }
  completeMessageEl.textContent = "Responses saved on the server.";
  completePathEl.textContent = savedPath;
  testPanelEl.classList.add("hidden");
  completeEl.classList.remove("hidden");
});

exportBtn.addEventListener("click", exportResponses);
saveBtn.addEventListener("click", () =>
  saveToServer({ checkpoint: true, silent: false }).catch((err) => alert(err.message))
);
completeDownloadBtn.addEventListener("click", exportResponses);

init().catch((err) => {
  loadingEl.textContent = `Failed to load: ${err}`;
});
