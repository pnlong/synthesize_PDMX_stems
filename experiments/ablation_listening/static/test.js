import {
  createScoreSlider,
  isScoreRated,
} from "/shared/slider.js";

const params = new URLSearchParams(window.location.search);
const REFERENCE_KEY = "__reference__";

const state = {
  meta: null,
  trialOrder: [],
  trialIndex: 0,
  trialDetail: null,
  ratings: {},
  storageKey: null,
  listenerId: "",
  sessionSeed: Number(params.get("seed") || document.getElementById("session-seed")?.value || "42"),
};

const setupEl = document.getElementById("setup");
const loadingEl = document.getElementById("loading");
const completeEl = document.getElementById("complete");
const testPanelEl = document.getElementById("test-panel");
const progressEl = document.getElementById("progress");
const saveStatusEl = document.getElementById("save-status");
const trialTitleEl = document.getElementById("trial-title");
const trialMetaEl = document.getElementById("trial-meta");
const trialNoteEl = document.getElementById("trial-note");
const rubricHintEl = document.getElementById("rubric-hint");
const referenceSectionEl = document.getElementById("reference-section");
const referenceAudioEl = document.getElementById("reference-audio");
const referenceSlidersEl = document.getElementById("reference-sliders");
const samplesEl = document.getElementById("samples");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const saveBtn = document.getElementById("save-btn");
const finishBtn = document.getElementById("finish-btn");
const startBtn = document.getElementById("start-btn");
const listenerInput = document.getElementById("listener-id");
const sessionSeedInput = document.getElementById("session-seed");
const completeMessageEl = document.getElementById("complete-message");
const completePathEl = document.getElementById("complete-path");

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Request failed: ${url}`);
  }
  return response.json();
}

function showSaveStatus(message, isError = false) {
  saveStatusEl.textContent = message;
  saveStatusEl.classList.toggle("error", isError);
  saveStatusEl.classList.remove("hidden");
  window.clearTimeout(showSaveStatus._timer);
  showSaveStatus._timer = window.setTimeout(() => {
    saveStatusEl.classList.add("hidden");
  }, isError ? 6000 : 2500);
}

function loadSavedRatings() {
  if (!state.storageKey) return {};
  try {
    return JSON.parse(localStorage.getItem(state.storageKey) || "{}");
  } catch {
    return {};
  }
}

function saveLocalRatings() {
  if (!state.storageKey) return;
  localStorage.setItem(state.storageKey, JSON.stringify(state.ratings));
}

function currentTrialId() {
  return state.trialOrder[state.trialIndex];
}

function trialRatings(trialId) {
  if (!state.ratings[trialId]) {
    state.ratings[trialId] = { samples: {} };
  }
  return state.ratings[trialId];
}

function isReferenceComplete(trialId) {
  const ref = trialRatings(trialId).samples[REFERENCE_KEY];
  return ref && isScoreRated(ref.realism);
}

function isVariantComplete(trialId, blindLabel) {
  const sample = trialRatings(trialId).samples[blindLabel];
  return sample && isScoreRated(sample.content) && isScoreRated(sample.realism);
}

function isTrialComplete(trialId) {
  const detail = state.trialDetail;
  if (!detail || detail.id !== trialId) return false;
  if (!isReferenceComplete(trialId)) return false;
  return detail.samples.every((s) => isVariantComplete(trialId, s.blind_label));
}

function completedTrialCount() {
  return state.trialOrder.filter((trialId) => {
    const entry = state.ratings[trialId];
    if (!entry?.samples) return false;
    const ref = entry.samples[REFERENCE_KEY];
    if (!ref || !isScoreRated(ref.realism)) return false;
    const variants = Object.entries(entry.samples).filter(([k]) => k !== REFERENCE_KEY);
    return variants.length >= 3 && variants.every(([, s]) =>
      isScoreRated(s.content) && isScoreRated(s.realism)
    );
  }).length;
}

function updateProgress() {
  progressEl.textContent = `Trial ${state.trialIndex + 1} of ${state.trialOrder.length} · ${completedTrialCount()} complete`;
  finishBtn.disabled = completedTrialCount() !== state.trialOrder.length;
  saveBtn.disabled = false;
}

function buildResponsesPayload(checkpoint = false) {
  const ratings = [];
  for (const trialId of state.trialOrder) {
    const entry = state.ratings[trialId];
    if (!entry?.samples) continue;

    const samples = [];
    const ref = entry.samples[REFERENCE_KEY];
    if (ref && isScoreRated(ref.realism)) {
      samples.push({
        is_reference: true,
        condition_id: ref.condition_id,
        realism: Number(ref.realism),
      });
    }

    for (const [key, sample] of Object.entries(entry.samples)) {
      if (key === REFERENCE_KEY) continue;
      if (!isScoreRated(sample.content) || !isScoreRated(sample.realism)) continue;
      samples.push({
        blind_label: sample.blind_label,
        condition_id: sample.condition_id,
        content: Number(sample.content),
        realism: Number(sample.realism),
      });
    }

    if (!samples.length) continue;
    ratings.push({
      trial_id: trialId,
      trial_type: entry.trial_type,
      category: entry.category,
      samples,
    });
  }
  return {
    test_id: state.meta?.test_id,
    listener_id: state.listenerId,
    session_seed: state.sessionSeed,
    checkpoint,
    ratings,
  };
}

async function saveToServer(checkpoint = true) {
  const payload = buildResponsesPayload(checkpoint);
  const response = await fetch("/api/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || "Save failed");
  }
  return response.json();
}

function renderAudio(container, url) {
  container.innerHTML = "";
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = url;
  container.append(audio);
}

async function loadTrial(index) {
  state.trialIndex = index;
  const trialId = currentTrialId();
  state.trialDetail = await fetchJson(`/api/trials/${encodeURIComponent(trialId)}?seed=${state.sessionSeed}`);
  renderTrial();
  updateProgress();
}

function renderTrial() {
  const detail = state.trialDetail;
  if (!detail) return;

  const typeLabel = detail.type === "mixture" ? "Mixture" : `Stem · ${detail.category || "unknown"}`;
  trialTitleEl.textContent = `${typeLabel} — ${detail.id}`;
  trialMetaEl.textContent = [
    detail.song_id,
    detail.track != null ? `track ${detail.track}` : null,
    detail.clip_seconds != null ? `${detail.clip_seconds}s clip` : null,
  ].filter(Boolean).join(" · ");
  trialNoteEl.textContent = detail.note || "";
  rubricHintEl.textContent =
    `${state.meta.rubrics.content.help} Blind samples only. Reference (A1): realism only.`;

  const saved = trialRatings(detail.id);
  saved.trial_type = detail.type;
  saved.category = detail.category;

  // Reference (A1) — visible, realism only
  referenceSectionEl.classList.remove("hidden");
  referenceAudioEl.innerHTML = "";
  if (detail.reference?.available) {
    renderAudio(referenceAudioEl, detail.reference.url);
  } else {
    referenceAudioEl.textContent = "Reference audio missing";
  }

  referenceSlidersEl.innerHTML = "";
  if (!saved.samples[REFERENCE_KEY]) {
    saved.samples[REFERENCE_KEY] = {
      condition_id: detail.reference.condition_id,
      is_reference: true,
    };
  }
  const refEntry = saved.samples[REFERENCE_KEY];

  createScoreSlider(referenceSlidersEl, {
    rubric: "realism",
    label: "Realism",
    help: detail.realism_rubric.help,
    value: refEntry.realism ?? null,
    onChange: (value) => {
      refEntry.realism = value;
      saveLocalRatings();
      updateProgress();
    },
  });

  // Blinded variants (A2/B1/B2) — content + realism
  samplesEl.innerHTML = "";
  for (const sample of detail.samples) {
    const card = document.createElement("article");
    card.className = "sample-card";
    if (!isVariantComplete(detail.id, sample.blind_label)) {
      card.classList.add("incomplete");
    }

    const header = document.createElement("div");
    header.className = "sample-header";
    const title = document.createElement("h3");
    title.textContent = `Sample ${sample.blind_label}`;
    header.append(title);
    card.append(header);

    const audioSlot = document.createElement("div");
    audioSlot.className = "audio-slot";
    if (sample.available) {
      renderAudio(audioSlot, sample.url);
    } else {
      audioSlot.textContent = "Audio missing";
    }
    card.append(audioSlot);

    const sliders = document.createElement("div");
    sliders.className = "sample-sliders";

    if (!saved.samples[sample.blind_label]) {
      saved.samples[sample.blind_label] = {
        blind_label: sample.blind_label,
        condition_id: sample.condition_id,
      };
    }
    const entry = saved.samples[sample.blind_label];

    const commit = () => {
      saveLocalRatings();
      updateProgress();
      card.classList.toggle("incomplete", !isVariantComplete(detail.id, sample.blind_label));
    };

    createScoreSlider(sliders, {
      rubric: "content",
      label: "Content",
      help: state.meta.rubrics.content.help,
      value: entry.content ?? null,
      onChange: (value) => {
        entry.content = value;
        commit();
      },
    });

    createScoreSlider(sliders, {
      rubric: "realism",
      label: "Realism",
      help: detail.realism_rubric.help,
      value: entry.realism ?? null,
      onChange: (value) => {
        entry.realism = value;
        commit();
      },
    });

    card.append(sliders);
    samplesEl.append(card);
  }

  prevBtn.disabled = state.trialIndex === 0;
  nextBtn.disabled = !isTrialComplete(detail.id);
  nextBtn.textContent =
    state.trialIndex === state.trialOrder.length - 1 ? "All trials done" : "Next trial →";
}

async function startTest() {
  state.listenerId = listenerInput.value.trim();
  state.sessionSeed = Number(sessionSeedInput.value || "42");
  if (!state.listenerId) {
    alert("Please enter a listener ID.");
    return;
  }

  setupEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");

  state.meta = await fetchJson(`/api/meta?seed=${state.sessionSeed}`);
  state.storageKey = state.meta.storage_key;
  state.trialOrder = state.meta.trial_order;

  let serverPayload = { ratings: [] };
  try {
    serverPayload = await fetchJson("/api/responses/session");
  } catch {
    serverPayload = { ratings: [] };
  }
  state.ratings = loadSavedRatings();
  for (const entry of serverPayload.ratings || []) {
    const mergedSamples = { ...(state.ratings[entry.trial_id]?.samples || {}) };
    for (const sample of entry.samples || []) {
      const key = sample.is_reference ? REFERENCE_KEY : sample.blind_label;
      mergedSamples[key] = sample;
    }
    state.ratings[entry.trial_id] = {
      trial_type: entry.trial_type,
      category: entry.category,
      samples: mergedSamples,
    };
  }

  loadingEl.classList.add("hidden");
  testPanelEl.classList.remove("hidden");

  let resumeIndex = 0;
  for (let i = 0; i < state.trialOrder.length; i += 1) {
    const trialId = state.trialOrder[i];
    const entry = state.ratings[trialId];
    const refOk = entry?.samples?.[REFERENCE_KEY] && isScoreRated(entry.samples[REFERENCE_KEY].realism);
    const variants = Object.entries(entry?.samples || {}).filter(([k]) => k !== REFERENCE_KEY);
    const variantsOk = variants.length >= 3 && variants.every(([, s]) =>
      isScoreRated(s.content) && isScoreRated(s.realism)
    );
    if (!refOk || !variantsOk) {
      resumeIndex = i;
      break;
    }
    resumeIndex = i + 1 < state.trialOrder.length ? i + 1 : i;
  }
  if (resumeIndex >= state.trialOrder.length) {
    resumeIndex = state.trialOrder.length - 1;
  }

  await loadTrial(resumeIndex);
}

startBtn.addEventListener("click", () => {
  startTest().catch((err) => {
    loadingEl.classList.add("hidden");
    setupEl.classList.remove("hidden");
    alert(err.message);
  });
});

prevBtn.addEventListener("click", async () => {
  if (state.trialIndex > 0) {
    try {
      await saveToServer(true);
      showSaveStatus("Progress saved");
    } catch (err) {
      showSaveStatus(err.message, true);
    }
    await loadTrial(state.trialIndex - 1);
  }
});

nextBtn.addEventListener("click", async () => {
  const detail = state.trialDetail;
  if (!detail || !isTrialComplete(detail.id)) {
    return;
  }
  try {
    await saveToServer(true);
    showSaveStatus("Progress saved");
  } catch (err) {
    showSaveStatus(err.message, true);
    return;
  }
  if (state.trialIndex < state.trialOrder.length - 1) {
    await loadTrial(state.trialIndex + 1);
  }
});

saveBtn.addEventListener("click", () => {
  saveToServer(true)
    .then((result) => showSaveStatus(`Saved ${result.saved}`))
    .catch((err) => showSaveStatus(err.message, true));
});

finishBtn.addEventListener("click", () => {
  saveToServer(false)
    .then((result) => {
      testPanelEl.classList.add("hidden");
      completeEl.classList.remove("hidden");
      completeMessageEl.textContent = "Thank you — responses saved on the server.";
      completePathEl.textContent = result.saved;
    })
    .catch((err) => showSaveStatus(err.message, true));
});

if (params.get("autostart") === "1") {
  startTest().catch(console.error);
}
