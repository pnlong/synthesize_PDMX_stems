const params = new URLSearchParams(window.location.search);
const SWEEP_TYPE = params.get("type") || "preset";
const RESPONSES_PARAM = params.get("responses") || "";

const state = {
  meta: null,
  responsesName: "",
  categoryOrder: [],
  categoryIndex: 0,
  soundfontIndex: 0,
  categoryDetail: null,
  decisions: {},
  storageKey: null,
};

const setupEl = document.getElementById("setup");
const setupHintEl = document.getElementById("setup-hint");
const responsesFieldEl = document.getElementById("responses-field");
const responsesLabelEl = document.getElementById("responses-label");
const loadingEl = document.getElementById("loading");
const completeEl = document.getElementById("complete");
const verifyPanelEl = document.getElementById("verify-panel");
const titleEl = document.getElementById("title");
const progressEl = document.getElementById("progress");
const saveStatusEl = document.getElementById("save-status");
const responsesSelectEl = document.getElementById("responses-select");
const startBtn = document.getElementById("start-btn");
const categoryTitleEl = document.getElementById("category-title");
const categoryMetaEl = document.getElementById("category-meta");
const soundfontNavSectionEl = document.getElementById("soundfont-nav-section");
const soundfontTabsEl = document.getElementById("soundfont-tabs");
const soundfontPositionEl = document.getElementById("soundfont-position");
const prevSoundfontBtn = document.getElementById("prev-soundfont-btn");
const nextSoundfontBtn = document.getElementById("next-soundfont-btn");
const currentSoundfontSectionEl = document.getElementById("current-soundfont-section");
const currentSoundfontHeaderEl = document.getElementById("current-soundfont-header");
const compareGridEl = document.getElementById("compare-grid");
const presetReferenceSectionEl = document.getElementById("preset-reference-section");
const presetBypassSectionEl = document.getElementById("preset-bypass-section");
const presetBypassControlsEl = document.getElementById("preset-bypass-controls");
const presetVariantsSectionEl = document.getElementById("preset-variants-section");
const stemsEl = document.getElementById("stems");
const variantsEl = document.getElementById("variants");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const saveBtn = document.getElementById("save-btn");
const completeMessageEl = document.getElementById("complete-message");
const completePathEl = document.getElementById("complete-path");
const completeDownloadBtn = document.getElementById("complete-download-btn");

function isPatchSoundfontMode() {
  return SWEEP_TYPE === "patch" && state.meta?.verification_mode === "soundfont_shortlist";
}

function isPresetRealifyMode() {
  return SWEEP_TYPE === "preset" && state.meta?.verification_mode === "preset_realify";
}

function presetVerifySource() {
  return "winners.yaml";
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Request failed: ${url}`);
  }
  return response.json();
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

function loadSavedDecisions() {
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

function saveDecisions() {
  if (!state.storageKey) {
    return;
  }
  localStorage.setItem(state.storageKey, JSON.stringify(state.decisions));
}

function decisionsFromServerPayload(payload) {
  const decisions = {};
  for (const entry of payload?.categories || []) {
    if (!entry?.category) {
      continue;
    }
    const stems = {};
    for (const stem of entry.stems || []) {
      if (!stem?.stem_id) {
        continue;
      }
      stems[stem.stem_id] = {
        bypass_realify: Boolean(stem.bypass_realify),
        track_name: stem.track_name || null,
        program: stem.program ?? 0,
        is_drum: Boolean(stem.is_drum),
      };
    }
    decisions[entry.category] = {
      approved: entry.approved || [],
      winner_variant_id: entry.winner_variant_id || null,
      bypass_realify: Boolean(entry.bypass_realify),
      stems,
      notes: entry.notes || "",
    };
  }
  return decisions;
}

function mergeDecisions(local, server) {
  const merged = { ...local };
  for (const [category, entry] of Object.entries(server || {})) {
    const localEntry = merged[category];
    const localApproved = localEntry?.approved?.length || 0;
    const serverApproved = entry?.approved?.length || 0;
    if (!localEntry || serverApproved > localApproved) {
      merged[category] = entry;
    }
  }
  return merged;
}

function currentCategory() {
  return state.categoryOrder[state.categoryIndex];
}

function categoryMeta(category) {
  return (state.meta?.categories || []).find((entry) => entry.category === category);
}

function shortlistForCategory(category) {
  const meta = categoryMeta(category);
  return meta?.shortlist || (meta?.variants || []).map((variant) => variant.variant_id);
}

function defaultDecision(category) {
  const meta = categoryMeta(category);
  if (!meta) {
    return {
      approved: [],
      winner_variant_id: null,
      bypass_realify: false,
      stems: {},
      notes: "",
    };
  }
  if (isPatchSoundfontMode()) {
    const shortlist = shortlistForCategory(category);
    return {
      approved: [...shortlist],
      winner_variant_id: null,
      bypass_realify: false,
      stems: {},
      notes: "",
    };
  }
  const passing = (meta.variants || []).filter((variant) => variant.passed_filter);
  const autoWinner = meta.auto_winner_variant_id;
  if (SWEEP_TYPE === "preset") {
    const variantId = autoWinner || passing[0]?.variant_id || null;
    return {
      approved: variantId ? [variantId] : [],
      winner_variant_id: variantId,
      bypass_realify: false,
      stems: {},
      notes: "",
    };
  }
  const approved = passing.map((variant) => variant.variant_id);
  let winner = autoWinner;
  if (winner && !approved.includes(winner)) {
    winner = approved[0] || null;
  }
  return {
    approved: [...approved],
    winner_variant_id: winner,
    bypass_realify: false,
    stems: {},
    notes: "",
  };
}

function ensureStemDecisions(category) {
  const decision = categoryDecision(category);
  if (!decision.stems) {
    decision.stems = {};
  }
  for (const stem of state.categoryDetail?.stems || []) {
    if (!decision.stems[stem.id]) {
      decision.stems[stem.id] = {
        bypass_realify: Boolean(decision.bypass_realify),
        track_name: stem.track_name || stem.note || null,
        program: stem.program ?? 0,
        is_drum: Boolean(stem.is_drum),
      };
    } else {
      decision.stems[stem.id].track_name =
        decision.stems[stem.id].track_name || stem.track_name || stem.note || null;
      decision.stems[stem.id].program = stem.program ?? decision.stems[stem.id].program ?? 0;
      decision.stems[stem.id].is_drum = Boolean(stem.is_drum);
    }
  }
}

function syncCategoryBypassFlag(category) {
  const decision = categoryDecision(category);
  const stems = state.categoryDetail?.stems || [];
  if (!stems.length || !decision.stems) {
    return;
  }
  decision.bypass_realify = stems.every(
    (stem) => Boolean(decision.stems[stem.id]?.bypass_realify)
  );
}

function setCategoryBypassAll(category, bypass) {
  const decision = categoryDecision(category);
  decision.bypass_realify = bypass;
  ensureStemDecisions(category);
  for (const stem of state.categoryDetail?.stems || []) {
    decision.stems[stem.id].bypass_realify = bypass;
  }
}

function stemBypassDecision(category, stemId) {
  ensureStemDecisions(category);
  return categoryDecision(category).stems[stemId];
}

function categoryDecision(category) {
  if (!state.decisions[category]) {
    state.decisions[category] = defaultDecision(category);
  }
  return state.decisions[category];
}

function isCategoryComplete(category) {
  const decision = categoryDecision(category);
  if (isPatchSoundfontMode()) {
    return decision.approved.length > 0;
  }
  if (decision.bypass_realify || isPresetRealifyMode()) {
    return true;
  }
  return (
    decision.approved.length > 0 &&
    decision.winner_variant_id &&
    decision.approved.includes(decision.winner_variant_id)
  );
}

function updateProgress() {
  const total = state.categoryOrder.length;
  const done = state.categoryOrder.filter((category) => isCategoryComplete(category)).length;
  progressEl.textContent = `Category ${state.categoryIndex + 1} of ${total} · ${done}/${total} complete`;
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

function formatStats(stats) {
  if (!stats) {
    return "";
  }
  const parts = [];
  if (stats.mean_rating != null) {
    parts.push(`rating ${stats.mean_rating}`);
  }
  parts.push(`content ${stats.mean_content} (min ${stats.min_content})`);
  parts.push(`realism ${stats.mean_realism}`);
  return parts.join(" · ");
}

function currentSoundfontId(category) {
  const shortlist = shortlistForCategory(category);
  if (!shortlist.length) {
    return null;
  }
  const index = Math.min(Math.max(state.soundfontIndex, 0), shortlist.length - 1);
  return shortlist[index];
}

function renderSoundfontTabs(category) {
  const shortlist = shortlistForCategory(category);
  const decision = categoryDecision(category);
  soundfontTabsEl.innerHTML = "";
  shortlist.forEach((soundfontId, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "soundfont-tab";
    if (index === state.soundfontIndex) {
      button.classList.add("active");
    }
    if (decision.approved.includes(soundfontId)) {
      button.classList.add("approved");
    } else {
      button.classList.add("rejected");
    }
    button.textContent = soundfontId;
    button.addEventListener("click", async () => {
      state.soundfontIndex = index;
      await loadCurrentSoundfont(category);
      renderCategory();
    });
    soundfontTabsEl.append(button);
  });
}

function renderPatchSoundfont(category) {
  const shortlist = shortlistForCategory(category);
  const soundfontId = currentSoundfontId(category);
  const meta = categoryMeta(category);
  const metaVariant = (meta?.variants || []).find((entry) => entry.variant_id === soundfontId);
  const detailVariant = (state.categoryDetail?.variants || []).find(
    (entry) => entry.variant_id === soundfontId
  );
  const decision = categoryDecision(category);
  const approved = decision.approved.includes(soundfontId);

  soundfontPositionEl.textContent = soundfontId
    ? `Soundfont ${state.soundfontIndex + 1} of ${shortlist.length}`
    : "";
  prevSoundfontBtn.disabled = state.soundfontIndex <= 0;
  nextSoundfontBtn.disabled = state.soundfontIndex >= shortlist.length - 1;

  currentSoundfontHeaderEl.innerHTML = "";
  const title = document.createElement("div");
  title.className = "soundfont-title";
  title.textContent = soundfontId || "No soundfont";
  const stats = document.createElement("div");
  stats.className = "meta";
  stats.textContent = formatStats(metaVariant?.stats);
  const approveLabel = document.createElement("label");
  approveLabel.className = "approve-label keep-label";
  const approveBox = document.createElement("input");
  approveBox.type = "checkbox";
  approveBox.checked = approved;
  approveBox.addEventListener("change", () => {
    const current = categoryDecision(category);
    if (approveBox.checked) {
      if (!current.approved.includes(soundfontId)) {
        current.approved.push(soundfontId);
      }
    } else {
      current.approved = current.approved.filter((id) => id !== soundfontId);
    }
    saveDecisions();
    renderSoundfontTabs(category);
    approveLabel.classList.toggle("rejected-state", !approveBox.checked);
    updateProgress();
  });
  approveLabel.classList.toggle("rejected-state", !approved);
  approveLabel.append(
    approveBox,
    document.createTextNode(approved ? " Keep in shortlist" : " Rejected — check to keep")
  );
  currentSoundfontHeaderEl.append(title, stats, approveLabel);

  compareGridEl.innerHTML = "";
  for (const stem of state.categoryDetail?.stems || []) {
    const card = document.createElement("div");
    card.className = "compare-card";
    appendStemHeading(card, stem);
    const note = document.createElement("div");
    note.className = "meta";
    note.textContent = [stem.note, `track ${stem.track}`].filter(Boolean).join(" · ");

    const refBlock = document.createElement("div");
    refBlock.className = "compare-block";
    const refLabel = document.createElement("div");
    refLabel.className = "compare-label";
    refLabel.textContent = "Reference (A1 basic)";
    const refAudio = document.createElement("div");
    refAudio.className = "audio-slot";
    renderAudio(refAudio, stem.reference);
    refBlock.append(refLabel, refAudio);

    const sfBlock = document.createElement("div");
    sfBlock.className = "compare-block";
    const sfLabel = document.createElement("div");
    sfLabel.className = "compare-label";
    sfLabel.textContent = soundfontId;
    const sfAudio = document.createElement("div");
    sfAudio.className = "audio-slot";
    renderAudio(sfAudio, detailVariant?.audio_by_stem?.[stem.id]);
    sfBlock.append(sfLabel, sfAudio);

    card.append(note, refBlock, sfBlock);
    compareGridEl.append(card);
  }
}

function renderPresetBypass(category) {
  presetBypassSectionEl.classList.remove("hidden");
  presetBypassControlsEl.innerHTML = "";
  ensureStemDecisions(category);
  syncCategoryBypassFlag(category);
  const decision = categoryDecision(category);

  const card = document.createElement("div");
  card.className = "bypass-card";

  const label = document.createElement("label");
  label.className = "bypass-label";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = Boolean(decision.bypass_realify);
  checkbox.addEventListener("change", () => {
    setCategoryBypassAll(category, checkbox.checked);
    renderCategory();
    updateProgress();
  });
  label.append(
    checkbox,
    document.createTextNode(" Bypass all stems in this category")
  );

  const hint = document.createElement("p");
  hint.className = "bypass-hint";
  hint.textContent =
    "Shortcut for bypassing every stem below. You can also bypass individual instruments; partial bypass becomes per-instrument routing rules at lock time.";

  card.append(label, hint);
  presetBypassControlsEl.append(card);
  presetVariantsSectionEl.classList.toggle("bypass-active", Boolean(decision.bypass_realify));
}

function formatPresetConfig(config) {
  if (!config) {
    return "";
  }
  const parts = [];
  if (config.init_noise_level != null) {
    parts.push(`noise ${config.init_noise_level}`);
  }
  if (config.prompt_variant) {
    parts.push(`prompt ${config.prompt_variant}`);
  }
  if (config.steps != null) {
    parts.push(`steps ${config.steps}`);
  }
  if (config.cfg_scale != null) {
    parts.push(`cfg ${config.cfg_scale}`);
  }
  return parts.join(" · ");
}

function stemInstrumentLabel(stem) {
  return stem.track_name || stem.note || `Track ${stem.track}`;
}

function appendStemHeading(container, stem, { showId = true } = {}) {
  const title = document.createElement("div");
  title.className = "stem-title";
  title.textContent = stemInstrumentLabel(stem);
  container.append(title);
  if (showId && stem.id) {
    const idEl = document.createElement("div");
    idEl.className = "stem-id meta";
    idEl.textContent = stem.id;
    idEl.title = stem.id;
    container.append(idEl);
  }
}

function lockedPresetLabel(meta) {
  const variantId = meta?.auto_winner_variant_id;
  const configText = formatPresetConfig(meta?.variants?.[0]?.config);
  return [variantId, configText].filter(Boolean).join(" · ");
}

function updatePresetSectionCopy() {
  const refTitle = presetReferenceSectionEl.querySelector("h3");
  const varTitle = presetVariantsSectionEl.querySelector("h3");
  const varHint = presetVariantsSectionEl.querySelector(".hint");
  if (!refTitle || !varTitle || !varHint) {
    return;
  }
  if (isPresetRealifyMode()) {
    refTitle.textContent = "Basic synthesis (no realify)";
    varTitle.textContent = "Locked preset";
    varHint.textContent =
      "Compare realified audio against the basic reference. Bypass realification if basic stems sound better.";
  } else {
    refTitle.textContent = "Reference stems";
    varTitle.textContent = "Filter-passing variants";
    varHint.textContent = "Check variants you approve. Pick one winner among approved.";
  }
}

function renderPresetRealifyCategory(category) {
  renderPresetBypass(category);
  ensureStemDecisions(category);
  syncCategoryBypassFlag(category);
  const meta = categoryMeta(category);
  const detail = state.categoryDetail;
  const lockedVariant = detail?.variants?.[0];
  const lockedMeta = meta?.variants?.[0];
  const decision = categoryDecision(category);

  stemsEl.innerHTML = "";
  for (const stem of detail?.stems || []) {
    const card = document.createElement("div");
    card.className = "stem-card";
    appendStemHeading(card, stem);
    const note = document.createElement("div");
    note.className = "meta";
    note.textContent = `track ${stem.track}`;

    const bypassLabel = document.createElement("label");
    bypassLabel.className = "stem-bypass-label";
    const bypassBox = document.createElement("input");
    bypassBox.type = "checkbox";
    bypassBox.checked = Boolean(stemBypassDecision(category, stem.id).bypass_realify);
    bypassBox.addEventListener("change", () => {
      stemBypassDecision(category, stem.id).bypass_realify = bypassBox.checked;
      syncCategoryBypassFlag(category);
      saveDecisions();
      renderCategory();
      updateProgress();
    });
    bypassLabel.append(
      bypassBox,
      document.createTextNode(" Bypass realification for this instrument")
    );

    const audioSlot = document.createElement("div");
    audioSlot.className = "audio-slot";
    renderAudio(audioSlot, stem.reference);
    card.append(note, bypassLabel, audioSlot);
    stemsEl.append(card);
  }

  presetVariantsSectionEl.classList.toggle(
    "bypass-active",
    Boolean(decision.bypass_realify)
  );

  variantsEl.innerHTML = "";
  if (!lockedVariant) {
    return;
  }

  const card = document.createElement("div");
  card.className = "variant-card auto-winner selected-winner";

  const header = document.createElement("div");
  header.className = "variant-header";

  const title = document.createElement("div");
  title.className = "variant-title";
  title.textContent = "Locked preset (realified)";
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = lockedVariant.variant_id;
  title.append(" ", badge);

  const stats = document.createElement("div");
  stats.className = "meta";
  stats.textContent = formatPresetConfig(lockedMeta?.config || lockedVariant.config);

  header.append(title, stats);
  card.append(header);

  const stemGrid = document.createElement("div");
  stemGrid.className = "variant-stems";
  for (const stem of detail?.stems || []) {
    const stemBlock = document.createElement("div");
    stemBlock.className = "variant-stem";
    appendStemHeading(stemBlock, stem);
    const audioSlot = document.createElement("div");
    audioSlot.className = "audio-slot";
    renderAudio(audioSlot, lockedVariant.audio_by_stem?.[stem.id]);
    stemBlock.append(audioSlot);
    stemGrid.append(stemBlock);
  }
  card.append(stemGrid);
  variantsEl.append(card);
}

function renderPresetCategory(category) {
  if (isPresetRealifyMode()) {
    renderPresetRealifyCategory(category);
    return;
  }
  renderPresetBypass(category);
  const decision = categoryDecision(category);
  const bypassed = Boolean(decision.bypass_realify);
  const meta = categoryMeta(category);
  const detail = state.categoryDetail;

  stemsEl.innerHTML = "";
  for (const stem of detail?.stems || []) {
    const card = document.createElement("div");
    card.className = "stem-card";
    appendStemHeading(card, stem);
    const note = document.createElement("div");
    note.className = "meta";
    note.textContent = `track ${stem.track}`;
    const audioSlot = document.createElement("div");
    audioSlot.className = "audio-slot";
    renderAudio(audioSlot, stem.reference);
    card.append(note, audioSlot);
    stemsEl.append(card);
  }

  variantsEl.innerHTML = "";
  for (const variant of detail?.variants || []) {
    const metaVariant = (meta?.variants || []).find(
      (entry) => entry.variant_id === variant.variant_id
    );
    const card = document.createElement("div");
    card.className = "variant-card";
    if (metaVariant?.is_auto_winner) {
      card.classList.add("auto-winner");
    }

    const header = document.createElement("div");
    header.className = "variant-header";

    const approveLabel = document.createElement("label");
    approveLabel.className = "approve-label";
    const approveBox = document.createElement("input");
    approveBox.type = "checkbox";
    approveBox.checked = decision.approved.includes(variant.variant_id);
    approveBox.disabled = bypassed;
    approveBox.addEventListener("change", () => {
      const current = categoryDecision(category);
      if (approveBox.checked) {
        if (!current.approved.includes(variant.variant_id)) {
          current.approved.push(variant.variant_id);
        }
      } else {
        current.approved = current.approved.filter((id) => id !== variant.variant_id);
        if (current.winner_variant_id === variant.variant_id) {
          current.winner_variant_id = current.approved[0] || null;
        }
      }
      saveDecisions();
      renderCategory();
      updateProgress();
    });
    approveLabel.append(approveBox, document.createTextNode(" Approve"));

    const title = document.createElement("div");
    title.className = "variant-title";
    title.textContent = variant.variant_id;
    if (metaVariant?.is_auto_winner) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "auto-winner";
      title.append(" ", badge);
    }

    const stats = document.createElement("div");
    stats.className = "meta";
    stats.textContent = formatStats(metaVariant?.stats);

    header.append(approveLabel, title, stats);
    card.append(header);

    const stemGrid = document.createElement("div");
    stemGrid.className = "variant-stems";
    for (const stem of detail?.stems || []) {
      const stemBlock = document.createElement("div");
      stemBlock.className = "variant-stem";
      appendStemHeading(stemBlock, stem);
      const audioSlot = document.createElement("div");
      audioSlot.className = "audio-slot";
      renderAudio(audioSlot, variant.audio_by_stem?.[stem.id]);
      stemBlock.append(audioSlot);
      stemGrid.append(stemBlock);
    }
    card.append(stemGrid);

    const winnerLabel = document.createElement("label");
    winnerLabel.className = "winner-label";
    const winnerRadio = document.createElement("input");
    winnerRadio.type = "radio";
    winnerRadio.name = `winner-${category}`;
    winnerRadio.checked = decision.winner_variant_id === variant.variant_id;
    winnerRadio.disabled = bypassed || !decision.approved.includes(variant.variant_id);
    winnerRadio.addEventListener("change", () => {
      categoryDecision(category).winner_variant_id = variant.variant_id;
      saveDecisions();
      updateProgress();
      card.classList.toggle("selected-winner", winnerRadio.checked);
    });
    winnerLabel.append(winnerRadio, document.createTextNode(" Winner for this category"));
    card.append(winnerLabel);
    card.classList.toggle("selected-winner", winnerRadio.checked);
    card.classList.toggle("not-approved", !decision.approved.includes(variant.variant_id));

    variantsEl.append(card);
  }
}

function renderCategory() {
  const category = currentCategory();
  const meta = categoryMeta(category);
  const decision = categoryDecision(category);
  const patchMode = isPatchSoundfontMode();

  categoryTitleEl.textContent = category;
  if (patchMode) {
    const shortlist = shortlistForCategory(category);
    const kept = decision.approved.length;
    categoryMetaEl.textContent = [
      meta?.verification_phase ? `phase ${meta.verification_phase}` : null,
      `${shortlist.length} in auto-shortlist`,
      `${kept} kept`,
      meta?.mean_rating_threshold != null
        ? `threshold ≥ ${meta.mean_rating_threshold}`
        : null,
    ].filter(Boolean).join(" · ");
  } else {
    const passingCount = (meta?.variants || []).filter((variant) => variant.passed_filter).length;
    const totalCount = meta?.variants?.length || 0;
    if (isPresetRealifyMode()) {
      categoryMetaEl.textContent = [
        meta?.verification_phase ? `phase ${meta.verification_phase}` : null,
        decision.bypass_realify ? "realification bypassed" : "locked preset from winners.yaml",
        lockedPresetLabel(meta),
      ].filter(Boolean).join(" · ");
    } else {
      categoryMetaEl.textContent = [
        meta?.verification_phase ? `phase ${meta.verification_phase}` : null,
        decision.bypass_realify ? "realification bypassed" : null,
        `${passingCount} passed filter`,
        meta?.auto_winner_variant_id ? `auto-winner ${meta.auto_winner_variant_id}` : null,
        `${totalCount} rated total`,
      ].filter(Boolean).join(" · ");
    }
  }

  soundfontNavSectionEl.classList.toggle("hidden", !patchMode);
  currentSoundfontSectionEl.classList.toggle("hidden", !patchMode);
  presetReferenceSectionEl.classList.toggle("hidden", patchMode);
  presetBypassSectionEl.classList.toggle("hidden", patchMode);
  presetVariantsSectionEl.classList.toggle("hidden", patchMode);
  updatePresetSectionCopy();

  if (patchMode) {
    renderSoundfontTabs(category);
    renderPatchSoundfont(category);
  } else {
    renderPresetCategory(category);
  }

  prevBtn.disabled = state.categoryIndex <= 0;
  nextBtn.textContent =
    state.categoryIndex >= state.categoryOrder.length - 1 ? "Finish" : "Next category →";
  updateProgress();
}

async function loadCurrentSoundfont(category) {
  const soundfontId = currentSoundfontId(category);
  if (!soundfontId) {
    state.categoryDetail = { stems: [], variants: [] };
    return;
  }
  const url = new URL(
    `/api/${SWEEP_TYPE}/verify/categories/${encodeURIComponent(category)}`,
    window.location.origin
  );
  url.searchParams.set("responses", state.responsesName);
  url.searchParams.set("variant_id", soundfontId);
  state.categoryDetail = await fetchJson(url.pathname + url.search);
}

async function loadCategory(index) {
  state.categoryIndex = index;
  state.soundfontIndex = 0;
  const category = currentCategory();
  if (isPatchSoundfontMode()) {
    await loadCurrentSoundfont(category);
  } else {
    state.categoryDetail = await fetchJson(
      `/api/${SWEEP_TYPE}/verify/categories/${encodeURIComponent(category)}?responses=${encodeURIComponent(state.responsesName)}`
    );
    if (isPresetRealifyMode()) {
      ensureStemDecisions(category);
      syncCategoryBypassFlag(category);
    }
  }
  renderCategory();
}

function buildExportPayload({ checkpoint = false } = {}) {
  const categories = [];
  for (const category of state.categoryOrder) {
    const decision = categoryDecision(category);
    const meta = categoryMeta(category);
    const shortlist = shortlistForCategory(category);
    ensureStemDecisions(category);
    syncCategoryBypassFlag(category);
    const stemEntries = Object.entries(decision.stems || {}).map(([stemId, stemDecision]) => ({
      stem_id: stemId,
      track_name: stemDecision.track_name || null,
      program: stemDecision.program ?? 0,
      is_drum: Boolean(stemDecision.is_drum),
      bypass_realify: Boolean(stemDecision.bypass_realify),
    }));
    categories.push({
      category,
      approved: [...decision.approved],
      rejected: shortlist.filter((id) => !decision.approved.includes(id)),
      winner_variant_id: decision.bypass_realify ? null : decision.winner_variant_id,
      bypass_realify: Boolean(decision.bypass_realify),
      stems: stemEntries,
      auto_winner_variant_id: meta?.auto_winner_variant_id || null,
      notes: decision.notes || "",
    });
  }
  return {
    mode: "verification",
    sweep_type: SWEEP_TYPE,
    source_responses: state.responsesName,
    manifest_id: state.meta?.manifest_id,
    verification_mode: state.meta?.verification_mode || "winner_pick",
    exported_at: new Date().toISOString(),
    checkpoint,
    categories,
  };
}

async function saveToServer({ checkpoint = false, silent = false } = {}) {
  const payload = buildExportPayload({ checkpoint });
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

function exportResponses() {
  const payload = buildExportPayload({ checkpoint: false });
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `verification_${SWEEP_TYPE}_${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function populateResponsesSelect() {
  const data = await fetchJson(`/api/${SWEEP_TYPE}/verify/responses`);
  responsesSelectEl.innerHTML = "";
  for (const file of data.files || []) {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = file.name;
    responsesSelectEl.append(option);
  }
  if (RESPONSES_PARAM) {
    responsesSelectEl.value = RESPONSES_PARAM;
  } else if (data.files?.length) {
    responsesSelectEl.selectedIndex = 0;
  }
}

function configureSetupCopy() {
  if (SWEEP_TYPE === "patch") {
    titleEl.textContent = "Patch soundfont shortlist review";
    setupHintEl.innerHTML =
      "Review each category's <strong>phase 1 soundfont shortlist</strong>. Listen dry (no FX), uncheck anything that slipped through, then <code>lock --verification</code>.";
    responsesLabelEl.textContent = "Phase 1 blind test responses";
    responsesFieldEl.classList.remove("hidden");
    startBtn.classList.remove("hidden");
  } else {
    titleEl.textContent = "Preset realify verification";
    setupHintEl.innerHTML =
      "Compare <strong>basic synthesis</strong> with your <strong>locked presets</strong> from <code>winners.yaml</code>. Bypass realification for categories where SA3 does not help.";
    responsesFieldEl.classList.add("hidden");
    startBtn.classList.add("hidden");
  }
}

async function startVerification() {
  const responsesName =
    SWEEP_TYPE === "preset" ? presetVerifySource() : responsesSelectEl.value;
  if (!responsesName) {
    alert("Select a blind test responses file first.");
    return;
  }

  setupEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");

  state.responsesName = responsesName;
  state.meta = await fetchJson(
    `/api/${SWEEP_TYPE}/verify/meta?responses=${encodeURIComponent(responsesName)}`
  );
  state.storageKey = `${state.meta.manifest_id}:verify:${responsesName}`;
  state.categoryOrder = (state.meta.categories || []).map((entry) => entry.category);

  const localDecisions = loadSavedDecisions();
  let serverPayload = null;
  try {
    serverPayload = await fetchJson(
      `/api/${SWEEP_TYPE}/verify/session?responses=${encodeURIComponent(responsesName)}`
    );
  } catch {
    serverPayload = null;
  }
  state.decisions = mergeDecisions(
    localDecisions,
    decisionsFromServerPayload(serverPayload)
  );
  saveDecisions();

  if (state.categoryOrder.length === 0) {
    loadingEl.textContent = "No categories found in winners.yaml.";
    return;
  }

  loadingEl.classList.add("hidden");
  verifyPanelEl.classList.remove("hidden");

  const resumeIndex = state.categoryOrder.findIndex((category) => !isCategoryComplete(category));
  const startIndex = resumeIndex === -1 ? 0 : resumeIndex;
  await loadCategory(startIndex);
}

configureSetupCopy();

async function bootVerifyPage() {
  if (SWEEP_TYPE === "preset") {
    try {
      await startVerification();
    } catch (err) {
      setupEl.classList.remove("hidden");
      loadingEl.classList.add("hidden");
      verifyPanelEl.classList.add("hidden");
      startBtn.classList.remove("hidden");
      startBtn.textContent = "Retry";
      setupHintEl.textContent = `Failed to start verification: ${err.message}`;
    }
    return;
  }

  try {
    await populateResponsesSelect();
  } catch (err) {
    setupHintEl.textContent = `Failed to list responses: ${err.message}`;
  }

  if (RESPONSES_PARAM) {
    startVerification().catch((err) => {
      setupEl.classList.remove("hidden");
      loadingEl.classList.add("hidden");
      alert(err.message);
    });
  }
}

bootVerifyPage();

startBtn.addEventListener("click", () => {
  startVerification().catch((err) => {
    loadingEl.textContent = `Failed to load: ${err.message}`;
    loadingEl.classList.remove("hidden");
  });
});

prevBtn.addEventListener("click", async () => {
  if (state.categoryIndex > 0) {
    await loadCategory(state.categoryIndex - 1);
  }
});

prevSoundfontBtn.addEventListener("click", async () => {
  if (state.soundfontIndex > 0) {
    state.soundfontIndex -= 1;
    await loadCurrentSoundfont(currentCategory());
    renderCategory();
  }
});

nextSoundfontBtn.addEventListener("click", async () => {
  const shortlist = shortlistForCategory(currentCategory());
  if (state.soundfontIndex < shortlist.length - 1) {
    state.soundfontIndex += 1;
    await loadCurrentSoundfont(currentCategory());
    renderCategory();
  }
});

nextBtn.addEventListener("click", async () => {
  const category = currentCategory();
  if (!isCategoryComplete(category)) {
    const message = isPatchSoundfontMode()
      ? "Keep at least one soundfont in this category before continuing."
      : "Approve at least one variant and pick a winner, or enable bypass realification.";
    alert(message);
    return;
  }
  try {
    await saveToServer({ checkpoint: true, silent: true });
  } catch (err) {
    showSaveStatus(`Save failed: ${err.message}`, true);
    const proceed = window.confirm(
      `Could not save progress to the server:\n${err.message}\n\nContinue anyway?`
    );
    if (!proceed) {
      return;
    }
  }
  if (state.categoryIndex < state.categoryOrder.length - 1) {
    await loadCategory(state.categoryIndex + 1);
    return;
  }
  let savedPath = null;
  try {
    const result = await saveToServer({ checkpoint: false, silent: true });
    savedPath = result.saved;
  } catch (err) {
    showSaveStatus(`Final save failed: ${err.message}`, true);
    completeMessageEl.textContent =
      "Could not save to the server. Download a backup copy below.";
    completePathEl.textContent = "";
    verifyPanelEl.classList.add("hidden");
    completeEl.classList.remove("hidden");
    return;
  }
  completeMessageEl.textContent = "Verification saved on the server.";
  completePathEl.textContent = savedPath;
  verifyPanelEl.classList.add("hidden");
  completeEl.classList.remove("hidden");
});

saveBtn.addEventListener("click", () =>
  saveToServer({ checkpoint: true, silent: false }).catch((err) => alert(err.message))
);
completeDownloadBtn.addEventListener("click", exportResponses);
