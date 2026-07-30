const params = new URLSearchParams(window.location.search);
const SWEEP_TYPE = params.get("type") || "patch";
const CATEGORY = params.get("category");
const ORDER = params.get("order") || "shuffle";
const SESSION_SEED = Number(params.get("seed") || "42");

const TIER_KEYS = {
  ArrowLeft: "strong_reject",
  ArrowRight: "strong_accept",
  ArrowDown: "weak_reject",
  ArrowUp: "weak_accept",
  "1": "strong_reject",
  "2": "weak_reject",
  "3": "weak_accept",
  "4": "strong_accept",
};

const TIER_FLASH_MS = 200;
const tierFlashTimers = {};

const state = {
  meta: null,
  cards: [],
  cardIndex: 0,
  votes: {},
  history: [],
  storageKey: null,
  audioUnlocked: false,
  saveChain: Promise.resolve(),
  serverDirty: false,
  finalizing: false,
};

const loadingEl = document.getElementById("loading");
const completeEl = document.getElementById("complete");
const panelEl = document.getElementById("swipe-panel");
const progressEl = document.getElementById("progress");
const saveStatusEl = document.getElementById("save-status");
const cardLabelEl = document.getElementById("card-label");
const cardMetaEl = document.getElementById("card-meta");
const audioSlotEl = document.getElementById("audio-slot");
const autoplayHintEl = document.getElementById("autoplay-hint");
const finishBtn = document.getElementById("finish-btn");
const completeMessageEl = document.getElementById("complete-message");
const completePathEl = document.getElementById("complete-path");
const audioEl = document.createElement("audio");
const preloaderEl = document.createElement("audio");

audioEl.controls = true;
audioEl.preload = "auto";
audioEl.loop = true;
preloaderEl.preload = "metadata";
audioSlotEl.append(audioEl);

const LOCAL_SAVE_DEBOUNCE_MS = 400;
const SERVER_SAVE_DEBOUNCE_MS = 1500;
let localSaveTimer = null;
let serverSaveTimer = null;
let preloadUrl = null;

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Request failed: ${url}`);
  }
  return response.json();
}

function voteCardId(vote) {
  if (!vote) {
    return null;
  }
  if (vote.card_id) {
    return vote.card_id;
  }
  if (vote.variant_id && vote.clip_id) {
    return `${vote.variant_id}|${vote.clip_id}`;
  }
  return null;
}

function updateSaveIndicator({ syncing = false, error = null, path = null } = {}) {
  if (!saveStatusEl) {
    return;
  }
  saveStatusEl.classList.remove("hidden", "error", "pending");
  const count = savedVoteCount();
  if (error) {
    saveStatusEl.textContent = `Save failed: ${error} (${count} votes in browser)`;
    saveStatusEl.classList.add("error");
    return;
  }
  if (syncing) {
    saveStatusEl.textContent = `Saving ${count} votes…`;
    saveStatusEl.classList.add("pending");
    return;
  }
  const time = new Date().toLocaleTimeString();
  let message = `${count} vote${count === 1 ? "" : "s"} saved · synced ${time}`;
  if (path) {
    const filename = String(path).split(/[/\\]/).pop();
    message += ` · ${filename}`;
  }
  saveStatusEl.textContent = message;
}

function loadLocalVotes() {
  if (!state.storageKey) {
    return {};
  }
  try {
    const raw = localStorage.getItem(state.storageKey);
    if (raw) {
      return JSON.parse(raw);
    }
  } catch {
    return {};
  }
  // Migrate only pre-session-id keys (manifest-id suffix), not other sweep phases.
  const prefix = `swipe_${SWEEP_TYPE}_`;
  let merged = {};
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (!key || !key.startsWith(prefix) || key === state.storageKey) {
      continue;
    }
    const suffix = key.slice(prefix.length);
    if (!/^\d+_\d+$/.test(suffix)) {
      continue;
    }
    try {
      const legacy = JSON.parse(localStorage.getItem(key));
      merged = { ...merged, ...legacy };
    } catch {
      // ignore invalid legacy payloads
    }
  }
  return merged;
}

function saveLocalVotes({ immediate = false } = {}) {
  if (!state.storageKey) {
    return;
  }
  window.clearTimeout(localSaveTimer);
  if (immediate) {
    localStorage.setItem(state.storageKey, JSON.stringify(state.votes));
    return;
  }
  localSaveTimer = window.setTimeout(() => {
    localStorage.setItem(state.storageKey, JSON.stringify(state.votes));
  }, LOCAL_SAVE_DEBOUNCE_MS);
}

function flushPendingSaves({ checkpoint = false } = {}) {
  saveLocalVotes({ immediate: true });
  window.clearTimeout(serverSaveTimer);
  state.serverDirty = false;
  if (!checkpoint) {
    state.finalizing = true;
  }
  // Drop any queued checkpoint saves so Finish cannot be overwritten.
  state.saveChain = Promise.resolve();
  return saveToServer({ checkpoint });
}

function scheduleSaveToServer({ checkpoint = true } = {}) {
  if (state.finalizing) {
    return;
  }
  state.serverDirty = true;
  window.clearTimeout(serverSaveTimer);
  serverSaveTimer = window.setTimeout(() => {
    saveToServer({ checkpoint }).catch(() => {});
  }, SERVER_SAVE_DEBOUNCE_MS);
}

function cleanupLegacyStorageKeys() {
  if (!state.storageKey) {
    return;
  }
  const prefix = `swipe_${SWEEP_TYPE}_`;
  for (let index = localStorage.length - 1; index >= 0; index -= 1) {
    const key = localStorage.key(index);
    if (key && key.startsWith(prefix) && key !== state.storageKey) {
      localStorage.removeItem(key);
    }
  }
}

function votesFromServerPayload(payload) {
  const votes = {};
  for (const vote of payload?.votes || []) {
    const cardId = voteCardId(vote);
    if (cardId) {
      votes[cardId] = { ...vote, card_id: cardId };
    }
  }
  return votes;
}

function mergeVotes(local, server) {
  return { ...server, ...local };
}

function cascadeGroupKey(card) {
  return `${card.category || ""}\0${card.variant_id || ""}`;
}

function siblingCards(card) {
  const groupKey = cascadeGroupKey(card);
  return state.cards.filter(
    (other) =>
      other.card_id !== card.card_id
      && cascadeGroupKey(other) === groupKey,
  );
}

function buildVote(card, tier, extra = {}) {
  return {
    card_id: card.card_id,
    category: card.category,
    stem_id: card.stem_id,
    clip_id: card.clip_id,
    variant_id: card.variant_id,
    tier,
    ...extra,
  };
}

function cascadeTier(triggerCard, tier, { overwrite = false } = {}) {
  const cascaded = [];
  for (const sibling of siblingCards(triggerCard)) {
    const previous = state.votes[sibling.card_id] || null;
    if (previous?.tier === tier) {
      continue;
    }
    if (previous && !overwrite) {
      continue;
    }
    cascaded.push({ card_id: sibling.card_id, previous });
    state.votes[sibling.card_id] = buildVote(sibling, tier, {
      cascaded_from: triggerCard.card_id,
    });
  }
  return cascaded;
}

function cascadeStrongReject(triggerCard) {
  return cascadeTier(triggerCard, "strong_reject", { overwrite: true });
}

function cascadeStrongAccept(triggerCard) {
  return cascadeTier(triggerCard, "strong_accept", { overwrite: false });
}

function reconcileTierCascades(tier) {
  let added = 0;
  for (const card of state.cards) {
    if (state.votes[card.card_id]?.tier !== tier) {
      continue;
    }
    for (const sibling of siblingCards(card)) {
      if (state.votes[sibling.card_id]) {
        continue;
      }
      state.votes[sibling.card_id] = buildVote(sibling, tier, {
        cascaded_from: card.card_id,
      });
      added += 1;
    }
  }
  return added;
}

function reconcileCascades() {
  return reconcileTierCascades("strong_reject") + reconcileTierCascades("strong_accept");
}

function currentCard() {
  return state.cards[state.cardIndex] || null;
}

function savedVoteCount() {
  return Object.keys(state.votes).length;
}

function votedCount() {
  return state.cards.filter((card) => state.votes[card.card_id]).length;
}

function isComplete() {
  return state.cards.length > 0 && votedCount() >= state.cards.length;
}

function firstUnvotedIndex(startFrom = 0) {
  for (let index = startFrom; index < state.cards.length; index += 1) {
    const card = state.cards[index];
    if (!state.votes[card.card_id]) {
      return index;
    }
  }
  return state.cards.length;
}

function nextUnvotedIndex() {
  return firstUnvotedIndex(state.cardIndex + 1);
}

function remainingCount() {
  return Math.max(state.cards.length - votedCount(), 0);
}

function effectiveOrder() {
  return state.meta?.order || ORDER;
}

function categoryStats(category) {
  if (!category) {
    return { total: 0, voted: 0, left: 0 };
  }
  let total = 0;
  let voted = 0;
  for (const card of state.cards) {
    if (card.category !== category) {
      continue;
    }
    total += 1;
    if (state.votes[card.card_id]) {
      voted += 1;
    }
  }
  return { total, voted, left: total - voted };
}

function categoryProgressText(category) {
  if (effectiveOrder() !== "sequential" || !category) {
    return "";
  }
  const { left, voted, total } = categoryStats(category);
  return ` · ${left} left in ${category} (${voted}/${total})`;
}

function updateProgress() {
  const total = state.cards.length;
  const done = votedCount();
  const left = remainingCount();
  const card = currentCard();
  progressEl.textContent = card
    ? `${done}/${total} voted · ${left} left${categoryProgressText(card.category)}`
    : `${done}/${total} voted · ${left} left`;
  finishBtn.disabled = done === 0;
}

function preloadNextAudio() {
  const nextIndex = nextUnvotedIndex();
  if (nextIndex >= state.cards.length) {
    preloadUrl = null;
    preloaderEl.removeAttribute("src");
    preloaderEl.load();
    return;
  }
  const next = state.cards[nextIndex];
  if (!next?.audio?.url) {
    return;
  }
  const url = new URL(next.audio.url, window.location.origin).href;
  if (preloadUrl === url) {
    return;
  }
  preloadUrl = url;
  preloaderEl.src = next.audio.url;
}

function stopPlayback() {
  audioEl.pause();
  audioEl.removeAttribute("src");
  audioEl.load();
  preloadUrl = null;
  preloaderEl.removeAttribute("src");
  preloaderEl.load();
}

function playCurrentCard() {
  const card = currentCard();
  if (!card?.audio?.url) {
    stopPlayback();
    return;
  }
  const url = new URL(card.audio.url, window.location.origin).href;
  audioEl.pause();
  if (audioEl.src !== url) {
    audioEl.src = card.audio.url;
  }
  if (!state.audioUnlocked) {
    autoplayHintEl.classList.remove("hidden");
    return;
  }
  autoplayHintEl.classList.add("hidden");
  audioEl.play().catch(() => {
    autoplayHintEl.classList.remove("hidden");
  });
  preloadNextAudio();
}

function renderCard() {
  const card = currentCard();
  if (!card) {
    if (isComplete()) {
      stopPlayback();
      panelEl.classList.add("hidden");
      completeEl.classList.remove("hidden");
      completeMessageEl.textContent = "All cards voted. Press Finish to save.";
    }
    return;
  }
  cardLabelEl.textContent = card.label;
  if (effectiveOrder() === "sequential" && card.category) {
    const { left, total } = categoryStats(card.category);
    cardMetaEl.textContent = `${card.category} · ${left} sample${left === 1 ? "" : "s"} left in category (${total} total)`;
  } else {
    cardMetaEl.textContent = card.category || "";
  }
  playCurrentCard();
  updateProgress();
}

function buildExportPayload({ checkpoint = false } = {}) {
  return {
    mode: "swipe",
    sweep_type: SWEEP_TYPE,
    manifest_id: state.meta.manifest_id,
    session_seed: SESSION_SEED,
    category: state.meta.category,
    order: state.meta.order,
    exported_at: new Date().toISOString(),
    checkpoint,
    votes: Object.values(state.votes),
  };
}

function saveToServer({ checkpoint = true } = {}) {
  if (state.finalizing && checkpoint) {
    return state.saveChain;
  }
  updateSaveIndicator({ syncing: true });
  state.saveChain = state.saveChain
    .then(async () => {
      const response = await fetch(`/api/${SWEEP_TYPE}/responses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildExportPayload({ checkpoint })),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || "Save failed");
      }
      const result = await response.json();
      state.serverDirty = false;
      updateSaveIndicator({ path: result.saved });
      return result;
    })
    .catch((err) => {
      updateSaveIndicator({ error: err.message });
      throw err;
    });
  return state.saveChain;
}

function unlockAudio() {
  if (state.audioUnlocked) {
    return;
  }
  state.audioUnlocked = true;
  autoplayHintEl.classList.add("hidden");
  playCurrentCard();
}

function toggleAudioPlayback() {
  unlockAudio();
  if (audioEl.paused) {
    audioEl.play().catch(() => {
      autoplayHintEl.classList.remove("hidden");
    });
  } else {
    audioEl.pause();
  }
}

function flashTierButton(tier) {
  const button = document.querySelector(`.swipe-pad-btn[data-tier="${tier}"]`);
  if (!button) {
    return;
  }
  button.classList.add("is-active");
  window.clearTimeout(tierFlashTimers[tier]);
  tierFlashTimers[tier] = window.setTimeout(() => {
    button.classList.remove("is-active");
  }, TIER_FLASH_MS);
}

function submitTier(tier) {
  flashTierButton(tier);
  const card = currentCard();
  if (!card) {
    return;
  }
  unlockAudio();
  const previous = state.votes[card.card_id] || null;
  const vote = buildVote(card, tier);
  let cascaded = [];
  if (tier === "strong_reject") {
    cascaded = cascadeStrongReject(card);
  } else if (tier === "strong_accept") {
    cascaded = cascadeStrongAccept(card);
  }
  state.history.push({
    cardIndex: state.cardIndex,
    previous,
    cascaded,
  });
  state.votes[card.card_id] = vote;
  saveLocalVotes();
  scheduleSaveToServer({ checkpoint: true });
  if (cascaded.length > 0) {
    const soundfont = soundfontKey(card);
    const verb = tier === "strong_reject" ? "auto-rejected" : "auto-accepted";
    updateSaveIndicator({
      syncing: false,
      path: "swipe_in_progress.json",
    });
    saveStatusEl.textContent = `${cascaded.length} more clip${cascaded.length === 1 ? "" : "s"} from ${soundfont} ${verb} in ${card.category}`;
  }
  const nextIndex = nextUnvotedIndex();
  if (nextIndex < state.cards.length) {
    state.cardIndex = nextIndex;
    renderCard();
    return;
  }
  state.cardIndex = state.cards.length;
  stopPlayback();
  updateProgress();
  panelEl.classList.add("hidden");
  completeEl.classList.remove("hidden");
  completeMessageEl.textContent = "All cards voted. Press Finish to save.";
}

function undoLastVote() {
  const last = state.history.pop();
  if (!last) {
    return;
  }
  state.cardIndex = last.cardIndex;
  if (last.previous) {
    state.votes[state.cards[state.cardIndex].card_id] = last.previous;
  } else {
    delete state.votes[state.cards[state.cardIndex].card_id];
  }
  for (const entry of last.cascaded || []) {
    if (entry.previous) {
      state.votes[entry.card_id] = entry.previous;
    } else {
      delete state.votes[entry.card_id];
    }
  }
  saveLocalVotes();
  scheduleSaveToServer({ checkpoint: true });
  panelEl.classList.remove("hidden");
  completeEl.classList.add("hidden");
  renderCard();
}

function handleKeydown(event) {
  if (event.target instanceof HTMLButtonElement) {
    return;
  }
  if (event.code === "Space") {
    event.preventDefault();
    toggleAudioPlayback();
    return;
  }
  if (event.code === "Backspace") {
    event.preventDefault();
    undoLastVote();
    return;
  }
  const tier = TIER_KEYS[event.key];
  if (tier) {
    event.preventDefault();
    submitTier(tier);
  }
}

async function init() {
  const scopeLabel = CATEGORY || "all categories";
  document.getElementById("title").textContent = `Soundfont swipe · ${scopeLabel}`;
  const query = new URLSearchParams({
    seed: String(SESSION_SEED),
    order: ORDER,
  });
  if (CATEGORY) {
    query.set("category", CATEGORY);
  }

  state.meta = await fetchJson(`/api/${SWEEP_TYPE}/swipe/meta?${query.toString()}`);
  state.storageKey = state.meta.storage_key;
  state.cards = state.meta.cards || [];

  let serverPayload = { votes: [] };
  try {
    serverPayload = await fetchJson(`/api/${SWEEP_TYPE}/swipe/responses/session`);
  } catch (err) {
    console.warn("Could not load server session:", err);
    serverPayload = { votes: [] };
  }
  const localVotes = loadLocalVotes();
  const serverVotes = votesFromServerPayload(serverPayload);
  state.votes = mergeVotes(localVotes, serverVotes);
  const backfilled = reconcileCascades();
  if (backfilled > 0) {
    saveLocalVotes({ immediate: true });
    scheduleSaveToServer({ checkpoint: true });
  } else {
    saveLocalVotes({ immediate: true });
  }
  cleanupLegacyStorageKeys();

  window.addEventListener("pagehide", () => {
    saveLocalVotes({ immediate: true });
    if (!state.serverDirty || state.finalizing) {
      return;
    }
    const payload = buildExportPayload({ checkpoint: true });
    navigator.sendBeacon(
      `/api/${SWEEP_TYPE}/responses`,
      new Blob([JSON.stringify(payload)], { type: "application/json" }),
    );
  });

  if (state.cards.length === 0) {
    loadingEl.textContent = CATEGORY
      ? `No swipe cards found for category "${CATEGORY}".`
      : "No swipe cards found. Run make_clips on the sweep output first.";
    return;
  }

  if (votedCount() > 0) {
    updateSaveIndicator();
  } else {
    saveStatusEl.textContent = "Votes save automatically after each swipe";
    saveStatusEl.classList.remove("hidden");
    saveStatusEl.classList.add("pending");
  }

  if (isComplete()) {
    loadingEl.classList.add("hidden");
    completeEl.classList.remove("hidden");
    completeMessageEl.textContent = "All cards voted. Press Finish to save.";
    finishBtn.disabled = false;
    updateProgress();
    document.addEventListener("keydown", handleKeydown);
    finishBtn.addEventListener("click", onFinish);
    return;
  }

  state.cardIndex = firstUnvotedIndex();
  const resumed = votedCount();
  if (resumed > 0) {
    progressEl.textContent = `Resumed · ${resumed}/${state.cards.length} voted · ${remainingCount()} left`;
  }
  loadingEl.classList.add("hidden");
  panelEl.classList.remove("hidden");
  renderCard();

  document.addEventListener("keydown", handleKeydown);
  panelEl.addEventListener("click", unlockAudio);
  for (const button of document.querySelectorAll(".swipe-pad-btn")) {
    button.addEventListener("click", () => {
      submitTier(button.dataset.tier);
    });
  }
  finishBtn.addEventListener("click", onFinish);
}

async function onFinish() {
  try {
    const result = await flushPendingSaves({ checkpoint: false });
    completeMessageEl.textContent = "Responses saved on the server.";
    completePathEl.textContent = result.saved;
    panelEl.classList.add("hidden");
    completeEl.classList.remove("hidden");
    updateSaveIndicator({ path: result.saved });
  } catch (err) {
    updateSaveIndicator({ error: err.message });
  }
}

init().catch((err) => {
  loadingEl.textContent = `Failed to load: ${err.message}`;
});
