const params = new URLSearchParams(window.location.search);
const SWEEP_TYPE = params.get("type") || "patch";
const ORDER = params.get("order") || "sequential";
const SESSION_SEED = Number(params.get("seed") || "42");

const TIER_KEYS = {
  ArrowLeft: "strong_reject",
  ArrowRight: "strong_accept",
  "1": "strong_reject",
  "4": "strong_accept",
};

const LOCAL_SAVE_DEBOUNCE_MS = 400;
const SERVER_SAVE_DEBOUNCE_MS = 1500;

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
audioEl.controls = true;
audioEl.preload = "auto";
audioEl.loop = true;
audioSlotEl.append(audioEl);

let localSaveTimer = null;
let serverSaveTimer = null;

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Request failed: ${url}`);
  }
  return response.json();
}

function loadLocalVotes() {
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

function votesFromServerPayload(payload) {
  const votes = {};
  for (const vote of payload?.votes || []) {
    const cardId = vote.card_id;
    if (cardId) {
      votes[cardId] = { ...vote, card_id: cardId };
    }
  }
  return votes;
}

function mergeVotes(local, server) {
  return { ...server, ...local };
}

function buildVote(card, tier) {
  return {
    card_id: card.card_id,
    category: card.category,
    stem_id: card.stem_id,
    clip_id: card.clip_id,
    variant_id: card.variant_id,
    tier,
  };
}

function currentCard() {
  return state.cards[state.cardIndex] || null;
}

function votedCount() {
  return state.cards.filter((card) => state.votes[card.card_id]).length;
}

function isComplete() {
  return state.cards.length > 0 && votedCount() >= state.cards.length;
}

function firstUnvotedIndex(startFrom = 0) {
  for (let index = startFrom; index < state.cards.length; index += 1) {
    if (!state.votes[state.cards[index].card_id]) {
      return index;
    }
  }
  return state.cards.length;
}

function updateProgress() {
  const total = state.cards.length;
  const done = votedCount();
  progressEl.textContent = `${done}/${total} reviewed · ${Math.max(total - done, 0)} left`;
  finishBtn.disabled = done === 0;
}

function updateSaveIndicator({ syncing = false, path = null, error = null } = {}) {
  saveStatusEl.classList.remove("hidden", "pending", "error");
  if (error) {
    saveStatusEl.textContent = `Save failed: ${error}`;
    saveStatusEl.classList.add("error");
    return;
  }
  if (syncing) {
    saveStatusEl.textContent = "Saving…";
    saveStatusEl.classList.add("pending");
    return;
  }
  const count = Object.keys(state.votes).length;
  saveStatusEl.textContent = path
    ? `${count} decisions saved · ${String(path).split(/[/\\]/).pop()}`
    : `${count} decisions saved`;
}

function buildExportPayload({ checkpoint = false } = {}) {
  return {
    mode: "verification",
    verification_mode: "soundfont_shortlist_swipe",
    sweep_type: SWEEP_TYPE,
    source_responses: state.meta?.source_responses || "winners.yaml",
    manifest_id: state.meta?.manifest_id,
    session_seed: SESSION_SEED,
    order: ORDER,
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

function stopPlayback() {
  audioEl.pause();
}

function playCurrentCard() {
  const card = currentCard();
  if (!card?.audio?.url) {
    audioSlotEl.classList.add("missing-audio");
    return;
  }
  audioSlotEl.classList.remove("missing-audio");
  if (audioEl.src !== card.audio.url) {
    audioEl.src = card.audio.url;
  }
  if (state.audioUnlocked) {
    audioEl.play().catch(() => {});
  }
}

function submitTier(tier) {
  const card = currentCard();
  if (!card) {
    return;
  }
  unlockAudio();
  state.history.push({ cardId: card.card_id, previous: state.votes[card.card_id] || null });
  state.votes[card.card_id] = buildVote(card, tier);
  saveLocalVotes();
  scheduleSaveToServer({ checkpoint: true });
  state.cardIndex = firstUnvotedIndex(state.cardIndex);
  renderCard();
}

function undoLastVote() {
  const last = state.history.pop();
  if (!last) {
    return;
  }
  if (last.previous) {
    state.votes[last.cardId] = last.previous;
  } else {
    delete state.votes[last.cardId];
  }
  const cardIndex = state.cards.findIndex((card) => card.card_id === last.cardId);
  if (cardIndex >= 0) {
    state.cardIndex = cardIndex;
  }
  saveLocalVotes();
  scheduleSaveToServer({ checkpoint: true });
  renderCard();
}

function unlockAudio() {
  if (state.audioUnlocked) {
    return;
  }
  state.audioUnlocked = true;
  autoplayHintEl.classList.add("hidden");
  playCurrentCard();
}

function renderCard() {
  const card = currentCard();
  if (!card) {
    if (isComplete()) {
      stopPlayback();
      panelEl.classList.add("hidden");
      completeEl.classList.remove("hidden");
      completeMessageEl.textContent = "All soundfonts reviewed. Press Finish to save.";
    }
    return;
  }
  cardLabelEl.textContent = card.label;
  cardMetaEl.textContent = `${card.category} · keep or reject this soundfont`;
  playCurrentCard();
  updateProgress();
}

async function onFinish() {
  if (!isComplete()) {
    window.alert("Review every soundfont before finishing.");
    return;
  }
  state.finalizing = true;
  saveLocalVotes({ immediate: true });
  window.clearTimeout(serverSaveTimer);
  try {
    const result = await saveToServer({ checkpoint: false });
    completePathEl.textContent = result.saved || "";
    panelEl.classList.add("hidden");
    completeEl.classList.remove("hidden");
    completeMessageEl.textContent = "Verification saved on the server.";
  } catch (err) {
    state.finalizing = false;
    window.alert(`Could not save verification:\n${err.message}`);
  }
}

function handleKeydown(event) {
  if (completeEl.classList.contains("hidden") === false) {
    return;
  }
  if (event.code === "Space") {
    event.preventDefault();
    unlockAudio();
    if (audioEl.paused) {
      audioEl.play().catch(() => {});
    } else {
      audioEl.pause();
    }
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

document.querySelectorAll(".swipe-pad-btn[data-tier]").forEach((button) => {
  button.addEventListener("click", () => {
    submitTier(button.dataset.tier);
  });
});

finishBtn.addEventListener("click", onFinish);
document.addEventListener("click", unlockAudio, { once: true });
document.addEventListener("keydown", handleKeydown);

async function init() {
  if (SWEEP_TYPE !== "patch") {
    loadingEl.textContent = "Verify swipe is only available for patch sweeps.";
    return;
  }

  const query = new URLSearchParams({
    seed: String(SESSION_SEED),
    order: ORDER,
  });
  state.meta = await fetchJson(`/api/${SWEEP_TYPE}/verify/swipe/meta?${query.toString()}`);
  state.storageKey = state.meta.storage_key;
  state.cards = state.meta.cards || [];

  let serverPayload = { votes: [] };
  try {
    serverPayload = await fetchJson(`/api/${SWEEP_TYPE}/verify/swipe/session`);
  } catch {
    serverPayload = { votes: [] };
  }
  state.votes = mergeVotes(loadLocalVotes(), votesFromServerPayload(serverPayload));
  saveLocalVotes({ immediate: true });

  if (state.cards.length === 0) {
    loadingEl.textContent = "No shortlisted soundfonts found. Record phase-1 winners first.";
    return;
  }

  state.cardIndex = firstUnvotedIndex(0);
  loadingEl.classList.add("hidden");
  panelEl.classList.remove("hidden");
  if (isComplete()) {
    completeEl.classList.remove("hidden");
    panelEl.classList.add("hidden");
    completeMessageEl.textContent = "All soundfonts reviewed. Press Finish to save.";
  } else {
    renderCard();
  }
  updateProgress();
}

init().catch((err) => {
  loadingEl.textContent = err.message;
});
