const params = new URLSearchParams(window.location.search);
const SWEEP_TYPE = params.get("type") || "patch";
const ORDER = params.get("order") || "sequential";
const SESSION_SEED = Number(params.get("seed") || "42");
const PASS_NUMBER = Math.max(1, Number(params.get("pass") || "1") || 1);
const SOURCE_VERIFICATION = params.get("from") || null;

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
  deferred: [],
  history: [],
  storageKey: null,
  audioUnlocked: false,
  saveChain: Promise.resolve(),
  serverDirty: false,
  finalizing: false,
  savedName: null,
};

const loadingEl = document.getElementById("loading");
const completeEl = document.getElementById("complete");
const panelEl = document.getElementById("swipe-panel");
const progressEl = document.getElementById("progress");
const categoryStatsEl = document.getElementById("category-stats");
const saveStatusEl = document.getElementById("save-status");
const cardLabelEl = document.getElementById("card-label");
const cardMetaEl = document.getElementById("card-meta");
const audioSlotEl = document.getElementById("audio-slot");
const autoplayHintEl = document.getElementById("autoplay-hint");
const finishBtn = document.getElementById("finish-btn");
const completeMessageEl = document.getElementById("complete-message");
const completePathEl = document.getElementById("complete-path");
const nextPassBtn = document.getElementById("next-pass-btn");
const titleEl = document.getElementById("title");

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
    return { votes: {}, deferred: [] };
  }
  try {
    const raw = localStorage.getItem(state.storageKey);
    if (!raw) {
      return { votes: {}, deferred: [] };
    }
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && parsed.votes) {
      return {
        votes: parsed.votes || {},
        deferred: Array.isArray(parsed.deferred) ? parsed.deferred : [],
      };
    }
    // Legacy: plain votes map
    return { votes: parsed || {}, deferred: [] };
  } catch {
    return { votes: {}, deferred: [] };
  }
}

function saveLocalVotes({ immediate = false } = {}) {
  if (!state.storageKey) {
    return;
  }
  const payload = {
    votes: state.votes,
    deferred: state.deferred,
  };
  window.clearTimeout(localSaveTimer);
  if (immediate) {
    localStorage.setItem(state.storageKey, JSON.stringify(payload));
    return;
  }
  localSaveTimer = window.setTimeout(() => {
    localStorage.setItem(state.storageKey, JSON.stringify(payload));
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

function removeDeferred(cardId) {
  state.deferred = state.deferred.filter((id) => id !== cardId);
}

function isDeferred(cardId) {
  return state.deferred.includes(cardId);
}

function categoryOrder() {
  if (state.meta?.categories?.length) {
    return state.meta.categories;
  }
  const seen = [];
  for (const card of state.cards) {
    if (card.category && !seen.includes(card.category)) {
      seen.push(card.category);
    }
  }
  return seen;
}

function unvotedIndicesInCategory(category) {
  const indices = [];
  for (let index = 0; index < state.cards.length; index += 1) {
    const card = state.cards[index];
    if (card.category === category && !state.votes[card.card_id]) {
      indices.push(index);
    }
  }
  return indices;
}

function pickWithinCategory(category, afterIndex) {
  const unvoted = unvotedIndicesInCategory(category);
  if (!unvoted.length) {
    return null;
  }

  const fresh = unvoted.filter((index) => !isDeferred(state.cards[index].card_id));
  const laterFresh = fresh.find((index) => index > afterIndex);
  if (laterFresh !== undefined) {
    return laterFresh;
  }

  // Skipped cards come back only after every other card in this category.
  for (const cardId of state.deferred) {
    if (state.votes[cardId]) {
      continue;
    }
    const index = state.cards.findIndex((card) => card.card_id === cardId);
    if (index >= 0 && state.cards[index].category === category) {
      return index;
    }
  }

  if (fresh.length > 0) {
    return fresh[0];
  }

  const later = unvoted.find((index) => index > afterIndex);
  if (later !== undefined) {
    return later;
  }
  return unvoted[0];
}

function nextIndexAfter(fromIndex, category) {
  // Always finish earlier/unfinished categories before continuing.
  // This also pulls you back if a skip left unvoted cards behind.
  for (const cat of categoryOrder()) {
    const afterIndex = cat === category ? fromIndex : -1;
    const pick = pickWithinCategory(cat, afterIndex);
    if (pick === null) {
      continue;
    }
    // Never land back on a card that was just voted (fromIndex).
    if (pick === fromIndex) {
      continue;
    }
    return pick;
  }
  return state.cards.length;
}

function activateCurrentCard() {
  const card = currentCard();
  if (card) {
    removeDeferred(card.card_id);
  }
}

function deferredCountInCategory(category) {
  return state.deferred.filter((cardId) => {
    if (state.votes[cardId]) {
      return false;
    }
    const card = state.cards.find((entry) => entry.card_id === cardId);
    return card && card.category === category;
  }).length;
}

function jumpToFirstUnfinished() {
  for (const cat of categoryOrder()) {
    const pick = pickWithinCategory(cat, -1);
    if (pick !== null) {
      state.cardIndex = pick;
      activateCurrentCard();
      return;
    }
  }
  state.cardIndex = state.cards.length;
}

function reconcileDeferredGaps() {
  // Unvoted cards behind the furthest voted card were left behind (e.g. an
  // old skip that jumped categories). Queue them for end-of-category return.
  let maxVotedIndex = -1;
  for (let index = 0; index < state.cards.length; index += 1) {
    if (state.votes[state.cards[index].card_id]) {
      maxVotedIndex = index;
    }
  }
  for (let index = 0; index < maxVotedIndex; index += 1) {
    const cardId = state.cards[index].card_id;
    if (!state.votes[cardId] && !isDeferred(cardId)) {
      state.deferred.push(cardId);
    }
  }
  saveLocalVotes({ immediate: true });
}

function categoryStats(category) {
  const cards = state.cards.filter((card) => card.category === category);
  let left = 0;
  let accepted = 0;
  let rejected = 0;
  for (const card of cards) {
    const vote = state.votes[card.card_id];
    if (!vote) {
      left += 1;
    } else if (vote.tier === "strong_accept") {
      accepted += 1;
    } else {
      rejected += 1;
    }
  }
  return { total: cards.length, left, accepted, rejected };
}

function allCategoryStats() {
  const categories = state.meta?.categories || [
    ...new Set(state.cards.map((card) => card.category).filter(Boolean)),
  ];
  return categories.map((category) => ({
    category,
    ...categoryStats(category),
  }));
}

function formatCategoryStat(entry) {
  return (
    `${entry.category}: ${entry.left} left · ` +
    `${entry.accepted} kept · ${entry.rejected} rejected`
  );
}

function updateProgress() {
  const total = state.cards.length;
  const done = votedCount();
  const passLabel = `Pass ${state.meta?.pass || PASS_NUMBER}`;
  progressEl.textContent =
    `${passLabel} · ${done}/${total} reviewed · ${Math.max(total - done, 0)} left`;
  finishBtn.disabled = done === 0;

  const stats = allCategoryStats();
  if (!stats.length) {
    categoryStatsEl.textContent = "";
    categoryStatsEl.classList.add("hidden");
    return;
  }
  categoryStatsEl.classList.remove("hidden");
  const current = currentCard()?.category;
  categoryStatsEl.innerHTML = "";
  for (const entry of stats) {
    const chip = document.createElement("span");
    chip.className = "category-stat-chip";
    if (entry.category === current) {
      chip.classList.add("active");
    }
    if (entry.left === 0 && entry.total > 0) {
      chip.classList.add("done");
    }
    chip.textContent = formatCategoryStat(entry);
    categoryStatsEl.append(chip);
  }
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
    source_verification: state.meta?.source_verification || SOURCE_VERIFICATION,
    pass: state.meta?.pass || PASS_NUMBER,
    shortlists: state.meta?.shortlists || {},
    manifest_id: state.meta?.manifest_id,
    session_seed: SESSION_SEED,
    order: ORDER,
    exported_at: new Date().toISOString(),
    checkpoint,
    votes: Object.values(state.votes),
    deferred: state.deferred,
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
      if (result.name) {
        state.savedName = result.name;
      }
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
  removeDeferred(card.card_id);
  state.history.push({
    type: "vote",
    cardId: card.card_id,
    previous: state.votes[card.card_id] || null,
  });
  state.votes[card.card_id] = buildVote(card, tier);
  saveLocalVotes();
  scheduleSaveToServer({ checkpoint: true });
  if (isComplete()) {
    state.cardIndex = state.cards.length;
  } else {
    state.cardIndex = nextIndexAfter(state.cardIndex, card.category);
    activateCurrentCard();
  }
  renderCard();
}

function skipCurrentCard() {
  const card = currentCard();
  if (!card || state.votes[card.card_id] || isComplete()) {
    return;
  }
  unlockAudio();
  const fromIndex = state.cardIndex;
  removeDeferred(card.card_id);
  state.deferred.push(card.card_id);
  saveLocalVotes();

  const nextIndex = nextIndexAfter(fromIndex, card.category);
  if (nextIndex === fromIndex) {
    // Only this card left in the category — stay put.
    removeDeferred(card.card_id);
    saveLocalVotes();
    return;
  }

  state.history.push({
    type: "skip",
    fromIndex,
    cardId: card.card_id,
  });
  state.cardIndex = nextIndex;
  activateCurrentCard();
  renderCard();
}

function undoLastVote() {
  const last = state.history.pop();
  if (!last) {
    return;
  }
  if (last.type === "skip") {
    removeDeferred(last.cardId);
    state.cardIndex = last.fromIndex;
    saveLocalVotes();
    renderCard();
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

function keptSummary() {
  const stats = allCategoryStats();
  const kept = stats.reduce((sum, entry) => sum + entry.accepted, 0);
  const categoriesWithKept = stats.filter((entry) => entry.accepted > 0).length;
  return { kept, categoriesWithKept, stats };
}

function nextPassUrl(savedName) {
  const next = new URLSearchParams({
    type: SWEEP_TYPE,
    order: ORDER,
    seed: String(SESSION_SEED),
    pass: String((state.meta?.pass || PASS_NUMBER) + 1),
    from: savedName,
  });
  return `/verify-swipe?${next.toString()}`;
}

function showComplete({ message, savedPath = null, offerNextPass = false } = {}) {
  stopPlayback();
  panelEl.classList.add("hidden");
  completeEl.classList.remove("hidden");
  completeMessageEl.textContent = message;
  completePathEl.textContent = savedPath || "";
  if (offerNextPass && state.savedName) {
    const { kept, categoriesWithKept } = keptSummary();
    nextPassBtn.classList.toggle("hidden", kept === 0);
    nextPassBtn.textContent =
      kept > 0
        ? `Start pass ${(state.meta?.pass || PASS_NUMBER) + 1} (${kept} kept across ${categoriesWithKept} categories)`
        : "No soundfonts kept";
    nextPassBtn.onclick = () => {
      window.location.href = nextPassUrl(state.savedName);
    };
  } else {
    nextPassBtn.classList.add("hidden");
  }
}

function renderCard() {
  if (isComplete()) {
    state.cardIndex = state.cards.length;
    showComplete({
      message: "All soundfonts reviewed. Press Finish to save this pass.",
      offerNextPass: false,
    });
    updateProgress();
    finishBtn.disabled = false;
    return;
  }

  let card = currentCard();
  // If navigation landed on an already-voted card, jump to unfinished work.
  if (!card || state.votes[card.card_id]) {
    jumpToFirstUnfinished();
    if (isComplete() || state.cardIndex >= state.cards.length) {
      showComplete({
        message: "All soundfonts reviewed. Press Finish to save this pass.",
        offerNextPass: false,
      });
      updateProgress();
      finishBtn.disabled = false;
      return;
    }
    card = currentCard();
  }
  if (!card) {
    return;
  }

  completeEl.classList.add("hidden");
  panelEl.classList.remove("hidden");
  const stats = categoryStats(card.category);
  const skipped = deferredCountInCategory(card.category);
  cardLabelEl.textContent = card.label;
  cardMetaEl.textContent = [
    `${card.category} · keep or reject`,
    `${stats.left} left · ${stats.accepted} kept · ${stats.rejected} rejected`,
    skipped > 0 ? `${skipped} skipped → end of category` : null,
  ].filter(Boolean).join(" · ");
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
    state.savedName = result.name || String(result.saved || "").split(/[/\\]/).pop();
    showComplete({
      message: `Pass ${state.meta?.pass || PASS_NUMBER} saved. Start another pass to keep filtering, or lock with this file.`,
      savedPath: result.saved || "",
      offerNextPass: true,
    });
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
  if (event.key === "s" || event.key === "S" || event.key === "ArrowDown") {
    event.preventDefault();
    skipCurrentCard();
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

document.getElementById("skip-btn").addEventListener("click", () => {
  skipCurrentCard();
});

finishBtn.addEventListener("click", onFinish);
document.addEventListener("click", unlockAudio, { once: true });
document.addEventListener("keydown", handleKeydown);

function passQuery() {
  const query = new URLSearchParams({
    seed: String(SESSION_SEED),
    order: ORDER,
    pass: String(PASS_NUMBER),
  });
  if (SOURCE_VERIFICATION) {
    query.set("from", SOURCE_VERIFICATION);
  }
  return query;
}

async function init() {
  if (SWEEP_TYPE !== "patch") {
    loadingEl.textContent = "Verify swipe is only available for patch sweeps.";
    return;
  }

  const query = passQuery();
  state.meta = await fetchJson(`/api/${SWEEP_TYPE}/verify/swipe/meta?${query.toString()}`);
  state.storageKey = state.meta.storage_key;
  state.cards = state.meta.cards || [];

  const pass = state.meta.pass || PASS_NUMBER;
  titleEl.textContent =
    pass > 1 || state.meta.source_verification
      ? `Patch shortlist verify · pass ${pass}`
      : "Patch shortlist verify";

  let serverPayload = { votes: [], deferred: [] };
  try {
    serverPayload = await fetchJson(
      `/api/${SWEEP_TYPE}/verify/swipe/session?${query.toString()}`
    );
  } catch {
    serverPayload = { votes: [], deferred: [] };
  }
  const local = loadLocalVotes();
  state.votes = mergeVotes(local.votes, votesFromServerPayload(serverPayload));
  const serverDeferred = Array.isArray(serverPayload.deferred) ? serverPayload.deferred : [];
  state.deferred = [...new Set([...(local.deferred || []), ...serverDeferred])]
    .filter((cardId) => !state.votes[cardId]);
  saveLocalVotes({ immediate: true });

  if (state.cards.length === 0) {
    loadingEl.textContent = state.meta.source_verification
      ? "No soundfonts left from the previous pass."
      : "No shortlisted soundfonts found. Record phase-1 winners first.";
    return;
  }

  // Auto-mark leftover unvoted cards behind the frontier as deferred so they
  // return at end-of-category rather than being stranded after a bad skip.
  reconcileDeferredGaps();
  jumpToFirstUnfinished();
  loadingEl.classList.add("hidden");
  panelEl.classList.remove("hidden");
  if (isComplete()) {
    showComplete({
      message: "All soundfonts reviewed. Press Finish to save this pass.",
      offerNextPass: false,
    });
    finishBtn.disabled = false;
  } else {
    renderCard();
  }
  updateProgress();
}

init().catch((err) => {
  loadingEl.textContent = err.message;
});
