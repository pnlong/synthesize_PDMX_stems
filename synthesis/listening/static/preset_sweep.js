const NOTES_KEY = "preset_sweep_notes_v1";

const state = {
  stems: [],
  filteredStems: [],
  variants: [],
  selectedId: null,
  stemDetail: null,
};

const stemListEl = document.getElementById("stem-list");
const searchEl = document.getElementById("search");
const categoryFilterEl = document.getElementById("category-filter");
const emptyStateEl = document.getElementById("empty-state");
const stemDetailEl = document.getElementById("stem-detail");
const stemTitleEl = document.getElementById("stem-title");
const stemMetaEl = document.getElementById("stem-meta");
const stemNoteEl = document.getElementById("stem-note");
const referenceCellEl = document.getElementById("reference-cell");
const variantsGridEl = document.getElementById("variants-grid");
const notesEl = document.getElementById("notes");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const exportNotesBtn = document.getElementById("export-notes");

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${url}`);
  }
  return response.json();
}

function loadNotes() {
  try {
    return JSON.parse(localStorage.getItem(NOTES_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveNote(stemId, text) {
  const notes = loadNotes();
  if (text.trim()) {
    notes[stemId] = text;
  } else {
    delete notes[stemId];
  }
  localStorage.setItem(NOTES_KEY, JSON.stringify(notes));
}

function stemSearchText(stem) {
  return [stem.id, stem.category, stem.note, stem.song_id]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function populateCategoryFilter() {
  const categories = [...new Set(state.stems.map((stem) => stem.category).filter(Boolean))].sort();
  for (const category of categories) {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    categoryFilterEl.append(option);
  }
}

function applyFilter() {
  const query = searchEl.value.trim().toLowerCase();
  const category = categoryFilterEl.value;
  state.filteredStems = state.stems.filter((stem) => {
    if (category && stem.category !== category) {
      return false;
    }
    if (!query) {
      return true;
    }
    return stemSearchText(stem).includes(query);
  });
  renderStemList();
  updateNavButtons();
}

function renderStemList() {
  stemListEl.innerHTML = "";
  for (const stem of state.filteredStems) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = state.selectedId === stem.id ? "active" : "";
    button.dataset.stemId = stem.id;

    const title = document.createElement("span");
    title.className = "song-item-title";
    title.textContent = stem.id;

    const meta = document.createElement("span");
    meta.className = "song-item-artist";
    meta.textContent = [stem.category, `track ${stem.track}`].filter(Boolean).join(" · ");

    button.append(title, meta);
    button.addEventListener("click", () => selectStem(stem.id));
    li.append(button);
    stemListEl.append(li);
  }
}

function renderAudioCell(cell, label, metaText, prompt) {
  const wrapper = document.createElement("div");
  const isAvailable = cell && cell.available && cell.url;
  wrapper.className = `preset-variant-card${isAvailable ? "" : " unavailable"}`;

  const labelEl = document.createElement("div");
  labelEl.className = "variant-label";
  labelEl.textContent = label;
  wrapper.append(labelEl);

  if (metaText) {
    const meta = document.createElement("div");
    meta.className = "variant-meta";
    meta.textContent = metaText;
    wrapper.append(meta);
  }

  if (isAvailable) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = cell.url;
    wrapper.append(audio);
  } else {
    const badge = document.createElement("span");
    badge.className = "unavailable-badge";
    badge.textContent = "Not generated";
    wrapper.append(badge);
  }

  if (prompt) {
    const caption = document.createElement("div");
    caption.className = "caption-block";
    caption.textContent = prompt;
    wrapper.append(caption);
  }

  return wrapper;
}

function renderStemDetail(detail) {
  stemTitleEl.textContent = detail.id;
  stemMetaEl.textContent = [
    detail.category,
    `track ${detail.track}`,
    detail.song_id,
  ].filter(Boolean).join(" · ");
  stemNoteEl.textContent = detail.note || "";

  referenceCellEl.innerHTML = "";
  referenceCellEl.append(
    renderAudioCell(detail.reference, "A1 raw", detail.filename, null)
  );

  variantsGridEl.innerHTML = "";
  for (const variant of detail.variants) {
    const meta = `noise ${variant.init_noise_level} · ${variant.prompt_variant}`;
    variantsGridEl.append(
      renderAudioCell(
        variant.audio,
        variant.variant_id,
        meta,
        variant.prompt
      )
    );
  }

  const notes = loadNotes();
  notesEl.value = notes[detail.id] || "";
}

function updateNavButtons() {
  const index = state.filteredStems.findIndex((stem) => stem.id === state.selectedId);
  prevBtn.disabled = index <= 0;
  nextBtn.disabled = index < 0 || index >= state.filteredStems.length - 1;
}

async function selectStem(stemId) {
  state.selectedId = stemId;
  renderStemList();
  emptyStateEl.classList.add("hidden");
  stemDetailEl.classList.remove("hidden");

  try {
    state.stemDetail = await fetchJson(
      `/api/preset-sweep/stems/${encodeURIComponent(stemId)}`
    );
    renderStemDetail(state.stemDetail);
  } catch (err) {
    stemTitleEl.textContent = "Failed to load stem";
    stemMetaEl.textContent = String(err);
    stemNoteEl.textContent = "";
    referenceCellEl.innerHTML = "";
    variantsGridEl.innerHTML = "";
  }

  updateNavButtons();
}

function navigate(delta) {
  const index = state.filteredStems.findIndex((stem) => stem.id === state.selectedId);
  const nextIndex = index + delta;
  if (nextIndex < 0 || nextIndex >= state.filteredStems.length) {
    return;
  }
  selectStem(state.filteredStems[nextIndex].id);
}

async function init() {
  const meta = await fetchJson("/api/preset-sweep/meta");
  state.variants = meta.variants || [];
  state.stems = await fetchJson("/api/preset-sweep/stems");
  state.filteredStems = [...state.stems];
  populateCategoryFilter();
  renderStemList();

  if (state.stems.length > 0) {
    await selectStem(state.stems[0].id);
  }
}

searchEl.addEventListener("input", applyFilter);
categoryFilterEl.addEventListener("change", applyFilter);
prevBtn.addEventListener("click", () => navigate(-1));
nextBtn.addEventListener("click", () => navigate(1));
notesEl.addEventListener("input", () => {
  if (state.selectedId) {
    saveNote(state.selectedId, notesEl.value);
  }
});
exportNotesBtn.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(loadNotes(), null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "preset_sweep_notes.json";
  link.click();
  URL.revokeObjectURL(url);
});

init().catch((err) => {
  emptyStateEl.textContent = `Failed to load preset sweep catalog: ${err}`;
});
