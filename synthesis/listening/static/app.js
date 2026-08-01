const state = {
  conditions: [],
  categories: [],
  selectedCategories: new Set(),
  songs: [],
  filteredSongs: [],
  selectedId: null,
  songDetail: null,
};

const songListEl = document.getElementById("song-list");
const searchEl = document.getElementById("search");
const categoryFiltersEl = document.getElementById("category-filters");
const categorySelectAllBtn = document.getElementById("category-select-all");
const emptyStateEl = document.getElementById("empty-state");
const songDetailEl = document.getElementById("song-detail");
const songTitleEl = document.getElementById("song-title");
const songMetaEl = document.getElementById("song-meta");
const songPathEl = document.getElementById("song-path");
const mixtureGridEl = document.getElementById("mixture-grid");
const stemsContainerEl = document.getElementById("stems-container");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${url}`);
  }
  return response.json();
}

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) {
    return null;
  }
  const total = Math.round(seconds);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

/** Blind label: PDMX basename hash (e.g. ``QmPfj…``), not title/artist. */
function displayId(song) {
  const id = song.id || "";
  const slash = id.lastIndexOf("/");
  return slash >= 0 ? id.slice(slash + 1) : id;
}

function trackCountLabel(song) {
  const n = song.n_tracks ?? 0;
  return `${n} track${n === 1 ? "" : "s"}`;
}

function songSearchText(song) {
  return [song.id, displayId(song)].filter(Boolean).join(" ").toLowerCase();
}

function allCategoriesSelected() {
  return (
    state.categories.length > 0 &&
    state.categories.every((cat) => state.selectedCategories.has(cat.id))
  );
}

function updateSelectAllButton() {
  categorySelectAllBtn.textContent = allCategoriesSelected()
    ? "Clear all"
    : "Select all";
}

function songMatchesCategories(song) {
  if (state.categories.length === 0 || allCategoriesSelected()) {
    return true;
  }
  if (state.selectedCategories.size === 0) {
    return false;
  }
  const cats = song.categories || [];
  return cats.some((cat) => state.selectedCategories.has(cat));
}

function applyFilter() {
  const query = searchEl.value.trim().toLowerCase();
  state.filteredSongs = state.songs.filter((song) => {
    if (!songMatchesCategories(song)) {
      return false;
    }
    if (!query) {
      return true;
    }
    return songSearchText(song).includes(query);
  });
  renderSongList();
  updateNavButtons();
}

function renderCategoryFilters() {
  categoryFiltersEl.innerHTML = "";
  for (const category of state.categories) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = category.id;
    input.checked = state.selectedCategories.has(category.id);
    input.addEventListener("change", () => {
      if (input.checked) {
        state.selectedCategories.add(category.id);
      } else {
        state.selectedCategories.delete(category.id);
      }
      updateSelectAllButton();
      applyFilter();
    });
    const text = document.createElement("span");
    text.textContent = category.label;
    label.append(input, text);
    categoryFiltersEl.append(label);
  }
  updateSelectAllButton();
}

function renderSongList() {
  songListEl.innerHTML = "";
  for (const song of state.filteredSongs) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = state.selectedId === song.id ? "active" : "";
    button.dataset.songId = song.id;

    const title = document.createElement("span");
    title.className = "song-item-title";
    title.textContent = displayId(song);

    const subtitle = document.createElement("span");
    subtitle.className = "song-item-subtitle";
    const cats = (song.categories || []).join(", ");
    subtitle.textContent = cats
      ? `${trackCountLabel(song)} · ${cats}`
      : trackCountLabel(song);

    button.append(title, subtitle);
    button.addEventListener("click", () => selectSong(song.id));
    li.append(button);
    songListEl.append(li);
  }
}

function renderConditionCell(condition, cell, caption, { showCaption = true } = {}) {
  const wrapper = document.createElement("div");
  const isAvailable = cell && cell.available && cell.url;
  wrapper.className = `condition-cell${isAvailable ? "" : " unavailable"}`;

  const label = document.createElement("div");
  label.className = "condition-label";
  label.textContent = `${condition.label} ${condition.name}`;

  wrapper.append(label);

  if (isAvailable) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = cell.url;
    wrapper.append(audio);

    const isRealify =
      condition.id === "basic_realify" || condition.id === "slakh_realify";
    if (showCaption && caption && isRealify) {
      const captionBlock = document.createElement("div");
      captionBlock.className = "caption-block";
      captionBlock.textContent = caption;
      wrapper.append(captionBlock);
    }
  } else {
    const badge = document.createElement("span");
    badge.className = "unavailable-badge";
    badge.textContent = condition.available ? "Audio missing" : "Not generated";
    wrapper.append(badge);
  }

  return wrapper;
}

function renderConditionGrid(cells, caption) {
  const grid = document.createElement("div");
  grid.className = "condition-grid";
  for (const condition of state.conditions) {
    const cell = cells[condition.id];
    grid.append(renderConditionCell(condition, cell, caption));
  }
  return grid;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderSongDetail(detail) {
  songTitleEl.textContent = displayId(detail);

  const metaParts = [trackCountLabel(detail)];
  const duration = formatDuration(detail.duration_seconds);
  if (duration) metaParts.push(duration);
  songMetaEl.textContent = metaParts.join(" · ");

  songPathEl.textContent = detail.id || "";

  mixtureGridEl.innerHTML = "";
  const showMixtures = detail.include_mixtures !== false;
  mixtureGridEl.classList.toggle("hidden", !showMixtures);
  if (showMixtures) {
    for (const condition of state.conditions) {
      mixtureGridEl.append(
        renderConditionCell(condition, detail.mixture[condition.id], null, {
          showCaption: false,
        })
      );
    }
  }

  stemsContainerEl.innerHTML = "";
  for (const stem of detail.stems) {
    const row = document.createElement("div");
    row.className = "stem-row";

    const header = document.createElement("div");
    header.className = "stem-row-header";
    const programText =
      stem.program != null ? ` · MIDI program ${stem.program}` : "";
    const categoryText = stem.category ? escapeHtml(stem.category) : "unknown";
    header.innerHTML = `${escapeHtml(stem.name)} <span>(${categoryText} · track ${stem.track}${programText})</span>`;
    row.append(header);

    row.append(renderConditionGrid(stem.conditions, stem.caption));
    stemsContainerEl.append(row);
  }
}

function updateNavButtons() {
  const index = state.filteredSongs.findIndex((song) => song.id === state.selectedId);
  prevBtn.disabled = index <= 0;
  nextBtn.disabled = index < 0 || index >= state.filteredSongs.length - 1;
}

async function selectSong(songId) {
  state.selectedId = songId;
  renderSongList();
  emptyStateEl.classList.add("hidden");
  songDetailEl.classList.remove("hidden");
  document.querySelector(".main")?.scrollTo(0, 0);

  try {
    state.songDetail = await fetchJson(`/api/songs/${encodeURIComponent(songId)}`);
    renderSongDetail(state.songDetail);
  } catch (err) {
    songTitleEl.textContent = "Failed to load song";
    songMetaEl.textContent = String(err);
    songPathEl.textContent = "";
    mixtureGridEl.innerHTML = "";
    stemsContainerEl.innerHTML = "";
  }

  updateNavButtons();
}

function navigate(delta) {
  const index = state.filteredSongs.findIndex((song) => song.id === state.selectedId);
  const nextIndex = index + delta;
  if (nextIndex < 0 || nextIndex >= state.filteredSongs.length) {
    return;
  }
  selectSong(state.filteredSongs[nextIndex].id);
}

async function init() {
  [state.conditions, state.categories, state.songs] = await Promise.all([
    fetchJson("/api/conditions"),
    fetchJson("/api/categories"),
    fetchJson("/api/songs"),
  ]);
  state.selectedCategories = new Set(state.categories.map((cat) => cat.id));
  renderCategoryFilters();
  applyFilter();

  if (state.filteredSongs.length > 0) {
    await selectSong(state.filteredSongs[0].id);
  }
}

searchEl.addEventListener("input", applyFilter);
categorySelectAllBtn.addEventListener("click", () => {
  if (allCategoriesSelected()) {
    state.selectedCategories.clear();
  } else {
    state.selectedCategories = new Set(state.categories.map((cat) => cat.id));
  }
  renderCategoryFilters();
  applyFilter();
});
prevBtn.addEventListener("click", () => navigate(-1));
nextBtn.addEventListener("click", () => navigate(1));

init().catch((err) => {
  emptyStateEl.textContent = `Failed to load catalog: ${err}`;
});
