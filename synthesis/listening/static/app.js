const state = {
  conditions: [],
  songs: [],
  filteredSongs: [],
  selectedId: null,
  songDetail: null,
};

const songListEl = document.getElementById("song-list");
const searchEl = document.getElementById("search");
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

function displayTitle(song) {
  return song.title || song.song_name || song.id;
}

function songSearchText(song) {
  return [song.title, song.song_name, song.artist_name, song.genres, song.id]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function applyFilter() {
  const query = searchEl.value.trim().toLowerCase();
  if (!query) {
    state.filteredSongs = [...state.songs];
  } else {
    state.filteredSongs = state.songs.filter((song) =>
      songSearchText(song).includes(query)
    );
  }
  renderSongList();
  updateNavButtons();
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
    title.textContent = displayTitle(song);

    const artist = document.createElement("span");
    artist.className = "song-item-artist";
    artist.textContent = song.artist_name || song.genres || `${song.n_tracks} tracks`;

    button.append(title, artist);
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

function renderSongDetail(detail) {
  songTitleEl.textContent = displayTitle(detail);

  const metaParts = [];
  if (detail.artist_name) metaParts.push(detail.artist_name);
  if (detail.genres) metaParts.push(detail.genres);
  metaParts.push(`${detail.n_tracks} track${detail.n_tracks === 1 ? "" : "s"}`);
  const duration = formatDuration(detail.duration_seconds);
  if (duration) metaParts.push(duration);
  if (detail.subtitle) metaParts.push(detail.subtitle);
  songMetaEl.textContent = metaParts.join(" · ");

  songPathEl.textContent = detail.id || "";

  mixtureGridEl.innerHTML = "";
  for (const condition of state.conditions) {
    mixtureGridEl.append(
      renderConditionCell(condition, detail.mixture[condition.id], null, {
        showCaption: false,
      })
    );
  }

  stemsContainerEl.innerHTML = "";
  for (const stem of detail.stems) {
    const row = document.createElement("div");
    row.className = "stem-row";

    const header = document.createElement("div");
    header.className = "stem-row-header";
    const programText =
      stem.program != null ? ` · MIDI program ${stem.program}` : "";
    header.innerHTML = `${stem.name} <span>(track ${stem.track}${programText})</span>`;
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
  [state.conditions, state.songs] = await Promise.all([
    fetchJson("/api/conditions"),
    fetchJson("/api/songs"),
  ]);
  state.filteredSongs = [...state.songs];
  renderSongList();

  if (state.songs.length > 0) {
    await selectSong(state.songs[0].id);
  }
}

searchEl.addEventListener("input", applyFilter);
prevBtn.addEventListener("click", () => navigate(-1));
nextBtn.addEventListener("click", () => navigate(1));

init().catch((err) => {
  emptyStateEl.textContent = `Failed to load catalog: ${err}`;
});
