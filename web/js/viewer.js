import { detections, frames, objects, regions } from "./data.js";

const status = document.querySelector("#status");
const playButton = document.querySelector("#play");
const slider = document.querySelector("#slider");
const diffBox = document.querySelector("#diff");
const discardedBox = document.querySelector("#discarded");
const caption = document.querySelector("#caption");
const image = document.querySelector("#frame");
const marks = document.querySelector("#marks");
const panel = document.querySelector("#panel");
const panelName = document.querySelector("#panel-name");
const panelBody = document.querySelector("#panel-body");

const SHAPE = {
  Known: "circle",
  "Unidentified candidate": "diamond",
  Discarded: "square",
};

let sequence = [];
let byObject = new Map();
let hits = [];
let index = 0;
let timer = 0;
const cache = new Map();

function stamp(value) {
  if (typeof value === "string") return value.replace(/\.\d+Z$/, "Z");
  const number = typeof value === "number" ? value : Number(value);
  const ms = Number.isFinite(number) && Math.abs(number) > 1e14 ? number / 1000 : number;
  const date = new Date(ms);
  if (!Number.isNaN(date.getTime())) return date.toISOString().replace(".000Z", "Z");
  return String(value);
}

function goldUrl(relative) {
  return new URL(`../gold/${relative}`, import.meta.url).href;
}

function loadImage(relative) {
  if (!relative) return Promise.resolve(null);
  if (!cache.has(relative)) {
    cache.set(
      relative,
      new Promise((resolve, reject) => {
        const picture = new Image();
        picture.onload = () => resolve(picture);
        picture.onerror = () => reject(new Error(`Missing ${relative}`));
        picture.src = goldUrl(relative);
      }),
    );
  }
  return cache.get(relative);
}

function shape(kind, x, y) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", kind === "circle" ? "circle" : "polygon");
  node.classList.add("mark", kind === "circle" ? "known" : kind === "diamond" ? "candidate" : "discarded");
  if (kind === "circle") {
    node.setAttribute("cx", x);
    node.setAttribute("cy", y);
    node.setAttribute("r", 8);
  } else if (kind === "diamond") {
    node.setAttribute("points", `${x},${y - 9} ${x + 8},${y} ${x},${y + 9} ${x - 8},${y}`);
  } else {
    node.setAttribute("points", `${x - 7},${y - 7} ${x + 7},${y - 7} ${x + 7},${y + 7} ${x - 7},${y + 7}`);
  }
  return node;
}

function showPanel(object) {
  panel.hidden = false;
  panelName.textContent = object.name;
  panelBody.replaceChildren();
  const fields = [
    ["Type", object.type],
    ["Status", object.status],
    ["Reason", object.reason],
    ["Catalog", object.catalog],
    ["p_real", object.p_real == null ? "not scored yet" : object.p_real],
  ];
  for (const [label, value] of fields) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value == null || value === "" ? "—" : String(value);
    panelBody.append(dt, dd);
  }
  if (typeof object.url === "string" && /^https?:\/\//.test(object.url)) {
    const dt = document.createElement("dt");
    dt.textContent = "Link";
    const dd = document.createElement("dd");
    const link = document.createElement("a");
    link.href = object.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = object.catalog || "Catalog";
    dd.append(link);
    panelBody.append(dt, dd);
  }
}

function draw(frame) {
  marks.replaceChildren();
  const width = image.naturalWidth || 1;
  const height = image.naturalHeight || 1;
  marks.setAttribute("viewBox", `0 0 ${width} ${height}`);
  marks.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const visible = hits.filter((hit) => hit.frame_id === frame.frame_id);
  for (const hit of visible) {
    const object = byObject.get(hit.object_id);
    if (!object) continue;
    if (object.status === "Discarded" && !discardedBox.checked) continue;
    const node = shape(SHAPE[object.status] || "square", hit.x_px, hit.y_px);
    node.addEventListener("click", () => showPanel(object));
    marks.append(node);
  }
}

async function render() {
  const frame = sequence[index];
  if (!frame) return;
  const relative = diffBox.checked && frame.png_diff ? frame.png_diff : frame.png;
  const picture = await loadImage(relative);
  image.src = picture.src;
  image.alt = `${frame.frame_id} ${frame.date_utc}`;
  draw(frame);
  caption.textContent = `${stamp(frame.date_utc)} · ${frame.wavelength_um} µm · ${index + 1} / ${sequence.length}`;
  slider.value = String(index);
  const url = new URL(location.href);
  url.searchParams.set("region", frame.region_id);
  url.searchParams.set("frame", frame.frame_id);
  history.replaceState(null, "", url);
  loadImage(sequence[index + 1]?.png);
  loadImage(sequence[index - 1]?.png);
}

function move(step) {
  if (!sequence.length) return;
  index = (index + step + sequence.length) % sequence.length;
  render();
}

function setPlaying(playing) {
  clearInterval(timer);
  timer = 0;
  playButton.textContent = playing ? "Pause" : "Play";
  if (playing) timer = setInterval(() => move(1), 900);
}

playButton.addEventListener("click", () => setPlaying(!timer));
slider.addEventListener("input", () => {
  setPlaying(false);
  index = Number(slider.value);
  render();
});
diffBox.addEventListener("change", () => render());
discardedBox.addEventListener("change", () => render());
document.querySelector("#close").addEventListener("click", () => {
  panel.hidden = true;
});
window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") {
    event.preventDefault();
    setPlaying(false);
    move(1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    setPlaying(false);
    move(-1);
  } else if (event.key === " ") {
    event.preventDefault();
    setPlaying(!timer);
  } else if (event.key === "Escape") {
    panel.hidden = true;
  }
});

try {
  const params = new URLSearchParams(location.search);
  const found = await regions();
  const region = found.find((item) => item.region_id === params.get("region")) || found[0];
  if (!region) throw new Error("No region");
  sequence = await frames(region.region_id);
  const [objectRows, detectionRows] = await Promise.all([
    objects(region.region_id),
    detections(region.region_id),
  ]);
  byObject = new Map(objectRows.map((row) => [row.object_id, row]));
  hits = detectionRows;
  slider.max = String(Math.max(sequence.length - 1, 0));
  const requested = sequence.findIndex((frame) => frame.frame_id === params.get("frame"));
  index = requested >= 0 ? requested : 0;
  status.textContent = region.name;
  await render();
} catch (error) {
  status.textContent = "The viewer could not open the published frames.";
  console.error(error);
}
