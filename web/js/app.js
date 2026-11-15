import { catalogs, differences, frames, meta, spectra, sums, targets, trajectory } from "./data.js?v=w3";
import { apply, onChange, t } from "./i18n.js?v=w3b";

Object.assign(window, { catalogs, differences, frames, meta, spectra, sums, targets, trajectory });

const ARCSEC_PER_PIXEL = 6.15;
const SVG = "http://www.w3.org/2000/svg";
const STATUS = {
  Known: { key: "statusKnown", tag: "green" },
  "Unidentified candidate": { key: "statusCandidate", tag: "amber" },
  Discarded: { key: "statusDiscarded", tag: "" },
};
const TYPE = {
  asteroid: "typeAsteroid",
  "variable star": "typeVariable",
  artifact: "typeArtifact",
  unknown: "typeUnknown",
};
const REASON = {
  "Within 2 px of a SkyBoT object on that date": "reasonSkybot",
  "Within 2 px of a SIMBAD variable star (V*)": "reasonSimbad",
  "No match in SkyBoT, SIMBAD or Gaia": "reasonNone",
  "Touches flagged pixels or the edge": "reasonEdge",
  "Seen in a single exposure": "reasonSingle",
};
const SKY = {
  pole: ["ngc-stars.jpg", "ngc-dust.jpg", "cygnus.jpg"],
  asteroid: ["cygnus.jpg", "vela.jpg", "ice.jpg"],
  variable_star: ["ice.jpg", "vela.jpg", "ngc-dust.jpg"],
};
const CARD = { pole: "ngc-stars.jpg", asteroid: "cygnus.jpg", variable_star: "ice.jpg" };
const FRAME_PX = 293;
const TARGETS = [
  { id: "barnard", code: "BA", name: "Barnard's Star", ra: 269.452077, dec: 4.693365 },
  { id: "eps_indi_b", code: "EI", name: "Epsilon Indi B", ra: 331.0441466, dec: -56.7828442 },
  { id: "wise_0855", code: "W8", name: "WISE 0855", ra: 133.8189082, dec: -7.247048 },
  { id: "rw_aur", code: "RW", name: "RW Aurigae", ra: 76.956526, dec: 30.4014382 },
  { id: "r_dra", code: "RD", name: "R Draconis", ra: 248.1676108, dec: 66.7549794 },
];
let activeTarget = "barnard";
function formatNumber(value) {
  return new Intl.NumberFormat(langCode()).format(value);
}
function langCode() {
  return document.documentElement.lang === "es" ? "es" : "en";
}

const playButton = document.querySelector("#play");
const slider = document.querySelector("#slider");
const diffBox = document.querySelector("#diff");
const discardedBox = document.querySelector("#discarded");
const image = document.querySelector("#frame");
const marks = document.querySelector("#marks");
const caption = document.querySelector("#caption");

let regionList = [];
let sequence = [];
let hits = [];
let byObject = new Map();
let catalog = [];
let index = 0;
let selectedId = "";
let timer = 0;
let blinkTimer = 0;
let blinkOn = false;
let blinkMs = 600;
let blinkShowNow = true;
let gifCount = 4;
let gifMs = 450;
let gifTimer = 0;
let gifStep = 0;
let gifFrames = [];
let gifBlob = null;
let loadToken = 0;
let detector = 1;
let sumRows = [];
const goldTargets = new Map();
const cache = new Map();
const BANDS = {
  1: [0.746, 1.117],
  2: [1.103, 1.65],
  3: [1.63, 2.435],
  4: [2.403, 3.847],
  5: [3.811, 4.43],
  6: [4.412, 5.01],
};

function stamp(value) {
  if (typeof value === "string") return value.replace(/\.\d+Z$/, "Z");
  const number = typeof value === "number" ? value : Number(value);
  const ms = Number.isFinite(number) && Math.abs(number) > 1e14 ? number / 1000 : number;
  const date = new Date(ms);
  if (!Number.isNaN(date.getTime())) return date.toISOString().replace(".000Z", "Z");
  return String(value);
}

function fecha(value) {
  const date = new Date(stamp(value));
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(langCode(), {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function assetUrl(file) {
  return new URL(`../img/${file}`, import.meta.url).href;
}

function skyFile(regionId, frameIndex) {
  const list = SKY[regionId] || SKY.pole;
  const at = ((frameIndex % list.length) + list.length) % list.length;
  return list[at];
}

function when(row) {
  return row?.mean_date_utc ?? row?.date_utc;
}

function pngUrl(row) {
  if (!row?.png) return "";
  return new URL(`../gold/${row.png}`, import.meta.url).href;
}

function skyFor(row) {
  return pngUrl(row);
}

function openingDetector(targetId) {
  return targetId === "wise_0855" ? 6 : 1;
}

function parseDetector(value, targetId) {
  if (value === "color") return "color";
  const number = Number(value);
  if (value != null && value !== "" && Number.isInteger(number) && number >= 0 && number <= 6) return number;
  return openingDetector(targetId);
}

function colorReady(rows, season) {
  const have = new Set(rows.filter((row) => row.season === season).map((row) => row.detector));
  return have.has(1) && have.has(3) && have.has(6);
}

function buildSequence(rows, chosen) {
  if (chosen === "color") {
    const seasons = [...new Set(rows.filter((row) => row.season >= 1).map((row) => row.season))].sort((a, b) => a - b);
    return seasons.filter((season) => colorReady(rows, season)).map((season) => {
      const parts = rows.filter((row) => row.season === season && [1, 3, 6].includes(row.detector));
      const dates = parts.map((row) => new Date(stamp(when(row))).getTime()).filter(Number.isFinite);
      const mean = dates.length ? new Date(dates.reduce((sum, item) => sum + item, 0) / dates.length).toISOString() : when(parts[0]);
      return {
        season,
        detector: "color",
        mean_date_utc: mean,
        n_frames: parts.reduce((sum, row) => sum + Number(row.n_frames || 0), 0),
        png: `png/${activeTarget}/color_s${season}.png`,
      };
    });
  }
  return rows
    .filter((row) => row.detector === chosen && row.season >= 1)
    .sort((a, b) => a.season - b.season);
}

function detectorLabel(value) {
  if (value === "color") return t("detectorColor");
  if (value === 0) return t("detectorAll");
  return `D${value}`;
}

function bandText(value) {
  if (value === "color") return t("bandColor");
  if (value === 0) return t("bandAll");
  const pair = BANDS[value];
  return pair ? `${pair[0]}–${pair[1]} µm` : "—";
}

function markDetectors() {
  const group = document.querySelector("#detectors");
  if (!group) return;
  const season = sequence[index]?.season;
  for (const button of group.querySelectorAll("button")) {
    const value = button.dataset.detector;
    const selected = value === "color" ? detector === "color" : Number(value) === detector;
    button.setAttribute("aria-pressed", selected ? "true" : "false");
    button.disabled = value === "color"
      ? !colorReady(sumRows, season)
      : !sumRows.some((row) => row.detector === Number(value) && row.season >= 1);
  }
}

function loadImage(url) {
  if (!url) return Promise.resolve(null);
  if (!cache.has(url)) {
    cache.set(
      url,
      new Promise((resolve, reject) => {
        const picture = new Image();
        picture.onload = () => resolve(picture);
        picture.onerror = () => reject(new Error(`Missing ${url}`));
        picture.src = url;
      }),
    );
  }
  return cache.get(url);
}

function regionName(id) {
  return regionList.find((region) => region.region_id === id)?.name || id;
}

function titleOf(id) {
  return TARGETS.find((item) => item.id === id)?.name || regionName(id);
}

function targetOf() {
  return TARGETS.find((item) => item.id === activeTarget) || TARGETS[0];
}

function goldFor(targetId) {
  const direct = regionList.find((region) => region.region_id === targetId);
  if (direct) return direct;
  const at = Math.max(0, TARGETS.findIndex((item) => item.id === targetId));
  return regionList[at % Math.max(regionList.length, 1)];
}

function hms(degrees) {
  const hours = degrees / 15;
  const whole = Math.floor(hours);
  const minutes = Math.floor((hours - whole) * 60);
  const seconds = ((hours - whole) * 60 - minutes) * 60;
  return `${whole}h ${String(minutes).padStart(2, "0")}m ${seconds.toFixed(1)}s`;
}

function dms(degrees) {
  const sign = degrees < 0 ? "-" : "+";
  const absolute = Math.abs(degrees);
  const whole = Math.floor(absolute);
  const minutes = Math.floor((absolute - whole) * 60);
  const seconds = ((absolute - whole) * 60 - minutes) * 60;
  return `${sign}${whole}° ${String(minutes).padStart(2, "0")}' ${seconds.toFixed(1)}"`;
}

function tally(frame) {
  if (!frame) return { known: 0, candidate: 0, discarded: 0 };
  const ids = new Set(hits.filter((hit) => hit.frame_id === frame.frame_id).map((hit) => hit.object_id));
  const rows = [...ids].map((id) => byObject.get(id)).filter(Boolean);
  return {
    known: rows.filter((row) => row.status === "Known").length,
    candidate: rows.filter((row) => row.status === "Unidentified candidate").length,
    discarded: rows.filter((row) => row.status === "Discarded").length,
  };
}

function fillEpochs() {
  const frame = sequence[index];
  const baseline = sequence[0];
  if (!frame || !baseline) return;
  const now = tally(frame);
  const base = tally(baseline);
  document.querySelector("#stat-known").textContent = "—";
  document.querySelector("#stat-candidate").textContent = "—";
  document.querySelector("#stat-discarded").textContent = "—";
  document.querySelector("#stat-frames").textContent = formatNumber(frame.n_frames || 0);
  document.querySelector("#base-known").textContent = "—";
  document.querySelector("#base-candidate").textContent = "—";
  document.querySelector("#base-discarded").textContent = "—";
  document.querySelector("#base-frames").textContent = formatNumber(baseline.n_frames || 0);
  document.querySelector("#epoch-now").textContent = fecha(when(frame));
  document.querySelector("#epoch-then").textContent = fecha(when(baseline));
  const delta = now.candidate - base.candidate;
  const signed = `${delta >= 0 ? "+" : ""}${delta}`;
  document.querySelector("#delta-candidates").textContent = t("candidatesDelta", { value: signed });
  document.querySelector("#delta-span").textContent = t("timeSpan", { value: intervalo(baseline, frame) });
}

function statusOf(object) {
  const info = STATUS[object?.status];
  return info ? { label: t(info.key), tag: info.tag } : { label: object?.status || "—", tag: "" };
}

function reasonOf(object) {
  if (!object?.reason) return "";
  return REASON[object.reason] ? t(REASON[object.reason]) : object.reason;
}

function setTag(node, text, kind) {
  node.hidden = false;
  node.className = `tag ${kind || ""}`.trim();
  node.textContent = text;
}

function shape(kind, x, y) {
  const node = document.createElementNS(SVG, kind === "discarded" ? "g" : kind === "circle" ? "circle" : "polygon");
  node.classList.add("mark", kind === "circle" ? "known" : kind === "diamond" ? "candidate" : "discarded");
  if (kind === "circle") {
    node.setAttribute("cx", x);
    node.setAttribute("cy", y);
    node.setAttribute("r", "8");
  } else if (kind === "diamond") {
    node.setAttribute("points", `${x},${y - 9} ${x + 8},${y} ${x},${y + 9} ${x - 8},${y}`);
  } else {
    for (const [x1, y1, x2, y2] of [
      [x - 6, y - 6, x + 6, y + 6],
      [x - 6, y + 6, x + 6, y - 6],
    ]) {
      const line = document.createElementNS(SVG, "line");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      node.append(line);
    }
  }
  return node;
}

function kindFor(object) {
  if (object.status === "Known") return "circle";
  if (object.status === "Unidentified candidate") return "diamond";
  return "discarded";
}

function place(hit) {
  const width = FRAME_PX;
  const height = FRAME_PX;
  const col = hit.x_px < width / 3 ? t("left") : hit.x_px > (2 * width) / 3 ? t("right") : t("central");
  const row = hit.y_px < height / 3 ? t("upper") : hit.y_px > (2 * height) / 3 ? t("lower") : t("central");
  if (col === t("central") && row === t("central")) return t("center");
  return t("place", { row, col });
}

function detectionOn(objectId, frameId) {
  return hits.find((hit) => hit.object_id === objectId && hit.frame_id === frameId);
}

function movement(object, fromFrame, toFrame) {
  const start = detectionOn(object.object_id, fromFrame.frame_id);
  const end = detectionOn(object.object_id, toFrame.frame_id);
  if (!start || !end) return null;
  const dx = (end.x_px - start.x_px) * ARCSEC_PER_PIXEL;
  const dy = (end.y_px - start.y_px) * ARCSEC_PER_PIXEL;
  return Math.hypot(dx, dy);
}

function intervalo(fromFrame, toFrame) {
  const ms = Math.abs(new Date(stamp(when(toFrame))) - new Date(stamp(when(fromFrame))));
  if (!Number.isFinite(ms)) return "—";
  const hours = ms / 36e5;
  if (hours < 48) return t("hours", { n: Math.max(1, Math.round(hours)) });
  return t("days", { n: Math.round(hours / 24) });
}

function draw(frame) {
  marks.replaceChildren();
  marks.setAttribute("viewBox", `0 0 ${FRAME_PX} ${FRAME_PX}`);
  marks.setAttribute("preserveAspectRatio", "none");
  for (const hit of hits.filter((item) => item.frame_id === frame.frame_id)) {
    const object = byObject.get(hit.object_id);
    if (!object) continue;
    if (object.status === "Discarded" && !discardedBox.checked) continue;
    const node = shape(kindFor(object), hit.x_px, hit.y_px);
    if (object.object_id === selectedId) node.classList.add("selected");
    node.addEventListener("click", () => {
      selectedId = object.object_id;
      fillPanel();
      draw(sequence[index]);
    });
    marks.append(node);
  }
}

function fillPanel() {
  const frame = sequence[index];
  const baseline = sequence[0];
  const object = byObject.get(selectedId);
  const status = document.querySelector("#panel-status");
  const prob = document.querySelector("#panel-prob");
  document.querySelector("#panel-date").textContent = frame ? fecha(when(frame)) : "—";
  const wave = frame ? bandText(frame.detector === "color" ? "color" : frame.detector) : "—";
  document.querySelector("#panel-wave").textContent = wave;
  document.querySelector("#wave-read").textContent = wave;
  document.querySelector("#panel-now").textContent = frame ? fecha(when(frame)) : "—";
  document.querySelector("#panel-then").textContent = baseline ? fecha(when(baseline)) : "—";
  document.querySelector("#panel-region").textContent = targetOf().name;
  document.querySelector("#panel-span").textContent =
    frame && baseline ? t("interval", { value: intervalo(baseline, frame) }) : t("intervalEmpty");
  if (!object) {
    setTag(status, t("selectMarker"), "amber");
    document.querySelector("#panel-name").textContent = t("noneChosen");
    document.querySelector("#panel-type").textContent = "—";
    prob.hidden = true;
    document.querySelector("#panel-notes").replaceChildren(t("detailsIdle"));
    document.querySelector("#panel-move").textContent = "—";
    document.querySelector("#move-read").textContent = "—";
    document.querySelector("#panel-place").textContent = "—";
    document.querySelector("#place-read").textContent = "—";
    return;
  }
  const info = statusOf(object);
  setTag(status, info.label, info.tag);
  document.querySelector("#panel-name").textContent = object.name;
  document.querySelector("#panel-type").textContent = TYPE[object.type] ? t(TYPE[object.type]) : object.type || "—";
  if (typeof object.p_real === "number") {
    prob.hidden = false;
    prob.textContent = `${(object.p_real * 100).toFixed(1)}% p_real`;
  } else {
    prob.hidden = true;
  }
  const notes = document.querySelector("#panel-notes");
  notes.replaceChildren();
  const lines = [reasonOf(object)];
  if (object.catalog) lines.push(t("catalogLine", { name: object.catalog }));
  if (object.status === "Unidentified candidate") lines.push(t("stillCandidate"));
  for (const line of lines.filter(Boolean)) {
    const span = document.createElement("span");
    span.textContent = line;
    notes.append(span);
  }
  const shift = frame && baseline ? movement(object, baseline, frame) : null;
  const moveText = shift == null ? "—" : `${shift.toFixed(1)}″`;
  document.querySelector("#panel-move").textContent = moveText;
  document.querySelector("#move-read").textContent = moveText;
  document.querySelector("#delta-move").textContent = shift == null ? t("noShift") : t("shiftOf", { value: shift.toFixed(1) });
  const hit = frame ? detectionOn(object.object_id, frame.frame_id) : null;
  const where = hit ? place(hit) : t("offFrame");
  document.querySelector("#panel-place").textContent = where;
  document.querySelector("#place-read").textContent = where;
}

function fillCompare() {
  const frame = sequence[index];
  const baseline = sequence[0];
  if (!frame || !baseline) return;
  const now = document.querySelector("#cmp-now");
  const then = document.querySelector("#cmp-then");
  const showingNow = !blinkOn || blinkShowNow;
  const left = showingNow ? frame : baseline;
  const leftIndex = left === frame ? index + (diffBox.checked ? 1 : 0) : 0;
  now.src = skyFor(left, leftIndex);
  then.src = skyFor(baseline, 0);
  document.querySelector("#cmp-now-label").textContent = t(showingNow ? "photoCurrent" : "photoBlink", { date: fecha(when(left)) });
  document.querySelector("#cmp-then-label").textContent = t("photoBase", { date: fecha(when(baseline)) });

  const recentIds = new Set(hits.filter((hit) => hit.frame_id === frame.frame_id).map((hit) => hit.object_id));
  const baseIds = new Set(hits.filter((hit) => hit.frame_id === baseline.frame_id).map((hit) => hit.object_id));
  const newcomers = [...recentIds]
    .filter((id) => !baseIds.has(id))
    .map((id) => byObject.get(id))
    .filter(Boolean);
  const candidates = newcomers.filter((object) => object.status === "Unidentified candidate");
  const named = (candidates.length ? candidates : newcomers).map((object) => object.name);
  document.querySelector("#event-new").textContent = named.length
    ? t("newcomers", { names: named.join(", ") })
    : t("noNewcomers");
  document.querySelector("#event-new-foot").textContent = named.length ? t("seenRecent") : t("noNew");

  const shifts = [...byObject.values()]
    .map((object) => ({ object, arcsec: movement(object, baseline, frame) }))
    .filter((item) => item.arcsec != null);
  const biggest = shifts.sort((a, b) => b.arcsec - a.arcsec)[0];
  document.querySelector("#event-move").textContent = biggest
    ? t("motionLine", { name: biggest.object.name, value: biggest.arcsec.toFixed(1) })
    : t("noRepeat");
  document.querySelector("#event-move-foot").textContent = biggest && biggest.arcsec >= 1 ? t("shiftMeasured") : t("noClearShift");

  document.querySelector("#event-diff").textContent = frame.png_diff ? t("diffOn") : t("diffOff");
  document.querySelector("#event-diff-foot").textContent = frame.png_diff ? t("availableShort") : t("unpublished");
  const findings = [named.length > 0, Boolean(biggest && biggest.arcsec >= 1), Boolean(frame.png_diff)].filter(Boolean).length;
  document.querySelector("#cmp-count").textContent = t(findings === 1 ? "finding" : "findings", { n: findings });
}

async function render() {
  const frame = sequence[index];
  if (!frame) return;
  const picture = await loadImage(pngUrl(frame));
  if (!picture) return;
  image.src = picture.src;
  const target = targetOf();
  const placed = goldTargets.get(activeTarget);
  const shown = frame.detector === "color" ? "color" : frame.detector;
  image.alt = t("sumAlt", { name: placed?.name || target.name, season: frame.season, detector: detectorLabel(shown) });
  marks.replaceChildren();
  caption.textContent = t("sumCaption", {
    date: stamp(when(frame)),
    detector: detectorLabel(shown),
    band: bandText(shown),
    frames: frame.n_frames ?? 0,
  });
  const readout = document.querySelector("#cnn-out");
  if (activeTarget === "wise_0855" && shown >= 1 && shown <= 4) readout.textContent = t("wiseNeighbor");
  else if (activeTarget === "wise_0855") readout.textContent = t("wiseGuide");
  else if (!selectedId) readout.textContent = t("cnnIdle");
  slider.value = String(index);
  document.querySelector("#sector").textContent = placed?.name || target.name;
  document.querySelector("#sky-ra").textContent = hms(placed?.ra_now ?? target.ra);
  document.querySelector("#sky-dec").textContent = dms(placed?.dec_now ?? target.dec);
  fillEpochs();
  const url = new URL(location.href);
  url.searchParams.set("target", activeTarget);
  url.searchParams.set("detector", String(detector));
  url.searchParams.set("season", String(frame.season));
  url.searchParams.delete("region");
  url.searchParams.delete("frame");
  history.replaceState(null, "", url);
  fillPanel();
  fillCompare();
  markDetectors();
  loadImage(pngUrl(sequence[index - 1]));
  loadImage(pngUrl(sequence[index + 1]));
}

function move(step) {
  if (!sequence.length) return;
  index = (index + step + sequence.length) % sequence.length;
  render();
}

function setPlaying(playing) {
  clearInterval(timer);
  timer = 0;
  playButton.textContent = playing ? t("pause") : t("play");
  if (playing) timer = setInterval(() => move(1), 900);
}

function sample(list, count) {
  if (list.length <= count) return list.slice();
  return Array.from({ length: count }, (_, step) => {
    const at = Math.round((step * (list.length - 1)) / (count - 1));
    return list[at];
  });
}

function updateGifLabel() {
  const seconds = ((gifFrames.length || gifCount) * gifMs) / 1000;
  document.querySelector("#gif-duration").textContent = t("duration", { seconds: seconds.toFixed(1) });
}

function restartPreview() {
  clearInterval(gifTimer);
  gifFrames = sample(sequence, gifCount);
  gifStep = 0;
  gifBlob = null;
  document.querySelector("#gif-save").disabled = true;
  const available = document.querySelector("#gif-available");
  if (available) available.textContent = t("captures", { n: sequence.length });
  updateGifLabel();
  const preview = document.querySelector("#gif-preview");
  const show = () => {
    const frame = gifFrames[gifStep % Math.max(gifFrames.length, 1)];
    if (frame) preview.src = skyFor(frame, sequence.indexOf(frame));
    gifStep += 1;
  };
  show();
  if (gifFrames.length) gifTimer = setInterval(show, gifMs);
}

async function makeGif() {
  const note = document.querySelector("#gif-note");
  const button = document.querySelector("#gif-make");
  button.disabled = true;
  note.textContent = t("makingGif");
  try {
    const { GIFEncoder, quantize, applyPalette } = await import(
      "https://cdn.jsdelivr.net/npm/gifenc@1.0.3/+esm"
    );
    const encoder = GIFEncoder();
    for (const frame of gifFrames) {
      const picture = await loadImage(skyFor(frame, sequence.indexOf(frame)));
      const canvas = document.createElement("canvas");
      canvas.width = picture.naturalWidth;
      canvas.height = picture.naturalHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(picture, 0, 0);
      const { data, width, height } = context.getImageData(0, 0, canvas.width, canvas.height);
      const palette = quantize(data, 256);
      const indexed = applyPalette(data, palette);
      encoder.writeFrame(indexed, width, height, { palette, delay: gifMs });
    }
    encoder.finish();
    gifBlob = new Blob([encoder.bytes()], { type: "image/gif" });
    document.querySelector("#gif-save").disabled = false;
    note.textContent = t("gifReady");
  } catch (error) {
    note.textContent = t("gifFail");
    console.error(error);
  } finally {
    button.disabled = false;
  }
}

function saveGif() {
  if (!gifBlob) return;
  const link = document.createElement("a");
  const region = sequence[index]?.region_id || "serafim";
  link.href = URL.createObjectURL(gifBlob);
  link.download = `serafim-${region}.gif`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function renderRegions() {
  const host = document.querySelector("#region-cards");
  host.replaceChildren();
  for (const target of TARGETS) {
    const active = target.id === activeTarget;
    const card = document.createElement("article");
    card.className = active ? "card active" : "card";
    if (active) {
      const flag = document.createElement("span");
      flag.className = "flag";
      flag.textContent = t("activeFlag");
      card.append(flag);
    }
    const head = document.createElement("div");
    head.className = "card-title";
    const title = document.createElement("h3");
    const code = document.createElement("span");
    code.className = "code";
    code.textContent = target.code;
    title.append(code, document.createTextNode(target.name));
    const badge = document.createElement("span");
    badge.className = active ? "tag amber" : "tag";
    badge.textContent = active ? t("inFocus") : t("available");
    head.append(title, badge);
    const text = document.createElement("p");
    text.textContent = t(`blurb_${target.id}`);
    const button = document.createElement("button");
    button.type = "button";
    button.className = active ? "amber wide" : "soft wide";
    button.textContent = active ? t("viewing") : t("loadTarget");
    button.addEventListener("click", () => {
      activeTarget = target.id;
      openRegion(target.id, { scroll: true, detector: openingDetector(target.id) });
    });
    card.append(head, text, button);
    host.append(card);
  }
}

function renderCatalog() {
  document.querySelector("#rows").replaceChildren();
  document.querySelector("#catalog-status").textContent = t("catalogLater");
}

async function focusObject(object) {
  if (sequence[0]?.region_id !== object.region_id) await openRegion(object.region_id);
  selectedId = object.object_id;
  const hit = hits.find((item) => item.object_id === object.object_id);
  if (hit) index = Math.max(0, sequence.findIndex((frame) => frame.frame_id === hit.frame_id));
  await render();
  document.querySelector("#visor").scrollIntoView({ behavior: "smooth" });
}

async function openRegion(targetId, options = {}) {
  const token = ++loadToken;
  setPlaying(false);
  if (!goldTargets.has(targetId) && !TARGETS.some((item) => item.id === targetId)) return;
  activeTarget = targetId;
  const rows = await sums(targetId);
  if (token !== loadToken) return;
  sumRows = rows;
  hits = [];
  byObject = new Map();
  selectedId = "";
  detector = options.detector ?? detector;
  sequence = buildSequence(sumRows, detector);
  const season = Number(options.season);
  const at = sequence.findIndex((row) => row.season === season);
  index = at >= 0 ? at : 0;
  slider.max = String(Math.max(sequence.length - 1, 0));
  renderRegions();
  markDetectors();
  restartPreview();
  await render();
  if (options.scroll) document.querySelector("#visor").scrollIntoView({ behavior: "smooth" });
}

function choose(group, pressed) {
  for (const button of group.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", button === pressed ? "true" : "false");
  }
}

playButton.addEventListener("click", () => setPlaying(!timer));
slider.addEventListener("input", () => {
  setPlaying(false);
  index = Number(slider.value);
  render();
});
diffBox.addEventListener("change", () => render());
document.querySelector("#detectors").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || button.disabled) return;
  const value = button.dataset.detector;
  const next = value === "color" ? "color" : Number(value);
  if (next === detector) return;
  setPlaying(false);
  const season = sequence[index]?.season;
  detector = next;
  sequence = buildSequence(sumRows, detector);
  const at = sequence.findIndex((row) => row.season === season);
  index = at >= 0 ? at : 0;
  slider.max = String(Math.max(sequence.length - 1, 0));
  markDetectors();
  restartPreview();
  render();
});
discardedBox.addEventListener("change", () => render());
document.querySelector("#blink").addEventListener("click", () => {
  blinkOn = !blinkOn;
  const button = document.querySelector("#blink");
  button.textContent = blinkOn ? t("blinkPause") : t("blink");
  clearInterval(blinkTimer);
  if (blinkOn) {
    blinkTimer = setInterval(() => {
      blinkShowNow = !blinkShowNow;
      fillCompare();
    }, blinkMs);
  } else {
    blinkShowNow = true;
    fillCompare();
  }
});
document.querySelector("#blink-jump").addEventListener("click", () => {
  document.querySelector("#comparador").scrollIntoView({ behavior: "smooth" });
  if (!blinkOn) document.querySelector("#blink").click();
});
document.querySelector("#blink-speed").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  choose(document.querySelector("#blink-speed"), button);
  blinkMs = Number(button.dataset.ms);
  if (blinkOn) {
    document.querySelector("#blink").click();
    document.querySelector("#blink").click();
  }
});
document.querySelector("#gif-count").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  choose(document.querySelector("#gif-count"), button);
  gifCount = Number(button.dataset.n);
  restartPreview();
});
document.querySelector("#gif-speed").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  choose(document.querySelector("#gif-speed"), button);
  gifMs = Number(button.dataset.ms);
  restartPreview();
});
document.querySelector("#gif-make").addEventListener("click", () => makeGif());
document.querySelector("#gif-save").addEventListener("click", () => saveGif());
document.querySelector("#analyze").addEventListener("click", () => {
  const object = byObject.get(selectedId);
  const out = document.querySelector("#cnn-out");
  if (!object) {
    out.textContent = t("pickMarker");
    return;
  }
  const score = Number(object.p_real);
  const scored = Number.isFinite(score) ? t("score", { value: (score * 100).toFixed(1) }) : t("noScore");
  out.textContent = `${object.name}: ${statusOf(object).label}.${scored} ${reasonOf(object)}`;
});
document.querySelector("#help-open").addEventListener("click", () => document.querySelector("#help").showModal());
document.querySelector("#help-close").addEventListener("click", () => document.querySelector("#help").close());
window.addEventListener("keydown", (event) => {
  if (event.target.closest("input, textarea")) return;
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
  }
});

apply();
onChange(() => {
  apply();
  renderRegions();
  if (regionList.length) renderCatalog();
  playButton.textContent = timer ? t("pause") : t("play");
  document.querySelector("#blink").textContent = blinkOn ? t("blinkPause") : t("blink");
  if (sequence.length) {
    updateGifLabel();
    const available = document.querySelector("#gif-available");
    if (available) available.textContent = t("captures", { n: sequence.length });
    render();
  }
});

try {
  const params = new URLSearchParams(location.search);
  const [found, info] = await Promise.all([targets(), meta()]);
  regionList = found;
  goldTargets.clear();
  for (const row of found) goldTargets.set(row.target_id, row);
  renderCatalog();
  const wanted = params.get("target") || params.get("region") || "barnard";
  const match = found.find((row) => row.target_id === wanted) || found[0];
  if (match) activeTarget = match.target_id;
  detector = parseDetector(params.get("detector"), activeTarget);
  document.querySelector("#updated").textContent = info?.run_date_utc
    ? t("tablesLoaded", { date: info.run_date_utc })
    : t("tablesLoaded", { date: "—" });
  if (match) await openRegion(match.target_id, { detector, season: params.get("season") });
  if (location.hash) {
    const place = document.querySelector(location.hash);
    if (place) place.scrollIntoView();
  }
} catch (error) {
  document.querySelector("#updated").textContent = t("tablesDown");
  document.querySelector("#catalog-status").textContent = t("tablesDown");
  console.error(error);
}
