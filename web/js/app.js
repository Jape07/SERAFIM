import { catalogs, differences, frames, meta, spectra, sums, targets, trajectory } from "./data.js?v=w3";
import { apply, lang, onChange, t } from "./i18n.js?v=gif1";

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
function put(selector, value) {
  const el = document.querySelector(selector);
  if (el) el.textContent = value;
  return el;
}

const MODES = ["sum", "difference", "all_seasons", "blink"];
const LAYERS = ["gaia", "catwise", "simbad", "skybot"];
const playButton = document.querySelector("#play");
const slider = document.querySelector("#slider");
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
let listedSources = [];
let catalogContext = null;
let timer = 0;
let blinkTimer = 0;
let blinkOn = false;
let blinkMs = 600;
let blinkShowNow = true;
let blinkPaused = false;
let blinkKey = "";
let blinkSlot = 0;
let viewToken = 0;
let gifCount = 4;
let gifMs = 450;
let gifTimer = 0;
let gifStep = 0;
let gifFrames = [];
let gifBlob = null;
let loadToken = 0;
let detector = 1;
let mode = "sum";
let sumRows = [];
let diffRows = [];
let guide = null;
let layersOn = new Set(LAYERS);
let selectedSource = null;
let magReach = 0;
const catalogCache = new Map();
const spectrumCache = new Map();
const pathCache = new Map();
const frameCache = new Map();
let spectrumRows = [];
let pathRows = [];
let searchPoint = null;
let fovArcmin = 15;
let searchState = "";
const projectionSeen = new Set();
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

function howToLook(targetId) {
  return guide?.targets?.find((item) => item.id === targetId)?.how_to_look || null;
}

function openingDetector(targetId) {
  const first = howToLook(targetId)?.detectors?.[0];
  return Number.isInteger(first) ? first : 1;
}

function openingMode(targetId) {
  const name = howToLook(targetId)?.mode;
  return MODES.includes(name) ? name : "sum";
}

function openingSeason(targetId) {
  const how = howToLook(targetId);
  const seasons = how?.seasons || [];
  if (how?.mode === "difference" && seasons.length >= 2) return `${seasons[0]}-${seasons[seasons.length - 1]}`;
  if (how?.mode === "all_seasons") return "0";
  const first = seasons.find((season) => season >= 1);
  return first == null ? "" : String(first);
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

function buildDifferences(rows, chosen) {
  if (typeof chosen !== "number" || chosen < 1 || chosen > 6) return [];
  return rows
    .filter((row) => row.detector === chosen)
    .sort((a, b) => {
      const longA = a.season_b - a.season_a > 1 ? 1 : 0;
      const longB = b.season_b - b.season_a > 1 ? 1 : 0;
      return longA - longB || a.season_a - b.season_a || a.season_b - b.season_b;
    });
}

function buildAllSeasons(rows, chosen) {
  if (chosen === "color") return [];
  const row = rows.find((item) => item.detector === chosen && item.season === 0);
  return row ? [row] : [];
}

function sequenceFor(name) {
  if (name === "difference") return buildDifferences(diffRows, detector);
  if (name === "all_seasons") return buildAllSeasons(sumRows, detector);
  return buildSequence(sumRows, detector);
}

function modeAvailable(name) {
  const rows = sequenceFor(name);
  return name === "blink" ? rows.length > 1 : rows.length > 0;
}

function parseMode(value) {
  return MODES.includes(value) ? value : "sum";
}

function sumAt(chosen, season) {
  return sumRows.find((row) => row.detector === chosen && row.season === season);
}

function seasonToken(frame = sequence[index]) {
  if (!frame) return "";
  if (mode === "difference") return `${frame.season_a}-${frame.season_b}`;
  return String(frame.season);
}

function holdSeason() {
  const frame = sequence[index];
  if (!frame) return "";
  if (mode === "difference") return String(frame.season_b);
  return String(frame.season);
}

function indexFor(rows, token) {
  if (!rows.length) return 0;
  if (mode === "difference") {
    const match = String(token ?? "").match(/^(\d+)-(\d+)$/);
    if (match) {
      const at = rows.findIndex((row) => row.season_a === Number(match[1]) && row.season_b === Number(match[2]));
      if (at >= 0) return at;
    }
    const season = Number(token);
    if (Number.isInteger(season) && season > 0) {
      const consecutive = rows.findIndex((row) => row.season_b === season && row.season_b - row.season_a === 1);
      if (consecutive >= 0) return consecutive;
      const any = rows.findIndex((row) => row.season_a === season || row.season_b === season);
      if (any >= 0) return any;
    }
    const span = rows.findIndex((row) => row.season_b - row.season_a > 1);
    return span >= 0 ? span : 0;
  }
  const season = Number(token);
  const at = rows.findIndex((row) => row.season === season);
  return at >= 0 ? at : 0;
}

function blinkPair() {
  const current = sequence[index];
  if (!current) return [];
  const partnerAt = index > 0 ? index - 1 : Math.min(1, sequence.length - 1);
  const partner = sequence[partnerAt];
  if (!partner || partner === current) return [current];
  return [current, partner];
}

function shownDetector(frame) {
  if (!frame) return detector;
  if (frame.detector === "color") return "color";
  return frame.detector ?? detector;
}

async function loadGuide() {
  try {
    const response = await fetch(new URL("../texts/targets.json", import.meta.url));
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    console.error(error);
    return null;
  }
}

function flagText(name) {
  if (lang() === "es") {
    const key = `flag_${name}`;
    const translated = t(key);
    if (translated !== key) return translated;
  }
  const entry = guide?.flags?.find((item) => item.name === name);
  return entry?.text || name;
}

function noteRows(frame) {
  if (!frame) return [];
  if (mode === "difference") {
    return [sumAt(frame.detector, frame.season_a), sumAt(frame.detector, frame.season_b)].filter(Boolean);
  }
  if (mode === "blink") return blinkPair();
  return [frame];
}

function flagNames(rows) {
  const names = [];
  for (const row of rows) {
    for (const part of String(row.flags || "").split(",")) {
      const name = part.trim();
      if (name && name !== "SOURCE" && !names.includes(name)) names.push(name);
    }
  }
  return names;
}

function rowSaturated(row) {
  return row?.saturated === true || row?.saturated === 1;
}

function captionFor(frame) {
  const shown = shownDetector(frame);
  if (mode === "difference") {
    return t("diffCaption", {
      detector: detectorLabel(shown),
      earlier: frame.season_a,
      later: frame.season_b,
      band: bandText(shown),
    });
  }
  if (mode === "blink") {
    const seasons = blinkPair().map((row) => row.season).sort((a, b) => a - b);
    return t("blinkCaption", {
      detector: detectorLabel(shown),
      first: seasons[0],
      second: seasons[seasons.length - 1],
      band: bandText(shown),
    });
  }
  return t(mode === "all_seasons" ? "allCaption" : "sumCaption", {
    date: stamp(when(frame)),
    detector: detectorLabel(shown),
    band: bandText(shown),
    frames: frame.n_frames ?? 0,
  });
}

function altFor(frame, name) {
  const shown = detectorLabel(shownDetector(frame));
  if (mode === "difference") {
    return t("diffAlt", { name, earlier: frame.season_a, later: frame.season_b, detector: shown });
  }
  if (mode === "blink") {
    const seasons = blinkPair().map((row) => row.season).sort((a, b) => a - b);
    return t("blinkAlt", { name, first: seasons[0], second: seasons[seasons.length - 1], detector: shown });
  }
  if (mode === "all_seasons") return t("allAlt", { name, detector: shown });
  return t("sumAlt", { name, season: frame.season, detector: shown });
}

function showNotes(frame) {
  const rows = noteRows(frame);
  const badge = document.querySelector("#saturated-badge");
  const host = document.querySelector("#flag-notes");
  if (!badge || !host) return;
  const saturated = rows.some(rowSaturated);
  badge.hidden = !saturated;
  badge.textContent = saturated ? (lang() === "es" ? t("saturatedBadge") : guide?.saturated_badge || t("saturatedBadge")) : "";
  host.replaceChildren();
  for (const name of flagNames(rows)) {
    const line = document.createElement("p");
    line.className = "hint";
    line.textContent = t("flagLine", { name, text: flagText(name) });
    host.append(line);
  }
}

function stopBlink() {
  clearInterval(blinkTimer);
  blinkTimer = 0;
  blinkKey = "";
}

function pairKey(pair) {
  return pair.map((row) => row.png).join("|");
}

function armBlink(pair, pictures) {
  const running = mode === "blink" && !blinkPaused && pair.length > 1 && pictures.every(Boolean);
  if (!running) {
    stopBlink();
    if (playButton) playButton.textContent = mode === "blink" ? t("play") : (timer ? t("pause") : t("play"));
    return;
  }
  if (playButton) playButton.textContent = t("pause");
  const key = pairKey(pair);
  if (blinkTimer && blinkKey === key) return;
  stopBlink();
  blinkKey = key;
  blinkSlot = 0;
  image.src = pictures[0].src;
  blinkTimer = setInterval(() => {
    blinkSlot = 1 - blinkSlot;
    const picture = pictures[blinkSlot];
    if (picture) image.src = picture.src;
  }, 700);
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

function colorPossible() {
  const seasons = [...new Set(sumRows.filter((row) => row.season >= 1).map((row) => row.season))];
  return seasons.some((season) => colorReady(sumRows, season));
}

function markDetectors() {
  const group = document.querySelector("#detectors");
  if (!group) return;
  for (const button of group.querySelectorAll("button")) {
    const value = button.dataset.detector;
    const selected = value === "color" ? detector === "color" : Number(value) === detector;
    button.setAttribute("aria-pressed", selected ? "true" : "false");
    button.disabled = value === "color"
      ? !colorPossible()
      : !sumRows.some((row) => row.detector === Number(value) && row.season >= 1);
  }
}

function markModes() {
  const group = document.querySelector("#modes");
  if (!group) return;
  for (const button of group.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", button.dataset.mode === mode ? "true" : "false");
    button.disabled = !modeAvailable(button.dataset.mode);
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
  if (!frame || !baseline || !document.querySelector("#stat-known")) return;
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
    if (object.status === "Discarded" && discardedBox && !discardedBox.checked) continue;
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
  put("#panel-date", frame ? fecha(when(frame)) : "—");
  const wave = frame ? bandText(frame.detector === "color" ? "color" : frame.detector) : "—";
  put("#panel-wave", wave);
  put("#wave-read", wave);
  const later = frame && mode === "difference" ? (sumAt(frame.detector, frame.season_b) || frame) : frame;
  const earlier = frame && mode === "difference" ? (sumAt(frame.detector, frame.season_a) || baseline) : baseline;
  put("#panel-now", later && when(later) ? fecha(when(later)) : "—");
  put("#panel-then", earlier && when(earlier) ? fecha(when(earlier)) : "—");
  put("#panel-region", targetOf().name);
  put("#panel-span", later && earlier && when(later) && when(earlier)
    ? t("interval", { value: intervalo(earlier, later) })
    : t("intervalEmpty"));
  const notes = document.querySelector("#panel-notes");
  if (!object) {
    if (status) setTag(status, t("selectMarker"), "amber");
    put("#panel-name", t("noneChosen"));
    put("#panel-type", "—");
    if (prob) prob.hidden = true;
    if (notes) notes.replaceChildren(t("detailsIdle"));
    put("#panel-move", "—");
    put("#move-read", "—");
    put("#panel-place", "—");
    put("#place-read", "—");
    return;
  }
  const info = statusOf(object);
  if (status) setTag(status, info.label, info.tag);
  put("#panel-name", object.name);
  put("#panel-type", TYPE[object.type] ? t(TYPE[object.type]) : object.type || "—");
  if (prob) {
    if (typeof object.p_real === "number") {
      prob.hidden = false;
      prob.textContent = `${(object.p_real * 100).toFixed(1)}% p_real`;
    } else {
      prob.hidden = true;
    }
  }
  if (!notes) {
    const shiftOnly = frame && baseline ? movement(object, baseline, frame) : null;
    const moveOnly = shiftOnly == null ? "—" : `${shiftOnly.toFixed(1)}″`;
    put("#panel-move", moveOnly);
    put("#move-read", moveOnly);
    put("#delta-move", shiftOnly == null ? t("noShift") : t("shiftOf", { value: shiftOnly.toFixed(1) }));
    const hitOnly = frame ? detectionOn(object.object_id, frame.frame_id) : null;
    const whereOnly = hitOnly ? place(hitOnly) : t("offFrame");
    put("#panel-place", whereOnly);
    put("#place-read", whereOnly);
    return;
  }
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
  put("#panel-move", moveText);
  put("#move-read", moveText);
  put("#delta-move", shift == null ? t("noShift") : t("shiftOf", { value: shift.toFixed(1) }));
  const hit = frame ? detectionOn(object.object_id, frame.frame_id) : null;
  const where = hit ? place(hit) : t("offFrame");
  put("#panel-place", where);
  put("#place-read", where);
}

function markCompare(selector, kind) {
  const svg = document.querySelector(selector);
  if (!svg) return;
  svg.replaceChildren();
  if (!catalogContext) return;
  const placed = goldTargets.get(activeTarget);
  const target = targetOf();
  const ra = selectedSource?.ra ?? placed?.ra_now ?? target.ra;
  const dec = selectedSource?.dec ?? placed?.dec_now ?? target.dec;
  const pixel = skyToPixel(catalogContext.wcs, ra, dec);
  const { width, height } = catalogContext;
  if (!pixel || pixel.x < 0 || pixel.y < 0 || pixel.x > width || pixel.y > height) return;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const radius = Math.min(width, height) * 0.16;
  if (kind === "found") {
    const glow = document.createElementNS(SVG, "circle");
    glow.setAttribute("cx", String(pixel.x));
    glow.setAttribute("cy", String(pixel.y));
    glow.setAttribute("r", String(radius));
    glow.setAttribute("fill", "none");
    glow.setAttribute("stroke", "rgba(245, 196, 0, 0.45)");
    glow.setAttribute("stroke-width", "1.1");
    const ring = document.createElementNS(SVG, "circle");
    ring.setAttribute("cx", String(pixel.x));
    ring.setAttribute("cy", String(pixel.y));
    ring.setAttribute("r", String(radius * 0.42));
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "#f5c400");
    ring.setAttribute("stroke-width", "1.5");
    const dot = document.createElementNS(SVG, "circle");
    dot.setAttribute("cx", String(pixel.x));
    dot.setAttribute("cy", String(pixel.y));
    dot.setAttribute("r", String(radius * 0.12));
    dot.setAttribute("fill", "#f5c400");
    svg.append(glow, ring, dot);
    return;
  }
  const ring = document.createElementNS(SVG, "circle");
  ring.setAttribute("cx", String(pixel.x));
  ring.setAttribute("cy", String(pixel.y));
  ring.setAttribute("r", String(radius * 0.7));
  ring.setAttribute("fill", "none");
  ring.setAttribute("stroke", "#94a3b8");
  ring.setAttribute("stroke-width", "1.2");
  ring.setAttribute("stroke-dasharray", "2.2 1.6");
  const arm = radius * 0.35;
  for (const [x1, y1, x2, y2] of [
    [pixel.x - arm, pixel.y - arm, pixel.x + arm, pixel.y + arm],
    [pixel.x - arm, pixel.y + arm, pixel.x + arm, pixel.y - arm],
  ]) {
    const line = document.createElementNS(SVG, "line");
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y2));
    line.setAttribute("stroke", "#94a3b8");
    line.setAttribute("stroke-width", "1.2");
    svg.append(line);
  }
  svg.append(ring);
}

function fillCompare() {
  const frame = sequence[index];
  const baseline = sequence[0];
  if (!frame || !baseline) return;
  let nowFrame = frame;
  let thenFrame = baseline;
  if (mode === "difference") {
    nowFrame = sumAt(frame.detector, frame.season_b) || frame;
    thenFrame = sumAt(frame.detector, frame.season_a) || frame;
  } else if (mode === "blink") {
    const pair = blinkPair();
    nowFrame = pair[0];
    thenFrame = pair[1] || pair[0];
  } else if (blinkOn && !blinkShowNow) {
    nowFrame = baseline;
  }
  const now = document.querySelector("#cmp-now");
  const then = document.querySelector("#cmp-then");
  if (now) now.src = skyFor(nowFrame);
  if (then) then.src = skyFor(thenFrame);
  put("#cmp-now-label", t(blinkOn && !blinkShowNow ? "photoBlink" : "photoCurrent", { date: fecha(when(nowFrame)) }));
  put("#cmp-then-label", t("photoBase", { date: fecha(when(thenFrame)) }));
  markCompare("#cmp-now-marks", "found");
  markCompare("#cmp-then-marks", "absent");

  fillFindings(thenFrame, nowFrame);
  fillSummary(thenFrame, nowFrame);
}

let summaryToken = 0;

function summarySeason(row) {
  if (!row || row.season == null || row.season === "") return null;
  const season = Number(row.season);
  return Number.isInteger(season) && season >= 0 ? season : null;
}

function frameCount(row) {
  const count = Number(row?.n_frames);
  return Number.isFinite(count) ? formatNumber(count) : "—";
}

function fillSummary(thenFrame, nowFrame) {
  if (!document.querySelector("#summary")) return;
  const token = ++summaryToken;
  const nowSeason = summarySeason(nowFrame);
  const thenSeason = summarySeason(thenFrame);
  put("#sum-now-date", nowFrame && when(nowFrame) ? fecha(when(nowFrame)) : "—");
  put("#sum-then-date", thenFrame && when(thenFrame) ? fecha(when(thenFrame)) : "—");
  put("#sum-now-season", nowSeason == null ? "—" : t("summarySeason", { n: nowSeason }));
  put("#sum-then-season", thenSeason == null ? "—" : t("summarySeasonBase", { n: thenSeason }));
  put("#sum-now-frames", frameCount(nowFrame));
  put("#sum-then-frames", frameCount(thenFrame));
  put("#sum-now-new", formatNumber(0));
  put("#sum-then-new", formatNumber(0));
  put("#sum-now-rej", formatNumber(0));
  put("#sum-then-rej", formatNumber(0));
  put("#sum-new-line", t("summaryNewNone"));
  const shown = shownDetector(nowFrame);
  const before = meanFlux(thenSeason, shown);
  const after = meanFlux(nowSeason, shown);
  const changed = before && after && (after / before >= 1.25 || after / before <= 0.8);
  put("#sum-bright-line", t(changed ? "summaryBrightOne" : "summaryBrightNone"));
  put("#sum-span-line", t("summarySpan", { value: nowFrame && thenFrame ? intervalo(thenFrame, nowFrame) : "—" }));
  Promise.all([
    nowSeason == null ? [] : catalogFor(activeTarget, nowSeason).catch(() => []),
    thenSeason == null ? [] : catalogFor(activeTarget, thenSeason).catch(() => []),
  ]).then(([nowRows, thenRows]) => {
    if (token !== summaryToken) return;
    put("#sum-now-known", formatNumber(nowRows.length));
    put("#sum-then-known", formatNumber(thenRows.length));
  });
}

function seasonOf(row) {
  if (!row || row.season == null || row.season === "") return null;
  const season = Number(row.season);
  return Number.isInteger(season) && season >= 1 ? season : null;
}

function skySep(ra1, dec1, ra2, dec2) {
  const rad = Math.PI / 180;
  const ra = ((Number(ra1) + Number(ra2)) / 2) * rad;
  const dx = (Number(ra2) - Number(ra1)) * Math.cos(ra) * 3600;
  const dy = (Number(dec2) - Number(dec1)) * 3600;
  return Math.hypot(dx, dy);
}

function pathAt(season) {
  return pathRows.find((row) => Number(row.season) === season) || null;
}

function flaggedSpectrum(row) {
  const names = String(row.flag || "").split(",").map((item) => item.trim());
  return names.includes("saturated") || names.includes("poor_fit");
}

function meanFlux(season, detectorNumber) {
  const band = BANDS[detectorNumber];
  if (!band) return null;
  const rows = spectrumRows.filter((row) => {
    if (Number(row.season) !== season || flaggedSpectrum(row)) return false;
    const wave = Number(row.wavelength_um);
    const flux = Number(row.flux_mjy);
    return wave >= band[0] && wave <= band[1] && Number.isFinite(flux) && flux > 0;
  });
  if (!rows.length) return null;
  return rows.reduce((sum, row) => sum + Number(row.flux_mjy), 0) / rows.length;
}

function fillFindings(thenFrame, nowFrame) {
  const mag = selectedSource && Number.isFinite(Number(selectedSource.mag))
    ? `${Number(selectedSource.mag).toFixed(1)} ${selectedSource.mag_band || ""}`.trim()
    : "";
  if (selectedSource) {
    put("#event-new", mag
      ? t("resultNamedMag", { name: selectedSource.name, catalog: selectedSource.catalog, mag })
      : t("resultNamed", { name: selectedSource.name, catalog: selectedSource.catalog }));
    put("#event-new-foot", t("resultMarked"));
  } else {
    put("#event-new", t("resultNone"));
    put("#event-new-foot", t("resultUnmarked"));
  }

  const earlier = pathAt(seasonOf(thenFrame));
  const later = pathAt(seasonOf(nowFrame));
  let moved = false;
  if (earlier && later && seasonOf(thenFrame) !== seasonOf(nowFrame)) {
    const measured = [earlier, later].every((row) => row.ra_measured != null && row.dec_measured != null);
    const pair = measured
      ? [earlier.ra_measured, earlier.dec_measured, later.ra_measured, later.dec_measured]
      : [earlier.ra_catalog, earlier.dec_catalog, later.ra_catalog, later.dec_catalog];
    const arcsec = skySep(...pair);
    moved = Number.isFinite(arcsec) && arcsec >= 1;
    const value = Number.isFinite(arcsec) ? arcsec.toFixed(1) : "—";
    put("#event-move", t(measured ? "resultShift" : "resultShiftCatalog", { value }));
    put("#event-move-foot", moved ? t("shiftMeasured") : t("noClearShift"));
  } else {
    put("#event-move", t("resultNoShift"));
    put("#event-move-foot", t("noClearShift"));
  }

  const shown = shownDetector(nowFrame);
  const before = meanFlux(seasonOf(thenFrame), shown);
  const after = meanFlux(seasonOf(nowFrame), shown);
  let brighter = false;
  if (before && after) {
    const ratio = after / before;
    brighter = ratio >= 1.25 || ratio <= 0.8;
    put("#event-diff", t("resultBright", { ratio: ratio.toFixed(2) }));
    put("#event-diff-foot", brighter ? t("resultChanged") : t("resultSteady"));
  } else {
    put("#event-diff", t("resultNoBright"));
    put("#event-diff-foot", t("unpublished"));
  }
  const findings = [Boolean(selectedSource), moved, brighter].filter(Boolean).length;
  put("#cmp-count", t(findings === 1 ? "finding" : "findings", { n: findings }));
}

function parseLayers(value) {
  if (value == null) return new Set(LAYERS);
  return new Set(value.split(",").map((item) => item.trim()).filter((item) => LAYERS.includes(item)));
}

function catalogSeason(frame) {
  if (!frame) return null;
  if (mode === "all_seasons") return 0;
  if (mode === "difference") return frame.season_b;
  return frame.season;
}

function catalogFor(target, season) {
  const key = `${target}:${season}`;
  if (!catalogCache.has(key)) {
    const pending = catalogs(target, season).catch((error) => {
      catalogCache.delete(key);
      throw error;
    });
    catalogCache.set(key, pending);
  }
  return catalogCache.get(key);
}

function skyToPixel(wcs, ra, dec) {
  const ra0 = (Number(wcs.CRVAL1) * Math.PI) / 180;
  const dec0 = (Number(wcs.CRVAL2) * Math.PI) / 180;
  const raR = (Number(ra) * Math.PI) / 180;
  const decR = (Number(dec) * Math.PI) / 180;
  const cosc = Math.sin(dec0) * Math.sin(decR) + Math.cos(dec0) * Math.cos(decR) * Math.cos(raR - ra0);
  if (!(cosc > 0)) return null;
  const xi = (Math.cos(decR) * Math.sin(raR - ra0) / cosc) * (180 / Math.PI);
  const eta = ((Math.cos(dec0) * Math.sin(decR) - Math.sin(dec0) * Math.cos(decR) * Math.cos(raR - ra0)) / cosc) * (180 / Math.PI);
  let fitsX;
  let fitsY;
  if (wcs.CD1_1 != null && wcs.CD2_2 != null) {
    const a = Number(wcs.CD1_1);
    const b = Number(wcs.CD1_2 || 0);
    const c = Number(wcs.CD2_1 || 0);
    const d = Number(wcs.CD2_2);
    const det = a * d - b * c;
    if (!det) return null;
    fitsX = Number(wcs.CRPIX1) + (d * xi - b * eta) / det;
    fitsY = Number(wcs.CRPIX2) + (-c * xi + a * eta) / det;
  } else {
    fitsX = Number(wcs.CRPIX1) + xi / Number(wcs.CDELT1);
    fitsY = Number(wcs.CRPIX2) + eta / Number(wcs.CDELT2);
  }
  return { x: fitsX - 1, y: Number(wcs.NAXIS2) - fitsY };
}

function layerName(id) {
  const key = { gaia: "layerGaia", catwise: "layerCatwise", simbad: "layerSimbad", skybot: "layerSkybot" }[id];
  return key ? t(key) : id;
}

function markerShape(catalog, x, y) {
  const node = document.createElementNS(SVG, "g");
  node.classList.add("marker", `marker-${catalog}`);
  node.dataset.catalog = catalog;
  node.setAttribute("transform", `translate(${x} ${y})`);
  let shape;
  if (catalog === "gaia") {
    shape = document.createElementNS(SVG, "circle");
    shape.setAttribute("r", "3.2");
  } else if (catalog === "catwise") {
    shape = document.createElementNS(SVG, "rect");
    shape.setAttribute("x", "-3");
    shape.setAttribute("y", "-3");
    shape.setAttribute("width", "6");
    shape.setAttribute("height", "6");
  } else if (catalog === "simbad") {
    shape = document.createElementNS(SVG, "polygon");
    shape.setAttribute("points", "0,-4.2 4.2,0 0,4.2 -4.2,0");
  } else {
    shape = document.createElementNS(SVG, "polygon");
    shape.setAttribute("points", "0,-4 3.6,3.2 -3.6,3.2");
  }
  node.append(shape);
  return node;
}

function markLayers() {
  const group = document.querySelector("#layers");
  if (!group) return;
  for (const button of group.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", layersOn.has(button.dataset.layer) ? "true" : "false");
  }
}

function paintCandidate(row) {
  const chip = document.querySelector("#candidate-chip");
  if (!chip) return;
  chip.hidden = !row;
  if (row) chip.textContent = `${t("candidateLabel")} ${row.name}`;
}

function showSource(row) {
  const readout = document.querySelector("#cnn-out");
  const notes = document.querySelector("#panel-notes");
  const name = document.querySelector("#panel-name");
  paintCandidate(row);
  if (!row) {
    if (notes) notes.hidden = true;
    if (name) name.hidden = true;
    return;
  }
  const mag = row.mag == null || row.mag === "" ? "—" : `${Number(row.mag).toFixed(1)} ${row.mag_band || ""}`.trim();
  if (readout) readout.textContent = t("catalogHit", { name: row.name, catalog: layerName(row.catalog), mag });
  if (name) {
    name.hidden = false;
    name.textContent = row.name;
  }
  if (!notes) return;
  notes.hidden = false;
  notes.replaceChildren();
  if (row.url) {
    const link = document.createElement("a");
    link.href = row.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = t("catalogLink");
    notes.append(link);
  } else {
    notes.textContent = t("catalogNoLink");
  }
}

function markerRules() {
  return guide?.targets?.find((item) => item.id === activeTarget)?.markers || null;
}

function magLimit(layer) {
  const rules = markerRules();
  const base = rules?.default_max_mag?.[layer];
  if (base == null) return null;
  const table = rules?.visible_max_mag?.[layer];
  let faint = Number(base);
  if (table) {
    const values = detector === 0 || detector === "color"
      ? Object.values(table).map(Number)
      : [Number(table[String(detector)] ?? base)];
    faint = Math.max(...values.filter(Number.isFinite), faint);
  }
  if (!(faint > base)) return Number(base);
  return Number(base) + (magReach / 100) * (faint - Number(base));
}

function rasterOf(picture) {
  if (picture._raster) return picture._raster;
  const canvas = document.createElement("canvas");
  canvas.width = picture.naturalWidth;
  canvas.height = picture.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(picture, 0, 0);
  picture._raster = context.getImageData(0, 0, canvas.width, canvas.height);
  return picture._raster;
}

function neutralPixel(raster, x, y) {
  const ix = Math.round(x);
  const iy = Math.round(y);
  if (ix < 0 || iy < 0 || ix >= raster.width || iy >= raster.height) return true;
  const index = (iy * raster.width + ix) * 4;
  const red = raster.data[index];
  const green = raster.data[index + 1];
  const blue = raster.data[index + 2];
  return Math.abs(red - 128) < 3 && Math.abs(green - 128) < 3 && Math.abs(blue - 128) < 3;
}

function checkProjection(wcs) {
  if (projectionSeen.has(activeTarget)) return;
  projectionSeen.add(activeTarget);
  const checks = markerRules()?.projection_check || [];
  for (const item of checks) {
    const pixel = skyToPixel(wcs, item.ra, item.dec);
    const dx = pixel ? Math.abs(pixel.x - item.png_x) : Infinity;
    const dy = pixel ? Math.abs(pixel.y - item.png_y) : Infinity;
    if (dx > 0.05 || dy > 0.05) console.warn("projection check", activeTarget, item, pixel);
  }
}

function trajectoryFor(target) {
  if (!pathCache.has(target)) {
    const pending = trajectory(target).catch((error) => {
      pathCache.delete(target);
      throw error;
    });
    pathCache.set(target, pending);
  }
  return pathCache.get(target);
}

function spectrumFor(target) {
  if (!spectrumCache.has(target)) {
    const pending = spectra(target).catch((error) => {
      spectrumCache.delete(target);
      throw error;
    });
    spectrumCache.set(target, pending);
  }
  return spectrumCache.get(target);
}

function drawSpectrum(rows) {
  const host = document.querySelector("#spectrum");
  const plot = document.querySelector("#spectrum-plot");
  if (!host || !plot) return;
  plot.replaceChildren();
  if (!rows?.length) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  const width = 280;
  const height = 110;
  const pad = 16;
  const xs = rows.map((row) => Number(row.wavelength_um));
  const ys = rows.map((row) => Number(row.flux_mjy));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const seasons = [...new Set(rows.map((row) => row.season))].sort((a, b) => a - b);
  const colors = ["#0B3D91", "#9BB6E6", "#FC3D21", "#6B7280", "#111827"];
  const xOf = (value) => pad + ((value - minX) / (maxX - minX || 1)) * (width - pad * 2);
  const yOf = (value) => height - pad - ((value - minY) / (maxY - minY || 1)) * (height - pad * 2);
  for (const season of seasons) {
    const color = colors[seasons.indexOf(season) % colors.length];
    for (const row of rows.filter((item) => item.season === season)) {
      const hollow = row.flag === "saturated" || row.flag === "poor_fit";
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("cx", String(xOf(row.wavelength_um)));
      dot.setAttribute("cy", String(yOf(row.flux_mjy)));
      dot.setAttribute("r", "2.4");
      dot.setAttribute("fill", hollow ? "none" : color);
      dot.setAttribute("stroke", color);
      dot.setAttribute("stroke-width", "1.2");
      plot.append(dot);
    }
  }
}

function framesFor(target) {
  if (!frameCache.has(target)) {
    const pending = frames(target).catch((error) => {
      frameCache.delete(target);
      throw error;
    });
    frameCache.set(target, pending);
  }
  return frameCache.get(target);
}

function drawLightCurve(rows, exposures) {
  const host = document.querySelector("#lightcurve");
  const plot = document.querySelector("#lightcurve-plot");
  if (!host || !plot) return;
  plot.replaceChildren();
  const byFrame = new Map((exposures || []).map((row) => [row.frame_id, row]));
  let points = (rows || []).map((row) => ({
    ...row,
    detector: byFrame.get(row.frame_id)?.detector,
    when: Date.parse(stamp(row.date_utc)),
    flux: Number(row.flux_mjy),
  })).filter((row) => Number.isFinite(row.when) && Number.isFinite(row.flux));
  if (typeof detector === "number" && detector >= 1 && detector <= 6) {
    const matched = points.filter((row) => Number(row.detector) === detector);
    if (matched.length) points = matched;
  }
  if (!points.length) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  const width = 280;
  const height = 110;
  const pad = 16;
  const xs = points.map((row) => row.when);
  const ys = points.map((row) => row.flux);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const seasons = [...new Set(points.map((row) => row.season))].sort((a, b) => a - b);
  const colors = ["#0B3D91", "#9BB6E6", "#FC3D21", "#6B7280", "#111827"];
  const xOf = (value) => pad + ((value - minX) / (maxX - minX || 1)) * (width - pad * 2);
  const yOf = (value) => height - pad - ((value - minY) / (maxY - minY || 1)) * (height - pad * 2);
  for (const season of seasons) {
    const color = colors[seasons.indexOf(season) % colors.length];
    for (const row of points.filter((item) => item.season === season)) {
      const hollow = row.flag != null && String(row.flag).trim() !== "";
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("cx", String(xOf(row.when)));
      dot.setAttribute("cy", String(yOf(row.flux)));
      dot.setAttribute("r", "2.4");
      dot.setAttribute("fill", hollow ? "none" : color);
      dot.setAttribute("stroke", color);
      dot.setAttribute("stroke-width", "1.2");
      plot.append(dot);
    }
  }
}

function parseRa(text) {
  const raw = String(text || "").trim().replace(/−/g, "-");
  if (!raw) return null;
  const sexagesimal = raw.match(/^([+-]?\d+(?:\.\d+)?)[h:](\d+(?:\.\d+)?)[m:](\d+(?:\.\d+)?)s?$/i);
  if (sexagesimal) {
    const sign = sexagesimal[1].startsWith("-") ? -1 : 1;
    const hours = Math.abs(Number(sexagesimal[1])) + Number(sexagesimal[2]) / 60 + Number(sexagesimal[3]) / 3600;
    return sign * hours * 15;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function parseDec(text) {
  const raw = String(text || "").trim().replace(/−/g, "-").replace(/º/g, "°");
  if (!raw) return null;
  const sexagesimal = raw.match(/^([+-]?\d+(?:\.\d+)?)[d°:](\d+(?:\.\d+)?)[m′'](\d+(?:\.\d+)?)[s″"]?$/i);
  if (sexagesimal) {
    const sign = sexagesimal[1].startsWith("-") ? -1 : 1;
    return sign * (Math.abs(Number(sexagesimal[1])) + Number(sexagesimal[2]) / 60 + Number(sexagesimal[3]) / 3600);
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function separationDeg(ra1, dec1, ra2, dec2) {
  const toRad = Math.PI / 180;
  const d1 = Number(dec1) * toRad;
  const d2 = Number(dec2) * toRad;
  const cosc = Math.sin(d1) * Math.sin(d2) + Math.cos(d1) * Math.cos(d2) * Math.cos((Number(ra1) - Number(ra2)) * toRad);
  return (Math.acos(Math.min(1, Math.max(-1, cosc))) * 180) / Math.PI;
}

function fieldFor(ra, dec) {
  let best = null;
  let bestSep = Infinity;
  for (const row of goldTargets.values()) {
    const half = (Number(row.size_deg) || 15 / 60) / 2;
    const sep = separationDeg(ra, dec, row.ra_now, row.dec_now);
    if (sep <= half + 1e-9 && sep < bestSep) {
      best = row;
      bestSep = sep;
    }
  }
  return best;
}

function pixelOf(point) {
  const placed = goldTargets.get(activeTarget);
  if (!placed?.wcs_json || !point) return null;
  const field = fieldFor(point.ra, point.dec);
  if (!field || field.target_id !== activeTarget) return null;
  let wcs = placed.wcs_json;
  if (typeof wcs === "string") wcs = JSON.parse(wcs);
  return skyToPixel(wcs, point.ra, point.dec);
}

function markFov() {
  const group = document.querySelector("#fov");
  if (!group) return;
  for (const button of group.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", Number(button.dataset.fov) === fovArcmin ? "true" : "false");
  }
}

function paintSearch() {
  const note = document.querySelector("#search-note");
  if (!note) return;
  if (searchState === "bad") note.textContent = t("searchBad");
  else if (searchState === "miss") note.textContent = t("searchMiss");
  else if (searchState === "hit") {
    const placed = goldTargets.get(activeTarget);
    note.textContent = t("searchHit", { name: placed?.name || activeTarget });
  } else note.textContent = "";
}

function applyFrame() {
  const sky = document.querySelector("#sky");
  const stage = document.querySelector("#sky-stage");
  if (!sky || !stage) return;
  const width = image.naturalWidth || 146;
  const height = image.naturalHeight || 146;
  let fx = 0.5;
  let fy = 0.5;
  const pixel = pixelOf(searchPoint);
  if (pixel && width && height) {
    fx = pixel.x / width;
    fy = pixel.y / height;
  }
  const zoom = 15 / fovArcmin;
  const box = sky.clientWidth || stage.parentElement?.clientWidth || 0;
  const tall = sky.clientHeight || box * 0.75;
  if (!box) return;
  const size = box * zoom;
  stage.style.transform = "none";
  stage.style.width = `${size}px`;
  stage.style.height = `${size}px`;
  stage.style.left = `${box / 2 - fx * size}px`;
  stage.style.top = `${tall / 2 - fy * size}px`;
}

function drawCross() {
  const old = marks.querySelector(".search-cross");
  if (old) old.remove();
  const pixel = pixelOf(searchPoint);
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  if (!pixel || !width || !height) return;
  if (pixel.x < 0 || pixel.y < 0 || pixel.x > width || pixel.y > height) return;
  const arm = Math.max(width, height) * 0.035;
  const node = document.createElementNS(SVG, "g");
  node.setAttribute("class", "search-cross");
  node.setAttribute("transform", `translate(${pixel.x} ${pixel.y})`);
  const horizontal = document.createElementNS(SVG, "line");
  horizontal.setAttribute("x1", String(-arm));
  horizontal.setAttribute("x2", String(arm));
  const vertical = document.createElementNS(SVG, "line");
  vertical.setAttribute("y1", String(-arm));
  vertical.setAttribute("y2", String(arm));
  node.append(horizontal, vertical);
  marks.append(node);
}

function writeUrl(frame) {
  const url = new URL(location.href);
  url.searchParams.set("target", activeTarget);
  url.searchParams.set("detector", String(detector));
  url.searchParams.set("season", seasonToken(frame));
  url.searchParams.set("mode", mode);
  url.searchParams.set("layers", LAYERS.filter((id) => layersOn.has(id)).join(","));
  if (searchPoint) {
    url.searchParams.set("ra", String(searchPoint.ra));
    url.searchParams.set("dec", String(searchPoint.dec));
    url.searchParams.set("fov", String(fovArcmin));
  } else {
    url.searchParams.delete("ra");
    url.searchParams.delete("dec");
    url.searchParams.delete("fov");
  }
  url.searchParams.delete("region");
  url.searchParams.delete("frame");
  history.replaceState(null, "", url);
}

function drawCatalog(rows, picture) {
  marks.replaceChildren();
  const width = picture?.naturalWidth;
  const height = picture?.naturalHeight;
  const placed = goldTargets.get(activeTarget);
  if (!width || !height || !placed?.wcs_json) {
    listedSources = [];
    catalogContext = null;
    return;
  }
  let wcs = placed.wcs_json;
  if (typeof wcs === "string") wcs = JSON.parse(wcs);
  checkProjection(wcs);
  const raster = rasterOf(picture);
  marks.setAttribute("viewBox", `0 0 ${width} ${height}`);
  marks.setAttribute("preserveAspectRatio", "none");
  catalogContext = { wcs, width, height, raster };
  listedSources = rows.filter((row) => {
    if (!layersOn.has(row.catalog)) return false;
    const limit = magLimit(row.catalog);
    return limit == null || row.mag == null || row.mag === "" || Number(row.mag) <= limit;
  });
  paintChosen();
}

function paintChosen() {
  for (const marker of marks.querySelectorAll(".marker")) marker.remove();
  if (!selectedSource || !catalogContext) return;
  const pixel = skyToPixel(catalogContext.wcs, selectedSource.ra, selectedSource.dec);
  const { width, height } = catalogContext;
  if (!pixel || pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height) return;
  if (neutralPixel(catalogContext.raster, pixel.x, pixel.y)) return;
  const node = markerShape(selectedSource.catalog, pixel.x, pixel.y);
  node.classList.add("selected");
  marks.append(node);
  const chip = document.querySelector("#candidate-chip");
  const stage = document.querySelector("#sky-stage");
  if (!chip || !stage) return;
  const x = stage.offsetLeft + (pixel.x / width) * stage.offsetWidth;
  const y = stage.offsetTop + (pixel.y / height) * stage.offsetHeight;
  chip.style.left = `${Math.max(8, x - 8)}px`;
  chip.style.top = `${Math.max(8, y - 34)}px`;
  markCompare("#cmp-now-marks", "found");
  markCompare("#cmp-then-marks", "absent");
}

function chooseSource(event) {
  if (!catalogContext || !listedSources.length) return;
  const matrix = marks.getScreenCTM();
  if (!matrix) return;
  const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
  let best = null;
  let bestDistance = Infinity;
  for (const row of listedSources) {
    const pixel = skyToPixel(catalogContext.wcs, row.ra, row.dec);
    if (!pixel) continue;
    const distance = Math.hypot(pixel.x - point.x, pixel.y - point.y);
    if (distance < bestDistance) {
      best = row;
      bestDistance = distance;
    }
  }
  const limit = Math.max(12, catalogContext.width * 0.08);
  if (!best || bestDistance > limit) return;
  selectedSource = best;
  showSource(best);
  paintChosen();
}

async function render() {
  const frame = sequence[index];
  if (!frame) return;
  const token = ++viewToken;
  const pair = mode === "blink" ? blinkPair() : [frame];
  const pictures = await Promise.all(pair.map((row) => loadImage(pngUrl(row)).catch(() => null)));
  if (token !== viewToken) return;
  const picture = pictures[0];
  if (!picture) return;
  const sameBlink = mode === "blink" && blinkTimer && blinkKey === pairKey(pair) && !blinkPaused;
  if (!sameBlink) image.src = picture.src;
  const target = targetOf();
  const placed = goldTargets.get(activeTarget);
  const shown = shownDetector(frame);
  const name = placed?.name || target.name;
  image.alt = altFor(frame, name);
  if (caption) caption.textContent = captionFor(frame);
  const first = sequence[0];
  const last = sequence[sequence.length - 1];
  const start = first && mode === "difference" ? (sumAt(first.detector, first.season_a) || first) : first;
  const end = last && mode === "difference" ? (sumAt(last.detector, last.season_b) || last) : last;
  put("#tick-start", start && when(start) ? fecha(when(start)) : "—");
  put("#tick-end", end && when(end) ? fecha(when(end)) : "—");
  showNotes(frame);
  const readout = document.querySelector("#cnn-out");
  slider.value = String(index);
  slider.max = String(Math.max(sequence.length - 1, 0));
  document.querySelector("#sector").textContent = name;
  document.querySelector("#sky-ra").textContent = hms(placed?.ra_now ?? target.ra);
  document.querySelector("#sky-dec").textContent = dms(placed?.dec_now ?? target.dec);
  fillEpochs();
  writeUrl(frame);
  fillPanel();
  fillCompare();
  markDetectors();
  markModes();
  markLayers();
  const season = catalogSeason(frame);
  try {
    const sources = season == null ? [] : await catalogFor(activeTarget, season);
    if (token !== viewToken) return;
    if (selectedSource && !sources.some((row) => row.catalog === selectedSource.catalog && row.name === selectedSource.name)) {
      selectedSource = null;
    } else if (selectedSource) {
      selectedSource = sources.find((row) => row.catalog === selectedSource.catalog && row.name === selectedSource.name) || null;
    }
    drawCatalog(sources, picture);
    drawCross();
    if (selectedSource) showSource(selectedSource);
    else {
      showSource(null);
      if (activeTarget === "wise_0855" && shown >= 1 && shown <= 4) readout.textContent = t("wiseNeighbor");
      else if (activeTarget === "wise_0855") readout.textContent = t("wiseGuide");
      else readout.textContent = t("cnnIdle");
    }
  } catch (error) {
    console.error(error);
    marks.replaceChildren();
    drawCross();
  }
  try {
    const [points, exposures, path] = await Promise.all([
      spectrumFor(activeTarget),
      framesFor(activeTarget),
      trajectoryFor(activeTarget),
    ]);
    if (token !== viewToken) return;
    spectrumRows = points;
    pathRows = path;
    drawSpectrum(points);
    drawLightCurve(points, exposures);
    fillCompare();
  } catch (error) {
    console.error(error);
    drawSpectrum([]);
    drawLightCurve([], []);
  }
  applyFrame();
  markFov();
  paintSearch();
  armBlink(pair, pictures);
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
  if (mode === "blink") return;
  if (playButton) playButton.textContent = playing ? t("pause") : t("play");
  if (playing) timer = setInterval(() => move(1), 900);
}

function togglePlay() {
  if (mode === "blink") {
    blinkPaused = !blinkPaused;
    render();
    return;
  }
  setPlaying(!timer);
}

function applyView(season) {
  if (!modeAvailable(mode)) mode = modeAvailable("sum") ? "sum" : mode;
  sequence = sequenceFor(mode);
  index = indexFor(sequence, season);
  slider.max = String(Math.max(sequence.length - 1, 0));
  markDetectors();
  markModes();
  restartPreview();
  return render();
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
  put("#gif-duration", t("duration", { seconds: seconds.toFixed(1) }));
}

function restartPreview() {
  clearInterval(gifTimer);
  gifFrames = sample(sequence, gifCount);
  gifStep = 0;
  gifBlob = null;
  const save = document.querySelector("#gif-save");
  if (!save) return;
  save.disabled = true;
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
  if (!host) return;
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
    const info = guide?.targets?.find((item) => item.id === target.id);
    const placed = goldTargets.get(target.id);
    const text = document.createElement("p");
    text.textContent = info?.what_to_look_at || t(`blurb_${target.id}`);
    const coords = document.createElement("p");
    coords.className = "card-coords";
    const ra = placed?.ra_now ?? info?.facts?.ra_now ?? target.ra;
    const dec = placed?.dec_now ?? info?.facts?.dec_now ?? target.dec;
    coords.textContent = `RA ${hms(ra)}   Dec ${dms(dec)}`;
    const look = document.createElement("p");
    look.className = "card-look";
    look.textContent = info?.how_to_look?.text || "";
    const kindKey = { motion: "kindMotion", spectrum: "kindSpectrum", brightness: "kindBrightness" }[info?.kind];
    const kind = document.createElement("span");
    if (kindKey) {
      kind.className = "tag indigo";
      kind.textContent = t(kindKey);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = active ? "amber wide" : "soft wide";
    button.textContent = active ? t("viewing") : t("loadTarget");
    button.addEventListener("click", () => {
      searchPoint = null;
      searchState = "";
      activeTarget = target.id;
      const season = openingSeason(target.id);
      openRegion(target.id, {
        scroll: true,
        detector: openingDetector(target.id),
        mode: openingMode(target.id),
        season,
      });
    });
    card.append(head, ...(kindKey ? [kind] : []), text, coords, look, button);
    host.append(card);
  }
}

function renderCatalog() {
  const rows = document.querySelector("#rows");
  const status = document.querySelector("#catalog-status");
  if (!rows || !status) return;
  rows.replaceChildren();
  status.textContent = t("catalogLater");
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
  stopBlink();
  if (!goldTargets.has(targetId) && !TARGETS.some((item) => item.id === targetId)) return;
  activeTarget = targetId;
  const [rows, diffs] = await Promise.all([sums(targetId), differences(targetId)]);
  if (token !== loadToken) return;
  sumRows = rows;
  diffRows = diffs;
  hits = [];
  byObject = new Map();
  selectedId = "";
  selectedSource = null;
  marks.replaceChildren();
  detector = options.detector ?? detector;
  if (options.mode) mode = parseMode(options.mode);
  blinkPaused = false;
  renderRegions();
  await applyView(options.season);
  if (options.scroll) document.querySelector("#visor").scrollIntoView({ behavior: "smooth" });
}

function choose(group, pressed) {
  for (const button of group.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", button === pressed ? "true" : "false");
  }
}

function csvCell(value) {
  if (value == null) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadRows(filename, rows) {
  if (!rows?.length) return false;
  const keys = Object.keys(rows[0]);
  const lines = [keys.join(",")];
  for (const row of rows) lines.push(keys.map((key) => csvCell(row[key])).join(","));
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1500);
  return true;
}

function setFov(value) {
  fovArcmin = value;
  markFov();
  applyFrame();
  if (sequence[index]) writeUrl(sequence[index]);
}

document.querySelector("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const ra = parseRa(document.querySelector("#search-ra").value);
  const dec = parseDec(document.querySelector("#search-dec").value);
  if (ra == null || dec == null) {
    searchState = "bad";
    searchPoint = null;
    paintSearch();
    return;
  }
  searchPoint = { ra, dec };
  const field = fieldFor(ra, dec);
  if (!field) {
    searchState = "miss";
    paintSearch();
    if (sequence[index]) writeUrl(sequence[index]);
    return;
  }
  searchState = "hit";
  await openRegion(field.target_id, { scroll: true });
});
document.querySelector("#fov").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  const value = Number(button?.dataset.fov);
  if (![3, 10, 15].includes(value)) return;
  fovArcmin = value;
  markFov();
  applyFrame();
  if (sequence[index]) writeUrl(sequence[index]);
});
playButton?.addEventListener("click", () => togglePlay());
slider.addEventListener("input", () => {
  setPlaying(false);
  index = Number(slider.value);
  render();
});
marks.addEventListener("click", chooseSource);
document.querySelector("#sky-search")?.addEventListener("click", () => {
  const form = document.querySelector("#search-form");
  if (!form) return;
  form.hidden = !form.hidden;
  if (!form.hidden) document.querySelector("#search-ra")?.focus();
});
document.querySelector("#fit-field")?.addEventListener("click", () => setFov(15));
document.querySelector("#cycle-fov")?.addEventListener("click", () => {
  const order = [15, 10, 3];
  setFov(order[(Math.max(order.indexOf(fovArcmin), 0) + 1) % order.length]);
});
document.querySelector("#compare-images")?.addEventListener("click", () => {
  const box = document.querySelector("#comparador");
  if (!box) return;
  box.hidden = false;
  fillCompare();
  box.scrollIntoView({ behavior: "smooth", block: "start" });
});
document.querySelector("#csv")?.addEventListener("click", () => {
  const rows = selectedSource ? [selectedSource] : listedSources;
  if (!downloadRows(`${activeTarget}-catalog.csv`, rows)) put("#cnn-out", t("csvEmpty"));
});
document.querySelector("#csv-pair")?.addEventListener("click", async () => {
  const frame = sequence[index];
  const baseline = sequence[0];
  if (!frame) {
    put("#pair-note", t("csvEmpty"));
    return;
  }
  const current = catalogSeason(frame);
  const base = baseline?.season ?? current;
  try {
    const [left, right] = await Promise.all([
      catalogs(activeTarget, base),
      catalogs(activeTarget, current),
    ]);
    const savedLeft = downloadRows(`${activeTarget}-baseline-s${base}.csv`, left);
    const savedRight = downloadRows(`${activeTarget}-current-s${current}.csv`, right);
    put("#pair-note", savedLeft || savedRight ? "" : t("csvEmpty"));
  } catch (error) {
    put("#pair-note", t("csvEmpty"));
    console.error(error);
  }
});
document.querySelector("#mag-reach")?.addEventListener("input", (event) => {
  magReach = Number(event.target.value);
  render();
});
document.querySelector("#layers")?.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const id = button.dataset.layer;
  if (!LAYERS.includes(id)) return;
  if (layersOn.has(id)) layersOn.delete(id);
  else layersOn.add(id);
  markLayers();
  render();
});
document.querySelector("#modes")?.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || button.disabled) return;
  const next = button.dataset.mode;
  if (!next || next === mode) return;
  setPlaying(false);
  const season = mode === "difference" && next !== "difference" ? holdSeason() : seasonToken();
  mode = next;
  blinkPaused = false;
  applyView(season);
});
document.querySelector("#detectors")?.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || button.disabled) return;
  const value = button.dataset.detector;
  const next = value === "color" ? "color" : Number(value);
  if (next === detector) return;
  setPlaying(false);
  let season = seasonToken();
  if ((next === "color" || next === 0) && mode === "difference") {
    season = holdSeason();
    mode = "sum";
  }
  if (next === "color" && mode === "all_seasons") mode = "sum";
  detector = next;
  blinkPaused = false;
  applyView(season);
});
discardedBox?.addEventListener("change", () => render());
document.querySelector("#blink")?.addEventListener("click", () => {
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
document.querySelector("#blink-jump")?.addEventListener("click", () => {
  document.querySelector("#comparador").scrollIntoView({ behavior: "smooth" });
  if (!blinkOn) document.querySelector("#blink").click();
});
document.querySelector("#blink-speed")?.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  choose(document.querySelector("#blink-speed"), button);
  blinkMs = Number(button.dataset.ms);
  if (blinkOn) {
    document.querySelector("#blink").click();
    document.querySelector("#blink").click();
  }
});
document.querySelector("#gif-count")?.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  choose(document.querySelector("#gif-count"), button);
  gifCount = Number(button.dataset.n);
  restartPreview();
});
document.querySelector("#gif-speed")?.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  choose(document.querySelector("#gif-speed"), button);
  gifMs = Number(button.dataset.ms);
  restartPreview();
});
document.querySelector("#gif-make")?.addEventListener("click", () => makeGif());
document.querySelector("#gif-save")?.addEventListener("click", () => saveGif());
document.querySelector("#analyze").addEventListener("click", () => {
  const out = document.querySelector("#cnn-out");
  if (selectedSource) {
    showSource(selectedSource);
    const score = Number(selectedSource.p_real);
    if (Number.isFinite(score)) out.textContent += t("score", { value: (score * 100).toFixed(1) });
    return;
  }
  const object = byObject.get(selectedId);
  if (!object) {
    out.textContent = t("pickMarker");
    return;
  }
  const score = Number(object.p_real);
  const scored = Number.isFinite(score) ? t("score", { value: (score * 100).toFixed(1) }) : t("noScore");
  out.textContent = `${object.name}: ${statusOf(object).label}.${scored} ${reasonOf(object)}`;
});
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
    togglePlay();
  }
});

apply();
onChange(() => {
  apply();
  renderRegions();
  if (regionList.length) renderCatalog();
  if (playButton) playButton.textContent = mode === "blink" && !blinkPaused ? t("pause") : timer ? t("pause") : t("play");
  put("#blink", blinkOn ? t("blinkPause") : t("blink"));
  if (sequence.length) {
    updateGifLabel();
    const available = document.querySelector("#gif-available");
    if (available) available.textContent = t("captures", { n: sequence.length });
    render();
  }
});

try {
  const params = new URLSearchParams(location.search);
  const [found, info, texts] = await Promise.all([targets(), meta(), loadGuide()]);
  regionList = found;
  guide = texts;
  goldTargets.clear();
  for (const row of found) goldTargets.set(row.target_id, row);
  const askedRa = parseRa(params.get("ra"));
  const askedDec = parseDec(params.get("dec"));
  const askedFov = Number(params.get("fov"));
  if ([3, 10, 15].includes(askedFov)) fovArcmin = askedFov;
  if (params.get("ra")) document.querySelector("#search-ra").value = params.get("ra");
  if (params.get("dec")) document.querySelector("#search-dec").value = params.get("dec");
  let wanted = params.get("target") || params.get("region") || "barnard";
  if (askedRa != null && askedDec != null) {
    searchPoint = { ra: askedRa, dec: askedDec };
    const field = fieldFor(askedRa, askedDec);
    if (field) {
      searchState = "hit";
      wanted = field.target_id;
    } else searchState = "miss";
  }
  renderCatalog();
  const match = found.find((row) => row.target_id === wanted) || found[0];
  if (match) activeTarget = match.target_id;
  detector = params.has("detector")
    ? parseDetector(params.get("detector"), activeTarget)
    : openingDetector(activeTarget);
  mode = params.has("mode") ? parseMode(params.get("mode")) : openingMode(activeTarget);
  layersOn = parseLayers(params.has("layers") ? params.get("layers") : null);
  if (!params.has("season")) params.set("season", openingSeason(activeTarget));
  put("#updated", info?.run_date_utc
    ? t("tablesLoaded", { date: info.run_date_utc })
    : t("tablesLoaded", { date: "—" }));
  if (match) await openRegion(match.target_id, { detector, season: params.get("season"), mode });
  if (location.hash) {
    const place = document.querySelector(location.hash);
    if (place) place.scrollIntoView();
  }
} catch (error) {
  put("#updated", t("tablesDown"));
  put("#catalog-status", t("tablesDown"));
  put("#cnn-out", t("tablesDown"));
  console.error(error);
}
