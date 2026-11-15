import { detections, frames, meta, objects, regions } from "./data.js";

const ARCSEC_PER_PIXEL = 6.15;
const SVG = "http://www.w3.org/2000/svg";
const STATUS = {
  Known: { label: "Objeto conocido", tag: "green" },
  "Unidentified candidate": { label: "Candidato sin identificar", tag: "amber" },
  Discarded: { label: "Descartado", tag: "" },
};
const TYPE = {
  asteroid: "Asteroide",
  "variable star": "Estrella variable",
  artifact: "Artefacto",
  unknown: "Sin clasificar",
};
const REASON = {
  "Within 2 px of a SkyBoT object on that date": "Coincide con un objeto de SkyBoT en esa fecha.",
  "Within 2 px of a SIMBAD variable star (V*)": "Coincide con una estrella variable de SIMBAD.",
  "No match in SkyBoT, SIMBAD or Gaia": "Sin coincidencia en SkyBoT, SIMBAD o Gaia.",
  "Touches flagged pixels or the edge": "Toca píxeles marcados o el borde del recorte.",
  "Seen in a single exposure": "Aparece en una sola exposición.",
};
const TITLE = {
  pole: "Polo Norte Eclíptico",
  asteroid: "Región asteroide",
  variable_star: "Región estrella variable",
};
const BLURB = {
  pole: "Zona de monitoreo constante de cielo profundo. Ideal para detectar cambios a largo plazo.",
  asteroid: "Búsqueda de cuerpos menores en movimiento. Pensada para asteroides que cruzan el campo.",
  variable_star: "Detección de fluctuaciones de brillo, novas y estrellas pulsantes.",
};
const ICON = { pole: "🌌", asteroid: "☄️", variable_star: "⭐" };
const SKY = {
  pole: ["ngc-stars.jpg", "ngc-dust.jpg", "cygnus.jpg"],
  asteroid: ["cygnus.jpg", "vela.jpg", "ice.jpg"],
  variable_star: ["ice.jpg", "vela.jpg", "ngc-dust.jpg"],
};
const CARD = { pole: "ngc-stars.jpg", asteroid: "cygnus.jpg", variable_star: "ice.jpg" };
const FRAME_PX = 293;
const TARGETS = [
  { id: "barnard", name: "Barnard's Star", icon: "✨", ra: 269.452077, dec: 4.693365, blurb: "Estrella cercana de gran movimiento propio visible a través de épocas. Ideal para medir desplazamientos temporales rápidos." },
  { id: "eps_indi_b", name: "Epsilon Indi B", icon: "🪐", ra: 331.0441466, dec: -56.7828442, blurb: "Sistema binario de enanas marrones frías. Excelente objetivo para comparar firmas espectrales y cinemática local." },
  { id: "wise_0855", name: "WISE 0855", icon: "❄️", ra: 133.8189082, dec: -7.247048, blurb: "Objeto frío subestelar y enana marrón de tipo Y. Detectable casi exclusivamente en observaciones infrarrojas profundas." },
  { id: "rw_aur", name: "RW Aurigae", icon: "☄️", ra: 76.956526, dec: 30.4014382, blurb: "Sistema estelar binario joven con disco circunestelar perturbado, mostrando oscurecimientos periódicos y eventos dinámicos." },
  { id: "r_dra", name: "R Draconis", icon: "⭐", ra: 248.1676108, dec: 66.7549794, blurb: "Estrella variable pulsante de largo período tipo Mira con marcadas oscilaciones de magnitud óptica e infrarroja." },
];
let activeTarget = "barnard";
const format = new Intl.NumberFormat("es");

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
const cache = new Map();

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
  return new Intl.DateTimeFormat("es", {
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

function skyFor(frame, frameIndex) {
  if (!frame) return "";
  return assetUrl(skyFile(frame.region_id, frameIndex));
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
  return TITLE[id] || regionName(id);
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
  document.querySelector("#stat-known").textContent = format.format(now.known);
  document.querySelector("#stat-candidate").textContent = format.format(now.candidate);
  document.querySelector("#stat-discarded").textContent = format.format(now.discarded);
  document.querySelector("#stat-frames").textContent = format.format(sequence.length);
  document.querySelector("#base-known").textContent = format.format(base.known);
  document.querySelector("#base-candidate").textContent = format.format(base.candidate);
  document.querySelector("#base-discarded").textContent = format.format(base.discarded);
  document.querySelector("#base-frames").textContent = "1";
  document.querySelector("#epoch-now").textContent = fecha(frame.date_utc);
  document.querySelector("#epoch-then").textContent = fecha(baseline.date_utc);
  const delta = now.candidate - base.candidate;
  document.querySelector("#delta-candidates").textContent =
    `${delta >= 0 ? "+" : ""}${delta} candidatos entre las dos fechas`;
  document.querySelector("#delta-span").textContent = `Intervalo temporal: ${intervalo(baseline, frame)}`;
  document.querySelector("#updated").textContent =
    `Diferencia: ${intervalo(baseline, frame)} · ${targetOf().name}`;
}

function statusOf(object) {
  return STATUS[object?.status] || { label: object?.status || "—", tag: "" };
}

function reasonOf(object) {
  if (!object?.reason) return "";
  return REASON[object.reason] || object.reason;
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
  const col = hit.x_px < width / 3 ? "izquierda" : hit.x_px > (2 * width) / 3 ? "derecha" : "central";
  const row = hit.y_px < height / 3 ? "superior" : hit.y_px > (2 * height) / 3 ? "inferior" : "central";
  if (col === "central" && row === "central") return "Centro";
  return `${row} ${col}`;
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
  const ms = Math.abs(new Date(stamp(toFrame.date_utc)) - new Date(stamp(fromFrame.date_utc)));
  if (!Number.isFinite(ms)) return "—";
  const hours = ms / 36e5;
  if (hours < 48) return `${Math.max(1, Math.round(hours))} h`;
  return `${Math.round(hours / 24)} días`;
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
  document.querySelector("#panel-date").textContent = frame ? fecha(frame.date_utc) : "—";
  document.querySelector("#panel-region").textContent = regionName(frame?.region_id);
  const wave = frame ? `${frame.wavelength_um} µm` : "—";
  document.querySelector("#panel-wave").textContent = wave;
  document.querySelector("#wave-read").textContent = wave;
  document.querySelector("#panel-now").textContent = frame ? fecha(frame.date_utc) : "—";
  document.querySelector("#panel-then").textContent = baseline ? fecha(baseline.date_utc) : "—";
  document.querySelector("#panel-span").textContent =
    frame && baseline ? `Intervalo: ${intervalo(baseline, frame)}` : "Intervalo: —";
  if (!object) {
    setTag(status, "Selecciona un marcador", "amber");
    document.querySelector("#panel-name").textContent = "Ningún objeto elegido";
    document.querySelector("#panel-type").textContent = "—";
    prob.hidden = true;
    document.querySelector("#panel-notes").replaceChildren("Los detalles aparecen al elegir un marcador del visor.");
    document.querySelector("#panel-move").textContent = "—";
    document.querySelector("#move-read").textContent = "—";
    document.querySelector("#panel-place").textContent = "—";
    document.querySelector("#place-read").textContent = "—";
    return;
  }
  const info = statusOf(object);
  setTag(status, info.label, info.tag);
  document.querySelector("#panel-name").textContent = object.name;
  document.querySelector("#panel-type").textContent = TYPE[object.type] || object.type || "—";
  if (typeof object.p_real === "number") {
    prob.hidden = false;
    prob.textContent = `${(object.p_real * 100).toFixed(1)}% p_real`;
  } else {
    prob.hidden = true;
  }
  const notes = document.querySelector("#panel-notes");
  notes.replaceChildren();
  const lines = [reasonOf(object)];
  if (object.catalog) lines.push(`Catálogo: ${object.catalog}.`);
  if (object.status === "Unidentified candidate") {
    lines.push("Sigue siendo un candidato: no se llama objeto nuevo.");
  }
  for (const line of lines.filter(Boolean)) {
    const span = document.createElement("span");
    span.textContent = line;
    notes.append(span);
  }
  const shift = frame && baseline ? movement(object, baseline, frame) : null;
  const moveText = shift == null ? "—" : `${shift.toFixed(1)}″`;
  document.querySelector("#panel-move").textContent = moveText;
  document.querySelector("#move-read").textContent = moveText;
  document.querySelector("#delta-move").textContent = shift == null ? "Sin desplazamiento medido" : `${shift.toFixed(1)}″ de desplazamiento`;
  const hit = frame ? detectionOn(object.object_id, frame.frame_id) : null;
  const where = hit ? place(hit) : "Fuera de esta foto";
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
  document.querySelector("#cmp-now-label").textContent = `Fotografía ${showingNow ? "actual" : "base"} (${fecha(left.date_utc)})`;
  document.querySelector("#cmp-then-label").textContent = `Fotografía comparativa (${fecha(baseline.date_utc)})`;

  const recentIds = new Set(hits.filter((hit) => hit.frame_id === frame.frame_id).map((hit) => hit.object_id));
  const baseIds = new Set(hits.filter((hit) => hit.frame_id === baseline.frame_id).map((hit) => hit.object_id));
  const newcomers = [...recentIds]
    .filter((id) => !baseIds.has(id))
    .map((id) => byObject.get(id))
    .filter(Boolean);
  const candidates = newcomers.filter((object) => object.status === "Unidentified candidate");
  const named = (candidates.length ? candidates : newcomers).map((object) => object.name);
  document.querySelector("#event-new").textContent = named.length
    ? `${named.join(", ")} ${named.length === 1 ? "aparece" : "aparecen"} en la exposición del visor y no en la base.`
    : "Ningún objeto de la exposición actual falta en la fotografía base.";
  document.querySelector("#event-new-foot").textContent = named.length ? "Visto en la foto reciente" : "Sin alta nueva";

  const shifts = [...byObject.values()]
    .map((object) => ({ object, arcsec: movement(object, baseline, frame) }))
    .filter((item) => item.arcsec != null);
  const biggest = shifts.sort((a, b) => b.arcsec - a.arcsec)[0];
  document.querySelector("#event-move").textContent = biggest
    ? `${biggest.object.name}: ${biggest.arcsec.toFixed(1)} segundos de arco entre la base y esta fecha.`
    : "Ningún objeto repetido tiene las dos posiciones medidas.";
  document.querySelector("#event-move-foot").textContent = biggest && biggest.arcsec >= 1 ? "Desplazamiento medido" : "Sin desplazamiento claro";

  document.querySelector("#event-diff").textContent = frame.png_diff
    ? "La diferencia medida sigue en las tablas. En pantalla, la casilla cambia a otra imagen de referencia de SPHEREx."
    : "Esta exposición no tiene imagen diferencia en las tablas.";
  document.querySelector("#event-diff-foot").textContent = frame.png_diff ? "Disponible" : "No publicada";
  const findings = [named.length > 0, Boolean(biggest && biggest.arcsec >= 1), Boolean(frame.png_diff)].filter(Boolean).length;
  document.querySelector("#cmp-count").textContent = `${findings} hallazgo${findings === 1 ? "" : "s"} entre las dos fechas`;
}

async function render() {
  const frame = sequence[index];
  if (!frame) return;
  const picture = await loadImage(skyFor(frame, index + (diffBox.checked ? 1 : 0)));
  image.src = picture.src;
  image.alt = `Imagen de referencia SPHEREx, ${titleOf(frame.region_id)}`;
  draw(frame);
  caption.textContent = `${stamp(frame.date_utc)} · ${frame.wavelength_um} µm · ${index + 1} / ${sequence.length}`;
  slider.value = String(index);
  const target = targetOf();
  document.querySelector("#sector").textContent = target.name;
  document.querySelector("#sky-ra").textContent = hms(target.ra);
  document.querySelector("#sky-dec").textContent = dms(target.dec);
  fillEpochs();
  const url = new URL(location.href);
  url.searchParams.set("region", frame.region_id);
  url.searchParams.set("frame", frame.frame_id);
  history.replaceState(null, "", url);
  fillPanel();
  fillCompare();
  loadImage(skyFor(sequence[(index + 1) % sequence.length], index + 1));
}

function move(step) {
  if (!sequence.length) return;
  index = (index + step + sequence.length) % sequence.length;
  render();
}

function setPlaying(playing) {
  clearInterval(timer);
  timer = 0;
  playButton.textContent = playing ? "Pausa" : "Reproducir";
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
  document.querySelector("#gif-duration").textContent =
    `Duración calculada: ${seconds.toFixed(1)} segundos (bucle)`;
}

function restartPreview() {
  clearInterval(gifTimer);
  gifFrames = sample(sequence, gifCount);
  gifStep = 0;
  gifBlob = null;
  document.querySelector("#gif-save").disabled = true;
  const available = document.querySelector("#gif-available");
  if (available) available.textContent = `${sequence.length} capturas`;
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
  note.textContent = "Generando el GIF…";
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
    note.textContent = "GIF listo para descargar.";
  } catch (error) {
    note.textContent = "No se pudo generar el GIF.";
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
      flag.textContent = "Región activa";
      card.append(flag);
    }
    const head = document.createElement("div");
    head.className = "card-title";
    const title = document.createElement("h3");
    title.textContent = `${target.icon} ${target.name}`;
    const badge = document.createElement("span");
    badge.className = active ? "tag amber" : "tag";
    badge.textContent = active ? "En foco" : "Disponible";
    head.append(title, badge);
    const text = document.createElement("p");
    text.textContent = target.blurb;
    const button = document.createElement("button");
    button.type = "button";
    button.className = active ? "amber wide" : "soft wide";
    button.textContent = active ? "Región en visualización" : "Cargar esta región";
    button.addEventListener("click", () => {
      activeTarget = target.id;
      const region = goldFor(target.id);
      if (region) openRegion(region.region_id, { scroll: true });
    });
    card.append(head, text, button);
    host.append(card);
  }
}

function renderCatalog() {
  const body = document.querySelector("#rows");
  body.replaceChildren();
  const names = new Map(regionList.map((region) => [region.region_id, region.name]));
  document.querySelector("#catalog-status").textContent =
    `${catalog.length} objetos en ${regionList.length} regiones.`;
  for (const row of catalog) {
    const info = statusOf(row);
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const wrap = document.createElement("span");
    wrap.className = "name-cell";
    const swatch = document.createElement("i");
    swatch.className = `swatch ${row.status === "Known" ? "known" : row.status === "Discarded" ? "discarded" : "candidate"}`;
    if (row.status === "Discarded") swatch.textContent = "×";
    const label = document.createElement("span");
    label.textContent = row.name;
    wrap.append(swatch, label);
    name.append(wrap);
    const state = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = `tag ${info.tag}`.trim();
    tag.textContent = info.label;
    state.append(tag);
    const where = document.createElement("td");
    where.textContent = names.get(row.region_id) || row.region_id;
    const when = document.createElement("td");
    when.textContent = fecha(row.last_date_utc || row.first_date_utc);
    const action = document.createElement("td");
    action.className = "right";
    const link = document.createElement("button");
    link.type = "button";
    link.className = "ghost";
    link.textContent = "Ver en visor";
    link.addEventListener("click", () => focusObject(row));
    action.append(link);
    tr.append(name, state, where, when, action);
    body.append(tr);
  }
}

async function focusObject(object) {
  if (sequence[0]?.region_id !== object.region_id) await openRegion(object.region_id);
  selectedId = object.object_id;
  const hit = hits.find((item) => item.object_id === object.object_id);
  if (hit) index = Math.max(0, sequence.findIndex((frame) => frame.frame_id === hit.frame_id));
  await render();
  document.querySelector("#visor").scrollIntoView({ behavior: "smooth" });
}

async function openRegion(regionId, options = {}) {
  const token = ++loadToken;
  setPlaying(false);
  const region = regionList.find((item) => item.region_id === regionId) || regionList[0];
  if (!region) return;
  const [frameRows, objectRows, detectionRows] = await Promise.all([
    frames(region.region_id),
    objects(region.region_id),
    detections(region.region_id),
  ]);
  if (token !== loadToken) return;
  sequence = frameRows;
  byObject = new Map(objectRows.map((row) => [row.object_id, row]));
  hits = detectionRows;
  slider.max = String(Math.max(sequence.length - 1, 0));
  const requested = options.frame
    ? sequence.findIndex((frame) => frame.frame_id === options.frame)
    : -1;
  if (requested >= 0) {
    index = requested;
  } else {
    const candidate = objectRows.find((row) => row.status === "Unidentified candidate");
    const hit = candidate && detectionRows.find((item) => item.object_id === candidate.object_id);
    index = hit ? Math.max(0, sequence.findIndex((frame) => frame.frame_id === hit.frame_id)) : 0;
    selectedId = candidate?.object_id || objectRows[0]?.object_id || "";
  }
  if (options.frame && !selectedId) selectedId = objectRows[0]?.object_id || "";
  renderRegions();
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
discardedBox.addEventListener("change", () => render());
document.querySelector("#blink").addEventListener("click", () => {
  blinkOn = !blinkOn;
  const button = document.querySelector("#blink");
  button.textContent = blinkOn ? "Pausar blink" : "Alternar / blink";
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
    out.textContent = "Elige un marcador del visor para analizarlo.";
    return;
  }
  const score = Number(object.p_real);
  const scored = Number.isFinite(score) ? ` p_real ${(score * 100).toFixed(1)}%.` : " Sin puntuación CNN todavía.";
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

try {
  const params = new URLSearchParams(location.search);
  const [found, info, groups] = await Promise.all([
    regions(),
    meta(),
    regions().then((list) => Promise.all(list.map((region) => objects(region.region_id)))),
  ]);
  regionList = found;
  catalog = groups.flat();
  renderCatalog();
  const wanted = params.get("region") || "barnard";
  if (TARGETS.some((item) => item.id === wanted)) activeTarget = wanted;
  const region = TARGETS.some((item) => item.id === wanted)
    ? goldFor(wanted)
    : found.find((item) => item.region_id === wanted) || found[0];
  await openRegion(region.region_id, { frame: params.get("frame") || "" });
  if (location.hash) {
    const target = document.querySelector(location.hash);
    if (target) target.scrollIntoView();
  }
} catch (error) {
  document.querySelector("#updated").textContent = "Las tablas publicadas no están disponibles.";
  console.error(error);
}
