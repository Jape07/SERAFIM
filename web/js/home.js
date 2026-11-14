import { meta, regions } from "./data.js";

const status = document.querySelector("#status");
const list = document.querySelector("#regions");

try {
  const [found, info] = await Promise.all([regions(), meta()]);
  status.textContent = info.n_frames
    ? `${info.n_frames} frames published (${info.run_date_utc}).`
    : "No frames published yet.";
  for (const region of found) {
    const card = document.createElement("article");
    card.className = "card";
    const title = document.createElement("h2");
    title.textContent = region.name;
    const place = document.createElement("p");
    place.textContent = `RA ${Number(region.ra).toFixed(2)}°, Dec ${Number(region.dec).toFixed(2)}°`;
    const link = document.createElement("a");
    link.className = "button";
    link.href = `viewer.html?region=${encodeURIComponent(region.region_id)}`;
    link.textContent = "Play the timeline";
    card.append(title, place, link);
    list.append(card);
  }
} catch (error) {
  status.textContent = "The published frames are not available yet.";
  console.error(error);
}
