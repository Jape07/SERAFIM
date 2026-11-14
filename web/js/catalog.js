import { objects, regions } from "./data.js";

const status = document.querySelector("#status");
const body = document.querySelector("#rows");

function text(value) {
  return value == null || value === "" ? "—" : String(value);
}

try {
  const found = await regions();
  const names = new Map(found.map((region) => [region.region_id, region.name]));
  const groups = await Promise.all(found.map((region) => objects(region.region_id)));
  const rows = groups.flat();
  status.textContent = found.length
    ? `${rows.length} objects across ${found.length} regions.`
    : "No region in the published gold.";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const values = [
      names.get(row.region_id) || row.region_id,
      row.name,
      row.status,
      row.type,
      row.n_detections,
      row.reason,
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = text(value);
      tr.append(td);
    }
    body.append(tr);
  }
} catch (error) {
  status.textContent = "The catalog could not be read.";
  console.error(error);
}
