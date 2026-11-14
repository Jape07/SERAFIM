import { objects, regions } from "./data.js";

const status = document.querySelector("#status");
const body = document.querySelector("#rows");

function text(value) {
  return value == null || value === "" ? "—" : String(value);
}

try {
  const found = await regions();
  const region = found[0];
  if (!region) {
    status.textContent = "No region in the published gold.";
  } else {
    const rows = await objects(region.region_id);
    status.textContent = `${rows.length} objects in ${region.name}.`;
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const value of [row.name, row.status, row.type, row.n_detections, row.reason]) {
        const td = document.createElement("td");
        td.textContent = text(value);
        tr.append(td);
      }
      body.append(tr);
    }
  }
} catch (error) {
  status.textContent = "The catalog could not be read.";
  console.error(error);
}
