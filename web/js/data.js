// Reads the gold Parquet files in the browser. Paths stay relative to this file.

import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.32.0/+esm";

const FILES = ["regions.parquet", "frames.parquet", "detections.parquet", "objects.parquet"];
const TIMEOUT_MS = 60000;

let ready;

function cell(value) {
  if (typeof value === "bigint") return Number(value);
  if (value instanceof Date) return value.toISOString();
  return value;
}

async function rows(result) {
  const table = await result;
  const batch = table.toArray().map((row) => {
    const source = typeof row.toJSON === "function" ? row.toJSON() : row;
    const plain = {};
    for (const key of Object.keys(source)) plain[key] = cell(source[key]);
    return plain;
  });
  if (typeof table.close === "function") table.close();
  return batch;
}

async function open() {
  const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
  const workerUrl = URL.createObjectURL(
    new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" }),
  );
  const worker = new Worker(workerUrl);
  const db = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  URL.revokeObjectURL(workerUrl);
  const conn = await db.connect();
  for (const name of FILES) {
    const url = new URL(`../gold/${name}`, import.meta.url);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Missing ${name}`);
    await db.registerFileBuffer(name, new Uint8Array(await response.arrayBuffer()));
  }
  return conn;
}

function connection() {
  ready ??= open().catch((error) => {
    ready = undefined;
    throw error;
  });
  return Promise.race([
    ready,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error("The gold tables took too long to open.")), TIMEOUT_MS);
    }),
  ]);
}

async function query(sql, params = []) {
  const conn = await connection();
  const statement = await conn.prepare(sql);
  const result = await statement.query(...params);
  await statement.close();
  return rows(result);
}

export function regions() {
  return query("SELECT * FROM 'regions.parquet' ORDER BY region_id");
}

export function frames(region) {
  return query(
    "SELECT * FROM 'frames.parquet' WHERE region_id = ? ORDER BY date_utc, frame_id",
    [region],
  );
}

export function objects(region) {
  return query(
    "SELECT * FROM 'objects.parquet' WHERE region_id = ? ORDER BY name",
    [region],
  );
}

export function detections(region) {
  return query(
    `SELECT d.* FROM 'detections.parquet' d
     JOIN 'frames.parquet' f ON d.frame_id = f.frame_id
     WHERE f.region_id = ?`,
    [region],
  );
}

export async function meta() {
  const url = new URL("../gold/meta.json", import.meta.url);
  const response = await fetch(url);
  if (!response.ok) throw new Error("Missing meta.json");
  return response.json();
}
