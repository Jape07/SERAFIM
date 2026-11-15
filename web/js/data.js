// Reads the gold Parquet files in the browser. Paths stay relative to this file.

import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.32.0/+esm";

const FILES = [
  "targets.parquet",
  "frames.parquet",
  "sums.parquet",
  "differences.parquet",
  "catalogs.parquet",
  "spectra.parquet",
  "trajectory.parquet",
];
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

export function targets() {
  return query("SELECT * FROM 'targets.parquet' ORDER BY target_id");
}

export function frames(target) {
  return query(
    "SELECT * FROM 'frames.parquet' WHERE target_id = ? ORDER BY date_utc, frame_id",
    [target],
  );
}

export function sums(target) {
  return query(
    "SELECT * FROM 'sums.parquet' WHERE target_id = ? ORDER BY detector, season",
    [target],
  );
}

export function differences(target) {
  return query(
    "SELECT * FROM 'differences.parquet' WHERE target_id = ? ORDER BY detector, season_a, season_b",
    [target],
  );
}

export function catalogs(target, season) {
  return query(
    "SELECT * FROM 'catalogs.parquet' WHERE target_id = ? AND season = ? ORDER BY catalog, name",
    [target, season],
  );
}

export function spectra(target) {
  return query(
    "SELECT * FROM 'spectra.parquet' WHERE target_id = ? ORDER BY wavelength_um, date_utc",
    [target],
  );
}

export function trajectory(target) {
  return query(
    "SELECT * FROM 'trajectory.parquet' WHERE target_id = ? ORDER BY season",
    [target],
  );
}

export async function meta() {
  const url = new URL("../gold/meta.json", import.meta.url);
  const response = await fetch(url);
  if (!response.ok) throw new Error("Missing meta.json");
  return response.json();
}
