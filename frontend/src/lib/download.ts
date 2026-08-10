import { resolveMediaUrl } from "./resolveMediaUrl";

export async function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function downloadFromUrl(url: string, filename: string) {
  const resolved = resolveMediaUrl(url) || url;
  const res = await fetch(resolved);
  if (!res.ok) throw new Error("Download failed");
  const blob = await res.blob();
  await downloadBlob(blob, filename);
}

export function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  return downloadBlob(blob, filename);
}

export function downloadText(text: string, filename: string) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  return downloadBlob(blob, filename);
}

/**
 * Minimal ZIP (store-only) builder — no external dependency.
 * Suitable for a small set of text/image files.
 */
export async function downloadZip(
  files: Array<{ name: string; blob: Blob }>,
  zipName: string
) {
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  const encoder = new TextEncoder();

  const crcTable = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c;
    }
    return table;
  })();

  const crc32 = (data: Uint8Array) => {
    let crc = 0xffffffff;
    for (let i = 0; i < data.length; i++) crc = crcTable[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  };

  const u16 = (n: number) => {
    const b = new Uint8Array(2);
    new DataView(b.buffer).setUint16(0, n, true);
    return b;
  };
  const u32 = (n: number) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, n, true);
    return b;
  };

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const data = new Uint8Array(await file.blob.arrayBuffer());
    const crc = crc32(data);
    const localHeader = new Uint8Array([
      ...u32(0x04034b50),
      ...u16(20),
      ...u16(0),
      ...u16(0),
      ...u16(0),
      ...u16(0),
      ...u32(crc),
      ...u32(data.length),
      ...u32(data.length),
      ...u16(nameBytes.length),
      ...u16(0),
      ...nameBytes,
    ]);

    parts.push(localHeader, data);

    const centralHeader = new Uint8Array([
      ...u32(0x02014b50),
      ...u16(20),
      ...u16(20),
      ...u16(0),
      ...u16(0),
      ...u16(0),
      ...u16(0),
      ...u32(crc),
      ...u32(data.length),
      ...u32(data.length),
      ...u16(nameBytes.length),
      ...u16(0),
      ...u16(0),
      ...u16(0),
      ...u16(0),
      ...u32(0),
      ...u32(offset),
      ...nameBytes,
    ]);
    central.push(centralHeader);
    offset += localHeader.length + data.length;
  }

  const centralSize = central.reduce((s, c) => s + c.length, 0);
  const end = new Uint8Array([
    ...u32(0x06054b50),
    ...u16(0),
    ...u16(0),
    ...u16(files.length),
    ...u16(files.length),
    ...u32(centralSize),
    ...u32(offset),
    ...u16(0),
  ]);

  const total = parts.reduce((s, p) => s + p.length, 0) + centralSize + end.length;
  const zip = new Uint8Array(total);
  let pos = 0;
  for (const p of parts) {
    zip.set(p, pos);
    pos += p.length;
  }
  for (const c of central) {
    zip.set(c, pos);
    pos += c.length;
  }
  zip.set(end, pos);

  await downloadBlob(new Blob([zip], { type: "application/zip" }), zipName);
}

export async function fetchAsBlob(url: string): Promise<Blob> {
  const resolved = resolveMediaUrl(url) || url;
  const res = await fetch(resolved);
  if (!res.ok) throw new Error("Failed to fetch asset");
  return res.blob();
}
