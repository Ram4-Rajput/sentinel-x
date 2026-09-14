import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
export const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;

export function bytesFmt(v: number): string {
  if (v < 1024) return `${v.toFixed(0)} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
  return `${(v / 1024 / 1024).toFixed(1)} MB`;
}

// Risk -> spectral color. Green (safe) -> amber -> red (critical).
export function riskColor(r: number): string {
  const c = Math.max(0, Math.min(1, r));
  if (c < 0.5) {
    // green -> amber
    const t = c / 0.5;
    return lerpHex("#3ddc97", "#ffb020", t);
  }
  const t = (c - 0.5) / 0.5;
  return lerpHex("#ffb020", "#ff4d5e", t);
}

function lerpHex(a: string, b: string, t: number): string {
  const ai = hexToRgb(a);
  const bi = hexToRgb(b);
  const r = Math.round(ai[0] + (bi[0] - ai[0]) * t);
  const g = Math.round(ai[1] + (bi[1] - ai[1]) * t);
  const bl = Math.round(ai[2] + (bi[2] - ai[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}
