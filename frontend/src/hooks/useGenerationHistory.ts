"use client";

import { useCallback, useEffect, useState } from "react";

export type HistoryModule = "custom-garment" | "virtual-tryon" | "semantic-analysis";

export type HistoryItem = {
  id: string;
  module: HistoryModule;
  thumbnailUrl: string;
  resultUrl?: string;
  title: string;
  model: string;
  status: "completed" | "failed";
  createdAt: string;
  durationMs?: number;
  metadata?: Record<string, unknown>;
  promptSummary?: string;
  resolution?: string;
};

const STORAGE_KEY = "fabricvision-generation-history";
const MAX_ITEMS = 24;

function readStorage(): HistoryItem[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryItem[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeStorage(items: HistoryItem[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
  } catch {
    // Quota exceeded — drop oldest half
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, Math.floor(MAX_ITEMS / 2))));
    } catch {
      /* ignore */
    }
  }
}

export function useGenerationHistory(module?: HistoryModule) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setItems(readStorage());
    setHydrated(true);
  }, []);

  const filtered = module ? items.filter((i) => i.module === module) : items;

  const addItem = useCallback((item: Omit<HistoryItem, "id" | "createdAt"> & { id?: string; createdAt?: string }) => {
    const entry: HistoryItem = {
      ...item,
      id: item.id || `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      createdAt: item.createdAt || new Date().toISOString(),
    };
    setItems((prev) => {
      const next = [entry, ...prev.filter((p) => p.id !== entry.id)].slice(0, MAX_ITEMS);
      writeStorage(next);
      return next;
    });
    return entry;
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((prev) => {
      const next = prev.filter((p) => p.id !== id);
      writeStorage(next);
      return next;
    });
  }, []);

  const clearModule = useCallback((mod: HistoryModule) => {
    setItems((prev) => {
      const next = prev.filter((p) => p.module !== mod);
      writeStorage(next);
      return next;
    });
  }, []);

  return { items: filtered, allItems: items, hydrated, addItem, removeItem, clearModule };
}
