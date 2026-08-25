import taxonomy from "./garmentTaxonomy.json";

export type GenderCategory = "men" | "women" | "unisex";

const DISPLAY_OVERRIDES: Record<string, string> = {
  t_shirt: "T-Shirt",
  polo_shirt: "Polo Shirt",
};

/** Reset targets must already exist in garment_taxonomy.json for that gender. */
const GENDER_DEFAULTS: Record<GenderCategory, string> = {
  men: "Shirt",
  women: "Dress",
  unisex: "T-Shirt",
};

type GenderBlock = {
  items?: string[];
  traditional?: string[];
  western?: string[];
};

function uniquePreserve(keys: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const key of keys) {
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(key);
  }
  return out;
}

export function normalizeGenderCategory(gender: string): GenderCategory {
  const key = (gender || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["men", "male", "mens", "man"].includes(key)) return "men";
  if (["women", "female", "womens", "woman"].includes(key)) return "women";
  if (["unisex", "neutral"].includes(key)) return "unisex";
  return "women";
}

export function taxonomyKeysForGender(gender: string): string[] {
  const cat = normalizeGenderCategory(gender);
  const block = (taxonomy.gender_categories as Record<string, GenderBlock>)[cat];
  if (!block) return [];
  if (Array.isArray(block.items) && block.items.length > 0) {
    return uniquePreserve(block.items);
  }
  const traditional = Array.isArray(block.traditional) ? block.traditional : [];
  const western = Array.isArray(block.western) ? block.western : [];
  return uniquePreserve([...traditional, ...western]);
}

export function toGarmentDisplayLabel(key: string): string {
  if (DISPLAY_OVERRIDES[key]) return DISPLAY_OVERRIDES[key];
  return key
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/** Display labels for the Garment Type dropdown, filtered by Gender. */
export function garmentsForGender(gender: string): string[] {
  return taxonomyKeysForGender(gender).map(toGarmentDisplayLabel);
}

/**
 * Keep the current garment when it belongs to the new gender.
 * Otherwise reset to that gender's taxonomy default (Dress / Shirt / T-Shirt).
 */
export function resolveGarmentTypeForGender(gender: string, current: string): string {
  const options = garmentsForGender(gender);
  if (current && options.includes(current)) return current;
  const preferred = GENDER_DEFAULTS[normalizeGenderCategory(gender)];
  if (preferred && options.includes(preferred)) return preferred;
  return options[0] ?? "";
}
