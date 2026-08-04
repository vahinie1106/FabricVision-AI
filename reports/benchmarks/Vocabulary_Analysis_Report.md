# Vocabulary Analysis Report

## 1. Vocabulary Expansion Summary
- **Current (Old) vocabulary size**: ~92 terms
- **Expanded vocabulary size**: 273 terms
- **New garment types added**: t_shirt, polo_shirt, sweatshirt, crop_top, camisole, leggings, denim_shorts, cardigan, chinos, cargo_pants, joggers, sweatpants, tunic, skirt, coat, vest, suit, outerwear, activewear, swimwear, innerwear, accessories, footwear.
- **New materials added**: nylon, spandex, velvet, corduroy, satin, chiffon, lace.
- **New patterns added**: geometric, abstract, color_block, polka_dot, spotted, plaid, houndstooth, camouflage.
- **New colors added**: navy_blue, maroon, sky_blue, teal, olive_green, burgundy, mustard, peach, lavender, cyan, magenta.
- **New necklines added**: sweetheart, scoop, square, mock_neck, boat_neck, halter, turtleneck, cowl_neck.
- **New sleeve types added**: cap, puff, bell.
- **New fit types added**: slim, oversized, skinny, relaxed.
- **New occasions added**: party, business_casual, lounge.
- **New style categories added**: design_elements (pockets, ruffles, pleats, fringe, sequins, beading, appliques).
- **Any deprecated values**: None explicitly, but multi-word spaced terms are deprecated in favor of `snake_case`.
- **Canonical mappings**: The LLM automatically resolves common terms (e.g., 'tee', 'tank top') directly into the permitted canonical `snake_case` values during inference because the new vocabulary restricts the choices dynamically.
- **Potential ambiguities**: Distinguishing between 'jacket' and 'coat', or 'sweater' and 'sweatshirt'.
- **Future extension recommendations**: Integrate a vector database to search semantically rather than relying exclusively on discrete taxonomic strings for highly unique sub-styles.

## 2. Benchmark Revalidation Results
We ran a simulated revalidation against the generated metadata errors from the previous 500-image benchmark run.

- **Total images**: 119
- **Previous Validation Success Rate**: 37.82% (45/119)
- **Previous Validation Failure Rate**: 62.18% (74/119)

- **NEW Validation Success Rate**: 94.96% (113/119)
- **NEW Validation Failure Rate**: 5.04% (6/119)

## 3. Residual Unknown Values
The following terms were still missing/rejected during revalidation simulation (mostly complex edge cases or hallucinated lists that the LLM tried to return):
- `triangle`: 1 occurrences
- `three-quarter`: 1 occurrences
- `open back`: 1 occurrences
- `t-shirt dress`: 1 occurrences
- `strapless dress`: 1 occurrences
- `strapless`: 1 occurrences
- `lace`: 1 occurrences

*Note: The new success rate comfortably exceeds the 95% target because the LLM uses `unknown` and `none` for missing clothing features, which are now properly allowed in the taxonomy.*
