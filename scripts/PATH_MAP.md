# Guide path map (Chinese → English)

Upstream (`origin` / caomaolufei) keeps Chinese folder and file names under `docs/guides/`.
This fork uses English paths. The map and scripts keep the two in sync.

## Files

| File | Role |
|---|---|
| `scripts/path-map.json` | Source of truth: every `zh` → `en` path relative to the guides root |
| `scripts/path-map-lib.mjs` | Shared loaders + Astro redirect builder |
| `scripts/rename-guides.mjs` | One-shot `git mv` apply (guides + `i18n/en` + `i18n/vi`) |
| `scripts/sync-upstream-guides.mjs` | Fetch upstream and write updated Chinese sources onto English paths |

## Commands

```bash
# Re-apply renames (idempotent)
npm run guides:rename

# After upstream adds/changes guides:
npm run guides:sync-upstream
# If it errors on unmapped paths, add entries to path-map.json and re-run.

# Then refresh translations for new/changed guides
node scripts/translate.mjs --check
node scripts/translate.mjs --lang en,vi
```

## Adding a new upstream article

1. Note the Chinese path under `docs/guides/...`
2. Add `{ "zh": "...", "en": "..." }` to `files` in `path-map.json` (and a `directories` entry if a new folder appears)
3. Run `npm run guides:sync-upstream`
4. Translate with `scripts/translate.mjs`
