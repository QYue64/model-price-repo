# model-price-repo

Filtered model pricing data for CRS and Sub2Api. The workflow keeps LiteLLM as a broad metadata source and overlays the latest OpenAI prices from the official pricing Markdown page.

## How it works

A GitHub Actions workflow runs every 10 minutes (and on manual trigger):

1. Downloads the full `model_prices_and_context_window.json` from LiteLLM for broad model metadata
2. Filters the LiteLLM data by the prefix rules in `config.json`
3. Downloads OpenAI's official [`pricing.md`](https://developers.openai.com/api/docs/pricing.md) and converts its Standard, Batch, Flex, and Fast tables into LiteLLM-compatible fields
4. Rebuilds the filtered set so changed upstream prices are replaced instead of preserved
5. Applies aliases and non-OpenAI custom models, then reapplies official OpenAI prices last
6. Writes the output JSON + SHA-256 hash, committing only if content changed

## Configuration

All settings live in [`config.json`](config.json):

| Field | Description |
|---|---|
| `upstream_url` | URL to the upstream litellm pricing JSON |
| `official_openai` | Official OpenAI pricing source, model prefixes, and long-context threshold |
| `output_file` | Output filename (default: `model_prices_and_context_window.json`) |
| `hash_file` | SHA-256 hash filename for change detection |
| `sync_mode` | `"additive"` (preserve old entries) or `"full"` (rebuild each run) |
| `update_existing` | Whether LiteLLM values may update existing entries; official OpenAI values always update |
| `prefix_filters` | List of prefixes — a model key must start with one to be included |
| `exclude_patterns` | Substring patterns to exclude (applied before prefix matching) |
| `aliases` | Map alias model keys to existing source models (deep copy pricing) |
| `custom_models` | Manually defined pricing objects, always injected |

### Adding new model prefixes

Edit the `prefix_filters` array in `config.json`:

```json
{
  "prefix_filters": [
    "claude-",
    "gpt-",
    "your-new-prefix/"
  ]
}
```

### Adding aliases

Aliases create copies of an existing model's pricing under a new key:

```json
{
  "aliases": {
    "claude-opus-4-6-thinking": {
      "source": "claude-opus-4-6",
      "description": "Thinking variant, same pricing"
    }
  }
}
```

If the source model doesn't exist in the filtered data, the alias is skipped with a warning.

## Running locally

```bash
python3 scripts/sync_prices.py --config config.json --repo-root .
```

No pip dependencies — uses Python standard library only.

The official source is a Markdown endpoint rather than an HTML page, which keeps
the parser deterministic while still following the published OpenAI pricing tables.

## CRS integration

Point CRS to the raw output file from this repo:

```
MODEL_PRICES_URL=https://raw.githubusercontent.com/<owner>/model-price-repo/main/model_prices_and_context_window.json
```

The output JSON structure is identical to what litellm produces (model key -> pricing object), so CRS `pricingService.js` works without changes.

## License

[MIT](LICENSE)
