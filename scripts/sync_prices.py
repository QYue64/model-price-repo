#!/usr/bin/env python3
"""Sync model pricing from LiteLLM and official OpenAI pricing, applying
prefix filters, aliases, and custom model definitions.

Usage:
    python3 scripts/sync_prices.py --config config.json --repo-root .
"""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal
import hashlib
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = [
    "upstream_url",
    "output_file",
    "hash_file",
    "sync_mode",
    "prefix_filters",
]


def load_config(path: str) -> dict:
    """Read and validate config.json."""
    if not os.path.isfile(path):
        log.error("Config file not found: %s", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        log.error("Config missing required keys: %s", ", ".join(missing))
        sys.exit(1)
    if cfg["sync_mode"] not in ("additive", "full"):
        log.error("Invalid sync_mode '%s'; must be 'additive' or 'full'", cfg["sync_mode"])
        sys.exit(1)
    return cfg


# ---------------------------------------------------------------------------
# Existing data
# ---------------------------------------------------------------------------


def load_existing(path: str) -> dict:
    """Load the current output file, or return {} on first run."""
    if not os.path.isfile(path):
        log.info("No existing output file; starting fresh.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_hash(path: str) -> str:
    """Read the stored SHA-256 hex digest, or return empty string."""
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Upstream fetch
# ---------------------------------------------------------------------------


def fetch_upstream(url: str) -> dict:
    """Download the full upstream pricing JSON."""
    log.info("Fetching upstream: %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "model-price-repo/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        log.error("Failed to fetch upstream: %s", exc)
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Upstream JSON is invalid: %s", exc)
        sys.exit(1)

    if not isinstance(data, dict):
        log.error("Upstream JSON is not an object (got %s)", type(data).__name__)
        sys.exit(1)

    log.info("Upstream contains %d model entries.", len(data))
    return data


def fetch_text(url: str, label: str) -> str:
    """从可信来源下载 UTF-8 文本文件。"""
    log.info("Fetching %s: %s", label, url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "model-price-repo/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, UnicodeDecodeError) as exc:
        log.error("Failed to fetch %s: %s", label, exc)
        sys.exit(1)


# 这些是 Sub2Api 消费的价格字段。应用官方数据前先删除它们，避免旧的
# 自定义价格在官方价格变更后继续残留。
OFFICIAL_PRICE_FIELDS = {
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_272k_tokens",
    "cache_read_input_token_cost",
    "cache_read_input_token_cost_above_272k_tokens",
    "long_context_input_token_threshold",
    "long_context_input_cost_multiplier",
    "long_context_output_cost_multiplier",
    "input_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens",
    "input_cost_per_token_priority",
    "output_cost_per_token_priority",
    "cache_creation_input_token_cost_priority",
    "cache_read_input_token_cost_priority",
    "input_cost_per_token_batches",
    "output_cost_per_token_batches",
    "cache_creation_input_token_cost_batches",
    "cache_read_input_token_cost_batches",
    "input_cost_per_token_flex",
    "output_cost_per_token_flex",
    "cache_creation_input_token_cost_flex",
    "cache_read_input_token_cost_flex",
}


def parse_price_per_million(value: str) -> float | None:
    """将 '$0.20' 这类每百万 Token 价格转换为单 Token 美元价格。"""
    normalized = value.strip().replace(",", "")
    if normalized in {"", "-", "—", "N/A", "n/a"}:
        return None
    normalized = normalized.removeprefix("$").strip()
    try:
        return round(float(normalized) / 1_000_000, 15)
    except ValueError:
        return None


def parse_markdown_row(line: str) -> list[str] | None:
    """解析一行由竖线分隔的 Markdown 表格。"""
    stripped = line.strip()
    if not stripped.startswith("|") or "|" not in stripped[1:]:
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
        return None
    return cells


def normalize_official_model_name(name: str) -> str:
    """移除官方模型名称末尾的上下文说明。"""
    name = re.sub(r"\s+\([^)]*\)\s*$", "", name.strip())
    return name.strip().lower()


def apply_official_tier(entry: dict, row: dict[str, str], tier: str, threshold: int) -> None:
    """将官方价格表的一行转换为 LiteLLM 兼容字段。"""
    field_prefix = {
        "standard": "",
        "batch": "_batches",
        "flex": "_flex",
        "fast": "_priority",
    }[tier]

    mapping = {
        "Short context input": f"input_cost_per_token{field_prefix}",
        "Short context cached input": f"cache_read_input_token_cost{field_prefix}",
        "Short context cache writes": f"cache_creation_input_token_cost{field_prefix}",
        "Short context output": f"output_cost_per_token{field_prefix}",
    }
    for column, output_key in mapping.items():
        value = parse_price_per_million(row.get(column, ""))
        if value is not None:
            entry[output_key] = value

    # Sub2Api 会把长上下文倍率应用到输入、缓存读取、缓存写入和输出。
    # 官方表格直接给出长上下文价格，因此计算倍率，不输出不受支持的
    # *_above_* 字段。
    if tier != "standard":
        return
    short_input = parse_price_per_million(row.get("Short context input", ""))
    short_output = parse_price_per_million(row.get("Short context output", ""))
    long_input = parse_price_per_million(row.get("Long context input", ""))
    long_output = parse_price_per_million(row.get("Long context output", ""))
    if short_input and long_input:
        entry["long_context_input_token_threshold"] = threshold
        entry["long_context_input_cost_multiplier"] = long_input / short_input
    if short_output and long_output:
        entry["long_context_input_token_threshold"] = threshold
        entry["long_context_output_cost_multiplier"] = long_output / short_output


def parse_official_openai_pricing(markdown: str, config: dict) -> dict:
    """解析 OpenAI 官方 Markdown 定价页并规范化其中的价格表。"""
    official_cfg = config.get("official_openai", {})
    prefixes = tuple(official_cfg.get("prefix_filters", ["gpt-", "o1", "o3", "o4"]))
    threshold = int(official_cfg.get("long_context_input_token_threshold", 272_000))

    current_tier: str | None = None
    headers: list[str] | None = None
    models: dict[str, dict] = {}
    heading_pattern = re.compile(r"^###\s+(Standard|Batch|Flex|Fast) pricing data\s*$", re.IGNORECASE)

    for line in markdown.splitlines():
        heading = heading_pattern.match(line.strip())
        if heading:
            current_tier = heading.group(1).lower()
            headers = None
            continue

        row = parse_markdown_row(line)
        if row is None:
            if not line.strip():
                headers = None
            continue

        if row[0].lower() == "model" and "Short context input" in row:
            headers = row
            continue
        if current_tier is None or headers is None or len(row) != len(headers):
            continue

        row_data = dict(zip(headers, row))
        model_name = normalize_official_model_name(row_data.get("Model", ""))
        if not model_name or not model_name.startswith(prefixes):
            continue

        entry = models.setdefault(
            model_name,
            {"litellm_provider": "openai", "mode": "chat"},
        )
        apply_official_tier(entry, row_data, current_tier, threshold)

    if not models:
        log.error("Official OpenAI pricing page contained no matching model rows")
        sys.exit(1)

    required_models = [
        normalize_official_model_name(name)
        for name in official_cfg.get("required_models", [])
    ]
    missing_models = [name for name in required_models if name not in models]
    if missing_models:
        log.error("Official OpenAI pricing is missing required models: %s", ", ".join(missing_models))
        sys.exit(1)

    log.info("Official OpenAI pricing contains %d matching models.", len(models))
    return models


def fetch_official_openai_pricing(url: str, config: dict) -> dict:
    """下载并解析 OpenAI 官方 Markdown 定价页。"""
    return parse_official_openai_pricing(fetch_text(url, "official OpenAI pricing"), config)


def merge_official_pricing(data: dict, official: dict) -> tuple[dict, int]:
    """覆盖官方价格，同时保留非价格元数据。"""
    updated = 0
    for model_name, pricing in official.items():
        target = data.setdefault(model_name, {})
        if not isinstance(target, dict):
            target = {}
            data[model_name] = target
        before = {key: target.get(key) for key in OFFICIAL_PRICE_FIELDS}
        for key in OFFICIAL_PRICE_FIELDS:
            target.pop(key, None)
        target.update(pricing)
        after = {key: target.get(key) for key in OFFICIAL_PRICE_FIELDS}
        if before != after:
            updated += 1
    log.info("Official OpenAI pricing applied to %d models.", updated)
    return data, updated


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_upstream(data: dict, config: dict) -> dict:
    """Apply prefix_filters and exclude_patterns to upstream data."""
    prefixes = tuple(config.get("prefix_filters", []))
    excludes = config.get("exclude_patterns", [])

    filtered = {}
    for key, value in data.items():
        # Exclude first
        if any(pat in key for pat in excludes):
            continue
        # Then check prefix match
        if prefixes and not key.startswith(prefixes):
            continue
        filtered[key] = value

    log.info("Filtered to %d models (from %d upstream).", len(filtered), len(data))
    return filtered


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_models(
    existing: dict,
    filtered: dict,
    sync_mode: str,
    update_existing: bool,
) -> tuple[dict, dict]:
    """Merge filtered upstream into existing data.

    Returns (merged_dict, stats_dict).
    """
    stats = {"added": 0, "updated": 0, "unchanged": 0, "total_upstream": len(filtered)}

    if sync_mode == "full":
        # Full mode: replace entirely with filtered upstream
        stats["added"] = len(filtered)
        return dict(filtered), stats

    # Additive mode
    merged = dict(existing)
    for key, value in filtered.items():
        if key not in merged:
            merged[key] = value
            stats["added"] += 1
        elif update_existing:
            if merged[key] != value:
                merged[key] = value
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1
        else:
            # update_existing=False: preserve existing fields, but absorb new fields from upstream
            if isinstance(merged[key], dict) and isinstance(value, dict):
                new_fields = {k: v for k, v in value.items() if k not in merged[key]}
                if new_fields:
                    merged[key].update(new_fields)
                    log.info("Model '%s': absorbed %d new field(s) from upstream: %s", key, len(new_fields), list(new_fields))
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1
            else:
                stats["unchanged"] += 1

    return merged, stats


# ---------------------------------------------------------------------------
# Aliases & custom models
# ---------------------------------------------------------------------------


def apply_aliases(data: dict, aliases: dict) -> dict:
    """Deep-copy source model data into alias keys."""
    for alias_key, alias_cfg in aliases.items():
        source = alias_cfg.get("source", "")
        if source not in data:
            log.warning(
                "Alias '%s': source model '%s' not found; skipping.",
                alias_key,
                source,
            )
            continue
        data[alias_key] = copy.deepcopy(data[source])
        log.info("Alias '%s' -> '%s' applied.", alias_key, source)
    return data


def apply_custom_models(data: dict, custom: dict) -> dict:
    """Inject custom model definitions (deep merge for existing, full set for new)."""
    for key, value in custom.items():
        if key in data and isinstance(data[key], dict) and isinstance(value, dict):
            data[key].update(value)
            log.info("Custom model '%s' merged (deep).", key)
        else:
            data[key] = value
            log.info("Custom model '%s' injected.", key)
    return data


def fill_cache_1hr_pricing(data: dict, config: dict) -> int:
    """Auto-fill missing cache_creation_input_token_cost_above_1hr for matching models.

    Uses a fixed ratio (default 1.6x) of the 5-minute cache write cost.
    Returns the number of models auto-filled.
    """
    auto_fill_cfg = config.get("cache_1hr_auto_fill")
    if not auto_fill_cfg:
        return 0

    prefix = auto_fill_cfg.get("model_prefix", "claude-")
    ratio = auto_fill_cfg.get("ratio", 1.6)
    count = 0

    for key, value in data.items():
        if not key.startswith(prefix):
            continue
        if not isinstance(value, dict):
            continue
        cost_5m = value.get("cache_creation_input_token_cost")
        if cost_5m is None:
            continue
        if value.get("cache_creation_input_token_cost_above_1hr") is not None:
            continue
        value["cache_creation_input_token_cost_above_1hr"] = float(
            Decimal(str(cost_5m)) * Decimal(str(ratio))
        )
        log.info("Auto-filled cache 1hr cost for '%s': %s * %s = %s", key, cost_5m, ratio, value["cache_creation_input_token_cost_above_1hr"])
        count += 1

    return count


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def compute_hash(json_bytes: bytes) -> str:
    """Return hex SHA-256 of the given bytes."""
    return hashlib.sha256(json_bytes).hexdigest()


def write_output(data: dict, json_path: str, hash_path: str, old_hash: str) -> tuple[bool, str]:
    """Write sorted JSON and SHA-256 hash file.

    Returns (changed: bool, new_hash: str).
    """
    json_bytes = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")
    new_hash = compute_hash(json_bytes)

    if new_hash == old_hash:
        log.info("No changes detected (hash matches).")
        return False, new_hash

    with open(json_path, "wb") as f:
        f.write(json_bytes)
    with open(hash_path, "w", encoding="utf-8") as f:
        f.write(new_hash + "\n")

    log.info("Output written: %s (%d models)", json_path, len(data))
    log.info("Hash written:   %s", hash_path)
    return True, new_hash


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync model pricing from upstream.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    config_path = os.path.join(repo_root, args.config)

    # 1. Load config
    config = load_config(config_path)

    output_path = os.path.join(repo_root, config["output_file"])
    hash_path = os.path.join(repo_root, config["hash_file"])

    # 2. Load existing data
    existing = load_existing(output_path)
    old_hash = load_existing_hash(hash_path)
    log.info("Existing output has %d models.", len(existing))

    # 3. Fetch upstream
    upstream = fetch_upstream(config["upstream_url"])

    # 4. Filter
    filtered = filter_upstream(upstream, config)

    # 5. Merge
    merged, stats = merge_models(
        existing,
        filtered,
        config["sync_mode"],
        config.get("update_existing", False),
    )
    log.info(
        "Merge stats: %d added, %d updated, %d unchanged.",
        stats["added"],
        stats["updated"],
        stats["unchanged"],
    )

    # 6. 自定义模型提供元数据和非官方厂商的价格覆盖。
    custom = config.get("custom_models", {})
    if custom:
        merged = apply_custom_models(merged, custom)

    # 7. 最后应用 OpenAI 官方价格，避免旧的自定义价格覆盖官方来源。
    official_cfg = config.get("official_openai", {})
    official_updated = 0
    if official_cfg.get("enabled", True):
        official_url = official_cfg.get("pricing_url")
        if not official_url:
            log.error("official_openai.enabled is true but pricing_url is empty")
            sys.exit(1)
        official = fetch_official_openai_pricing(official_url, config)
        merged, official_updated = merge_official_pricing(merged, official)

    # 8. 官方价格应用后再复制别名，使官方模型别名继承最新价格。
    aliases = config.get("aliases", {})
    if aliases:
        merged = apply_aliases(merged, aliases)

    # 9. Auto-fill cache 1hr pricing
    cache_1hr_count = fill_cache_1hr_pricing(merged, config)

    # 10. Write output
    changed, new_hash = write_output(merged, output_path, hash_path, old_hash)

    # 11. Report
    log.info("--- Sync Report ---")
    log.info("Total models in output: %d", len(merged))
    log.info("Added:     %d", stats["added"])
    log.info("Updated:   %d", stats["updated"])
    log.info("Unchanged: %d", stats["unchanged"])
    log.info("Aliases:   %d", len(aliases))
    log.info("Cache 1hr auto-filled: %d", cache_1hr_count)
    log.info("Custom:    %d", len(custom))
    log.info("Official OpenAI updated: %d", official_updated)

    # Machine-readable output for CI
    print(f"CHANGED={str(changed).lower()}")
    print(f"HASH={new_hash}")


if __name__ == "__main__":
    main()
