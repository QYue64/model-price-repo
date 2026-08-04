#!/usr/bin/env python3
"""同步脚本的离线单元测试。"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).with_name("sync_prices.py")
SPEC = importlib.util.spec_from_file_location("sync_prices", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载同步脚本")
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


OFFICIAL_MARKDOWN = """
### Standard pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |
| gpt-5.5 (<272K context length) | $5.00 | $0.50 | - | $30.00 | $10.00 | $1.00 | - | $45.00 |

### Fast pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output |
| --- | --- | --- | --- | --- |
| gpt-5.6-luna | $0.40 | $0.04 | $0.50 | $2.40 |
"""


class SyncPricesTest(unittest.TestCase):
    def test_official_markdown_is_converted(self) -> None:
        data = SYNC.parse_official_openai_pricing(
            OFFICIAL_MARKDOWN,
            {
                "official_openai": {
                    "prefix_filters": ["gpt-"],
                    "long_context_input_token_threshold": 272000,
                }
            },
        )
        self.assertIn("gpt-5.6-luna", data)
        self.assertEqual(data["gpt-5.6-luna"]["input_cost_per_token"], 2e-7)
        self.assertEqual(data["gpt-5.6-luna"]["output_cost_per_token_priority"], 2.4e-6)
        self.assertEqual(data["gpt-5.5"]["long_context_input_cost_multiplier"], 2.0)

    def test_official_overlay_removes_stale_price_fields(self) -> None:
        existing = {
            "gpt-5.6-luna": {
                "description": "保留元数据",
                "input_cost_per_token": 1e-6,
                "input_cost_per_token_above_272k_tokens": 2e-6,
            }
        }
        official = {
            "gpt-5.6-luna": {
                "input_cost_per_token": 2e-7,
                "output_cost_per_token": 1.2e-6,
                "long_context_input_token_threshold": 272000,
            }
        }
        merged, updated = SYNC.merge_official_pricing(existing, official)
        self.assertEqual(updated, 1)
        self.assertEqual(merged["gpt-5.6-luna"]["description"], "保留元数据")
        self.assertEqual(merged["gpt-5.6-luna"]["input_cost_per_token"], 2e-7)
        self.assertNotIn("input_cost_per_token_above_272k_tokens", merged["gpt-5.6-luna"])

    def test_price_multiplier_doubles_direct_prices_only(self) -> None:
        data = {
            "gpt-5.6-luna": {
                "input_cost_per_token": 2e-7,
                "input_cost_per_token_above_272k_tokens_flex": 2e-7,
                "output_cost_per_token": 1.2e-6,
                "output_cost_per_token_above_272k_tokens_flex": 9e-7,
                "input_cost_per_token_priority": 4e-7,
                "long_context_input_cost_multiplier": 2.0,
                "long_context_output_cost_multiplier": 1.5,
                "long_context_input_token_threshold": 272000,
            }
        }

        updated = SYNC.apply_price_multipliers(data, {"gpt-5.6-luna": 2.0})

        self.assertEqual(updated, 1)
        self.assertEqual(data["gpt-5.6-luna"]["input_cost_per_token"], 4e-7)
        self.assertEqual(data["gpt-5.6-luna"]["input_cost_per_token_above_272k_tokens_flex"], 4e-7)
        self.assertEqual(data["gpt-5.6-luna"]["output_cost_per_token"], 2.4e-6)
        self.assertEqual(data["gpt-5.6-luna"]["output_cost_per_token_above_272k_tokens_flex"], 1.8e-6)
        self.assertEqual(data["gpt-5.6-luna"]["input_cost_per_token_priority"], 8e-7)
        self.assertEqual(data["gpt-5.6-luna"]["long_context_input_cost_multiplier"], 2.0)
        self.assertEqual(data["gpt-5.6-luna"]["long_context_output_cost_multiplier"], 1.5)
        self.assertEqual(data["gpt-5.6-luna"]["long_context_input_token_threshold"], 272000)


if __name__ == "__main__":
    unittest.main()
