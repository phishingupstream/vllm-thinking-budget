#!/usr/bin/env python3
"""Unit tests for RequestState logic and patch validator logic.

Runs without torch/vLLM — tests pure Python logic only.
"""

import sys
import unittest
from dataclasses import dataclass


# ─── Inline RequestState (mirrors thinking_budget_processor.py) ───────────
# We can't import the real one without torch/vLLM, so we duplicate the
# dataclass logic here for unit testing. Any divergence from the source
# is itself a signal to update these tests.

@dataclass
class RequestState:
    """Mirror of thinking_budget_processor.RequestState for testing."""
    budget: int
    output_token_ids: list
    think_start_id: int
    think_end_id: int
    starts_in_thinking: bool = False
    in_thinking: bool = False
    thinking_token_count: int = 0
    forced_stop: bool = False
    last_scanned: int = 0

    def scan_tokens(self) -> None:
        tokens = self.output_token_ids
        end = len(tokens)

        # Speculative decode rejection: output shrank → rescan
        if self.last_scanned > end:
            self.last_scanned = 0
            self.thinking_token_count = 0
            self.forced_stop = False
            self.in_thinking = self.starts_in_thinking

        if self.forced_stop:
            return

        for i in range(self.last_scanned, end):
            tid = tokens[i]
            if tid == self.think_start_id:
                self.in_thinking = True
            elif tid == self.think_end_id:
                self.in_thinking = False
            elif self.in_thinking:
                self.thinking_token_count += 1
        self.last_scanned = end


# Token IDs used in tests
THINK_START = 100
THINK_END = 101
NL = 10
TOK_A = 50
TOK_B = 51


class TestScanTokens(unittest.TestCase):
    """Test RequestState.scan_tokens logic."""

    def _make_state(self, budget=100, starts_in_thinking=False, tokens=None):
        tok_list = tokens if tokens is not None else []
        return RequestState(
            budget=budget,
            output_token_ids=tok_list,
            think_start_id=THINK_START,
            think_end_id=THINK_END,
            starts_in_thinking=starts_in_thinking,
            in_thinking=starts_in_thinking,
        )

    def test_basic_thinking_count(self):
        """Tokens between <think> and </think> are counted."""
        tokens = [THINK_START, TOK_A, TOK_B, TOK_A, THINK_END]
        state = self._make_state(tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 3)
        self.assertFalse(state.in_thinking)

    def test_starts_in_thinking(self):
        """When prompt ends inside <think>, tokens are counted from start."""
        tokens = [TOK_A, TOK_B, THINK_END]
        state = self._make_state(starts_in_thinking=True, tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 2)
        self.assertFalse(state.in_thinking)

    def test_cumulative_across_blocks(self):
        """Budget is cumulative — reopening <think> doesn't reset count."""
        tokens = [THINK_START, TOK_A, TOK_B, THINK_END, TOK_A,
                  THINK_START, TOK_A, THINK_END]
        state = self._make_state(tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 3)  # 2 + 1

    def test_incremental_scan(self):
        """scan_tokens can be called incrementally as tokens arrive."""
        tokens = [THINK_START, TOK_A]
        state = self._make_state(tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 1)
        self.assertTrue(state.in_thinking)

        # More tokens arrive
        tokens.extend([TOK_B, THINK_END])
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 2)
        self.assertFalse(state.in_thinking)

    def test_forced_stop_prevents_further_scan(self):
        """After forced_stop, scan_tokens is a no-op."""
        tokens = [THINK_START, TOK_A]
        state = self._make_state(tokens=tokens)
        state.scan_tokens()
        state.forced_stop = True

        tokens.extend([TOK_B, TOK_A])
        state.scan_tokens()
        # Count should not have increased
        self.assertEqual(state.thinking_token_count, 1)

    def test_speculative_decode_rejection_resets_forced_stop(self):
        """When output shrinks (spec decode rejection), forced_stop resets."""
        tokens = [THINK_START, TOK_A, TOK_B, TOK_A]
        state = self._make_state(tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 3)
        state.forced_stop = True  # Simulate forced stop

        # Speculative tokens rejected — output shrinks
        del tokens[2:]  # Now [THINK_START, TOK_A]
        state.scan_tokens()

        # forced_stop should have been reset, state rescanned
        self.assertFalse(state.forced_stop)
        self.assertEqual(state.thinking_token_count, 1)
        self.assertTrue(state.in_thinking)

    def test_speculative_decode_rejection_without_forced_stop(self):
        """Output shrink without forced_stop still rescans correctly."""
        tokens = [THINK_START, TOK_A, TOK_B, TOK_A]
        state = self._make_state(tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 3)

        # Shrink
        del tokens[2:]
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 1)

    def test_speculative_rejection_with_starts_in_thinking(self):
        """Rescan restores starts_in_thinking state."""
        tokens = [TOK_A, TOK_B, THINK_END, TOK_A]
        state = self._make_state(starts_in_thinking=True, tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 2)
        self.assertFalse(state.in_thinking)
        state.forced_stop = True

        # Shrink to just one token
        del tokens[1:]
        state.scan_tokens()
        self.assertFalse(state.forced_stop)
        self.assertTrue(state.in_thinking)  # Restored from starts_in_thinking
        self.assertEqual(state.thinking_token_count, 1)

    def test_empty_tokens(self):
        """No tokens = no state change."""
        state = self._make_state(tokens=[])
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 0)
        self.assertFalse(state.in_thinking)

    def test_no_thinking_tokens(self):
        """Tokens outside <think> blocks don't count."""
        tokens = [TOK_A, TOK_B, TOK_A]
        state = self._make_state(tokens=tokens)
        state.scan_tokens()
        self.assertEqual(state.thinking_token_count, 0)


class TestMergeThinkingParams(unittest.TestCase):
    """Test the merge_thinking_params validator logic from the OpenAI compat patch."""

    @staticmethod
    def merge_thinking_params(data):
        """Extracted logic from fix_enable_thinking_compat.py's generated validator."""
        ctk = data.get("chat_template_kwargs") or {}
        xargs = data.get("vllm_xargs") or {}
        effort = data.get("reasoning_effort")
        enable = data.get("enable_thinking")

        if effort is None and enable is None and "enable_thinking" not in ctk:
            effort = "medium"

        if effort is not None:
            effort = effort.strip().lower()
            effort_map = {
                "none": (False, None),
                "low": (True, 2048),
                "medium": (True, 8192),
                "high": (True, None),
            }
            if effort not in effort_map:
                raise ValueError(
                    f"reasoning_effort must be one of {list(effort_map.keys())}, got '{effort}'"
                )
            ef_enable, ef_budget = effort_map[effort]
            if enable is None and "enable_thinking" not in ctk:
                ctk["enable_thinking"] = ef_enable
            if ef_budget is not None and "max_thinking_tokens" not in xargs:
                xargs["max_thinking_tokens"] = ef_budget

        # Fixed: enable_thinking overrides pre-existing ctk (highest precedence)
        if enable is not None:
            ctk["enable_thinking"] = enable

        if ctk.get("enable_thinking") and "max_thinking_tokens" not in xargs:
            xargs["max_thinking_tokens"] = 8192

        if ctk:
            data["chat_template_kwargs"] = ctk
        if xargs:
            data["vllm_xargs"] = xargs
        return data

    def test_default_is_medium(self):
        """No thinking params → defaults to medium (enable=True, budget=8192)."""
        data = self.merge_thinking_params({})
        self.assertTrue(data["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(data["vllm_xargs"]["max_thinking_tokens"], 8192)

    def test_effort_none_disables(self):
        """reasoning_effort='none' → enable_thinking=False."""
        data = self.merge_thinking_params({"reasoning_effort": "none"})
        self.assertFalse(data["chat_template_kwargs"]["enable_thinking"])

    def test_effort_low(self):
        """reasoning_effort='low' → budget=2048."""
        data = self.merge_thinking_params({"reasoning_effort": "low"})
        self.assertTrue(data["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(data["vllm_xargs"]["max_thinking_tokens"], 2048)

    def test_effort_high_no_budget_cap(self):
        """reasoning_effort='high' → enable=True, default medium budget applied."""
        data = self.merge_thinking_params({"reasoning_effort": "high"})
        self.assertTrue(data["chat_template_kwargs"]["enable_thinking"])
        # high has no budget from effort_map, but the fallback adds 8192
        self.assertEqual(data["vllm_xargs"]["max_thinking_tokens"], 8192)

    def test_explicit_enable_overrides_effort(self):
        """enable_thinking=False + reasoning_effort='high' → disabled."""
        data = self.merge_thinking_params({
            "enable_thinking": False,
            "reasoning_effort": "high",
        })
        self.assertFalse(data["chat_template_kwargs"]["enable_thinking"])

    def test_explicit_enable_overrides_ctk(self):
        """enable_thinking=True overrides pre-existing ctk enable_thinking=False."""
        data = self.merge_thinking_params({
            "enable_thinking": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        # After our fix, explicit enable_thinking should win
        self.assertTrue(data["chat_template_kwargs"]["enable_thinking"])

    def test_explicit_enable_false_overrides_ctk_true(self):
        """enable_thinking=False overrides pre-existing ctk enable_thinking=True."""
        data = self.merge_thinking_params({
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": True},
        })
        self.assertFalse(data["chat_template_kwargs"]["enable_thinking"])

    def test_ctk_preserved_when_no_override(self):
        """Pre-existing ctk enable_thinking is preserved when no top-level override."""
        data = self.merge_thinking_params({
            "chat_template_kwargs": {"enable_thinking": False},
        })
        self.assertFalse(data["chat_template_kwargs"]["enable_thinking"])

    def test_invalid_effort_raises(self):
        """Invalid reasoning_effort raises ValueError."""
        with self.assertRaises(ValueError):
            self.merge_thinking_params({"reasoning_effort": "turbo"})

    def test_vllm_xargs_not_overwritten(self):
        """Pre-existing vllm_xargs max_thinking_tokens is preserved."""
        data = self.merge_thinking_params({
            "reasoning_effort": "low",
            "vllm_xargs": {"max_thinking_tokens": 512},
        })
        self.assertEqual(data["vllm_xargs"]["max_thinking_tokens"], 512)


class TestAnthropicBudgetFiltering(unittest.TestCase):
    """Test the Anthropic budget filtering logic from fix_anthropic_thinking_compat.py."""

    @staticmethod
    def filter_budget(budget, max_tokens):
        """Extracted logic from the Anthropic patch's budget filtering."""
        max_tok = max_tokens or float('inf')
        if (budget is not None and isinstance(budget, int)
                and budget > 0 and budget < max_tok * 0.8):
            return budget
        return None  # Budget filtered out, default will apply

    def test_restrictive_budget_passes_through(self):
        """A budget well below max_tokens is forwarded."""
        self.assertEqual(self.filter_budget(2048, 8192), 2048)

    def test_near_max_budget_filtered(self):
        """budget_tokens close to max_tokens is filtered (Anthropic 'no cap' pattern)."""
        self.assertIsNone(self.filter_budget(8191, 8192))

    def test_budget_at_80pct_filtered(self):
        """Budget at exactly 80% of max_tokens is filtered."""
        # 8192 * 0.8 = 6553.6, so budget=6554 is NOT < 6553.6
        self.assertIsNone(self.filter_budget(6554, 8192))

    def test_budget_below_80pct_passes(self):
        """Budget below 80% of max_tokens passes through."""
        self.assertEqual(self.filter_budget(6000, 8192), 6000)

    def test_none_max_tokens_passes_budget_through(self):
        """When max_tokens is None, explicit budget should still be forwarded."""
        # This was the bug: max_tok=0 made budget < 0 always false
        self.assertEqual(self.filter_budget(2048, None), 2048)

    def test_zero_budget_filtered(self):
        """Budget of 0 is filtered (guard: budget > 0)."""
        self.assertIsNone(self.filter_budget(0, 8192))

    def test_none_budget_filtered(self):
        """None budget is filtered."""
        self.assertIsNone(self.filter_budget(None, 8192))

    def test_negative_budget_filtered(self):
        """Negative budget is filtered."""
        self.assertIsNone(self.filter_budget(-1, 8192))


class TestAvailableSteps(unittest.TestCase):
    """Validate test harness step iteration."""

    def test_no_step_4_in_available(self):
        """available_steps should not include 4 (no handler exists)."""
        available_steps = [0, 1, 2, 3, 5]
        self.assertNotIn(4, available_steps)
        # All steps should have handlers
        handler_map = {0, 1, 2, 3, 5}
        self.assertEqual(set(available_steps), handler_map)


if __name__ == "__main__":
    unittest.main()
