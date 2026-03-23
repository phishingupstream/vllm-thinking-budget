# SPDX-License-Identifier: Apache-2.0
"""V1 Logits Processor that caps thinking tokens for Qwen3.5 models.

When thinking mode is enabled with structured output, the model can generate
unlimited thinking tokens that consume the entire max_tokens budget. This
processor caps thinking tokens and forces </think> after a budget is reached.

The budget is cumulative across all thinking blocks in a single generation.
If the model closes and reopens a <think> block, previously spent tokens
still count toward the budget.

Usage:
    Client passes budget via vllm_xargs:
        extra_body={"vllm_xargs": {"max_thinking_tokens": 512}}

    Or in raw JSON request body:
        {"vllm_xargs": {"max_thinking_tokens": 512}}

    Server loads via --logits-processors flag or config YAML:
        logits-processors:
          - "thinking_budget_processor:ThinkingBudgetLogitsProcessor"

    PYTHONPATH must include the parent of the plugins directory
    (e.g. PYTHONPATH=/path/to/this/repo).

Budget semantics:
    - budget=0: forces </think> as first output token (no thinking)
    - budget=1: forces \n then </think> (0 real thinking tokens)
    - budget=N (N>=2): allows N-2 real thinking tokens, then \n, then </think>
    - The forced \n before </think> ensures well-formatted output
    - Soft nudge (boosted \n and </think> logits) starts at 80% of budget
      for budgets > 10; skipped for small budgets where hard cutoff suffices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger("vllm.thinking_budget")

# Number of prompt tail tokens to scan for <think>/<think> state detection.
# Must cover the assistant prefix tokens (e.g. Qwen3.5: <|im_start|>assistant\n<think>\n)
_PROMPT_TAIL_SCAN = 8


@dataclass
class RequestState:
    """Per-request thinking budget state."""
    budget: int                          # max thinking tokens (0 = immediate stop)
    output_token_ids: list[int]          # reference to vLLM's output token list
    think_start_id: int
    think_end_id: int
    starts_in_thinking: bool = False     # initial state from prompt prefix (immutable)
    in_thinking: bool = False
    thinking_token_count: int = 0        # cumulative across all thinking blocks
    forced_stop: bool = False            # True after we forced </think>
    last_scanned: int = 0                # index up to which we've scanned output_token_ids

    def scan_tokens(self) -> None:
        """Scan new output tokens to track thinking state.

        Cumulative: thinking_token_count never resets on new <think> blocks.
        After forced_stop, no further state changes are made.
        """
        if self.forced_stop:
            return

        tokens = self.output_token_ids
        end = len(tokens)

        # Handle speculative decode token rejection: if vLLM truncated
        # output_token_ids (rejected speculative tokens), rescan from scratch.
        if self.last_scanned > end:
            logger.debug(
                "output_token_ids shrank from %d to %d, rescanning",
                self.last_scanned, end,
            )
            self.last_scanned = 0
            self.thinking_token_count = 0
            # Restore initial prompt prefix state before rescanning output
            self.in_thinking = self.starts_in_thinking

        for i in range(self.last_scanned, end):
            tid = tokens[i]
            if tid == self.think_start_id:
                self.in_thinking = True
                # Don't reset thinking_token_count — budget is cumulative
            elif tid == self.think_end_id:
                self.in_thinking = False
            elif self.in_thinking:
                self.thinking_token_count += 1
        self.last_scanned = end


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    """Limits thinking tokens between <think> and </think>.

    Behavior:
    - At 80% of budget (for budgets > 10): soft nudge — boosts \\n and </think>
    - At budget-1: forces newline token (for budgets >= 2)
    - At budget: forces </think> token
    - budget=0: forces </think> immediately (no \\n prefix)
    - No budget in request: pure passthrough (no impact)
    - Budget is cumulative: re-entering <think> doesn't reset the count
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        ea = params.extra_args
        if ea is None:
            return
        budget = ea.get("max_thinking_tokens")
        if budget is not None:
            if not isinstance(budget, int) or budget < 0:
                raise ValueError(
                    f"max_thinking_tokens must be a non-negative integer, got {budget}"
                )

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        self.device = device

        # Resolve token IDs from vLLM's already-loaded cached tokenizer.
        # This avoids loading a duplicate tokenizer into CPU RAM and
        # eliminates redundant disk I/O at startup.
        model_name = vllm_config.model_config.model
        try:
            from vllm.transformers_utils.tokenizer import (
                cached_tokenizer_from_config,
            )
        except ImportError:
            from vllm.tokenizers import cached_tokenizer_from_config
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.think_start_id = tokenizer.convert_tokens_to_ids("<think>")
        self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        self.nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        logger.info(
            "ThinkingBudgetLogitsProcessor initialized: "
            "think_start=%d, think_end=%d, nl=%d, model=%s",
            self.think_start_id, self.think_end_id, self.nl_id, model_name,
        )

        # Per-request state, keyed by batch index
        self.requests: dict[int, RequestState] = {}
        # Track how many requests have active budgets (for fast passthrough)
        self.active_count = 0

    def _detect_thinking_from_prompt(
        self, prompt_tok_ids: list[int] | None,
    ) -> bool:
        """Detect if generation starts inside a thinking block.

        Scans the tail of the prompt for the last <think> or </think> token.
        If <think> comes after the last </think> (or there's no </think>),
        the model starts in thinking mode.

        Handles Qwen3.5's assistant prefix pattern: <|im_start|>assistant\\n<think>\\n
        """
        if not prompt_tok_ids:
            return False

        scan_end = len(prompt_tok_ids)
        scan_start = max(0, scan_end - _PROMPT_TAIL_SCAN)

        # Scan backwards: first <think> or </think> found determines state
        for i in range(scan_end - 1, scan_start - 1, -1):
            tid = prompt_tok_ids[i]
            if tid == self.think_start_id:
                return True
            if tid == self.think_end_id:
                return False

        return False

    def is_argmax_invariant(self) -> bool:
        # We modify logits when budget is active
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not batch_update:
            # No batch changes, but still scan tokens for existing requests
            for req in self.requests.values():
                req.scan_tokens()
            return

        # Process removals first
        for index in batch_update.removed:
            removed = self.requests.pop(index, None)
            if removed is not None:
                self.active_count -= 1

        # Process additions
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            # Remove any existing request at this index
            old = self.requests.pop(index, None)
            if old is not None:
                self.active_count -= 1

            # Check for thinking budget in extra_args
            # (vllm_xargs in API request maps to extra_args in SamplingParams)
            ea = params.extra_args
            budget = None
            if ea is not None:
                budget = ea.get("max_thinking_tokens")

            if budget is not None:
                starts_in_thinking = self._detect_thinking_from_prompt(
                    prompt_tok_ids,
                )

                state = RequestState(
                    budget=budget,
                    output_token_ids=output_tok_ids,
                    think_start_id=self.think_start_id,
                    think_end_id=self.think_end_id,
                    starts_in_thinking=starts_in_thinking,
                    in_thinking=starts_in_thinking,
                )
                self.requests[index] = state
                self.active_count += 1
                logger.info(
                    "Tracking request at index %d: budget=%d, "
                    "starts_in_thinking=%s",
                    index, budget, starts_in_thinking,
                )

        # Process moves
        for a_idx, b_idx, directionality in batch_update.moved:
            a_state = self.requests.pop(a_idx, None)
            b_state = self.requests.pop(b_idx, None)

            if a_state is not None:
                self.requests[b_idx] = a_state
            if b_state is not None:
                if directionality == MoveDirectionality.SWAP:
                    self.requests[a_idx] = b_state
                else:
                    # Unidirectional: b_state is being replaced
                    self.active_count -= 1

        # Scan tokens for all tracked requests
        for req in self.requests.values():
            req.scan_tokens()

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.active_count == 0:
            return logits

        for idx, req in self.requests.items():
            if req.forced_stop or not req.in_thinking:
                continue

            budget = req.budget
            count = req.thinking_token_count

            # budget=0: force </think> immediately, no \n prefix
            if budget == 0:
                mask = torch.full_like(logits[idx], float("-inf"))
                mask[self.think_end_id] = 0.0
                logits[idx] = mask
                req.forced_stop = True
                logger.info(
                    "Request %d: forced </think> at thinking token %d/%d",
                    idx, count, budget,
                )
                continue

            # Soft nudge: boost \n and </think> logits as budget approaches.
            # Only for budgets > 10 (small budgets go straight to hard cutoff).
            # Threshold is 80% to give a wider nudge window.
            nudge_start = int(budget * 0.8)
            if budget > 10 and count >= nudge_start and count < budget - 1:
                progress = (count - nudge_start) / (budget - 1 - nudge_start)
                scale = 1.0 + progress  # 1.0 → 2.0
                logits[idx, self.nl_id] += scale * 5.0
                logits[idx, self.think_end_id] += scale * 5.0

            # Hard cutoff: force \n at budget-1
            if count >= budget - 1 and count < budget:
                mask = torch.full_like(logits[idx], float("-inf"))
                mask[self.nl_id] = 0.0
                logits[idx] = mask

            # Hard cutoff: force </think> at budget
            elif count >= budget:
                mask = torch.full_like(logits[idx], float("-inf"))
                mask[self.think_end_id] = 0.0
                logits[idx] = mask
                req.forced_stop = True
                logger.info(
                    "Request %d: forced </think> at thinking token %d/%d",
                    idx, count, budget,
                )

        return logits
