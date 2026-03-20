#!/usr/bin/env python3
"""Patch vLLM ChatCompletionRequest to accept enable_thinking and reasoning_effort.

Problem:
    OpenAI-compatible clients send these top-level fields that vLLM ignores:
    - enable_thinking (bool): third-party clients
    - reasoning_effort (str): OpenAI standard for o1/o3/GPT-5 ("low"/"medium"/"high")

    vLLM's native mechanism is chat_template_kwargs for thinking control and
    vllm_xargs for max_thinking_tokens. Clients shouldn't need to know this.

Fix:
    Add both fields to ChatCompletionRequest with a model_validator that maps:
    - enable_thinking → chat_template_kwargs["enable_thinking"]
    - reasoning_effort → chat_template_kwargs["enable_thinking"] + vllm_xargs["max_thinking_tokens"]

    Mapping (reasoning_effort → max_thinking_tokens):
    - "none"   → enable_thinking=false
    - "low"    → enable_thinking=true, max_thinking_tokens=2048
    - "medium" → enable_thinking=true, max_thinking_tokens=8192
    - "high"   → enable_thinking=true (no budget cap)

    Precedence: explicit enable_thinking > reasoning_effort > chat_template_kwargs
"""
import re

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py"

with open(file_path, "r") as f:
    content = f.read()

if "[PATCH] reasoning_effort compat" in content:
    print("Already patched")
    exit(0)

# 1. Add enable_thinking field after chat_template_kwargs field
old_field = '''    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )'''

new_field = '''    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    # [PATCH] reasoning_effort compat: accept top-level fields from OpenAI-compatible clients.
    enable_thinking: bool | None = Field(
        default=None,
        description=(
            "Compat: enable or disable model reasoning/thinking. "
            "Merged into chat_template_kwargs['enable_thinking'] at validation time."
        ),
    )
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "OpenAI-compatible reasoning effort level: 'none', 'low', 'medium', 'high'. "
            "Maps to enable_thinking + max_thinking_tokens budget."
        ),
    )'''

if old_field not in content:
    print("ERROR: chat_template_kwargs field not found - vLLM version may differ")
    exit(1)

content = content.replace(old_field, new_field, 1)

# 2. Add a model_validator to merge enable_thinking into chat_template_kwargs.
#    Insert before the first @model_validator in the class.
old_validator_anchor = '''    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):'''

new_validator = '''    # [PATCH] reasoning_effort compat
    @model_validator(mode="before")
    @classmethod
    def merge_thinking_params(cls, data):
        # [PATCH] reasoning_effort compat: map reasoning_effort and enable_thinking
        # into chat_template_kwargs and vllm_xargs for the thinking budget processor.
        #
        # Precedence: explicit enable_thinking > reasoning_effort > chat_template_kwargs
        ctk = data.get("chat_template_kwargs") or {}
        xargs = data.get("vllm_xargs") or {}
        effort = data.get("reasoning_effort")
        enable = data.get("enable_thinking")

        # Default to medium if no thinking control is specified
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
            # Set enable_thinking if not explicitly provided
            if enable is None and "enable_thinking" not in ctk:
                ctk["enable_thinking"] = ef_enable
            # Set max_thinking_tokens if not already in vllm_xargs
            if ef_budget is not None and "max_thinking_tokens" not in xargs:
                xargs["max_thinking_tokens"] = ef_budget

        # Merge explicit enable_thinking (highest precedence)
        if enable is not None and "enable_thinking" not in ctk:
            ctk["enable_thinking"] = enable

        # If thinking is enabled but no budget was set (e.g. Anthropic client
        # sent budget_tokens ~ max_tokens which was filtered out, or client
        # only sent enable_thinking=True), apply the "medium" default budget
        # so the thinking_budget_processor has a real cap.
        if ctk.get("enable_thinking") and "max_thinking_tokens" not in xargs:
            xargs["max_thinking_tokens"] = 8192  # "medium" default

        if ctk:
            data["chat_template_kwargs"] = ctk
        if xargs:
            data["vllm_xargs"] = xargs
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):'''

if old_validator_anchor not in content:
    print("ERROR: validate_stream_options not found - vLLM version may differ")
    exit(1)

content = content.replace(old_validator_anchor, new_validator, 1)

with open(file_path, "w") as f:
    f.write(content)

print("PATCHED: Added enable_thinking + reasoning_effort compat to ChatCompletionRequest")
