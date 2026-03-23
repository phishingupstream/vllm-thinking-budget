#!/usr/bin/env python3
"""Patch vLLM AnthropicMessagesRequest to accept thinking configuration.

Problem:
    Anthropic clients send thinking control as:
        {"thinking": {"type": "enabled", "budget_tokens": 2048}}

    vLLM's Anthropic endpoint ignores this — no thinking field on
    AnthropicMessagesRequest, and _build_base_request doesn't pass
    thinking params to the internal ChatCompletionRequest.

Fix:
    1. Add a `thinking` field to AnthropicMessagesRequest
    2. In _build_base_request, map thinking config to ChatCompletionRequest fields:
       - thinking.type="enabled" + budget_tokens → enable_thinking=True + vllm_xargs
       - thinking.type="disabled" → enable_thinking=False

    The existing merge_thinking_params validator on ChatCompletionRequest
    then routes these into chat_template_kwargs and SamplingParams.extra_args.
"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/protocol.py"

with open(file_path, "r") as f:
    content = f.read()

if "[PATCH] anthropic thinking compat" in content:
    print("Already patched")
    exit(0)

# 1. Add thinking field to AnthropicMessagesRequest
old_fields = """    top_k: int | None = None
    top_p: float | None = None"""

new_fields = """    top_k: int | None = None
    top_p: float | None = None
    # [PATCH] anthropic thinking compat: accept thinking configuration
    thinking: dict[str, Any] | None = None"""

if old_fields not in content:
    print("ERROR: AnthropicMessagesRequest fields not found - vLLM version may differ")
    exit(1)

content = content.replace(old_fields, new_fields, 1)

# Ensure Any is imported (it should be already, but be safe)
if "from typing import Any" not in content and "Any" not in content.split("from typing import")[1].split("\n")[0] if "from typing import" in content else True:
    # Any should already be imported since the class uses dict[str, Any]
    pass

with open(file_path, "w") as f:
    f.write(content)

print("PATCHED (1/2): Added thinking field to AnthropicMessagesRequest")

# 2. Patch serving.py to pass thinking config through to ChatCompletionRequest
serving_path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py"

with open(serving_path, "r") as f:
    serving = f.read()

if "[PATCH] anthropic thinking compat" in serving:
    print("Already patched (serving)")
    exit(0)

# Find the _build_base_request return statement and add thinking params
old_return = """        return ChatCompletionRequest(
            model=anthropic_request.model,
            messages=openai_messages,
            max_tokens=anthropic_request.max_tokens,
            max_completion_tokens=anthropic_request.max_tokens,
            stop=anthropic_request.stop_sequences,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
        )"""

new_return = """        # [PATCH] anthropic thinking compat: map thinking config to
        # enable_thinking + vllm_xargs for the thinking budget processor.
        enable_thinking = None
        vllm_xargs = None
        thinking = getattr(anthropic_request, 'thinking', None)
        if thinking and isinstance(thinking, dict):
            thinking_type = thinking.get('type')
            if thinking_type == 'enabled':
                enable_thinking = True
                budget = thinking.get('budget_tokens')
                max_tok = anthropic_request.max_tokens or float('inf')
                # Anthropic clients set budget_tokens = max_tokens - 1 to mean
                # "no cap" (thinking counts against max_tokens in Anthropic API).
                # In vLLM, max_thinking_tokens is a separate cap, so passing
                # through a near-max budget defeats the processor. Only forward
                # genuinely restrictive budgets; otherwise let the
                # merge_thinking_params validator apply a sensible default.
                if (budget is not None and isinstance(budget, int)
                        and budget > 0 and budget < max_tok * 0.8):
                    vllm_xargs = {'max_thinking_tokens': budget}
            elif thinking_type == 'disabled':
                enable_thinking = False

        req_kwargs: dict = dict(
            model=anthropic_request.model,
            messages=openai_messages,
            max_tokens=anthropic_request.max_tokens,
            max_completion_tokens=anthropic_request.max_tokens,
            stop=anthropic_request.stop_sequences,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
        )
        if enable_thinking is not None:
            req_kwargs['enable_thinking'] = enable_thinking
        if vllm_xargs is not None:
            req_kwargs['vllm_xargs'] = vllm_xargs

        return ChatCompletionRequest(**req_kwargs)"""

if old_return not in serving:
    print("ERROR: _build_base_request return not found - vLLM version may differ")
    exit(1)

serving = serving.replace(old_return, new_return, 1)

with open(serving_path, "w") as f:
    f.write(serving)

print("PATCHED (2/2): Added thinking config passthrough in _build_base_request")
