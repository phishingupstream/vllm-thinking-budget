#!/usr/bin/env python3
"""Test harness for the thinking budget logits processor.

Tests thinking token budgets against the vLLM endpoint.
Budget is passed via vllm_xargs: {"max_thinking_tokens": N}

Usage:
    python3 tests/test_thinking_budget.py [--url http://localhost:8000] [--step N]
    python3 tests/test_thinking_budget.py --step 3  # Just hard cutoff tests
    python3 tests/test_thinking_budget.py --step 5  # Just tool calling tests
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

import httpx


@dataclass
class TestResult:
    name: str
    budget: int | None
    thinking_tokens: int  # from usage.completion_tokens_details or estimated
    thinking_chars: int
    content_chars: int
    finish_reason: str
    latency_ms: float
    passed: bool
    note: str = ""
    completion_tokens: int = 0


SIMPLE_PROMPT = "What is 2+2? Answer in one word."
MEDIUM_PROMPT = "Explain why the sky is blue in exactly two sentences."
HARD_PROMPT = "What is the integral of x^2 * e^x dx? Show your work briefly."
TOOL_PROMPT = "What is the current weather in London?"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["location"],
            },
        },
    }
]


def send_request(
    base_url: str,
    prompt: str,
    budget: int | None = None,
    tools: list | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
) -> TestResult:
    """Send a single request via raw HTTP and capture results."""
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
    }
    if budget is not None:
        body["vllm_xargs"] = {"max_thinking_tokens": budget}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return TestResult(
            name="",
            budget=budget,
            thinking_tokens=0,
            thinking_chars=0,
            content_chars=0,
            finish_reason="error",
            latency_ms=elapsed,
            passed=False,
            note=str(e)[:200],
        )

    elapsed = (time.monotonic() - start) * 1000
    choice = data["choices"][0]
    msg = choice["message"]

    reasoning = msg.get("reasoning") or ""
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

    thinking_chars = len(reasoning)
    content_chars = len(content)

    # Get token counts from usage
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    # Estimate thinking tokens: ~4 chars per token (rough)
    thinking_tokens_est = thinking_chars // 4

    note = ""
    if tool_calls:
        note = f"tool_calls={len(tool_calls)}"
        for tc in tool_calls:
            fn = tc.get("function", {})
            note += f" [{fn.get('name', '?')}({fn.get('arguments', '')})]"

    return TestResult(
        name="",
        budget=budget,
        thinking_tokens=thinking_tokens_est,
        thinking_chars=thinking_chars,
        content_chars=content_chars,
        finish_reason=choice.get("finish_reason", "unknown"),
        latency_ms=elapsed,
        passed=True,
        note=note,
        completion_tokens=completion_tokens,
    )


def run_baseline(base_url: str) -> list[TestResult]:
    """Step 0: Baseline characterization."""
    print("\n=== Step 0: Baseline (no budget) ===\n")
    results = []

    for name, prompt in [
        ("simple", SIMPLE_PROMPT),
        ("medium", MEDIUM_PROMPT),
        ("hard", HARD_PROMPT),
    ]:
        r = send_request(base_url, prompt)
        r.name = f"baseline_{name}"
        results.append(r)
        print(
            f"  {r.name}: thinking={r.thinking_chars}c/~{r.thinking_tokens}t, "
            f"content={r.content_chars}c, completion_tokens={r.completion_tokens}, "
            f"finish={r.finish_reason}, latency={r.latency_ms:.0f}ms"
        )

    # Tool calling baseline
    r = send_request(base_url, TOOL_PROMPT, tools=TOOLS)
    r.name = "baseline_tool"
    results.append(r)
    print(
        f"  {r.name}: thinking={r.thinking_chars}c/~{r.thinking_tokens}t, "
        f"content={r.content_chars}c, finish={r.finish_reason}, "
        f"latency={r.latency_ms:.0f}ms, {r.note}"
    )

    return results


def run_step1_noop(base_url: str) -> list[TestResult]:
    """Step 1: Verify no-op (no budget set) - processor loaded but inactive."""
    print("\n=== Step 1: No-Op Verification (no budget in request) ===\n")
    results = []

    for name, prompt in [
        ("simple", SIMPLE_PROMPT),
        ("medium", MEDIUM_PROMPT),
    ]:
        r = send_request(base_url, prompt)
        r.name = f"noop_{name}"
        results.append(r)
        print(
            f"  {r.name}: thinking={r.thinking_chars}c/~{r.thinking_tokens}t, "
            f"content={r.content_chars}c, finish={r.finish_reason}, "
            f"completion_tokens={r.completion_tokens}"
        )
        r.passed = r.finish_reason in ("stop", "length")

    return results


def run_step2_tracking(base_url: str) -> list[TestResult]:
    """Step 2: State tracking (huge budget = effective passthrough)."""
    print("\n=== Step 2: State Tracking (huge budget = passthrough) ===\n")
    results = []

    for name, prompt in [
        ("simple", SIMPLE_PROMPT),
        ("medium", MEDIUM_PROMPT),
    ]:
        r = send_request(base_url, prompt, budget=50000)
        r.name = f"tracking_{name}"
        results.append(r)
        print(
            f"  {r.name}: thinking={r.thinking_chars}c/~{r.thinking_tokens}t, "
            f"content={r.content_chars}c, finish={r.finish_reason}, "
            f"completion_tokens={r.completion_tokens}"
        )
        r.passed = r.finish_reason in ("stop", "length")

    return results


def run_step3_cutoff(base_url: str) -> list[TestResult]:
    """Step 3: Hard cutoff tests."""
    print("\n=== Step 3: Hard Cutoff Tests ===\n")
    results = []

    test_cases = [
        ("T3.1_no_budget", MEDIUM_PROMPT, None, "Should match baseline"),
        ("T3.2_budget_2048", HARD_PROMPT, 2048, "Thinking truncated ~2048"),
        ("T3.3_budget_512", HARD_PROMPT, 512, "Moderate thinking"),
        ("T3.4_budget_128", HARD_PROMPT, 128, "Very short thinking"),
        ("T3.5_budget_0", SIMPLE_PROMPT, 0, "Immediate </think>"),
        ("T3.6_huge_budget", MEDIUM_PROMPT, 50000, "Should match baseline"),
    ]

    for name, prompt, budget, desc in test_cases:
        r = send_request(base_url, prompt, budget=budget)
        r.name = name
        results.append(r)

        # Validation
        if budget is not None and budget < 10000:
            # Check thinking chars are plausible for budget
            # ~4 chars/token, so budget*4 is rough max chars
            max_chars = (budget + 5) * 6  # generous margin
            if budget > 0 and r.thinking_chars > max_chars:
                r.passed = False
                r.note = f"OVER BUDGET: ~{r.thinking_tokens}t >> {budget}"

        if budget == 0:
            # With budget=0, there should be minimal/no thinking
            if r.thinking_chars > 50:
                r.passed = False
                r.note = f"Budget=0 but thinking_chars={r.thinking_chars}"

        # Content should exist after budget kicks in
        if budget is not None and budget < 5000 and r.content_chars == 0:
            if r.finish_reason != "tool_calls":
                r.note += " WARNING: no content after forced </think>"

        print(
            f"  {name} (budget={budget}): thinking={r.thinking_chars}c/~{r.thinking_tokens}t, "
            f"content={r.content_chars}c, completion_tokens={r.completion_tokens}, "
            f"finish={r.finish_reason} {'PASS' if r.passed else 'FAIL'} "
            f"{r.note or desc}"
        )

    return results


def run_step5_tool_calling(base_url: str) -> list[TestResult]:
    """Step 5: Tool calling with budgets."""
    print("\n=== Step 5: Tool Calling + Budget ===\n")
    results = []

    for name, budget in [
        ("tool_no_budget", None),
        ("tool_budget_2048", 2048),
        ("tool_budget_512", 512),
        ("tool_budget_128", 128),
        ("tool_budget_0", 0),
    ]:
        r = send_request(base_url, TOOL_PROMPT, budget=budget, tools=TOOLS)
        r.name = name
        results.append(r)

        # Check tool call parsed
        has_tool = "tool_calls=" in r.note
        if not has_tool and r.finish_reason != "stop":
            r.note += " (no tool call)"

        print(
            f"  {name} (budget={budget}): thinking={r.thinking_chars}c/~{r.thinking_tokens}t, "
            f"finish={r.finish_reason} {'PASS' if r.passed else 'FAIL'} {r.note}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Test thinking budget processor")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="vLLM base URL"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Run specific step (0,1,2,3,5). Default: run all.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Quick health check
    try:
        resp = httpx.get(f"{args.url}/v1/models", timeout=10.0)
        models = [m["id"] for m in resp.json()["data"]]
        print(f"Connected to {args.url}, models: {models}")
    except Exception as e:
        print(f"ERROR: Cannot connect to {args.url}: {e}")
        sys.exit(1)

    all_results = []
    available_steps = [0, 1, 2, 3, 5]
    steps = available_steps if args.step is None else [args.step]

    for step in steps:
        if step == 0:
            all_results.extend(run_baseline(args.url))
        elif step == 1:
            all_results.extend(run_step1_noop(args.url))
        elif step == 2:
            all_results.extend(run_step2_tracking(args.url))
        elif step == 3:
            all_results.extend(run_step3_cutoff(args.url))
        elif step == 5:
            all_results.extend(run_step5_tool_calling(args.url))

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"Results saved to {args.output}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
