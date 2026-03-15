"""Single forecasting agent: 5-stage pipeline with agentic search."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from dataclasses import dataclass

import anthropic

from .events import EventType, NullHandler, PipelineEvent, Stage, track_api_cost
from .search import execute_search
from .types import (
    AgentTrace,
    BaseRateOutput,
    DecomposeOutput,
    InsideViewOutput,
    Question,
    ResearchOutput,
    SearchRound,
    SearchTrace,
    SynthesisOutput,
    clean_schema,
)
from .utils import extract_json, load_memory, load_prompt

MODEL = "claude-opus-4-6"
FAST_MODEL = "claude-haiku-4-5-20251001"
MAX_SEARCH_ROUNDS = 3


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "anthropic", "openai", "google"
    model_id: str
    api_key_env: str


# Agent 0: Claude Opus, Agent 1: GPT 5.4 Pro, Agent 2: Gemini 3.1 Pro Preview
AGENT_MODELS: dict[int, ModelConfig] = {
    0: ModelConfig(provider="anthropic", model_id="claude-opus-4-6", api_key_env="ANTHROPIC_API_KEY"),
    1: ModelConfig(provider="openai", model_id="gpt-5.4", api_key_env="OPENAI_API_KEY"),
    2: ModelConfig(provider="google", model_id="gemini-3.1-pro-preview", api_key_env="GEMINI_API_KEY"),
}


async def _run_stage(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    user_message: str,
    output_model: type,
    temperature: float = 1.0,
    handler=None,
) -> object:
    """Run a single pipeline stage using tool_use for structured output."""
    schema = output_model.model_json_schema()
    # Remove $defs and other JSON Schema features that aren't needed for tool input
    tool_schema = clean_schema(schema)

    tool = {
        "name": "provide_output",
        "description": f"Provide the {output_model.__name__} output",
        "input_schema": tool_schema,
    }

    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "provide_output"},
    )

    if response.usage:
        await track_api_cost(handler, "anthropic", MODEL, response.usage.input_tokens, response.usage.output_tokens)

    # Extract tool use input
    for block in response.content:
        if block.type == "tool_use":
            return output_model.model_validate(block.input)

    raise RuntimeError(f"No tool_use block in response for {output_model.__name__}")


def _openai_strict_fixup(schema: dict) -> None:
    """Recursively enforce OpenAI strict mode: additionalProperties false,
    required includes every property, defaults removed (not allowed in strict)."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        # Strict mode: every property must be in required
        schema["required"] = list(schema["properties"].keys())
        for prop in schema["properties"].values():
            # Remove default values — OpenAI strict mode forbids them
            prop.pop("default", None)
            _openai_strict_fixup(prop)
    if schema.get("type") == "array" and "items" in schema:
        _openai_strict_fixup(schema["items"])
    # Handle anyOf/oneOf (e.g. Optional fields)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for variant in schema[key]:
                _openai_strict_fixup(variant)


async def _run_stage_openai(
    model_config: ModelConfig,
    system_prompt: str,
    user_message: str,
    output_model: type,
    temperature: float = 1.0,
    handler=None,
) -> object:
    """Run a single pipeline stage using OpenAI function calling for structured output."""
    import openai

    schema = output_model.model_json_schema()
    tool_schema = clean_schema(schema)
    # OpenAI strict mode requires additionalProperties: false and all properties
    # listed as required on every object, recursively
    _openai_strict_fixup(tool_schema)

    tool = {
        "type": "function",
        "function": {
            "name": "provide_output",
            "description": f"Provide the {output_model.__name__} output",
            "parameters": tool_schema,
            "strict": True,
        },
    }

    api_key = os.environ.get(model_config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var {model_config.api_key_env}")

    client = openai.AsyncOpenAI(api_key=api_key, timeout=120.0)
    response = await client.chat.completions.create(
        model=model_config.model_id,
        temperature=temperature,
        max_completion_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "provide_output"}},
    )

    if response.usage:
        await track_api_cost(handler, "openai", model_config.model_id, response.usage.prompt_tokens, response.usage.completion_tokens)

    # Extract function call arguments
    msg = response.choices[0].message
    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        return output_model.model_validate(args)

    raise RuntimeError(f"No tool_calls in OpenAI response for {output_model.__name__}")


async def _run_stage_gemini(
    model_config: ModelConfig,
    system_prompt: str,
    user_message: str,
    output_model: type,
    temperature: float = 1.0,
    handler=None,
) -> object:
    """Run a single pipeline stage using Google GenAI function calling for structured output."""
    from google import genai
    from google.genai import types as genai_types

    schema = output_model.model_json_schema()
    tool_schema = clean_schema(schema)

    # Build the function declaration for Gemini
    func_decl = genai_types.FunctionDeclaration(
        name="provide_output",
        description=f"Provide the {output_model.__name__} output",
        parameters=tool_schema,
    )
    tool = genai_types.Tool(function_declarations=[func_decl])

    api_key = os.environ.get(model_config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var {model_config.api_key_env}")

    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=4096,
        tools=[tool],
        tool_config=genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["provide_output"],
            ),
        ),
    )

    # google-genai client's generate_content is sync; run in executor
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model_config.model_id,
            contents=user_message,
            config=config,
        ),
    )

    if response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
        await track_api_cost(handler, "google", model_config.model_id, input_tokens, output_tokens)

    # Extract function call from response
    for part in response.candidates[0].content.parts:
        if part.function_call:
            args = dict(part.function_call.args) if part.function_call.args else {}
            return output_model.model_validate(args)

    raise RuntimeError(f"No function_call in Gemini response for {output_model.__name__}")


async def _run_stage_dispatch(
    model_config: ModelConfig,
    system_prompt: str,
    user_message: str,
    output_model: type,
    temperature: float = 1.0,
    handler=None,
) -> object:
    """Dispatch a stage call to the right provider."""
    if model_config.provider == "anthropic":
        client = anthropic.AsyncAnthropic(timeout=120.0)
        return await _run_stage(
            client, system_prompt, user_message, output_model,
            temperature=temperature, handler=handler,
        )
    elif model_config.provider == "openai":
        return await _run_stage_openai(
            model_config, system_prompt, user_message, output_model,
            temperature=temperature, handler=handler,
        )
    elif model_config.provider == "google":
        return await _run_stage_gemini(
            model_config, system_prompt, user_message, output_model,
            temperature=temperature, handler=handler,
        )
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")


async def _run_agentic_search(
    question: Question,
    initial_queries: list[str],
    close_date: str,
    api_key: str | None = None,
    handler=None,
    agent_id: int | None = None,
) -> SearchTrace:
    """Execute iterative search: run initial queries in parallel, then follow up on gaps."""
    _handler = handler or NullHandler()

    # Fire all initial queries in parallel
    for q in initial_queries:
        await _handler.handle(PipelineEvent(
            event_type=EventType.SEARCH_START, question_id=question.question_id,
            agent_id=agent_id, stage=Stage.RESEARCH, data={"query": q},
        ))
    initial_results = await asyncio.gather(*[
        execute_search(query, close_date, api_key, handler=_handler, question_title=question.title)
        for query in initial_queries
    ])
    for q in initial_queries:
        await _handler.handle(PipelineEvent(
            event_type=EventType.SEARCH_DONE, question_id=question.question_id,
            agent_id=agent_id, stage=Stage.RESEARCH, data={"query": q},
        ))

    rounds: list[SearchRound] = [
        SearchRound(
            round_number=i + 1,
            query=query,
            result=result,
            reasoning="Initial query from decomposition",
        )
        for i, (query, result) in enumerate(zip(initial_queries, initial_results))
    ]

    # Additional rounds driven by Haiku deciding what to search next
    if len(rounds) < MAX_SEARCH_ROUNDS:
        client = anthropic.AsyncAnthropic(timeout=120.0)
        for round_num in range(len(rounds) + 1, MAX_SEARCH_ROUNDS + 1):
            search_summary = "\n\n".join([
                f"**Search {r.round_number}: {r.query}**\n{r.result.content[:1000]}"
                for r in rounds
            ])

            follow_up_response = await client.messages.create(
                model=FAST_MODEL,
                max_tokens=1024,
                temperature=1.0,
                system="You are a research assistant. Given the question and search results so far, decide if more searching would help. If yes, respond with a JSON object: {\"search\": true, \"query\": \"your query\", \"reasoning\": \"why\"}. If no, respond with: {\"search\": false, \"reasoning\": \"why not\"}.",
                messages=[{"role": "user", "content": f"Question: {question.title}\n\nSearch results so far:\n{search_summary}\n\nShould we search for more information? Respond with JSON only."}],
            )

            if follow_up_response.usage:
                await track_api_cost(_handler, "anthropic", FAST_MODEL, follow_up_response.usage.input_tokens, follow_up_response.usage.output_tokens)

            text = follow_up_response.content[0].text
            try:
                decision = extract_json(text)
            except json.JSONDecodeError:
                break

            if not decision.get("search", False):
                break

            await _handler.handle(PipelineEvent(
                event_type=EventType.SEARCH_START, question_id=question.question_id,
                agent_id=agent_id, stage=Stage.RESEARCH, data={"query": decision["query"]},
            ))
            result = await execute_search(decision["query"], close_date, api_key, handler=_handler, question_title=question.title)
            await _handler.handle(PipelineEvent(
                event_type=EventType.SEARCH_DONE, question_id=question.question_id,
                agent_id=agent_id, stage=Stage.RESEARCH, data={"query": decision["query"]},
            ))
            rounds.append(SearchRound(
                round_number=round_num,
                query=decision["query"],
                result=result,
                reasoning=decision.get("reasoning", ""),
            ))

    total_citations = sum(len(r.result.citations) for r in rounds)
    return SearchTrace(rounds=rounds, total_citations=total_citations)


async def run_agent(
    question: Question,
    agent_id: int,
    memory: str | None = None,
    handler=None,
) -> AgentTrace:
    """Run full 5-stage forecasting pipeline for one agent."""
    _handler = handler or NullHandler()
    model_config = AGENT_MODELS.get(agent_id, AGENT_MODELS[0])
    memory = memory if memory is not None else load_memory()
    memory_section = f"\n\n## Accumulated Forecasting Lessons\n{memory}" if memory else ""

    _evt = lambda et, stage=None, **data: _handler.handle(PipelineEvent(
        event_type=et, question_id=question.question_id,
        question_title=question.title, agent_id=agent_id, stage=stage, data=data,
    ))

    # Stage 1: Decompose
    await _evt(EventType.AGENT_STAGE_START, Stage.DECOMPOSE)
    decompose_prompt = load_prompt("decompose")
    decompose_msg = f"Question: {question.title}\n\nClose date: {question.close_date}\nDomain: {question.domain.value}\nTags: {', '.join(question.tags)}{memory_section}"

    decompose = await _run_stage_dispatch(
        model_config, decompose_prompt, decompose_msg,
        DecomposeOutput, temperature=1.0, handler=_handler,
    )
    await _evt(EventType.AGENT_STAGE_DONE, Stage.DECOMPOSE,
               sub_questions=decompose.sub_questions)

    # Stage 2: Research (agentic search loop)
    await _evt(EventType.AGENT_STAGE_START, Stage.RESEARCH)
    search_trace = await _run_agentic_search(
        question, decompose.initial_search_queries, question.close_date,
        handler=_handler, agent_id=agent_id,
    )

    # Synthesize research findings via LLM (still part of research stage)
    research_prompt = load_prompt("research")
    search_summary = "\n\n".join([
        f"**Search {r.round_number}: {r.query}**\n{r.result.content}"
        for r in search_trace.rounds
    ])
    research_msg = (
        f"Question: {question.title}\n\n"
        f"Sub-questions:\n" + "\n".join(f"- {sq}" for sq in decompose.sub_questions) +
        f"\n\nSearch Results:\n{search_summary}{memory_section}"
    )

    # For research output, we need to inject the search trace after
    research_raw = await _run_stage_dispatch(
        model_config, research_prompt, research_msg,
        ResearchOutput, temperature=1.0, handler=_handler,
    )
    # Override the search trace with the actual one
    research = ResearchOutput(
        key_findings=research_raw.key_findings,
        evidence_for=research_raw.evidence_for,
        evidence_against=research_raw.evidence_against,
        search_trace=search_trace,
        information_gaps=research_raw.information_gaps,
    )
    await _evt(EventType.AGENT_STAGE_DONE, Stage.RESEARCH,
               key_findings=research.key_findings[:3])

    # Stage 3: Base Rate
    await _evt(EventType.AGENT_STAGE_START, Stage.BASE_RATE)
    base_rate_prompt = load_prompt("base_rate")
    base_rate_msg = (
        f"Question: {question.title}\n\n"
        f"Key findings:\n" + "\n".join(f"- {f}" for f in research.key_findings) +
        f"\n\nEvidence for Yes:\n" + "\n".join(f"- {e}" for e in research.evidence_for) +
        f"\n\nEvidence against:\n" + "\n".join(f"- {e}" for e in research.evidence_against) +
        f"{memory_section}"
    )

    base_rate = await _run_stage_dispatch(
        model_config, base_rate_prompt, base_rate_msg,
        BaseRateOutput, temperature=1.0, handler=_handler,
    )
    await _evt(EventType.AGENT_STAGE_DONE, Stage.BASE_RATE,
               base_rate_estimate=base_rate.base_rate_estimate)

    # Stage 4: Inside View
    await _evt(EventType.AGENT_STAGE_START, Stage.INSIDE_VIEW)
    inside_view_prompt = load_prompt("inside_view")
    inside_view_msg = (
        f"Question: {question.title}\n\n"
        f"Base rate estimate: {base_rate.base_rate_estimate:.2%}\n"
        f"Reference classes: {', '.join(base_rate.reference_classes)}\n\n"
        f"Key findings:\n" + "\n".join(f"- {f}" for f in research.key_findings) +
        f"\n\nEvidence for Yes:\n" + "\n".join(f"- {e}" for e in research.evidence_for) +
        f"\n\nEvidence against:\n" + "\n".join(f"- {e}" for e in research.evidence_against) +
        f"\n\nInformation gaps:\n" + "\n".join(f"- {g}" for g in research.information_gaps) +
        f"{memory_section}"
    )

    inside_view = await _run_stage_dispatch(
        model_config, inside_view_prompt, inside_view_msg,
        InsideViewOutput, temperature=1.0, handler=_handler,
    )
    await _evt(EventType.AGENT_STAGE_DONE, Stage.INSIDE_VIEW,
               factors_for=inside_view.factors_for, factors_against=inside_view.factors_against)

    # Stage 5: Synthesize
    await _evt(EventType.AGENT_STAGE_START, Stage.SYNTHESIZE)
    synthesize_prompt = load_prompt("synthesize")
    synthesize_msg = (
        f"Question: {question.title}\n\n"
        f"Base rate: {base_rate.base_rate_estimate:.2%}\n"
        f"Base rate reasoning: {base_rate.reasoning}\n\n"
        f"Inside view: {inside_view.inside_view_estimate:.2%}\n"
        f"Inside view reasoning: {inside_view.reasoning}\n\n"
        f"Factors for Yes:\n" + "\n".join(f"- {f}" for f in inside_view.factors_for) +
        f"\n\nFactors against:\n" + "\n".join(f"- {f}" for f in inside_view.factors_against) +
        f"{memory_section}"
    )

    synthesis = await _run_stage_dispatch(
        model_config, synthesize_prompt, synthesize_msg,
        SynthesisOutput, temperature=1.0, handler=_handler,
    )
    await _evt(EventType.AGENT_STAGE_DONE, Stage.SYNTHESIZE,
               final_probability=synthesis.final_probability)

    return AgentTrace(
        agent_id=agent_id,
        question_id=question.question_id,
        model_id=model_config.model_id,
        decompose=decompose,
        research=research,
        base_rate=base_rate,
        inside_view=inside_view,
        synthesis=synthesis,
        raw_probability=synthesis.final_probability,
    )
