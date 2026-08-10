"""
================================================================================
 APEX HACKATHON AI — LangGraph Implementation
================================================================================
 A bounded cyclic multi-agent graph that turns a hackathon problem statement
 into an APPROVED, $0-budget, novel technical design plus full deliverables
 (6-slide SIH pitch deck outline + runnable repository scaffold).

 GRAPH TOPOLOGY
 --------------
   START -> adversarial_researcher -> shoestring_architect -> red_team_judge
   red_team_judge --(APPROVED)-------------> deliverables_master -> END
   red_team_judge --(REJECT_DESIGN)--------> shoestring_architect
   red_team_judge --(REJECT_CONCEPT)-------> adversarial_researcher
   red_team_judge --(FORCED_EXIT, i>=max)--> best_effort_finalizer -> deliverables_master

 SETUP
 -----
 1) Install dependencies (pick the provider packages you need):

    pip install langgraph langchain-core pydantic python-dotenv
    pip install langchain-google-genai      # provider: gemini
    pip install langchain-openai            # provider: openrouter (GLM etc.)
    pip install langchain-groq              # provider: groq
    pip install langchain-anthropic         # provider: anthropic
    pip install ddgs                        # OPTIONAL: live web search for the Researcher/Architect

 2) Create a `.env` file next to this script:

    # ── .env template ─────────────────────────────────────────────
    # Which provider to use: gemini | openrouter | groq | anthropic
    LLM_PROVIDER=gemini

    # Provider API keys (only the one matching LLM_PROVIDER is required)
    GOOGLE_API_KEY=your-google-api-key
    OPENROUTER_API_KEY=your-openrouter-api-key
    GROQ_API_KEY=your-groq-api-key
    ANTHROPIC_API_KEY=your-anthropic-api-key

    # Optional model overrides (sensible defaults are used if unset)
    GEMINI_MODEL=gemini-2.0-flash
    OPENROUTER_MODEL=z-ai/glm-4.6
    GROQ_MODEL=llama-3.3-70b-versatile
    ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
    # ──────────────────────────────────────────────────────────────

 3) Run:

    python apex_hackathon_ai.py
================================================================================
"""

from __future__ import annotations

import json
import operator
import os
import time
from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ==============================================================================
# 1. PYDANTIC SCHEMAS — guarantee route_from_judge never sees malformed data
# ==============================================================================


class MarketFlaw(BaseModel):
    """A documented weakness in an existing product/solution in this space."""

    product_name: str = Field(description="Name of the existing product, repo, or prior SIH winner.")
    flaw_description: str = Field(description="The specific, concrete weakness or gap in that product.")
    exploitation_angle: str = Field(description="How a new hackathon solution can exploit this flaw.")


class ResearcherOutput(BaseModel):
    """Structured output of the Adversarial Researcher node."""

    market_flaws: List[MarketFlaw] = Field(
        description="3-6 documented flaws in existing solutions, competitors, or prior winners."
    )
    innovation_gap: str = Field(
        description=(
            "A single, sharply-worded thesis describing the unoccupied innovation space "
            "that a genuinely novel solution can claim. Must NOT be satisfiable by a thin "
            "wrapper around an existing API."
        )
    )


class TechDesign(BaseModel):
    """Structured output of the Shoestring Architect node."""

    solution_summary: str = Field(description="2-4 sentence summary of the proposed solution.")
    tech_stack: List[str] = Field(
        description='Concrete free/open-source components, e.g. ["SQLite", "FastAPI", "llama.cpp Q4"].'
    )
    cost_estimate_usd: float = Field(
        description="Total monetary cost to build AND run the MVP. Must be ~0.0."
    )
    offline_strategy: str = Field(
        description="Explicit plan for low-bandwidth / rural / offline operation."
    )
    novelty_argument: str = Field(
        description=(
            "Explicit defense against the charge of being 'just an API wrapper'. "
            "Must name the architectural or algorithmic contribution that is original."
        )
    )
    maintenance_plan: str = Field(
        description="How the system stays operable in real life without prohibitive maintenance or ops cost."
    )


class JudgeCritique(BaseModel):
    """Structured output of the Red Team Judge node."""

    iteration: int = Field(default=0, description="Filled in programmatically by the judge node.")
    novelty_score: int = Field(ge=1, le=10, description="1-10. Pass threshold: >= 7.")
    feasibility_score: int = Field(ge=1, le=10, description="1-10. Pass threshold: >= 8.")
    scale_score: int = Field(ge=1, le=10, description="1-10. Pass threshold: >= 7.")
    fatal_flaw: str = Field(description="Single sentence: the judge's harshest, most specific objection.")
    required_fix: str = Field(description="One actionable instruction for the node receiving the rejection.")


class ScoredDesign(BaseModel):
    """A (design, scores) archive entry used by the Best-Effort Finalizer."""

    design: TechDesign
    novelty_score: int
    feasibility_score: int
    scale_score: int

    @property
    def total_score(self) -> int:
        return self.novelty_score + self.feasibility_score + self.scale_score


class Deliverables(BaseModel):
    """Structured output of the Deliverables Master node."""

    pitch_deck: List[str] = Field(
        description="Exactly 6 slide outlines in SIH format. Each entry: 'Slide N — Title: bullet; bullet; bullet'."
    )
    repo_tree: str = Field(description="ASCII tree of the repository scaffold.")
    mvp_scaffold_files: dict = Field(
        description="Mapping of file path -> complete file content for the runnable MVP skeleton."
    )
    readme: str = Field(description="Complete README.md content including setup and run instructions.")


# ==============================================================================
# 2. TYPED STATE — explicit reducers for append-only channels
# ==============================================================================


class ApexState(TypedDict, total=False):
    # ── OVERWRITTEN each pass (last-write-wins) ───────────────────────────────
    problem_statement: str
    innovation_gap: str
    current_design: Optional[TechDesign]
    judge_verdict: Literal["APPROVED", "REJECT_CONCEPT", "REJECT_DESIGN", "FORCED_EXIT", "PENDING"]
    deliverables: Optional[Deliverables]

    # ── APPENDED via reducers ─────────────────────────────────────────────────
    market_flaws: Annotated[List[MarketFlaw], operator.add]
    critique_log: Annotated[List[JudgeCritique], operator.add]
    design_history: Annotated[List[ScoredDesign], operator.add]

    # ── COUNTERS (overwritten, incremented by Judge) ──────────────────────────
    iteration_count: int
    max_iterations: int


# ==============================================================================
# 3. UNIVERSAL LLM FACTORY + RETRY WRAPPER
# ==============================================================================


def get_llm(node_role: str):
    """
    Dynamically initialize a chat model based on the LLM_PROVIDER env var.

    Supported providers:
      - "gemini"     -> langchain_google_genai.ChatGoogleGenerativeAI
      - "openrouter" -> langchain_openai.ChatOpenAI pointed at OpenRouter (GLM etc.)
      - "groq"       -> langchain_groq.ChatGroq
      - "anthropic"  -> langchain_anthropic.ChatAnthropic

    The judge runs cold (temperature 0) for deterministic rubric application;
    creative nodes run warmer.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    temperature = 0.0 if node_role in ("red_team_judge",) else 0.7

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "z-ai/glm-4.6"),
            temperature=temperature,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider}'. "
            "Use one of: gemini, openrouter, groq, anthropic."
        )


def invoke_structured_with_retry(node_role: str, schema, system_prompt: str, user_prompt: str, max_retries: int = 3):
    """
    Call the LLM with structured output and basic exponential-backoff retry.
    Guarantees a validated Pydantic object or raises after max_retries.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            llm = get_llm(node_role)
            structured_llm = llm.with_structured_output(schema)
            result = structured_llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            if result is None:
                raise ValueError("LLM returned None for structured output.")
            return result
        except Exception as exc:  # noqa: BLE001 — deliberate broad retry boundary
            last_error = exc
            wait = 2 ** attempt
            print(f"[retry] {node_role} attempt {attempt}/{max_retries} failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"{node_role} failed after {max_retries} attempts: {last_error}")


# ==============================================================================
# 4. OPTIONAL WEB SEARCH TOOL (graceful degradation if ddgs is not installed)
# ==============================================================================


def web_search(query: str, max_results: int = 5) -> str:
    """
    Lightweight web search via DuckDuckGo (ddgs). If the package is missing
    or the network call fails, returns a note so the graph still runs end-to-end.
    """
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                results.append(f"- {title}: {body} ({href})")
        if results:
            return "\n".join(results)
        return "(web search returned no results)"
    except Exception as exc:  # noqa: BLE001 — search is optional by design
        return f"(web search unavailable: {exc}. Reason from internal knowledge instead.)"


# ==============================================================================
# 5. NODE IMPLEMENTATIONS
# ==============================================================================

# ── Node 1: Adversarial Researcher ────────────────────────────────────────────

RESEARCHER_SYSTEM_PROMPT = """You are the ADVERSARIAL RESEARCHER in a hackathon strategy pipeline.

Your job: attack the existing solution landscape for the given problem statement and
extract a genuinely novel INNOVATION GAP.

MANDATORY NOVELTY SAFEGUARD (non-negotiable):
- You MUST explicitly FAIL and DISCARD any candidate direction that is merely a
  "thin wrapper around an existing API" (e.g., "we call GPT-4 / Gemini with a custom
  prompt and show the answer"). Such directions are AUTOMATIC DISQUALIFICATIONS.
- You MUST discard any direction that is an unoriginal copy of an existing product,
  a prior Smart India Hackathon winner, or a well-known GitHub project.
- The innovation gap you output must require an original architectural or algorithmic
  contribution — something a judge could not dismiss as "this already exists".

METHOD:
1. Enumerate existing products, open-source repos, and prior hackathon winners in this space.
2. For each, extract a DOCUMENTED, SPECIFIC flaw (not a vague "it could be better").
3. Synthesize the flaws into one sharp innovation-gap thesis that a $0-budget student
   team could plausibly claim.

If a critique log from a Red Team Judge is provided, you MUST read every entry and
avoid re-proposing any direction that was already rejected. Each rejection narrows
your search space — respect it.

Output strictly in the requested structured format."""


def adversarial_researcher(state: ApexState) -> dict:
    print(f"\n{'=' * 70}\n[NODE] Adversarial Researcher (iteration {state.get('iteration_count', 0)})\n{'=' * 70}")

    problem = state["problem_statement"]
    critique_log = state.get("critique_log", [])
    prior_flaws = state.get("market_flaws", [])

    search_dump = web_search(f"existing solutions products open source projects for: {problem}", max_results=5)

    critique_text = "None yet — this is the first research pass."
    if critique_log:
        critique_text = "\n".join(
            f"- Iteration {c.iteration}: verdict scores N={c.novelty_score}/F={c.feasibility_score}/S={c.scale_score}. "
            f"Fatal flaw: {c.fatal_flaw} Required fix: {c.required_fix}"
            for c in critique_log
        )

    prior_flaws_text = "None yet."
    if prior_flaws:
        prior_flaws_text = "\n".join(
            f"- {f.product_name}: {f.flaw_description}" for f in prior_flaws
        )

    user_prompt = f"""PROBLEM STATEMENT:
{problem}

WEB SEARCH FINDINGS (may be partial):
{search_dump}

FULL JUDGE CRITIQUE LOG (you MUST avoid repeating rejected concepts):
{critique_text}

MARKET FLAWS ALREADY DISCOVERED IN PRIOR ITERATIONS (extend, do not duplicate):
{prior_flaws_text}

Produce 3-6 NEW market flaws and a single innovation-gap thesis that survives the
novelty safeguard. Remember: any "thin API wrapper" direction is an automatic discard."""

    result: ResearcherOutput = invoke_structured_with_retry(
        "adversarial_researcher", ResearcherOutput, RESEARCHER_SYSTEM_PROMPT, user_prompt
    )

    print(f"[researcher] innovation gap: {result.innovation_gap[:200]}")
    for f in result.market_flaws:
        print(f"[researcher] flaw: {f.product_name} -> {f.flaw_description[:120]}")

    return {
        "innovation_gap": result.innovation_gap,
        "market_flaws": result.market_flaws,  # appended via operator.add
    }


# ── Node 2: Shoestring Architect ──────────────────────────────────────────────

ARCHITECT_SYSTEM_PROMPT = """You are the SHOESTRING ARCHITECT in a hackathon strategy pipeline.

Your job: turn the innovation gap into a complete, buildable technical design.

MANDATORY $0 BUDGET SAFEGUARD (non-negotiable):
- Total cost to build AND operate the MVP MUST be $0. Strictly enforce open-source,
  free-tier, or serverless-free tooling only. Approved examples: SQLite, PostgreSQL
  (self-hosted/free tier), FastAPI, Streamlit, Flask, llama.cpp with quantized (Q4/Q5)
  local models, ONNX Runtime, scikit-learn, Hugging Face free models, GitHub Pages,
  Cloudflare Workers free tier.
- You MUST REJECT and never propose enterprise cloud infrastructure: AWS RDS, paid
  GPU instances, managed Kubernetes, paid SaaS APIs, or anything with a credit-card
  requirement to operate. If a component has a paid tier, you must justify why the
  free tier suffices for the MVP and demo.

MANDATORY PRACTICAL FEASIBILITY SAFEGUARD (non-negotiable):
- The design must be practically buildable by a small student team during a hackathon
  and operable in real life WITHOUT prohibitive maintenance or operational costs.
- Every component must run on commodity hardware (a mid-range laptop or a low-cost
  device). If you propose a local model, state the quantization level and RAM/VRAM
  budget explicitly.
- Include an explicit offline / low-bandwidth strategy suitable for rural deployment.

NOVELTY DEFENSE:
- Include a novelty_argument that explicitly defends the design against the charge of
  being "just an API wrapper". Name the original architectural or algorithmic
  contribution. If your design's novelty reduces to "we call a hosted LLM with a
  custom prompt", it is INVALID — redesign before answering.

If a critique log is provided, read EVERY entry and fix every required_fix that
targets the design. Do not repeat any rejected stack choice.

Output strictly in the requested structured format."""


def shoestring_architect(state: ApexState) -> dict:
    print(f"\n{'=' * 70}\n[NODE] Shoestring Architect (iteration {state.get('iteration_count', 0)})\n{'=' * 70}")

    problem = state["problem_statement"]
    gap = state.get("innovation_gap", "")
    critique_log = state.get("critique_log", [])
    flaws = state.get("market_flaws", [])

    pricing_check = web_search("free open source self-hosted alternatives licensing cost", max_results=3)

    critique_text = "None yet — this is the first design pass."
    if critique_log:
        critique_text = "\n".join(
            f"- Iteration {c.iteration}: N={c.novelty_score}/F={c.feasibility_score}/S={c.scale_score}. "
            f"Fatal flaw: {c.fatal_flaw} Required fix: {c.required_fix}"
            for c in critique_log
        )

    flaws_text = "\n".join(f"- {f.product_name}: {f.flaw_description} -> exploit: {f.exploitation_angle}" for f in flaws)

    prior_design_text = "None."
    if state.get("current_design"):
        prior_design_text = state["current_design"].model_dump_json(indent=2)

    user_prompt = f"""PROBLEM STATEMENT:
{problem}

INNOVATION GAP TO CLAIM:
{gap}

MARKET FLAWS TO EXPLOIT:
{flaws_text}

FULL JUDGE CRITIQUE LOG (fix every required_fix aimed at the design; never repeat a rejected stack):
{critique_text}

PREVIOUS DESIGN (if rejected, do not resubmit it unchanged):
{prior_design_text}

REFERENCE NOTES ON FREE TOOLING (may be partial):
{pricing_check}

Produce the complete TechDesign. cost_estimate_usd MUST be 0.0 or justified as
effectively zero. Every stack item must be free/open-source and commodity-hardware ready."""

    design: TechDesign = invoke_structured_with_retry(
        "shoestring_architect", TechDesign, ARCHITECT_SYSTEM_PROMPT, user_prompt
    )

    print(f"[architect] summary: {design.solution_summary[:200]}")
    print(f"[architect] stack: {design.tech_stack}")
    print(f"[architect] cost: ${design.cost_estimate_usd}")

    return {"current_design": design}


# ── Node 3: Red Team Judge (deliberately tool-free) ───────────────────────────

JUDGE_SYSTEM_PROMPT = """You are the RED TEAM JUDGE in a hackathon strategy pipeline.
You have NO tools. You evaluate purely from the state given to you. Your power is
ruthless, consistent rubric application. You are the harshest judge on the panel.

RUBRIC (score each 1-10):
1. NOVELTY (pass >= 7):
   - AUTOMATIC FAILURE (score <= 4): any design whose novelty_argument reduces to
     "we call GPT-4 / Gemini / any hosted LLM API with a custom prompt" — that is a
     thin API wrapper and MUST be failed and discarded. Same for unoriginal copies
     of existing products or prior hackathon winners.
   - High scores require a named, original architectural or algorithmic contribution.
2. FEASIBILITY (pass >= 8 — deliberately the strictest gate, because cost overruns
   are the most common hackathon-killer):
   - The design must be practically buildable and operable in real life without
     prohibitive maintenance or operational costs.
   - $0 budget is mandatory. Any dependence on paid infrastructure (AWS RDS, paid
     GPUs, paid API quotas beyond a free tier) caps this score at 5.
   - Components must run on commodity hardware; unverifiable hardware claims lose points.
3. SCALE (pass >= 7):
   - Judge the offline/low-bandwidth strategy and the path from MVP to real deployment.
   - A demo that collapses beyond 10 users, or has no rural/offline story when the
     problem demands one, fails this gate.

OUTPUT REQUIREMENTS:
- fatal_flaw: exactly one sentence — your single harshest, most specific objection.
- required_fix: one concrete, actionable instruction for whichever node receives the
  rejection (the Researcher if novelty failed, the Architect otherwise).
- Do not inflate scores to be polite. A mediocre design must fail."""


def red_team_judge(state: ApexState) -> dict:
    print(f"\n{'=' * 70}\n[NODE] Red Team Judge (evaluating iteration {state.get('iteration_count', 0)})\n{'=' * 70}")

    design = state["current_design"]
    problem = state["problem_statement"]
    gap = state.get("innovation_gap", "")

    prior_critiques = state.get("critique_log", [])
    prior_text = "None."
    if prior_critiques:
        prior_text = "\n".join(
            f"- Iteration {c.iteration}: N={c.novelty_score}/F={c.feasibility_score}/S={c.scale_score}: {c.fatal_flaw}"
            for c in prior_critiques
        )

    user_prompt = f"""PROBLEM STATEMENT:
{problem}

CLAIMED INNOVATION GAP:
{gap}

DESIGN UNDER EVALUATION:
{design.model_dump_json(indent=2)}

YOUR PRIOR CRITIQUES (be consistent — do not approve a design that still contains a
flaw you previously called fatal, and do not invent brand-new objections for issues
you previously accepted):
{prior_text}

Apply the rubric. Score novelty, feasibility, and scale. State the fatal flaw and the
required fix."""

    critique: JudgeCritique = invoke_structured_with_retry(
        "red_team_judge", JudgeCritique, JUDGE_SYSTEM_PROMPT, user_prompt
    )

    # The Judge increments iteration_count on EVERY evaluation, including approvals,
    # so the loop-breaker counter can never be bypassed.
    new_iteration = state.get("iteration_count", 0) + 1
    critique.iteration = new_iteration

    passed = (
        critique.novelty_score >= 7
        and critique.feasibility_score >= 8
        and critique.scale_score >= 7
    )
    if new_iteration >= state.get("max_iterations", 4) and not passed:
        verdict = "FORCED_EXIT"
    elif passed:
        verdict = "APPROVED"
    elif critique.novelty_score < 7:
        verdict = "REJECT_CONCEPT"
    else:
        verdict = "REJECT_DESIGN"

    print(
        f"[judge] iteration {new_iteration}: novelty={critique.novelty_score} "
        f"feasibility={critique.feasibility_score} scale={critique.scale_score} -> {verdict}"
    )
    print(f"[judge] fatal flaw: {critique.fatal_flaw}")
    print(f"[judge] required fix: {critique.required_fix}")

    scored = ScoredDesign(
        design=design,
        novelty_score=critique.novelty_score,
        feasibility_score=critique.feasibility_score,
        scale_score=critique.scale_score,
    )

    return {
        "critique_log": [critique],       # appended via operator.add
        "design_history": [scored],       # appended via operator.add
        "iteration_count": new_iteration, # overwritten
        "judge_verdict": verdict,
    }


# ── Node 4: Best-Effort Finalizer (pure state operation, no LLM, no tools) ────


def best_effort_finalizer(state: ApexState) -> dict:
    print(f"\n{'=' * 70}\n[NODE] Best-Effort Finalizer (FORCED_EXIT after {state['iteration_count']} iterations)\n{'=' * 70}")

    history = state.get("design_history", [])
    if not history:
        # Degenerate case: nothing was ever scored. Keep whatever design exists.
        print("[finalizer] design_history empty; keeping current design as-is.")
        return {"judge_verdict": "FORCED_EXIT"}

    best = max(history, key=lambda s: s.total_score)
    print(
        f"[finalizer] selected best-scoring design (total={best.total_score}: "
        f"N={best.novelty_score}/F={best.feasibility_score}/S={best.scale_score}) "
        f"out of {len(history)} attempts."
    )
    return {
        "current_design": best.design,
        "judge_verdict": "FORCED_EXIT",
    }


# ── Node 5: Deliverables Master ───────────────────────────────────────────────

DELIVERABLES_SYSTEM_PROMPT = """You are the DELIVERABLES MASTER in a hackathon strategy pipeline.
The design you receive is FINAL (approved by the Red Team Judge, or the best-effort
selection after forced exit). Do NOT redesign it. Your job is packaging.

MANDATORY HACKATHON DELIVERABLES SAFEGUARD (non-negotiable):
Your output MUST follow standard hackathon submission requirements:

1. PITCH DECK — exactly 6 slides in Smart India Hackathon (SIH) format:
   - Slide 1 — Title & Team: solution name, one-line tagline, problem statement ID.
   - Slide 2 — Problem & Existing Gaps: the pain point and why current solutions fail.
   - Slide 3 — Proposed Solution & Novelty: the core idea and the explicit novelty argument.
   - Slide 4 — Technical Architecture: stack, data flow, offline strategy.
   - Slide 5 — Feasibility & Cost: $0 budget breakdown, commodity-hardware claims, maintenance plan.
   - Slide 6 — Impact & Scale: beneficiaries, rollout path, scale story.
   Each slide entry must be a single string: "Slide N — Title: bullet; bullet; bullet".

2. REPOSITORY SCAFFOLD — a runnable MVP skeleton:
   - repo_tree: an ASCII directory tree.
   - mvp_scaffold_files: a mapping of file path -> COMPLETE file content. Include at
     minimum: the main application entrypoint, a requirements.txt, and one core module.
     Files must be syntactically valid and runnable (a working skeleton, not pseudocode).
   - readme: a complete README.md with setup and run instructions matching the scaffold.

Keep every file consistent with the approved tech stack. Do not introduce paid services."""


def deliverables_master(state: ApexState) -> dict:
    print(f"\n{'=' * 70}\n[NODE] Deliverables Master (verdict: {state.get('judge_verdict')})\n{'=' * 70}")

    design = state["current_design"]
    problem = state["problem_statement"]

    user_prompt = f"""PROBLEM STATEMENT:
{problem}

FINAL APPROVED DESIGN (do not alter it):
{design.model_dump_json(indent=2)}

Produce the full deliverables package: the 6-slide SIH pitch deck outline, the
repository ASCII tree, complete runnable scaffold files, and the README."""

    deliverables: Deliverables = invoke_structured_with_retry(
        "deliverables_master", Deliverables, DELIVERABLES_SYSTEM_PROMPT, user_prompt
    )

    print(f"[deliverables] {len(deliverables.pitch_deck)} slides, {len(deliverables.mvp_scaffold_files)} scaffold files.")

    # FileWrite tool: persist the scaffold to disk under ./apex_output/
    output_dir = os.path.join(os.getcwd(), "apex_output")
    os.makedirs(output_dir, exist_ok=True)

    for rel_path, content in deliverables.mvp_scaffold_files.items():
        safe_rel = rel_path.lstrip("/").replace("..", "")
        abs_path = os.path.join(output_dir, safe_rel)
        os.makedirs(os.path.dirname(abs_path) or output_dir, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"[deliverables] wrote {abs_path}")

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(deliverables.readme)
    print(f"[deliverables] wrote {readme_path}")

    deck_path = os.path.join(output_dir, "PITCH_DECK.md")
    with open(deck_path, "w", encoding="utf-8") as fh:
        fh.write("# SIH Pitch Deck (6 Slides)\n\n")
        for slide in deliverables.pitch_deck:
            fh.write(f"## {slide}\n\n")
    print(f"[deliverables] wrote {deck_path}")

    return {"deliverables": deliverables}


# ==============================================================================
# 6. EDGE ROUTING — strict priority order
# ==============================================================================


def route_from_judge(state: ApexState) -> str:
    """
    Single conditional edge sourced from the Red Team Judge.
    Evaluated in STRICT priority order. Pydantic-validated critiques guarantee
    this function can never raise a KeyError or a parsing error.
    """
    # PRIORITY 1 — Loop breaker (checked BEFORE quality gates)
    if state["iteration_count"] >= state["max_iterations"] and state["judge_verdict"] != "APPROVED":
        return "best_effort_finalizer"

    critique = state["critique_log"][-1]

    # PRIORITY 2 — Full approval: ALL three gates must pass
    if (
        critique.novelty_score >= 7
        and critique.feasibility_score >= 8
        and critique.scale_score >= 7
    ):
        return "deliverables_master"

    # PRIORITY 3 — Concept-level failure -> restart research
    if critique.novelty_score < 7:
        return "adversarial_researcher"

    # PRIORITY 4 — Implementation-level failure -> redesign only
    return "shoestring_architect"


# ==============================================================================
# 7. GRAPH CONSTRUCTION & COMPILATION
# ==============================================================================


def build_graph():
    graph = StateGraph(ApexState)

    graph.add_node("adversarial_researcher", adversarial_researcher)
    graph.add_node("shoestring_architect", shoestring_architect)
    graph.add_node("red_team_judge", red_team_judge)
    graph.add_node("best_effort_finalizer", best_effort_finalizer)
    graph.add_node("deliverables_master", deliverables_master)

    graph.add_edge(START, "adversarial_researcher")
    graph.add_edge("adversarial_researcher", "shoestring_architect")
    graph.add_edge("shoestring_architect", "red_team_judge")

    graph.add_conditional_edges(
        "red_team_judge",
        route_from_judge,
        {
            "best_effort_finalizer": "best_effort_finalizer",
            "deliverables_master": "deliverables_master",
            "adversarial_researcher": "adversarial_researcher",
            "shoestring_architect": "shoestring_architect",
        },
    )

    # Forced-exit fallback still ships deliverables (from the best-scoring design).
    graph.add_edge("best_effort_finalizer", "deliverables_master")

    # Deliverables Master has exactly one outgoing edge: END. No loop re-entry.
    graph.add_edge("deliverables_master", END)

    return graph.compile()


# ==============================================================================
# 8. RUNNER
# ==============================================================================


def run_apex(problem_statement: str, max_iterations: int = 4) -> ApexState:
    app = build_graph()

    initial_state: ApexState = {
        "problem_statement": problem_statement,
        "innovation_gap": "",
        "current_design": None,
        "judge_verdict": "PENDING",
        "deliverables": None,
        "market_flaws": [],
        "critique_log": [],
        "design_history": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
    }

    # recursion_limit sized generously: each loop pass touches at most 3 nodes,
    # plus the terminal fan-out, so 4 iterations fit comfortably under 50.
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})
    return final_state


if __name__ == "__main__":
    SAMPLE_PROBLEM = (
        "SIH Problem Statement: Rural primary health centers in India lack reliable "
        "tools for early screening of anemia and malnutrition in children under 5. "
        "Health workers operate with intermittent connectivity, low-end Android "
        "phones, and no budget for cloud services. Build a technology solution that "
        "enables accurate, offline-first screening and longitudinal tracking, and "
        "syncs opportunistically when connectivity appears."
    )

    print("=" * 70)
    print(" APEX HACKATHON AI — starting run")
    print("=" * 70)
    print(f"Problem: {SAMPLE_PROBLEM}\n")

    result = run_apex(SAMPLE_PROBLEM, max_iterations=4)

    print("\n" + "=" * 70)
    print(" FINAL RESULT")
    print("=" * 70)
    print(f"Verdict:      {result['judge_verdict']}")
    print(f"Iterations:   {result['iteration_count']}")
    print(f"Designs seen: {len(result['design_history'])}")

    final_design = result["current_design"]
    if final_design is not None:
        print("\n--- FINAL DESIGN ---")
        print(json.dumps(final_design.model_dump(), indent=2))

    deliverables = result.get("deliverables")
    if deliverables is not None:
        print("\n--- PITCH DECK (6 slides) ---")
        for slide in deliverables.pitch_deck:
            print(f"  {slide}")
        print("\n--- REPO TREE ---")
        print(deliverables.repo_tree)
        print("\nScaffold files written to ./apex_output/")

    print("\n--- CRITIQUE LOG ---")
    for c in result["critique_log"]:
        print(
            f"  [iter {c.iteration}] N={c.novelty_score} F={c.feasibility_score} "
            f"S={c.scale_score} | {c.fatal_flaw}"
        )

    print("\nDone.")
