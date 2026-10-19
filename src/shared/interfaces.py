"""Contracts between the four parts (see docs/interfaces.md).

Every part codes against these types and protocols, so parts can be built
separately and plugged together. `stubs.py` has simple fakes for each one.

Ownership:
    Part 1  QuestionSource (part1.index.QuestionIndex), TopicExtractor
    Part 2  run_interview, InterviewConfig handling, the scorecard
    Part 3  Grader, Diagnoser, LLMClient
    Part 4  plan_answer, CandidateSimulator, Respondent for simulated candidates
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

# --------------------------------------------------------------------------
# Plain data (JSON-shaped dicts; formats are defined by the JSON schemas)
# --------------------------------------------------------------------------

Skill = Literal["py", "sql", "ds"]
SKILLS: tuple[Skill, ...] = ("py", "sql", "ds")

Question = dict[str, Any]   # schemas/question.schema.json
Job = dict[str, Any]        # an entry of data/jobs.json
Candidate = dict[str, Any]  # schemas/candidate.schema.json
InterviewResult = dict[str, Any]  # schemas/interview_result.schema.json

Cause = Literal["gap", "misconception", "misread", "slip"]
Outcome = Literal["correct", "partial", "wrong", "misconception", "misread", "slip"]
TurnKind = Literal["main", "follow_up"]


class Grade(TypedDict, total=False):
    score: int            # 0, 1 or 2 (majority of votes); required
    votes: list[int]      # every individual mark; required
    agreement: float      # share of votes equal to score
    rationale: str
    grader: str


class DiagnosisVote(TypedDict, total=False):
    cause: Cause
    misconception_id: str | None
    follow_up: str | None


class Diagnosis(TypedDict, total=False):
    cause: Cause                   # required: the winning cause
    misconception_id: str | None   # winning id when cause == "misconception"
    follow_up: str | None          # question to ask next, or None
    vote_share: float              # share of runs agreeing with the winner
    votes: list[DiagnosisVote]
    confidence_expressed: Literal["low", "medium", "high"]  # stretch feature


class FollowUp(TypedDict, total=False):
    prompt: str
    answer: str
    diagnosis: Diagnosis


class AnswerPlan(TypedDict, total=False):
    """What a simulated candidate will do for one reply. Hidden from the engine."""
    outcome: Outcome
    misconception_id: str | None
    expected_score: int     # what a perfect grader should give: 2, 1 or 0
    effective_level: int    # candidate's level on the question's topics (min over topics)
    attack: str | None      # safety-attack text appended to this reply, if any


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict


class LLMResponse(TypedDict, total=False):
    text: str
    parsed: Any             # parsed JSON when a schema was requested
    tool_calls: list[ToolCall]
    raw_message: dict       # assistant message as returned; send back unchanged
    model: str
    input_tokens: int
    output_tokens: int      # includes hidden reasoning tokens when the provider reports them
    cached: bool


# --------------------------------------------------------------------------
# Interfaces
# --------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat call (OpenAI-compatible endpoints: Gemini, Groq, NVIDIA NIM, OpenRouter...).

    Implementations must cache responses by every request argument (so re-runs cost
    nothing) and retry on rate limits. `cache_tag` only changes the cache key, so
    repeated votes at temperature > 0 are distinct calls.

    Tool calling uses the OpenAI format: `tools` is a list of
    {"type": "function", "function": {"name", "description", "parameters"}}. The response
    lists `tool_calls` as {"id", "name", "arguments": dict}. `raw_message` is the assistant
    message exactly as returned; send it back unchanged in the next request's `messages`
    (Gemini needs its extra fields, e.g. thought signatures), followed by
    {"role": "tool", "tool_call_id": id, "content": "..."} per call.
    """

    def complete(self, messages: list[dict], *, model: str, temperature: float = 0.0,
                 json_schema: dict | None = None, max_tokens: int = 1024,
                 tools: list[dict] | None = None, reasoning_effort: str | None = None,
                 cache_tag: str | None = None) -> LLMResponse: ...


@runtime_checkable
class QuestionSource(Protocol):
    """Part 1's search index (part1.index.QuestionIndex already matches this)."""

    def search(self, query: str | None = None, skill: str | None = None,
               levels: list[int] | None = None, prefer_topics: list[str] | None = None,
               exclude_ids: set[str] | frozenset = frozenset(), k: int = 5) -> list[tuple[Question, float]]: ...


@runtime_checkable
class QuestionGenerator(Protocol):
    """Fallback when the bank has nothing for a required topic. Returns a question
    marked source="generated-live"; the engine gives its answers less weight."""

    def generate(self, skill: Skill, level: int, topic: str) -> Question: ...


@runtime_checkable
class TopicExtractor(Protocol):
    """Job description -> topic ids from data/topics.json (part1.extract_topics)."""

    def __call__(self, job_description: str) -> list[str]: ...


@runtime_checkable
class Grader(Protocol):
    """Marks a whole exchange (main answer plus any follow-ups) 0/1/2 against the rubric."""

    def grade(self, question: Question, answer: str, follow_ups: list[FollowUp] = ()) -> Grade: ...


@runtime_checkable
class Diagnoser(Protocol):
    """After a wrong or vague reply: why was it wrong, and what to ask next.
    Picks a cause from fixed options and a misconception id from the question's list,
    run several times with a majority vote (see InterviewConfig.diagnosis_votes)."""

    def diagnose(self, question: Question, answer: str, history: list[FollowUp] = ()) -> Diagnosis: ...


@runtime_checkable
class Respondent(Protocol):
    """Whoever answers: a human in the app, or a simulated candidate in experiments.
    The engine only ever sees the reply text."""

    def reply(self, prompt: str, *, question: Question | None, kind: TurnKind, turn: int) -> str: ...


@runtime_checkable
class CandidateSimulator(Protocol):
    """Turns a hidden AnswerPlan into natural spoken text (an LLM from a different
    family than the grader). It must follow the plan exactly."""

    def answer(self, candidate: Candidate, question: Question, prompt: str,
               plan: AnswerPlan, history: list[FollowUp] = ()) -> str: ...


# --------------------------------------------------------------------------
# Configuration: one InterviewConfig per system being compared
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSettings:
    """Model name per role. Provider endpoints/keys come from environment variables
    (see docs/interfaces.md), never from code."""
    interviewer: str = "gemini-3.5-flash-lite"      # agent driver, live question generation
    diagnoser: str = "gemini-3.5-flash-lite"
    topic_extractor: str = "gemini-3.5-flash-lite"  # keep fixed across the base-model comparison
    grader: str = "finetuned-grader"              # A3 and, until fine-tuning exists, live use: "gemini-3.5-flash-lite"
    simulator: str = "gpt-oss-120b"               # different family from the grader


@dataclass(frozen=True)
class InterviewConfig:
    system: str = "A2"
    driver: Literal["code", "agent"] = "code"   # agent: an LLM calls the rule-enforcing tools

    # How questions are chosen
    mode: Literal["adaptive", "fixed"] = "adaptive"
    retrieval: Literal["job-aware", "job-blind"] = "job-aware"
    fixed_levels: tuple[int, ...] = (1, 2, 2, 3)   # fixed mode: levels asked per skill
    skill_order: tuple[Skill, ...] = SKILLS         # fixed mode: rotation order
    start_level: int = 2                            # prior guess for every skill

    # When to stop
    stop_rule: Literal["uncertainty", "fixed_count", "decision"] = "uncertainty"
    uncertainty_threshold: float = 0.6   # Rasch standard error (logits); tune on the pilot
    max_questions: int = 15              # main questions, adaptive mode
    fixed_count: int = 12                # main questions for fixed mode / fixed_count stop rule

    # Follow-ups and diagnosis
    # structured: voted diagnosis + targeted follow-ups (A2)
    # single: one diagnosis run on the main reply, generic follow-up only (A4)
    # off: no diagnosis, no follow-ups (B1)
    diagnosis: Literal["structured", "single", "off"] = "structured"
    generic_follow_up: bool = True       # single: ask one generic "explain your reasoning"
    max_follow_ups: int = 2
    diagnosis_votes: int = 3
    misconception_confirmations: int = 2  # replies on which the same id must win

    # Misconception recall: prefer questions that probe topics where an earlier answer was
    # weak, and questions that list misconceptions nobody has probed yet (spec section 4)
    probe_missed_topics: bool = True

    # Agent driver: stop after this many LLM calls (spec section 1)
    max_agent_steps: int = 150

    # Grading
    grader_votes: int = 3

    # Question generation fallback
    allow_generated_questions: bool = True
    generated_weight: float = 0.5        # evidence weight of an answer to a generated question

    # Topics
    use_topic_extraction: bool = True    # False: use jobs.json required_topics directly

    models: ModelSettings = field(default_factory=ModelSettings)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **changes) -> "InterviewConfig":
        return replace(self, **changes)


_A2 = InterviewConfig(system="A2")

SYSTEMS: dict[str, InterviewConfig] = {
    # Baseline: fixed 12 questions (4 per skill), same job-aware bank
    "B1": _A2.with_(system="B1", mode="fixed", stop_rule="fixed_count", diagnosis="off",
                    generic_follow_up=False, probe_missed_topics=False),
    # Adaptive question choice, but no early stopping
    "A1": _A2.with_(system="A1", stop_rule="fixed_count"),
    # Full system
    "A2": _A2,
    # Full system with a prompted grader instead of the fine-tuned one
    "A3": _A2.with_(system="A3", models=replace(_A2.models, grader="gemini-3.5-flash-lite")),
    # Full system without the candidate model: skill-only scorecard, generic follow-up,
    # misconceptions guessed from single replies
    "A4": _A2.with_(system="A4", diagnosis="single", diagnosis_votes=1, misconception_confirmations=1),
    # Full system with job-blind retrieval
    "A5": _A2.with_(system="A5", retrieval="job-blind"),
}


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

@dataclass
class Components:
    """Everything the engine needs, injected so any piece can be swapped for a stub."""
    question_source: QuestionSource
    grader: Grader
    diagnoser: Diagnoser | None = None
    topic_extractor: TopicExtractor | None = None
    question_generator: QuestionGenerator | None = None
    llm: LLMClient | None = None
    misconceptions: dict[str, dict] = field(default_factory=dict)  # id -> misconception record


def run_interview(job: Job, respondent: Respondent, config: InterviewConfig,
                  components: Components, *, candidate_id: str | None = None,
                  rng: random.Random | None = None) -> InterviewResult:
    """Run one interview and return a dict matching schemas/interview_result.schema.json.

    Implemented by Part 2 (engine.run.run_interview). It must never read candidate keys or
    truth data; the experiment runner adds `truth` fields afterwards from the respondent's log.
    """
    from engine.run import run_interview as run_engine   # imported here to avoid a circular import
    return run_engine(job, respondent, config, components, candidate_id=candidate_id, rng=rng)


def plan_answer(candidate: Candidate, question: Question, *, kind: TurnKind, turn: int,
                previous: AnswerPlan | None, rng: random.Random) -> AnswerPlan:
    """Decide in code what a simulated candidate does for one reply (Part 4 owns the
    final rules). A reference implementation lives in shared.stubs.reference_plan_answer."""
    raise NotImplementedError("Part 4 implements the answer policy")
