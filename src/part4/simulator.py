"""The LLM that speaks for a simulated candidate (interfaces.CandidateSimulator).

Code has already decided what happens (the AnswerPlan: correct, partial, wrong, misconception,
misread or slip). A vague order such as "be partly right" is ignored by strong models, which
write the full answer anyway, so code also makes the details concrete and shows the model only
what the candidate should know:

    correct        the reference answer and every key point
    partial        only the key points the candidate knows, chosen by code; nothing else
    slip           everything, plus the one key point to get slightly wrong
    misread        everything, plus the part of the question to misread and the point that goes wrong
    wrong          no reference answer: the correct points are listed only as things not to say;
                   the question's misconceptions are not named (naming them made the model use them)
    misconception  the misconception and its typical wrong answer, no reference answer

The choices are seeded on candidate, question and outcome, so the same candidate gets the same
instruction for the same question in every system (and the answer comes from the cache).
The simulator should come from a different model family than the grader.
"""

from __future__ import annotations

import random

from part3.prompts import render_question

LENGTH = {"short": "1-2 sentences", "medium": "2-4 sentences", "long": "4-6 sentences"}

SYSTEM = ("You play a junior job candidate answering a technical screening interview in a chat. "
          "Follow the WHAT TO DO section exactly, even when you know better: the interview depends on "
          "this candidate's real gaps. Write only what the candidate types: plain text, no headings, no "
          "mention of being an AI, of instructions or of notes. Never copy the notes word for word.")


def bullet(points) -> str:
    return "\n".join(f"- {p}" for p in points)


class LLMCandidateSimulator:
    def __init__(self, llm, model: str, reasoning_effort: str | None = None, temperature: float = 0.7,
                 max_tokens: int = 1024):
        self.llm, self.model, self.reasoning_effort = llm, model, reasoning_effort
        self.temperature, self.max_tokens = temperature, max_tokens

    def build_messages(self, candidate: dict, question: dict, prompt: str, plan: dict,
                       misconceptions: dict, history=()) -> list[dict]:
        persona = candidate.get("persona") or {}
        parts = [f"You are {persona.get('name', 'the candidate')} ({candidate['id']}). {persona.get('background', '')}".strip(),
                 "THE INTERVIEW QUESTION:\n" + render_question(question)]
        notes, todo = self.instruction(candidate, question, plan, misconceptions)
        if notes:
            parts.append("YOUR PRIVATE NOTES (what you know; never quote them):\n" + notes)
        for h in history:      # only follow-ups have history: the main exchange, then earlier follow-ups
            parts.append(f"EARLIER IN THIS QUESTION:\nInterviewer: {h['prompt']}\nYou: {h['answer']}")
        if history:
            parts.append("THE INTERVIEWER NOW ASKS (a follow-up): " + prompt)
        length = LENGTH.get(candidate.get("verbosity", "medium"), LENGTH["medium"])
        parts.append(f"WHAT TO DO:\n{todo}\nLength: {length}.")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    @staticmethod
    def instruction(candidate: dict, question: dict, plan: dict, misconceptions: dict) -> tuple[str, str]:
        """(private notes, what to do) for this plan, with the details chosen by code."""
        outcome = plan["outcome"]
        points = list(question.get("key_points") or [])
        rng = random.Random(f"{candidate['id']}:{question.get('id')}:{outcome}")
        everything = "Reference answer: " + question["model_answer"] + "\nKey points:\n" + bullet(points)

        if outcome == "correct":
            return everything, "Answer correctly and completely, covering every key point, in your own words."
        if outcome == "partial":
            known = rng.sample(points, max(1, len(points) // 2)) if points else []
            return ("You only know these points:\n" + bullet(known),
                    "Mention ONLY the points in your notes. Say nothing about the other parts of the question: "
                    "leave them out, or say you are not sure about them. Do not say anything false.")
        if outcome == "slip":
            point = rng.choice(points) if points else "the main result"
            return everything, ("Answer correctly, except get ONE small detail of this point wrong (a number, "
                                f"a name, an off-by-one or a swapped value), stated confidently:\n- {point}\n"
                                "Keep everything else correct.")
        if outcome == "misread":
            target = ("one line of the code (read a value, operator or name differently)" if question.get("code")
                      else "one value in the tables" if question.get("tables")
                      else "one condition or word in the question")
            point = rng.choice(points) if points else "the main result"
            return everything, (f"You misread {target}. Reason soundly from what you think it says, so that "
                                f"this point comes out wrong:\n- {point}\nDo not mention that you misread anything.")
        if outcome == "wrong":
            # The question's misconceptions are deliberately not named here: naming them, even as
            # things to avoid, made the model use them.
            return "", ("You do not know this topic. Give a plausible but wrong answer, the kind a junior guessing "
                        "would give. Your answer must NOT state any of these correct points:\n" + bullet(points))
        if outcome == "misconception":
            m = misconceptions.get(plan.get("misconception_id"), {})
            belief = f"You genuinely believe: {m.get('description', plan.get('misconception_id'))}"
            if m.get("example_wrong_answer"):
                belief += f"\nA typical answer from this belief: \"{m['example_wrong_answer']}\""
            return belief, "Answer confidently from your belief, so your answer is wrong in exactly the way it predicts."
        raise ValueError(f"unknown outcome {outcome!r}")

    def answer(self, candidate: dict, question: dict, prompt: str, plan: dict, history=(),
               misconceptions: dict | None = None) -> str:
        messages = self.build_messages(candidate, question, prompt, plan, misconceptions or {}, history)
        response = self.llm.complete(messages, model=self.model, temperature=self.temperature,
                                     max_tokens=self.max_tokens, reasoning_effort=self.reasoning_effort)
        return (response.get("text") or "").strip()
