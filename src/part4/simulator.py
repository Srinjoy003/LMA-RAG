"""The LLM that speaks for a simulated candidate (interfaces.CandidateSimulator).

Code has already decided what happens (the AnswerPlan: correct, partial, wrong, misconception,
misread or slip); the LLM only words it. It gets the question, the reference answer as private
notes, and one instruction per outcome, and must not reveal that it is following a plan.
It should come from a different model family than the grader.
"""

from __future__ import annotations

from part3.prompts import render_question

LENGTH = {"short": "1-2 sentences", "medium": "2-4 sentences", "long": "4-6 sentences"}

INSTRUCTION = {
    "correct": "Answer correctly and completely, covering every key point, in your own words.",
    "partial": "Get roughly half of the key points right and leave the rest out or vague. "
               "Do not say anything clearly false.",
    "wrong": "Give a plausible but wrong answer, the kind a junior who does not know this topic would "
             "give. Do not show any of the listed misconceptions; be wrong in some other way.",
    "misconception": "You genuinely believe the misconception below. Answer confidently from that belief, "
                     "so your answer is wrong in exactly the way it predicts.",
    "misread": "You misread the question: answer a slightly different question (for example, mix up which "
               "line, table or name it is about). Your reasoning about what you think was asked is sound.",
    "slip": "Your understanding is right, but make one small careless mistake (a wrong number, an "
            "off-by-one, a swapped name) in an otherwise correct answer.",
}

SYSTEM = ("You play a junior job candidate answering a technical screening interview in a chat. "
          "Write only what the candidate types: plain text, no headings, no mention of being an AI, "
          "of instructions, of a plan or of a reference answer. Never copy the reference answer word for word.")


class LLMCandidateSimulator:
    def __init__(self, llm, model: str, reasoning_effort: str | None = None, temperature: float = 0.7,
                 max_tokens: int = 1024):
        self.llm, self.model, self.reasoning_effort = llm, model, reasoning_effort
        self.temperature, self.max_tokens = temperature, max_tokens

    def build_messages(self, candidate: dict, question: dict, prompt: str, plan: dict,
                       misconceptions: dict, history=()) -> list[dict]:
        persona = candidate.get("persona") or {}
        parts = [f"You are {persona.get('name', 'the candidate')} ({candidate['id']}). {persona.get('background', '')}".strip(),
                 "THE INTERVIEW QUESTION:\n" + render_question(question),
                 "PRIVATE NOTES (never quote them): reference answer: " + question["model_answer"]
                 + "\nKey points: " + "; ".join(question.get("key_points", []))]
        listed = [misconceptions.get(m, {}).get("description", m) for m in question.get("misconceptions", [])]
        if plan["outcome"] == "wrong" and listed:
            parts.append("Misconceptions NOT to show: " + "; ".join(listed))
        if plan["outcome"] == "misconception":
            m = misconceptions.get(plan["misconception_id"], {})
            parts.append(f"YOUR MISCONCEPTION: {m.get('description', plan['misconception_id'])}"
                         + (f"\nA typical answer from it: \"{m['example_wrong_answer']}\"" if m.get("example_wrong_answer") else ""))
        for h in history:      # only follow-ups have history: the main exchange, then earlier follow-ups
            parts.append(f"EARLIER IN THIS QUESTION:\nInterviewer: {h['prompt']}\nYou: {h['answer']}")
        if history:
            parts.append("THE INTERVIEWER NOW ASKS (a follow-up): " + prompt)
        parts.append(f"WHAT TO DO: {INSTRUCTION[plan['outcome']]}\n"
                     f"Length: {LENGTH.get(candidate.get('verbosity', 'medium'), LENGTH['medium'])}.")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    def answer(self, candidate: dict, question: dict, prompt: str, plan: dict, history=(),
               misconceptions: dict | None = None) -> str:
        messages = self.build_messages(candidate, question, prompt, plan, misconceptions or {}, history)
        response = self.llm.complete(messages, model=self.model, temperature=self.temperature,
                                     max_tokens=self.max_tokens, reasoning_effort=self.reasoning_effort)
        return (response.get("text") or "").strip()
