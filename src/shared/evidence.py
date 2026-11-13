"""Validate evidence provenance, not the semantic truth of an LLM's judgement."""


def claim_span(claim: dict, text: str, allowed) -> dict | None:
    mid, quote, reason = (claim.get(k) for k in ("misconception_id", "supporting_quote", "reason"))
    if mid not in allowed or not isinstance(quote, str) or not quote.strip():
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None
    start = text.find(quote)
    if start < 0:
        return None
    return {"misconception_id": mid, "supporting_quote": quote, "quote_start": start,
            "quote_end": start + len(quote), "reason": reason}
