"""Validate evidence provenance, not the semantic truth of an LLM's judgement."""

# Formatting a model may change when it copies a quote: inline-code backticks and quotation marks.
_IGNORED = set("`'\"‘’“”")


def _normalise(text: str) -> tuple[str, list[int]]:
    """`text` without formatting differences (every kind of space is one plain space, backticks
    and quotation marks dropped, case folded), and for each character kept its index in `text`."""
    out, where = [], []
    for i, ch in enumerate(text):
        if ch in _IGNORED:
            continue
        if ch.isspace():                  # includes no-break and narrow no-break spaces
            if not out or out[-1] == " ":
                continue
            ch = " "
        out.append(ch.casefold() if len(ch.casefold()) == 1 else ch)
        where.append(i)
    while out and out[-1] == " ":
        out.pop()
        where.pop()
    return "".join(out), where


def find_quote(quote: str, text: str) -> tuple[int, int] | None:
    """(start, end) of `quote` in `text`: an exact match, or else one that differs only in
    formatting (spaces, backticks, quotation marks, case). None if the words are not there."""
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    needle, _ = _normalise(quote)
    haystack, where = _normalise(text)
    at = haystack.find(needle) if needle.strip() else -1
    if at < 0:
        return None
    return where[at], where[at + len(needle) - 1] + 1


def claim_span(claim: dict, text: str, allowed) -> dict | None:
    """The claim with its quote located in `text`, or None if the claim is unsupported. The
    stored quote is the candidate's own words (`text[quote_start:quote_end]`)."""
    mid, quote, reason = (claim.get(k) for k in ("misconception_id", "supporting_quote", "reason"))
    if mid not in allowed or not isinstance(quote, str) or not quote.strip():
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None
    span = find_quote(quote, text)
    if span is None:
        return None
    start, end = span
    return {"misconception_id": mid, "supporting_quote": text[start:end], "quote_start": start,
            "quote_end": end, "reason": reason}
