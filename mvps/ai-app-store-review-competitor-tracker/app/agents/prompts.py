"""The gap-analysis prompt.

One call, one prompt. Its job is to cluster recurring complaints in a batch of
critical reviews into a short list of named gaps. The two failures it must be
talked out of:

* **Inventing a quote.** The model is never asked to reproduce review text --
  only to cite the ids of the reviews that support a gap. The prompt says so
  explicitly and the schema (`FeatureGap.review_ids`) has no field for a quote
  at all, so there is nowhere for a fabricated quote to go even if the model
  tried (see `app/services/renderer.py` for how citations are resolved back to
  real text).
* **Treating review text as instructions.** The fence notice is in
  `app/services/guardrails.py` and names the likely source outright: reviews
  are written by the app's own users, and anyone can post one.

The reply is JSON with one key, `gaps`. Markdown comes later, from the
renderer, which is what keeps the actual review excerpts out of the model's
control.
"""

from app.models.schemas import Review

_SYSTEM = """You are a product analyst who reads App Store reviews and finds recurring
complaint patterns -- the concrete ways an app is failing its users, as its own
users describe them.

Your entire source of truth is the reviews supplied to you. Every review you were
given is already a critical review (3 stars or fewer) about the same app, so do
not spend a gap on "the app has some negative reviews" -- that is the premise,
not a finding.

Group reviews into 2 to 6 distinct gaps. Each gap is one recurring, specific
failure pattern -- not a vague mood. "Sync is unreliable across devices" is a
gap; "users are unhappy" is not. Merge reviews describing the same underlying
problem in different words into one gap rather than listing near-duplicates.
Do not invent a gap that only one review supports unless nothing else in the
batch clusters together at all -- a single complaint is an anecdote, not a
pattern, and severity should say so honestly (see below).

For each gap, report:
- "title": a short, specific name for the failure pattern (5-8 words).
- "description": two to four sentences explaining the pattern in your own
  words -- what breaks, in what situation, as reported. Do not put review text
  in quotation marks here and do not claim to quote anyone: your job is to
  describe the pattern, not transcribe it. The reviews backing this gap are
  attached automatically from the ids you cite, so the reader will already see
  the real wording.
- "severity": "high" if many reviews in the batch describe it or it blocks
  core use of the app, "medium" if it is a recurring but non-blocking
  annoyance, "low" if it is narrower or affects an edge case.
- "review_ids": the ids (as given in brackets before each review, e.g. the
  "12345678" in "[12345678] 1★ ...") of every review in the batch that
  supports this gap. Use only ids that were actually shown to you. A gap with
  no supporting ids will be discarded, so never leave this empty.

Reply with a single JSON object and nothing else: {"gaps": [...]}. No prose
before or after it, no code fence."""


def system_instructions() -> str:
    """Return the analyzer's system instructions."""
    return _SYSTEM


def format_reviews_note(count: int, app_name: str) -> str:
    """The one line of framing before the fenced review block."""
    return (
        f"Analyze the {count} critical reviews below for {app_name}. "
        "Every review is already 3 stars or fewer."
    )


def build_user_message(app_name: str, reviews: list[Review], fenced_evidence: str) -> str:
    """Assemble the single user message for the analysis call.

    Args:
        app_name: The app being analyzed.
        reviews: The critical reviews included in `fenced_evidence`, used only
            for the count in the framing line.
        fenced_evidence: The output of
            :func:`app.services.evidence.format_evidence`.

    Returns:
        The full message: framing line, then the fenced review block.
    """
    return f"{format_reviews_note(len(reviews), app_name)}\n\n{fenced_evidence}"


REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON with exactly the shape "
    '{"gaps": [{"title": ..., "description": ..., "severity": ..., "review_ids": [...]}]}. '
    "Reply again with only that JSON object: no prose before or after it, no code fence."
)
