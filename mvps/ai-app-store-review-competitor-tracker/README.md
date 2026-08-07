# App Review Competitor Tracker

Pulls a competitor's most recent reviews — from the **iOS App Store** or **Google Play** — and runs a
single feature-gap analysis over the critical ones: a short, cited list of the concrete ways the app is
failing its users, in their own words. Built with **Streamlit**, **Pydantic v2**, and **Microsoft Agent
Framework**. No search-provider key required: app lookup and review fetching, on both stores, are free,
keyless, public (if unofficial) endpoints.

Full design doc: [`docs/01-prd.md`](docs/01-prd.md) (Android/Google Play support is in the Addendum —
the MVP originally shipped iOS-only and was expanded once live testing showed Play's data was more
reliable).

---

## ⚠️ Known limitation: iOS reviews are unreliable — read this before you file a bug

Apple's public customer-reviews feed is unofficial. Two things were confirmed by direct testing on
**2026-08-07** ([`docs/01-prd.md` §7](docs/01-prd.md#7-the-load-bearing-constraint-apples-review-feed-is-degraded-and-the-design-accepts-that)
has the full trace):

- **Reproduced every time it was tried:** only the **bare URL, US storefront, no `page=` parameter**
  ever returns real data. Any other country code, or any `page=` value including `page=1`, returns an
  empty feed instead of an error. This is why the iOS side of this app is fixed to the US storefront.
- **Not fully explained:** even that exact request is **unreliable under sustained request volume**
  from one source — it worked consistently for the first, sparsely-spaced calls in this investigation,
  then started failing intermittently (for different apps, unpredictably) once testing got heavier. A
  browser-shaped `User-Agent` and the `cc` query param were both tested as the cause and neither held
  up under a controlled retest. This looks like informal, IP-based rate limiting or bot-suspicion on
  Apple's side, not something this client can detect or negotiate with.

`app/appstore/reviews.py` sends a browser-shaped `User-Agent` and retries once, briefly — cheap,
plausible partial mitigations, not confirmed fixes. This app has no way to tell "this app genuinely has
zero reviews" apart from "the feed is temporarily uncooperative," so it doesn't try: an empty sample is
always shown plainly as a fact, never as an error the user has to interpret.

**Google Play does not have this problem.** Direct testing found Play's review data reliable across all
12 storefronts this app offers (US, UK, India, Canada, Australia, Germany, France, Brazil, Japan,
Mexico, Indonesia, Nigeria), on every attempt, with no sign of the rate-limiting shape above — see the
Addendum in the PRD. If iOS is being unreliable, try the same competitor on Android.

## What it does

1. **Pick a store and storefront** — iOS (fixed to the US storefront, per the limitation above — Apple's
   feed does not work for any other country, not merely "not yet offered") or Google Play (12
   storefronts, all confirmed working).
2. **Resolve** an app by name (with disambiguation if more than one matches), a store URL, a numeric
   App Store id, or a Play Store package name (e.g. `com.spotify.music`).
3. **Fetch** its most recent reviews (one request; the iOS client retries once on an empty response).
4. **Compute rating stats** in code — distribution, % critical, sample date range. Never touches the
   model; a star count is arithmetic, not a place for an LLM to be wrong.
5. **Analyze** the ≤3★ reviews in one model call, clustering recurring complaints into 2–6 named
   feature gaps, each with a severity and citations by **review id** — the model is never asked to
   quote a review, only to point at which ones support a pattern.
6. **Render** one report: rating snapshot, gaps with real excerpts pulled from the app's own fetched
   data, and a closing summary — not a dump of all 50 fetched reviews, which added length without
   adding information once the excerpts already under each gap say the same thing.

## Zero invented quotes, by construction

The model cites review ids; it never writes an excerpt. The renderer resolves every cited id against
the reviews the app actually fetched and **drops any gap whose ids don't resolve** — the same pattern
this repo's `ai-competitor-analyzer` uses for URLs, adapted to reviews. A test
(`tests/test_renderer.py::test_no_excerpt_text_appears_that_was_not_in_a_fetched_review`) asserts this
directly: every excerpt in a rendered report is a substring of some fetched review's content. This
guarantee is identical for both platforms — it operates on the normalized `Review` model, not on
anything store-specific.

Reviews are also the untrusted input this whole app is built around — unmoderated, public,
user-generated text fed to a model, on either store. They are fenced and labelled as data, never
instructions, before the model sees them (`app/services/guardrails.py`).

## Quickstart

This repo's projects use `pyenv` for the Python version and `uv` for the environment/packages.

```bash
cd mvps/app-store-review-competitor-tracker
pyenv install -s 3.11.10 && pyenv local 3.11.10
uv venv --python 3.11.10
source .venv/bin/activate
uv pip install -r requirements-dev.txt
cp .env.example .env          # set GROQ_API_KEY — the only required credential
streamlit run streamlit_app.py
```

### Docker

```bash
docker build -t review-tracker .
docker run --rm -p 8501:8501 --env-file .env review-tracker
```

### Testing

```bash
ruff check .
pytest                 # offline, mocked HTTP and mocked model
pytest -m live         # hits the real App Store and Play Store endpoints (docs/01-prd.md §7)
```

## Configuration

Only one credential is required — see `.env.example` for the full list with defaults.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MODEL_PROVIDER` | `groq` \| `openai` \| `anthropic` \| `openrouter` \| `gemini` \| `ollama` \| `foundry` | `groq` |
| `MODEL_NAME` | Provider-native model id | `llama-3.3-70b-versatile` |
| `GROQ_API_KEY` | Required when `MODEL_PROVIDER=groq` (the default) | — |
| `APPSTORE_COUNTRY` | iOS storefront — see the limitation above; not offered as a UI choice | `us` |
| `PLAYSTORE_COUNTRY` | Default Android storefront offered in the UI (12 options, all confirmed — see `.env.example`) | `us` |
| `MAX_REVIEWS` | Ceiling on reviews used from one fetch | `50` |
| `MIN_CRITICAL_REVIEWS` | Floor on ≤3★ reviews before a gap analysis runs | `5` |

## Explicitly out of scope

- Historical tracking, scheduling, alerting, or diffing between runs — every run is a one-off snapshot.
- Multi-app or multi-store or multi-country comparison in one run — one app, one storefront, per run.
- Review pagination beyond what each store's feed actually serves — no scraping a store's web page
  beyond what each platform client already does, no headless browser.
- A suggested roadmap or backlog — the app names gaps, it does not prioritize them for you.
- Any write action against either store — read-only, public data only.

See [`docs/01-prd.md` §5](docs/01-prd.md#5-explicitly-out-of-scope) and the Addendum for the full list
and reasoning, including why Google Play was *not* left out of scope.

## Repository layout

```
app/
  core/         # Settings, secret-redacting logging, exception hierarchy
  appstore/     # iTunes search/lookup resolver, the single-request iOS review fetch
  playstore/    # Google Play search/lookup resolver and review fetch (via google-play-scraper)
  models/       # Pydantic domain models (Platform, AppIdentity, Review, FeatureGap, Report, ...)
  agents/       # Provider-agnostic chat client, the gap-analysis prompt and call
  services/     # normalize, guardrails, stats, evidence packing, pipeline, renderer
ui/             # Streamlit widgets: platform/store picker, search + disambiguation, report view
docs/01-prd.md  # Problem statement, scope, design, success criteria, Android Addendum
tests/          # Offline unit + pipeline tests; `-m live` tests hit the real endpoints
```
