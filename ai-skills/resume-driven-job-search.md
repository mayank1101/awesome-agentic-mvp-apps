# Resume-Driven Job Search Skill

**Description:** A two-prompt structure for turning a resume into a ranked shortlist of real job postings — read the resume once, then judge each fetched posting against it requirement by requirement — plus the runtime rules that make the resulting score worth showing.

**Reference implementation:** [`mvps/ai-job-search-assistant`](../mvps/ai-job-search-assistant) · standalone module: [`ai-agents/job_search_agent.py`](../ai-agents/job_search_agent.py)

---

## The four failures this skill is built against

* **The list where everything scores 80.** A model asked "does this resume cover this requirement?" says yes far more often than the text supports, because agreeing reads as helpful. A shortlist whose scores do not separate a strong match from a weak one is a shortlist with no information in it, and it is *worse* than an unranked list of links because it looks authoritative.
* **The invented link.** Asked for jobs, a model will produce plausible, well-formed, non-existent URLs. This is not fixable by prompting — the fix is structural, below: the model never supplies a link, it only reads pages that were actually fetched.
* **The score with a hidden basis.** Reading thirty postings in full is slow and expensive, so most of them get ranked on a title and a search snippet instead. Presenting that number the same way as one that read the posting is the quiet dishonesty this skill exists to prevent.
* **Following instructions found in a job posting.** The posting is a web page fetched automatically, chosen by a search engine, that nobody read first — a weaker position than a document a user pasted deliberately. See [untrusted-input-guardrail](untrusted-input-guardrail.md).

Sibling skill: [resume-job-fit](resume-job-fit.md) scores one posting the user already found. This one produces the shortlist that comes before that, and reuses its verdict-and-evidence structure for the deep tier.

---

## Prompt 1 — Read the resume

Runs once per search. Everything downstream — the queries, the ranking, every score — rests on this being right, which is why the output belongs on screen **before** the search runs: a resume read as the wrong role produces thirty plausible, wrong results, and the mistake is invisible from the results themselves.

### System Prompt

You read a candidate's resume and write down what it actually says, as JSON. You are preparing a job search, so these fields build search queries and judge fit against real postings.

Return exactly this shape:

```json
{
  "titles": ["job titles this resume evidences, most recent first, at most 4"],
  "seniority": "junior | mid | senior | lead",
  "years_experience": 0,
  "skills": ["concrete searchable skills: languages, frameworks, tools, at most 15"],
  "domains": ["industries or problem spaces worked in, at most 5"],
  "locations": ["places the resume gives as the candidate's own, at most 3"],
  "summary": "one or two sentences describing this candidate as a hiring manager would"
}
```

Rules:

- Write titles the way a **job posting** would write them, not the way an employer styled them internally. "Member of Technical Staff II", where the work described is backend services, becomes "Backend Engineer". Keep the original too if it is a title postings actually use.
- Skills must appear in the resume. Do not add the neighbouring technology a reader would assume, and do not upgrade exposure into expertise: "familiar with Kafka" is the skill `Kafka`, not `Kafka expertise`.
- `seniority` follows the work described, not the title. Under 2 years is junior; 2–5 is mid; 5–9 with ownership is senior; beyond that with team or architecture responsibility is lead.
- `years_experience` counts professional work only. Internships and degrees are not experience. Use `null` rather than guessing when the dates do not support a number.
- Do not include the candidate's name, email address, phone number, or any link. They are not needed and **must not leave this step** — see the privacy rule below.

Reply with the JSON object only. No prose, no code fence.

---

## Prompt 2 — Judge one posting

Runs once per posting that is read in full. **One call, not two.** Extracting requirements and judging them in separate calls is cleaner and doubles the cost of the most expensive part of a run, and the second call only re-reads text the first already had in context.

### System Prompt

You compare one job posting against one candidate's resume and report, requirement by requirement, what the resume does and does not cover. **You do not produce a score.** A score is computed from your verdicts by the application.

Return exactly this shape:

```json
{
  "company": "employer name as the posting states it, or empty string",
  "title": "role title as the posting states it",
  "location": "where the role is, as stated, or empty string",
  "remote": false,
  "requirements": [
    {"id": "R-01", "text": "one requirement in one line", "must_have": true,
     "category": "hard_skill | experience | education | domain | soft_skill | responsibility"}
  ],
  "assessments": [
    {"requirement_id": "R-01", "status": "covered | partial | missing",
     "evidence": "the resume line that shows it, quoted exactly, or empty string",
     "note": "one short sentence of reasoning"}
  ]
}
```

**Extracting requirements**

- Take them from the posting's requirements, qualifications, and responsibilities. Ignore benefits, equal-opportunity statements, company history, and application instructions.
- `must_have` is true when the posting states it as required, essential, or minimum; false when it is preferred, a plus, a bonus, or nice to have. When the posting does not distinguish, treat the first list as required.
- Merge restatements of one thing into one requirement. Split a bullet asking for two unrelated things into two. Emit at most 20, must-haves first.
- Give every requirement an id of the form `R-01`, and emit exactly one assessment per requirement using the same id.

**Judging coverage**

| Status | When |
| :--- | :--- |
| `covered` | The resume shows this directly. The evidence line, on its own, would convince a hiring manager. |
| `partial` | The resume shows something adjacent, less of it, or at smaller scale. Five years asked for, three shown. Kubernetes asked for, Docker shown. |
| `missing` | The resume does not show it. **This is a normal, expected answer.** A report where every requirement is covered tells the candidate nothing and is almost always wrong. |

- `evidence` must be copied from the resume, **word for word**, one line at most. A paraphrase cannot be checked; a quote can.
- Never write evidence for a `missing` requirement, and never quote the **posting** — a posting stating a requirement is not evidence that the candidate meets it.
- Judge only what the resume states. Do not credit a skill because it usually accompanies a stated one, and do not credit seniority because a title sounds senior.

Reply with the JSON object only. No prose, no code fence.

---

## User message shape

Both documents are fenced, and the posting comes **before** the resume — the requirements have to be read out of the posting first, because a model that met the resume first tends to extract the requirements the resume happens to satisfy.

```
Job posting (from https://boards.greenhouse.io/acme/jobs/4551201, listed as 'Senior Backend Engineer'):
<<<UNTRUSTED_DOCUMENT
{fetched page text, normalised and capped}
UNTRUSTED_DOCUMENT>>>

Candidate resume:
<<<UNTRUSTED_DOCUMENT
{extracted resume text}
UNTRUSTED_DOCUMENT>>>

Extract this posting's requirements and judge each one against the resume.
Return the JSON object described in your instructions.
```

---

## What the runtime must own

Prompting cannot guarantee any of the following. Each one is application code in the reference implementation.

### The score is arithmetic, never a model output

The model returns verdicts; the number is computed from them:

```
credit         = covered 1.0 · partial 0.5 · missing 0.0
must_have      = 100 × Σcredit(must-haves)  / count(must-haves)
nice_to_have   = 100 × Σcredit(preferred)   / count(preferred)
score          = 0.8 × must_have + 0.2 × nice_to_have
```

Weights **renormalise** when a posting states no preferred requirements — scoring against a section the posting never had costs the candidate points for nothing. The consequences of computing it this way: the same resume and posting score identically twice, the number is explainable line by line, and a model in a generous mood can inflate one verdict rather than a total.

### Claimed coverage is checked before it counts

A `covered` verdict whose evidence quote does not appear in the resume is demoted to `partial` before it earns credit, and the demotion is **shown**, not hidden. Compare with whitespace and punctuation squashed on both sides: PDF extraction inserts kerning spaces inside words, so a quote that is character-for-character correct still fails a naive containment check.

Where embeddings are available, a second backstop catches a `covered` verdict that carries *no* quote for a requirement no resume line comes near. Calibrate it as a gross-mismatch check rather than a judge — measured with hosted embeddings, genuine coverage sits at 0.71–0.90 and genuine misses at 0.62–0.76, overlapping ranges, so any threshold sharp enough to catch every miss also demotes real matches. A verified quote skips this check entirely; it is the stronger evidence.

### Links come from retrieval, never from the model

The model is never asked for a URL and never given the chance to supply one. Every link on the page is a URL the search provider returned and the fetcher actually fetched. This is what turns "no invented jobs" from an instruction into a property.

### The site whitelist is a restriction, not a filter

Pass it to the search API (`include_domains`), so an off-list page is never fetched and never paid for — then re-check every returned result anyway. "The API honoured it last time" is not the guarantee the product made.

### Non-postings are rejected on URL shape, before fetching

Roughly a third of what a job-site search returns is not a job: board landing pages, company profiles, "42 jobs at Acme" indexes. Scoring a resume against a category page produces a confident number about nothing. Judge by URL structure (`/jobs/{id}`, a UUID segment, a job id in the query string) rather than content, so a rejected page costs nothing. Then de-duplicate on **both** the canonical URL and the normalised title — one role is genuinely posted on an ATS *and* on a big board, under two different URLs.

### Two tiers, and the tier is always visible

Rank everything cheaply; read only the strongest few in full. Deep-scoring thirty postings takes minutes and exhausts a free-tier daily token budget in a couple of runs, which looks like a broken app rather than an expensive one. Every row must state which tier produced its number, in words. Never blend the two into one figure, and never sort a snippet-ranked row above one that was actually read.

### Nothing identifying goes into a search query

A query leaves the process and is logged by a third party. Build queries from titles, skills, domains, and the user's own filters only — never from the summary or a highlight, both of which can quote a resume line verbatim. This is why prompt 1 is instructed not to emit name, email, or phone at all: the surest way not to leak a detail is not to have it in the object that builds queries.

### Ids are re-assigned after parsing

Models number things inconsistently: `1`, `R1`, `Req-3`, verdicts for requirements that were never emitted, requirements with no verdict at all. Every one of those parses as valid JSON and every one corrupts a score silently. Re-index the requirements yourself, pair each with exactly one verdict, and apply any cap **after** ordering must-haves first, so trimming removes the preferred tail rather than the screen-out criteria.

### Partial results beat no results

A posting whose page will not load keeps its snippet score and says why. A job whose scoring call fails keeps its row and carries the error. A run that hits its time budget returns what it scored with the rest ranked. The only failures worth stopping for are the ones that leave nothing to show: an unreadable resume, and a search that returned nothing at all.

### The two untrusted documents are treated differently

A flagged **resume** may stop the run — there is one, and its owner is standing right there to fix it. A flagged **posting** never stops anything: it is one row of thirty, the candidate did not write it, and hiding a job because its page contains an odd sentence hides a job the candidate might want. Annotate the row and carry on; the fence is what contains it.

### A repair turn, then stop

If the reply is not valid JSON in the required shape, resend once with the broken output and the validation error attached: *"Reply again with only the JSON object: no prose before or after it, no code fence, no extra keys."* A second failure is a real failure — past that point the model cannot produce the shape, and looping burns a budget to arrive at the same place.
