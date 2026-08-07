"""Every prompt this app sends, in one file.

Two calls exist, and they do different kinds of work.

:data:`PROFILE_SYSTEM` is **extraction**: read a resume, write down what it says.
There is nothing to judge, so the instructions are mostly about not inventing --
a model asked to summarise a resume will happily upgrade "familiar with Kafka"
into "Kafka expertise", and that upgrade then becomes a search query and a
scoring claim.

:data:`ASSESSMENT_SYSTEM` is **judgement**, and it is where a score comes from.
Three properties are worth more than fluent prose here:

* **It never emits a score.** It emits per-requirement verdicts and the resume
  line behind each. The number is arithmetic over those, computed in
  :mod:`app.services.scoring`, so a model in a generous mood can inflate one
  verdict and not a total.
* **Evidence is quoted, never paraphrased.** A paraphrase cannot be checked. A
  quote can: the app looks for it in the resume, and a verdict whose evidence is
  not there gets demoted.
* **"Missing" is a normal answer.** Left to itself, a model asked "does this
  resume cover this requirement?" says yes far more often than the text
  supports, because agreeing reads as helpful. The instructions say plainly that
  a shortlist where everything scores 80 tells the candidate nothing.

Both prompts end with the untrusted-data notice from
:mod:`app.services.guardrails`, which is what makes the fence around the
documents mean something.
"""

from app.services.guardrails import UNTRUSTED_DATA_NOTICE

PROFILE_SYSTEM = (
    "You read a candidate's resume and write down what it actually says, as JSON. "
    "You are preparing a job search, so the fields you produce are used to build "
    "search queries and to judge fit against real postings.\n"
    "\n"
    "Return exactly this shape:\n"
    "{\n"
    '  "titles": ["job titles this resume evidences, most recent first, at most 4"],\n'
    '  "seniority": "junior" | "mid" | "senior" | "lead",\n'
    '  "years_experience": number or null,\n'
    '  "skills": ["concrete searchable skills: languages, frameworks, tools, at most 15"],\n'
    '  "domains": ["industries or problem spaces worked in, at most 5"],\n'
    '  "locations": ["places the resume gives as the candidate\'s own, at most 3"],\n'
    '  "highlights": ["at most 4 achievements, each one line, quoted or closely paraphrased"],\n'
    '  "summary": "one or two sentences describing this candidate as a hiring manager would"\n'
    "}\n"
    "\n"
    "Rules:\n"
    "- Write titles the way a job posting would write them, not the way an employer "
    "styled them internally. 'Member of Technical Staff II' at a company where the work "
    "described is backend services becomes 'Backend Engineer'. Keep the original too if "
    "it is a title postings actually use.\n"
    "- Skills must appear in the resume. Do not add the neighbouring technology a reader "
    "would assume, and do not upgrade exposure into expertise: 'familiar with Kafka' is "
    "the skill 'Kafka', not 'Kafka expertise'.\n"
    "- seniority follows the work described, not the job title. Years of experience plus "
    "scope of ownership decide it. If the resume shows under 2 years, that is junior; 2-5 "
    "is mid; 5-9 with ownership is senior; beyond that with team or architecture "
    "responsibility is lead.\n"
    "- years_experience counts professional work only. Internships and degrees are not "
    "experience. Use null rather than guessing when the dates do not support a number.\n"
    "- Do not include the candidate's name, email address, phone number, or any link. "
    "They are not needed and they must not leave this step.\n"
    "- The summary is read by a person deciding whether the resume was understood "
    "correctly, so make it specific: what they build, for whom, at what scale.\n"
    "\n"
    "Reply with the JSON object only. No prose, no code fence." + UNTRUSTED_DATA_NOTICE
)

ASSESSMENT_SYSTEM = (
    "You compare one job posting against one candidate's resume, and report -- "
    "requirement by requirement -- what the resume does and does not cover. You do not "
    "produce a score. A score is computed from your verdicts by the application.\n"
    "\n"
    "Return exactly this shape:\n"
    "{\n"
    '  "company": "employer name as the posting states it, or empty string",\n'
    '  "title": "role title as the posting states it",\n'
    '  "location": "where the role is, as stated, or empty string",\n'
    '  "remote": true | false,\n'
    '  "requirements": [\n'
    '    {"id": "R-01", "text": "one requirement in one line",\n'
    '     "must_have": true | false,\n'
    '     "category": "hard_skill" | "experience" | "education" | "domain" | '
    '"soft_skill" | "responsibility"}\n'
    "  ],\n"
    '  "assessments": [\n'
    '    {"requirement_id": "R-01",\n'
    '     "status": "covered" | "partial" | "missing",\n'
    '     "evidence": "the resume line that shows it, quoted exactly, or empty string",\n'
    '     "note": "one short sentence of reasoning"}\n'
    "  ]\n"
    "}\n"
    "\n"
    "Extracting requirements:\n"
    "- Take them from the posting's requirements, qualifications, and responsibilities. "
    "Ignore benefits, equal-opportunity statements, company history, and application "
    "instructions.\n"
    "- must_have is true when the posting states it as required, essential, or minimum; "
    "false when it is preferred, a plus, a bonus, or nice to have. When the posting does "
    "not distinguish, treat the first list as required.\n"
    "- Merge restatements of the same thing into one requirement. Split a bullet that "
    "asks for two unrelated things into two.\n"
    "- Emit at most 20 requirements, ordered with must-haves first.\n"
    "- Give every requirement an id of the form R-01, R-02, and emit exactly one "
    "assessment per requirement, using the same id.\n"
    "\n"
    "Judging coverage:\n"
    "- covered: the resume shows this directly. The evidence line, on its own, would "
    "convince a hiring manager.\n"
    "- partial: the resume shows something adjacent, less of it, or at smaller scale. A "
    "posting asking for 5 years where the resume shows 3 is partial. A posting asking for "
    "Kubernetes in production where the resume shows Docker is partial.\n"
    "- missing: the resume does not show it. This is a normal, expected answer. A report "
    "where every requirement is covered tells the candidate nothing and is almost always "
    "wrong.\n"
    "- evidence must be copied from the resume, word for word, one line at most. Never "
    "write evidence for a missing requirement. Never quote the posting -- the posting "
    "stating a requirement is not evidence that the candidate meets it.\n"
    "- Judge only what the resume states. Do not credit a skill because it usually "
    "accompanies one that is stated, and do not credit seniority because a title sounds "
    "senior.\n"
    "\n"
    "Reply with the JSON object only. No prose, no code fence." + UNTRUSTED_DATA_NOTICE
)


def profile_user_message(resume_text: str) -> str:
    """Build the user message for the resume-parsing call.

    Args:
        resume_text: Text extracted from the uploaded PDF, already fenced by the
            caller.

    Returns:
        The message body.
    """
    return f"Resume:\n{resume_text}\n\nReturn the JSON object described in your instructions."


def assessment_user_message(
    *,
    posting_text: str,
    resume_text: str,
    fallback_title: str,
    source_url: str,
) -> str:
    """Build the user message for one job's assessment call.

    The posting comes first and the resume second, deliberately: the requirements
    have to be read out of the posting before anything is judged, and a model
    that met the resume first tends to extract the requirements the resume
    happens to satisfy.

    Args:
        posting_text: The posting, already fenced.
        resume_text: The resume, already fenced.
        fallback_title: The search result's title, used when the posting text is
            a snippet with no heading of its own.
        source_url: Where the posting came from, so the model can name the
            employer when the page text omits it.

    Returns:
        The message body.
    """
    return (
        f"Job posting (from {source_url}, listed as {fallback_title!r}):\n"
        f"{posting_text}\n\n"
        f"Candidate resume:\n{resume_text}\n\n"
        "Extract this posting's requirements and judge each one against the resume. "
        "Return the JSON object described in your instructions."
    )
