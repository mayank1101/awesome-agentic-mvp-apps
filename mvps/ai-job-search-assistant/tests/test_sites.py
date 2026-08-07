import pytest

from app.services import sites


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("boards.greenhouse.io", "boards.greenhouse.io"),
        ("https://boards.greenhouse.io/", "boards.greenhouse.io"),
        ("www.LinkedIn.com", "linkedin.com"),
        ("  naukri.com/jobs  ", "naukri.com"),
        ("http://jobs.lever.co:443/acme", "jobs.lever.co"),
        ("not a domain", None),
        ("localhost", None),
        ("", None),
    ],
)
def test_normalize_domain(raw: str, expected: str | None) -> None:
    assert sites.normalize_domain(raw) == expected


def test_normalize_sites_dedupes_and_reports_rejects() -> None:
    accepted, rejected = sites.normalize_sites(
        ["https://linkedin.com", "www.linkedin.com", "nonsense", "jobs.lever.co", "  "]
    )
    assert accepted == ["linkedin.com", "jobs.lever.co"]
    assert rejected == ["nonsense"]


def test_canonical_url_strips_tracking_and_normalises() -> None:
    canonical = sites.canonical_url(
        "HTTPS://WWW.Boards.Greenhouse.io/acme/jobs/123/?utm_source=x&gh_src=y&keep=1#apply"
    )
    assert canonical == "https://boards.greenhouse.io/acme/jobs/123?keep=1"


def test_canonical_url_keeps_job_id_query() -> None:
    canonical = sites.canonical_url("https://acme.com/careers?gh_jid=42&utm_campaign=z")
    assert canonical == "https://acme.com/careers?gh_jid=42"


@pytest.mark.parametrize(
    "url",
    [
        "https://boards.greenhouse.io/acme/jobs/4567890",
        "https://job-boards.greenhouse.io/acme/jobs/4567890",
        "https://jobs.lever.co/acme/2f1c9a44-1111-2222-3333-444455556666",
        "https://jobs.ashbyhq.com/acme/9f8e7d6c-aaaa-bbbb-cccc-ddddeeeeffff",
        "https://apply.workable.com/j/9AB12CD34E",
        "https://www.linkedin.com/jobs/view/3912345678",
        "https://www.naukri.com/job-listings-backend-engineer-acme-bengaluru-1234567",
        "https://wellfound.com/jobs/1234567-backend-engineer",
        "https://acme.com/careers?gh_jid=42",
    ],
)
def test_is_probable_posting_accepts_real_postings(url: str) -> None:
    assert sites.is_probable_posting(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://boards.greenhouse.io/acme",
        "https://jobs.lever.co/acme",
        "https://www.linkedin.com/company/acme",
        "https://www.linkedin.com/jobs/search?keywords=backend",
        "https://wellfound.com/jobs",
        "https://example.com/blog/how-we-hire",
        "https://example.com/careers",
        "not-a-url",
    ],
)
def test_is_probable_posting_rejects_listing_pages(url: str) -> None:
    assert sites.is_probable_posting(url) is False


def test_dedupe_key_matches_the_same_role_across_boards() -> None:
    _, left = sites.dedupe_key("https://a.example/1", "Senior Backend Engineer")
    _, right = sites.dedupe_key("https://b.example/2", "Senior  Backend   Engineer!")
    assert left == right


def test_dedupe_key_url_half_ignores_tracking() -> None:
    left, _ = sites.dedupe_key("https://boards.greenhouse.io/acme/jobs/1?utm_source=x", "A")
    right, _ = sites.dedupe_key("https://boards.greenhouse.io/acme/jobs/1/", "B")
    assert left == right


def test_the_ashby_embed_variant_is_stripped() -> None:
    # A live run returned this URL; extraction fails on the widget variant and
    # works on the page itself, so the job was scored on a snippet for a reason
    # that had nothing to do with the job.
    canonical = sites.canonical_url(
        "https://jobs.ashbyhq.com/savvymoney/126f643c-2e75-4d55-b662-8c1beaccf62e?embed=js"
    )
    assert canonical == ("https://jobs.ashbyhq.com/savvymoney/126f643c-2e75-4d55-b662-8c1beaccf62e")
