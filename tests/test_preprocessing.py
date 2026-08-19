from utils.text_preprocessing import (
    canonical_window,
    normalize_text,
    prepare_revision2_texts,
    remove_structural_noise,
    remove_token_dense_lines,
    remove_token_dense_lines_batch,
    revision2_document_representation,
)


def test_normalize_text_handles_html_urls_email_and_whitespace():
    result = normalize_text("A&nbsp; B\nhttps://example.com  me@example.com")
    assert result == "A B <URL> <EMAIL>"


def test_canonical_window_is_the_single_word_bound():
    assert canonical_window("one  two\nthree four", 3) == "one two three"


def test_structural_noise_removes_runs_but_preserves_embedded_content():
    text = (
        "************************************************ COLOR 19'' ZENITH TV for SALE *************************************************\n"
        + " ." * 30
        + "\nnormal prose stays here"
    )
    cleaned, removed_runs, removed_lines = remove_structural_noise(text)
    assert removed_runs == 2
    assert removed_lines == 1
    assert "COLOR 19'' ZENITH TV for SALE" in cleaned
    assert "normal prose stays here" in cleaned


def test_token_dense_line_cleanup_preserves_normal_prose():
    text = "normal prose stays here\n" + ("-=" * 30) + "\nsecond normal line"

    def token_count(line: str) -> int:
        return 100 if "-=" in line else len(line.split())

    cleaned, removed = remove_token_dense_lines(
        text,
        token_count,
        min_chars=40,
        max_tokens_per_word=20.0,
    )
    assert removed == 1
    assert "normal prose stays here" in cleaned
    assert "second normal line" in cleaned
    assert "-=" not in cleaned


def test_batched_token_dense_cleanup_preserves_document_alignment():
    texts = [
        "first prose\n" + ("+-" * 30),
        "second prose only",
    ]

    def token_counts(lines: list[str]) -> list[int]:
        return [100 if "+-" in line else len(line.split()) for line in lines]

    cleaned, removed = remove_token_dense_lines_batch(
        texts,
        token_counts,
        min_chars=40,
        max_tokens_per_word=20.0,
    )
    assert cleaned == ["first prose", "second prose only"]
    assert removed == [1, 0]


def test_serving_rfc_detection_does_not_drop_plain_text_before_blank_line():
    raw = "NASA: mission planning is discussed here.\n\nThe second paragraph stays too."
    representation, subject, thread_subject = revision2_document_representation(
        raw,
        assume_rfc_headers=False,
    )
    assert representation == raw
    assert subject == ""
    assert thread_subject == ""


def test_shared_revision2_preprocessing_applies_subject_structural_and_dense_cleanup():
    raw = (
        "From: poster@example.com\n"
        "Subject: Re: Space mission update\n"
        "Organization: Example Org\n"
        "\n"
        "NASA prepares the spacecraft.\n"
        + ("ENCODED" * 12)
        + "\nMission control remains nominal."
    )

    def token_counts(lines: list[str]) -> list[int]:
        return [100 if "ENCODED" in line else len(line.split()) for line in lines]

    prepared, stats = prepare_revision2_texts(
        [raw],
        token_counts,
        assume_rfc_headers=True,
        token_dense_cleanup=True,
    )

    assert prepared == [
        "Space mission update NASA prepares the spacecraft. Mission control remains nominal."
    ]
    assert stats["removed_token_dense_lines"] == 1
    assert "poster" not in prepared[0]
    assert "Example Org" not in prepared[0]
