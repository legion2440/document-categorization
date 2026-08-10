from utils.text_preprocessing import (
    canonical_window,
    normalize_text,
    remove_token_dense_lines,
    remove_token_dense_lines_batch,
)


def test_normalize_text_handles_html_urls_email_and_whitespace():
    result = normalize_text("A&nbsp; B\nhttps://example.com  me@example.com")
    assert result == "A B <URL> <EMAIL>"


def test_canonical_window_is_the_single_word_bound():
    assert canonical_window("one  two\nthree four", 3) == "one two three"


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
