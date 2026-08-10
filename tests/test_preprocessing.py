from utils.text_preprocessing import canonical_window, normalize_text, truncate_for_translation


def test_normalize_text_handles_html_urls_email_and_whitespace():
    result = normalize_text("A&nbsp; B\nhttps://example.com  me@example.com")
    assert result == "A B <URL> <EMAIL>"


def test_canonical_window_is_the_explicit_word_bound():
    assert canonical_window("one  two\nthree four", 3) == "one two three"


def test_truncate_translation_input_is_bounded():
    result = truncate_for_translation("word " * 100, max_chars=40)
    assert len(result) <= 40
    assert not result.endswith(" ")
