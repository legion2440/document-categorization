from __future__ import annotations

import numpy as np

from utils.revision2_protocol import (
    cramers_v_category_binary,
    has_re_prefix,
    message_number_from_filename,
    normalized_thread_subject,
    parse_rfc_header_block,
    parsed_date_timestamp,
    spearman_correlation,
)


def test_header_parser_unfolds_subject_and_separates_body():
    raw = (
        "From: person@example.com\n"
        "Subject: Re: IBM PS/2 model\n"
        "  70 question\n"
        "Organization: Example Org\n"
        "Date: Mon, 1 Mar 1993 12:30:00 GMT\n"
        "\n"
        "This is the body.\n"
    )
    headers, body = parse_rfc_header_block(raw)
    assert headers["subject"] == "Re: IBM PS/2 model 70 question"
    assert headers["from"] == "person@example.com"
    assert body == "This is the body.\n"


def test_thread_subject_normalization_removes_re_prefix_only_for_thread_identity():
    subject = "  Re: RE:  Graphics   Card  "
    assert has_re_prefix(subject)
    assert normalized_thread_subject(subject) == "graphics card"


def test_message_number_and_date_parsing():
    assert message_number_from_filename("/tmp/comp.graphics/12345") == 12345
    assert message_number_from_filename(b"/tmp/comp.graphics/12345") == 12345
    assert message_number_from_filename("/tmp/comp.graphics/not-a-number") is None
    assert parsed_date_timestamp("Mon, 1 Mar 1993 12:30:00 GMT") is not None
    assert parsed_date_timestamp("not a date") is None


def test_spearman_detects_monotonic_proxy():
    assert spearman_correlation(
        np.asarray([10, 20, 30, 40], dtype=float),
        np.asarray([100, 200, 300, 400], dtype=float),
    ) == 1.0


def test_cramers_v_detects_category_reply_prefix_association():
    independent = cramers_v_category_binary(
        ["a", "a", "b", "b"],
        [False, True, False, True],
    )
    associated = cramers_v_category_binary(
        ["a", "a", "b", "b"],
        [False, False, True, True],
    )
    assert independent == 0.0
    assert associated > 0.9
