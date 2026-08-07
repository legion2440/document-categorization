from models.tagger import DocumentTagger


class Token:
    def __init__(self, lemma, *, is_alpha=True, is_stop=False):
        self.lemma_ = lemma
        self.text = lemma
        self.is_alpha = is_alpha
        self.is_stop = is_stop


def test_keyword_candidates_are_context_frequency_ranked():
    doc = [Token("space"), Token("space"), Token("orbit"), Token("the", is_stop=True), Token("42", is_alpha=False)]
    assert DocumentTagger._keyword_candidates(doc) == ["space", "orbit"]
