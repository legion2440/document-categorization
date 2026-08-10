"""Context-aware multilingual tagging using spaCy NER plus lexical signals."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

LANGUAGE_MODELS = {"en": "en_core_web_sm", "es": "es_core_news_sm"}
LANGUAGE_DETECTION_PREFIX_CHARS = 300


@dataclass(frozen=True)
class Entity:
    text: str
    label: str


@dataclass(frozen=True)
class TaggingResult:
    language: str
    tags: list[str]
    entities: list[Entity]


@lru_cache(maxsize=4)
def _load_spacy(language: str):
    import spacy

    model_name = LANGUAGE_MODELS.get(language)
    if not model_name:
        raise ValueError(f"Unsupported language: {language}")
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is missing. Run `python scripts/download_models.py`."
        ) from exc

    # The tagging contract needs tokenization, lexical attributes and NER. Surface-form
    # keyword ranking does not require POS tagging, parsing or lemmatization, so keep
    # only the components needed by the NER path.
    for component in tuple(nlp.pipe_names):
        if component not in {"tok2vec", "ner"}:
            nlp.disable_pipe(component)
    return nlp


def detect_language_code(text: str, prefix_chars: int = LANGUAGE_DETECTION_PREFIX_CHARS) -> str:
    from langdetect import DetectorFactory, LangDetectException, detect

    if prefix_chars <= 0:
        raise ValueError("prefix_chars must be positive")
    DetectorFactory.seed = 42
    sample = text[:prefix_chars]
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def detect_language(text: str) -> str:
    language = detect_language_code(text)
    return language if language in LANGUAGE_MODELS else "en"


class DocumentTagger:
    def __init__(
        self,
        max_keyword_tags: int = 6,
        max_entity_tags: int = 8,
        *,
        pipe_batch_size: int = 128,
        n_process: int = 1,
    ):
        if pipe_batch_size <= 0:
            raise ValueError("pipe_batch_size must be positive")
        if n_process == 0 or n_process < -1:
            raise ValueError("n_process must be -1 or a positive integer")
        self.max_keyword_tags = max_keyword_tags
        self.max_entity_tags = max_entity_tags
        self.pipe_batch_size = pipe_batch_size
        self.n_process = n_process

    @staticmethod
    def _keyword_candidates(doc) -> list[str]:
        tokens = [
            token.text.casefold().strip()
            for token in doc
            if token.is_alpha and not token.is_stop and len(token.text) >= 3
        ]
        return [term for term, _ in Counter(tokens).most_common()]

    def tag(self, text: str, language: str | None = None) -> TaggingResult:
        language = language or detect_language(text)
        nlp = _load_spacy(language)
        doc = nlp(text)
        entities = [Entity(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
        entity_tags = []
        for ent in entities:
            normalized = ent.text.casefold()
            if normalized not in entity_tags:
                entity_tags.append(normalized)
        keywords = [k for k in self._keyword_candidates(doc) if k not in entity_tags]
        tags = entity_tags[: self.max_entity_tags] + keywords[: self.max_keyword_tags]
        return TaggingResult(language=language, tags=tags, entities=entities)

    def tag_batch(self, texts: list[str], languages: list[str] | None = None) -> list[TaggingResult]:
        if languages is None:
            languages = [detect_language(text) for text in texts]
        if len(texts) != len(languages):
            raise ValueError("texts and languages must have the same length")

        output: list[TaggingResult | None] = [None] * len(texts)
        for language in sorted(set(languages)):
            indices = [i for i, lang in enumerate(languages) if lang == language]
            nlp = _load_spacy(language)
            docs = nlp.pipe(
                (texts[i] for i in indices),
                batch_size=self.pipe_batch_size,
                n_process=self.n_process,
            )
            for idx, doc in zip(indices, docs):
                entities = [Entity(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
                entity_tags = list(dict.fromkeys(ent.text.casefold() for ent in entities))
                keywords = [k for k in self._keyword_candidates(doc) if k not in entity_tags]
                output[idx] = TaggingResult(
                    language=language,
                    tags=entity_tags[: self.max_entity_tags] + keywords[: self.max_keyword_tags],
                    entities=entities,
                )
        if any(item is None for item in output):
            raise RuntimeError("spaCy batch tagging did not produce one result per input document")
        return [item for item in output if item is not None]
