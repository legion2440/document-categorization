"""Context-aware multilingual tagging using spaCy NER plus lexical signals."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

LANGUAGE_MODELS = {"en": "en_core_web_sm", "es": "es_core_news_sm"}

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
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is missing. Run `python scripts/download_models.py`."
        ) from exc


def detect_language(text: str) -> str:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 42
    try:
        language = detect(text)
    except LangDetectException:
        return "en"
    return language if language in LANGUAGE_MODELS else "en"


class DocumentTagger:
    def __init__(self, max_keyword_tags: int = 6, max_entity_tags: int = 8):
        self.max_keyword_tags = max_keyword_tags
        self.max_entity_tags = max_entity_tags

    @staticmethod
    def _keyword_candidates(doc) -> list[str]:
        tokens = [
            token.lemma_.lower().strip()
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
            docs = nlp.pipe((texts[i] for i in indices), batch_size=64)
            for idx, doc in zip(indices, docs):
                entities = [Entity(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
                entity_tags = list(dict.fromkeys(ent.text.casefold() for ent in entities))
                keywords = [k for k in self._keyword_candidates(doc) if k not in entity_tags]
                output[idx] = TaggingResult(
                    language=language,
                    tags=entity_tags[: self.max_entity_tags] + keywords[: self.max_keyword_tags],
                    entities=entities,
                )
        return [item for item in output if item is not None]
