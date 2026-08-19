"""Real-time classification + tagging pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from models.tagger import DocumentTagger, TAGGING_WINDOW_WORDS, detect_language
from models.text_classifier import (
    ClassifierConfig,
    build_model,
    load_runtime_config,
    tokenize_with_budget,
)
from utils.text_preprocessing import (
    CLASSIFICATION_WINDOW_WORDS,
    canonical_window,
    prepare_revision2_texts,
    revision2_content_token_counts,
    revision2_document_representation,
)

INFERENCE_BUCKET_LENGTHS = (64, 128, 192, 256, 384, 512)
INFERENCE_PRECISION_POLICIES = ("float32", "mixed_float16")


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    language: str
    tags: list[str]
    entities: list[dict[str, str]]


def _runtime_bucket_lengths(max_length: int) -> tuple[int, ...]:
    lengths = [length for length in INFERENCE_BUCKET_LENGTHS if length < max_length]
    lengths.append(max_length)
    return tuple(sorted(set(lengths)))


def _bucket_for_length(length: int, buckets: tuple[int, ...]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    raise ValueError(f"Tokenized length {length} exceeds configured buckets {buckets}")


def attention_balanced_batch_sizes(
    buckets: tuple[int, ...],
    longest_batch_size: int,
    *,
    max_batch_size: int = 64,
) -> dict[int, int]:
    """Keep batch*sequence^2 bounded by the proven-safe longest-bucket workload."""
    if not buckets:
        raise ValueError("At least one inference bucket is required")
    if longest_batch_size <= 0 or max_batch_size <= 0:
        raise ValueError("Inference batch sizes must be positive")
    if max_batch_size < longest_batch_size:
        raise ValueError("max_batch_size cannot be smaller than longest_batch_size")

    longest = max(buckets)
    attention_budget = longest_batch_size * longest * longest
    profile: dict[int, int] = {}
    for bucket in buckets:
        candidate = max(longest_batch_size, attention_budget // (bucket * bucket))
        candidate = min(candidate, max_batch_size)
        power_of_two = 1 << (int(candidate).bit_length() - 1)
        profile[bucket] = max(longest_batch_size, power_of_two)
    profile[longest] = longest_batch_size
    return profile


def prepare_inference_texts(
    texts: list[str],
    languages: list[str] | None = None,
    *,
    token_counts: Callable[[list[str]], list[int]] = revision2_content_token_counts,
) -> tuple[list[str], list[str]]:
    """Prepare raw serving inputs with the same Revision 2 policy used for training.

    RFC-style input receives the frozen Subject+body representation. Arbitrary
    plain text is preserved. English uses the train-calibrated token-density
    cleanup; Spanish mirrors the post-translation structural/window finalization.
    """
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("all texts must be strings")
    if languages is not None and len(languages) != len(texts):
        raise ValueError("texts and languages must have the same length")

    if languages is None:
        detection_inputs = []
        for text in texts:
            representation, _, _ = revision2_document_representation(
                text,
                assume_rfc_headers=False,
            )
            detection_inputs.append(canonical_window(representation, CLASSIFICATION_WINDOW_WORDS))
        resolved_languages = [detect_language(text) for text in detection_inputs]
    else:
        resolved_languages = [str(language) for language in languages]

    dense_mask = [language == "en" for language in resolved_languages]
    prepared, _ = prepare_revision2_texts(
        texts,
        token_counts if any(dense_mask) else None,
        assume_rfc_headers=False,
        token_dense_cleanup=dense_mask,
    )
    if any(not text for text in prepared):
        raise ValueError("Documents must contain non-empty text after preprocessing")
    return prepared, resolved_languages


class DocumentCategorizationPipeline:
    def __init__(
        self,
        checkpoint_dir: str | Path = "models/checkpoints",
        *,
        weights_name: str = "text_classifier_best.h5",
        classifier_batch_size: int | None = None,
        classifier_batch_sizes: dict[int, int] | None = None,
        precision_policy: str = "float32",
        jit_compile: bool = False,
        tagger_window_words: int = TAGGING_WINDOW_WORDS,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.json"
        weights_path = checkpoint_dir / weights_name
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                "Trained classifier artifacts are missing. Run `python scripts/train_revision2.py` first."
            )
        if precision_policy not in INFERENCE_PRECISION_POLICIES:
            raise ValueError(
                f"precision_policy must be one of {INFERENCE_PRECISION_POLICIES}, got {precision_policy!r}"
            )
        if tagger_window_words <= 0:
            raise ValueError("tagger_window_words must be positive")

        runtime = load_runtime_config(config_path)
        self.labels = list(runtime["labels"])
        self.config = ClassifierConfig(
            model_name=runtime["model_name"],
            max_length=int(runtime["max_length"]),
            learning_rate=float(runtime["learning_rate"]),
            epochs=int(runtime["epochs"]),
            batch_size=int(runtime["batch_size"]),
            weight_decay=float(runtime.get("weight_decay", 0.01)),
            warmup_ratio=float(runtime.get("warmup_ratio", 0.10)),
            gradient_clip_norm=float(runtime.get("gradient_clip_norm", 1.0)),
            random_seed=int(runtime.get("random_seed", 42)),
        )
        self.bucket_lengths = _runtime_bucket_lengths(self.config.max_length)
        if classifier_batch_size is not None and classifier_batch_sizes is not None:
            raise ValueError("Use either classifier_batch_size or classifier_batch_sizes, not both")
        if classifier_batch_sizes is not None:
            normalized = {int(bucket): int(size) for bucket, size in classifier_batch_sizes.items()}
            if set(normalized) != set(self.bucket_lengths):
                raise ValueError(
                    f"classifier_batch_sizes must define exactly these buckets: {self.bucket_lengths}"
                )
            if any(size <= 0 for size in normalized.values()):
                raise ValueError("classifier batch sizes must be positive")
            self.classifier_batch_sizes = normalized
        else:
            batch_size = int(classifier_batch_size or self.config.batch_size)
            if batch_size <= 0:
                raise ValueError("classifier_batch_size must be positive")
            self.classifier_batch_sizes = {bucket: batch_size for bucket in self.bucket_lengths}

        import tf_keras

        tf_keras.mixed_precision.set_global_policy(precision_policy)
        self.precision_policy = tf_keras.mixed_precision.global_policy().name
        self.jit_compile = bool(jit_compile)
        self.tagger_window_words = int(tagger_window_words)
        self.tokenizer, self.model = build_model(len(self.labels), self.config)
        self.model.load_weights(weights_path)
        self._compiled_classifier_functions: dict[tuple[int, int], object] = {}
        self.tagger = DocumentTagger(window_words=self.tagger_window_words)
        self._stage_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tagger-stage")

    @property
    def compiled_classifier_shapes(self) -> list[list[int]]:
        return [[batch, sequence] for batch, sequence in sorted(self._compiled_classifier_functions)]

    def _compiled_classifier(self, batch_size: int, bucket_length: int):
        key = (batch_size, bucket_length)
        compiled = self._compiled_classifier_functions.get(key)
        if compiled is not None:
            return compiled

        import tensorflow as tf

        @tf.function(
            input_signature=[
                tf.TensorSpec([batch_size, bucket_length], tf.int32, name="input_ids"),
                tf.TensorSpec([batch_size, bucket_length], tf.int32, name="attention_mask"),
            ],
            jit_compile=True,
        )
        def run(input_ids, attention_mask):
            outputs = self.model(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                training=False,
            )
            return tf.cast(outputs.logits, tf.float32)

        self._compiled_classifier_functions[key] = run
        return run

    def _classifier_logits(
        self,
        batch_ids: np.ndarray,
        batch_mask: np.ndarray,
        bucket_length: int,
    ):
        import tensorflow as tf

        input_ids = tf.convert_to_tensor(batch_ids)
        attention_mask = tf.convert_to_tensor(batch_mask)
        if self.jit_compile:
            batch_size = int(batch_ids.shape[0])
            return self._compiled_classifier(batch_size, bucket_length)(input_ids, attention_mask)
        outputs = self.model(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            training=False,
        )
        return tf.cast(outputs.logits, tf.float32)

    def _fixed_batch_arrays(
        self,
        items: list[tuple[int, list[int], list[int]]],
        bucket_length: int,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        pad_token_id = int(self.tokenizer.pad_token_id or 0)
        input_ids = np.full((batch_size, bucket_length), pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((batch_size, bucket_length), dtype=np.int32)

        for row, (_, ids, mask) in enumerate(items):
            length = len(ids)
            input_ids[row, :length] = ids
            attention_mask[row, :length] = mask

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        for row in range(len(items), batch_size):
            input_ids[row, 0] = int(cls_token_id if cls_token_id is not None else pad_token_id)
            attention_mask[row, 0] = 1
            if bucket_length > 1 and sep_token_id is not None:
                input_ids[row, 1] = int(sep_token_id)
                attention_mask[row, 1] = 1

        return input_ids, attention_mask

    def _classify_batch_with_stats(
        self,
        texts: list[str],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        import tensorflow as tf

        tokenized, clipped = tokenize_with_budget(self.tokenizer, texts, self.config.max_length)
        input_ids = tokenized["input_ids"]
        attention_masks = tokenized["attention_mask"]
        groups: dict[int, list[tuple[int, list[int], list[int]]]] = {
            bucket: [] for bucket in self.bucket_lengths
        }
        for index, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
            bucket = _bucket_for_length(len(ids), self.bucket_lengths)
            groups[bucket].append((index, list(ids), list(mask)))

        predicted = np.full(len(texts), -1, dtype=np.int32)
        confidence = np.full(len(texts), np.nan, dtype=np.float32)
        bucket_documents: dict[str, int] = {}
        bucket_batches: dict[str, int] = {}
        bucket_model_rows: dict[str, int] = {}
        model_rows = 0

        for bucket in self.bucket_lengths:
            items = groups[bucket]
            if not items:
                continue
            batch_size = self.classifier_batch_sizes[bucket]
            bucket_documents[str(bucket)] = len(items)
            batches = 0
            rows = 0
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                batch_ids, batch_mask = self._fixed_batch_arrays(chunk, bucket, batch_size)
                logits = self._classifier_logits(batch_ids, batch_mask, bucket)
                probabilities = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1).numpy()[: len(chunk)]
                ids = probabilities.argmax(axis=-1)
                for row, (original_index, _, _) in enumerate(chunk):
                    label_id = int(ids[row])
                    predicted[original_index] = label_id
                    confidence[original_index] = float(probabilities[row, label_id])
                batches += 1
                rows += batch_size
                model_rows += batch_size
            bucket_batches[str(bucket)] = batches
            bucket_model_rows[str(bucket)] = rows

        expected_order = np.arange(len(texts), dtype=np.int32)
        restored_order = np.flatnonzero(predicted >= 0).astype(np.int32)
        if not np.array_equal(restored_order, expected_order):
            raise RuntimeError("Fixed-bucket inference did not produce one prediction per input document")

        stats: dict[str, object] = {
            "documents": len(texts),
            "clipped_documents": int(clipped),
            "precision_policy": self.precision_policy,
            "jit_compile": self.jit_compile,
            "classifier_batch_sizes": {str(k): v for k, v in self.classifier_batch_sizes.items()},
            "bucket_lengths": list(self.bucket_lengths),
            "bucket_documents": bucket_documents,
            "bucket_batches": bucket_batches,
            "bucket_model_rows": bucket_model_rows,
            "model_rows": model_rows,
            "dummy_rows": model_rows - len(texts),
        }
        return predicted, confidence, stats

    def _classify_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        predicted, confidence, _ = self._classify_batch_with_stats(texts)
        return predicted, confidence

    def warmup_classifier(self) -> None:
        """Warm every fixed [batch, sequence] shape used by classifier inference."""
        pad_token_id = int(self.tokenizer.pad_token_id or 0)
        cls_token_id = int(self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else pad_token_id)
        sep_token_id = self.tokenizer.sep_token_id
        for bucket in self.bucket_lengths:
            batch_size = self.classifier_batch_sizes[bucket]
            input_ids = np.full((batch_size, bucket), pad_token_id, dtype=np.int32)
            attention_mask = np.zeros((batch_size, bucket), dtype=np.int32)
            input_ids[:, 0] = cls_token_id
            attention_mask[:, 0] = 1
            if bucket > 1 and sep_token_id is not None:
                input_ids[:, 1] = int(sep_token_id)
                attention_mask[:, 1] = 1
            self._classifier_logits(input_ids, attention_mask, bucket)

    def process_batch(
        self,
        texts: list[str],
        languages: list[str] | None = None,
        *,
        parallel_stages: bool = False,
    ) -> list[Prediction]:
        prepared, resolved_languages = prepare_inference_texts(texts, languages)

        if parallel_stages:
            tagging_future = self._stage_pool.submit(
                self.tagger.tag_batch,
                prepared,
                resolved_languages,
            )
            ids, confidence = self._classify_batch(prepared)
            tagging = tagging_future.result()
        else:
            ids, confidence = self._classify_batch(prepared)
            tagging = self.tagger.tag_batch(prepared, resolved_languages)

        return [
            Prediction(
                category=self.labels[int(label_id)],
                confidence=float(score),
                language=tag.language,
                tags=tag.tags,
                entities=[{"text": ent.text, "label": ent.label} for ent in tag.entities],
            )
            for label_id, score, tag in zip(ids, confidence, tagging)
        ]

    def process(self, text: str, language: str | None = None) -> Prediction:
        return self.process_batch([text], [language] if language else None)[0]

    def close(self) -> None:
        self._stage_pool.shutdown(wait=True)
