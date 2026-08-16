# Revision 2 pre-registration

Registered before any Revision 2 preprocessing, training, checkpoint selection, calibration, or second final test evaluation.

## Immutable Revision 1 result

Revision 1 remains the primary pre-registered result and is preserved in `reports/revision1_performance_metrics.json`.

- test accuracy: `0.819534984789222`
- macro F1: `0.8191512653974548`
- throughput: `134.08416548759763 docs/s`
- baseline accuracy: `0.7781399391568883`
- relative accuracy improvement over baseline: `0.05319743088522766`

Revision 1 failed only the registered `accuracy >= 0.85` gate among the primary assignment thresholds. It is not replaced or hidden by Revision 2.

## Final-test policy

Revision 2 is allowed exactly one final test evaluation after its protocol, model, checkpoint, calibration method, and runtime are frozen. There will be no third final test evaluation regardless of the Revision 2 result. If Revision 2 fails any assignment threshold, the failure is recorded and the experiment stops.

Revision 1 and Revision 2 results must be published together. Revision 1 test errors and predictions are not inputs to Revision 2 design or model selection.

## Revision 2 rationale

The original project voluntarily used a stricter setup than the assignment explicitly requires: header removal, official bydate test, mechanically size-selected categories, and a 150-word classification window. Revision 2 does not relax the test threshold, weaken the baseline, manually choose easier categories, expand the classifier window, or introduce a post-test ensemble.

Revision 2 changes two methodological choices that can be justified without reference to Revision 1 test errors:

1. Treat `Subject` as a legitimate document-title signal while removing routing/identity metadata.
2. Replace random stratified validation with a temporal proxy built only from official-train documents.

The machine-readable decision rules are frozen in `config/revision2_protocol.json`.

## Document representation

Fetch 20 Newsgroups with sklearn removal limited to `footers` and `quotes`. Parse the remaining RFC-style header block ourselves.

Only `Subject` is allowed into model content. Known routing/identity fields are explicitly denied in configuration and unknown header fields are dropped. The resulting classifier text is `Subject` followed by body. The classifier window remains 150 words.

The Spanish copy translates the combined title+body input. Revision 1 translation cache entries are not reused; Revision 2 uses a separate cache namespace.

## Train-only diagnostics before implementation

Before Revision 2 preprocessing or training, run diagnostics on `subset="train"` only. The diagnostic step must not fetch, open, count, or score the official test split.

The diagnostics establish three facts using rules already fixed in `config/revision2_protocol.json`:

- whether leading `Re:` is sufficiently category-associated to strip from model content;
- whether filename message number is a defensible temporal proxy for parsed `Date:`;
- whether normalized thread subjects cross the candidate temporal train/validation boundary often enough to require grouped splitting.

If parseable `Date:` coverage is below the registered minimum, the project stops before retraining and revises the train-only protocol without consulting test metrics.

## Category selection

After Revision 2 cleaning, re-run the original mechanical category rule exactly. Categories are ranked by cleaned official-train source count descending with lexicographic tie-break, then the minimal prefix reaching at least 11,000 cleaned official train+test source documents is selected.

If the category set changes because `Subject` rescues previously empty documents, the mechanically produced new set is accepted. No category may be manually added or removed based on model performance.

## Validation and training

Validation comes only from official train. The registered temporal-key decision determines the ordering. The latest 15% per category becomes validation.

If the registered subject-overlap threshold is reached, documents are grouped by category plus normalized thread subject and whole late groups are assigned to validation; no group is split. EN/ES pairs always remain in the same split.

Training remains:

- `microsoft/mdeberta-v3-base`
- 5 epochs
- learning rate `2e-5`
- batch size `2`
- AdamW
- weight decay `0.01`
- warmup ratio `0.10`
- gradient clipping `1.0`

Checkpoint selection is highest validation correct-document count, then lower validation loss, then earlier epoch. There is no ensemble and no train+validation retrain.

## Calibration and runtime

Temperature scaling is re-fitted on the new Revision 2 validation split by minimum negative log-likelihood. Revision 1 temperature `2.5834003220128468` is not reused. Calibration must preserve argmax.

Runtime semantics remain fixed: float32, XLA, parallel classifier/tagger stages, 150-word classifier window, 75-word tagger window, and the registered attention-balanced batch profile.

## Failure definition

Revision 2 succeeds only if its one final test evaluation satisfies all registered assignment gates:

- accuracy `>= 0.85`
- macro F1 `>= 0.80`
- throughput `>= 100 docs/s`
- accuracy `>= 0.80` for every supported language
- relative accuracy improvement over the unchanged baseline `>= 5%`

The absolute percentage-point improvement is reported separately but is not the registered interpretation of the assignment's `>=5%` wording.

If Revision 2 produces, for example, `83.5%` accuracy, that number is published as the final Revision 2 result and the project stops. No third test run is permitted.
