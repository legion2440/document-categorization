# Revision 2 Amendment 01 — Validation fallback

Registered before any Revision 2 preprocessing, training, checkpoint selection, calibration, or second test evaluation.

## Why the original temporal branch stopped

The Revision 2 pre-registration required the official-train `Date:` field to be parseable for at least 80% of source documents before either parsed dates or message numbers could be used as a temporal ordering signal.

The train-only diagnostic examined all 11,314 official training documents and found zero parseable `Date:` headers. It did not read the held-out test split. Therefore the registered stop condition fired and no temporal proxy is accepted. Message-number ordering is not assumed to be temporal without train-only evidence linking it to dates.

The same diagnostic measured Cramér's V between category and a leading `Re:` marker at 0.3788, above the pre-registered 0.10 threshold, so leading `Re:` markers are removed from the model Subject in Revision 2.

## Replacement validation design

Revision 2 uses a deterministic thread-grouped stratified fallback built only from official train:

1. Normalize Subject for thread identity with NFKC, case-folding, whitespace collapse, and repeated leading `Re:` removal.
2. Form groups by `category + normalized Subject`; a missing Subject becomes a unique per-document group.
3. Process categories in stable sorted order, deterministically shuffle whole groups with NumPy seed 42, and assign complete groups to validation until that category contains at least 15% of its source documents in validation.
4. Never split a thread group between train and validation.
5. English and Spanish rows derived from the same source document remain in the same split.

This fallback is not selected using Revision 1 test predictions, errors, per-category metrics, test document inspection, model accuracy, model F1, or class-separability measurements.

## What remains unchanged

Revision 1's 81.9535% test accuracy remains the immutable primary result of the original frozen protocol. Revision 2 is still the second and final allowed held-out test evaluation. A third test run is forbidden regardless of Revision 2 outcome.

The model family, five-epoch training budget, learning rate, batch size, optimizer, 150-word classifier window, baseline definition, no-ensemble rule, no train+validation retrain rule, temperature-scaling method, and production runtime policy remain fixed as registered before this amendment.
