# What changes at 100,000 articles

The first component to fail is the JSON registry. Each mention currently scans
same-label entities for fuzzy matches, so resolution tends toward quadratic work as
the registry grows. Rewriting the entire registry and disk cache after a run also
becomes expensive, and one process cannot keep model inference saturated reliably.

For a 100,000-article corpus I would:

1. Move canonical entities, aliases, mention-resolution records, and cache entries
   into PostgreSQL or SQLite first, with unique indexes on `(label, normalized_key)`
   and `(doc_id, start, end, label)`.
2. Retrieve fuzzy candidates from a blocking index (label, acronym, surname, token
   prefixes, or character n-grams) instead of comparing every mention to every
   entity. Keep expensive scoring to a small candidate set.
3. Store raw and processed documents in object storage, put document IDs on a work
   queue, and make every stage idempotent so workers can retry safely.
4. Batch spaCy and GLiNER on GPU workers, measure throughput by model and batch size,
   and separate CPU normalization/resolution workers from model inference.
5. Snapshot and version the registry. Route conflicting writes through transactions
   with uniqueness constraints so two workers cannot create the same entity.
6. Replace the JSON review queue with a durable table that records ownership,
   decisions, audit history, and model/prompt versions.
7. Track per-stage latency, cache hit rate, candidate counts, ambiguity rate, drift,
   and scores on a fixed regression set before each release.

The external behavior and data contract need not change; only the storage and stage
implementations do. That is why the project keeps strict boundaries between stages.

