# Experiment results

| Configuration | Corpus | Strict P/R/F1 | Relaxed P/R/F1 | Notes |
|---|---|---:|---:|---|
| Keyword baseline (`1.0.0`) | Included 3-document fixture, 13 mentions | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | Sanity check only; the lexicon intentionally covers the fixture. |
| spaCy `en_core_web_sm` | Real 50-document gold set | Pending | Pending | Run after replacing the sample corpus. |
| spaCy `en_core_web_trf` | Real 50-document gold set | Pending | Pending | Run after replacing the sample corpus. |
| GLiNER medium | Real 50-document gold set | Pending | Pending | Sweep label wording and threshold first. |
| spaCy + GLiNER merge | Real 50-document gold set | Pending | Pending | spaCy owns core labels; GLiNER owns domain labels. |

The sample score proves the glue, offsets, scorer, and resolution path are working; it
is not evidence of real-world NER quality. Timestamped experiment JSON is written to
`eval/results/` and intentionally not overwritten.

