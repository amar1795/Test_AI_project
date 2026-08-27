# Annotation guidelines

These rules freeze the conventions used by the included gold fixture. Re-check all
existing annotations before changing any rule.

- Annotate every explicit `PERSON`, `ORG`, and `GPE` mention. Domain labels such as
  `FACILITY` are annotated when configured.
- Include an immediately preceding honorific (`Dr.`, `Mr.`, `Justice`) in a PERSON
  span. Normalization removes it only from the matching key.
- Annotate the full legal organization surface, including corporate suffixes.
  Normalization removes suffixes only from the matching key.
- Treat armed services and named government bodies as `ORG` in the core label set.
- Do not infer unspoken entities, resolve pronouns, or annotate adjectival fragments
  such as `Delhi-based` as GPE.
- Offsets are zero-based, document-level, start-inclusive, and end-exclusive.
  `text[start:end]` must reproduce the annotation exactly.

