# Error analysis template

Generate a review set with:

```powershell
python eval/analyze_errors.py --pred data/output --gold data/gold
```

For each error, record one category and a short observation. Suggested categories:

- missed Indian or transliterated person name
- honorific or boundary mismatch
- government or military organization mislabeled
- ORG versus GPE disagreement
- nested or overlapping span
- abbreviation or acronym
- domain entity absent from the model's fixed labels
- gold annotation error

Do not change extraction rules until at least 30 false positives and 30 false
negatives have been reviewed on the real corpus.

