import unittest

from src.extract_gliner import merge_mentions


class ExtractTests(unittest.TestCase):
    def test_longest_span_wins_and_ids_are_stable(self):
        mentions = [
            {"id": "", "text": "Bhabha", "start": 0, "end": 6, "label": "ORG", "source": "gliner", "confidence": 0.9, "sent_id": 0},
            {"id": "", "text": "Bhabha Atomic Research Centre", "start": 0, "end": 29, "label": "ORG", "source": "spacy", "confidence": 1.0, "sent_id": 0},
        ]
        merged = merge_mentions(mentions)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "Bhabha Atomic Research Centre")
        self.assertEqual(merged[0]["id"], "m0")


if __name__ == "__main__":
    unittest.main()

