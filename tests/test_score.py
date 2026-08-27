import unittest

from eval.score import score_mentions


class ScoreTests(unittest.TestCase):
    def test_strict_and_relaxed_are_one_to_one(self):
        gold = [{"doc_id": "d", "start": 5, "end": 10, "label": "PERSON", "text": "Alice"}]
        predicted = [
            {"doc_id": "d", "start": 5, "end": 10, "label": "PERSON", "text": "Alice"},
            {"doc_id": "d", "start": 6, "end": 10, "label": "PERSON", "text": "lice"},
        ]
        score = score_mentions(predicted, gold)
        self.assertEqual(score["strict"]["OVERALL"]["true_positive"], 1)
        self.assertEqual(score["relaxed"]["OVERALL"]["true_positive"], 1)
        self.assertEqual(score["strict"]["OVERALL"]["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()

