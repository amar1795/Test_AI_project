import unittest

from src.segment import find_sentence_id, regex_segment


class SegmentTests(unittest.TestCase):
    def test_offsets_always_slice_original_text(self):
        text = "First sentence!  Second sentence?\nLast one."
        sentences = regex_segment(text)
        self.assertEqual([item["text"] for item in sentences], ["First sentence!", "Second sentence?", "Last one."])
        for sentence in sentences:
            self.assertEqual(text[sentence["start"]:sentence["end"]], sentence["text"])

    def test_sentence_lookup_uses_document_offsets(self):
        sentences = regex_segment("Alpha. Beta entity here.")
        self.assertEqual(find_sentence_id(sentences, 12, 18), 1)


if __name__ == "__main__":
    unittest.main()

