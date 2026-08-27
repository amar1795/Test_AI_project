import unittest

from src.disambiguate_llm import disambiguate, parse_resolution_response
from src.schema import ContractError


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, prompt):
        return next(self.responses)


class LLMTests(unittest.TestCase):
    def test_markdown_fenced_json_is_accepted(self):
        value = parse_resolution_response(
            '```json\n{"entity_id":"ent_00001","confidence":0.8,"reasoning":"context matches"}\n```'
        )
        self.assertEqual(value["entity_id"], "ent_00001")

    def test_missing_field_is_rejected(self):
        with self.assertRaises(ContractError):
            parse_resolution_response('{"entity_id": null, "confidence": 0.1}')

    def test_invalid_response_retries_then_succeeds(self):
        client = FakeClient([
            "not json",
            '{"entity_id":null,"confidence":0.2,"reasoning":"insufficient evidence"}',
        ])
        mention = {"text": "Sharma", "sent_id": 0}
        document = {"sentences": [{"id": 0, "text": "Sharma spoke."}]}
        result = disambiguate(client, mention, document, [], sleep=lambda _: None)
        self.assertIsNone(result["entity_id"])


if __name__ == "__main__":
    unittest.main()

