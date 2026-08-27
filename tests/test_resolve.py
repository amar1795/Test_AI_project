import json
import tempfile
import unittest
from pathlib import Path

from src.config import load_config
from src.resolve import EntityRegistry


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "entities.json"
        self.config = load_config()
        self.document = {
            "doc_id": "doc-a",
            "published_at": "2026-03-14T00:00:00Z",
        }

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def mention(text, start, label="PERSON"):
        return {"id": "m0", "text": text, "start": start, "end": start + len(text), "label": label}

    def test_initials_resolve_and_registry_is_idempotent(self):
        registry = EntityRegistry(self.path, self.config)
        first = registry.resolve(self.mention("Dr. Rajesh Sharma", 0), self.document)
        second_document = {"doc_id": "doc-b", "published_at": "2026-03-15T00:00:00Z"}
        second = registry.resolve(self.mention("R. Sharma", 20), second_document)
        self.assertEqual(first.entity_id, second.entity_id)
        registry.save()
        before = self.path.read_text(encoding="utf-8")

        reloaded = EntityRegistry(self.path, self.config)
        repeated = reloaded.resolve(self.mention("R. Sharma", 20), second_document)
        reloaded.save()
        self.assertEqual(second.method, repeated.method)
        self.assertEqual(before, self.path.read_text(encoding="utf-8"))

    def test_different_labels_never_merge(self):
        registry = EntityRegistry(self.path, self.config)
        person = registry.resolve(self.mention("Jordan", 0, "PERSON"), self.document)
        organization = registry.resolve(self.mention("Jordan", 10, "ORG"), self.document)
        self.assertNotEqual(person.entity_id, organization.entity_id)

    def test_blocked_alias_short_circuits(self):
        registry = EntityRegistry(self.path, self.config)
        result = registry.resolve(self.mention("Rajesh Sharma", 0), self.document)
        registry.entities[result.entity_id]["blocked_aliases"] = ["Sharma"]
        ambiguous = registry.resolve(self.mention("Sharma", 50), self.document)
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertEqual(ambiguous.method, "blocked_alias")


if __name__ == "__main__":
    unittest.main()

