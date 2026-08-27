import tempfile
import unittest
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineIntegrationTests(unittest.TestCase):
    def test_sample_pipeline_is_offset_safe_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = {
                "input_dir": PROJECT_ROOT / "data" / "raw",
                "normalized_dir": root / "normalized",
                "output_dir": root / "output",
                "registry_path": root / "registry.json",
                "cache_path": root / "cache.json",
            }
            first_summary = run_pipeline(**arguments)
            registry_before = (root / "registry.json").read_text(encoding="utf-8")
            outputs_before = {
                path.name: path.read_text(encoding="utf-8")
                for path in (root / "output").glob("*.json")
            }

            second_summary = run_pipeline(**arguments)
            self.assertEqual(first_summary["documents_processed"], 3)
            self.assertEqual(second_summary["cache_hit_rate"], 1.0)
            self.assertEqual(registry_before, (root / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(
                outputs_before,
                {path.name: path.read_text(encoding="utf-8") for path in (root / "output").glob("*.json")},
            )

            manifest = read_json(root / "output" / "manifest.json")
            for doc_id in manifest["document_ids"]:
                document = read_json(root / "output" / f"{doc_id}.json")
                for mention in document["mentions"]:
                    self.assertEqual(document["text"][mention["start"]:mention["end"]], mention["text"])


if __name__ == "__main__":
    unittest.main()

