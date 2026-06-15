import unittest
from pathlib import Path


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "bernini" / "pipeline.py"


class LoRAMemoryTests(unittest.TestCase):
    def test_lora_forward_uses_inplace_add_to_avoid_large_temporaries(self):
        source = PIPELINE_PATH.read_text(encoding="utf-8")
        self.assertIn("return output.add_(lora, alpha=self.multiplier)", source)
        self.assertNotIn("return output + lora.to(dtype=output.dtype) * self.multiplier", source)


if __name__ == "__main__":
    unittest.main()
