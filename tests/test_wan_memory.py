import unittest
from pathlib import Path


TRANSFORMER_PATH = Path(__file__).resolve().parents[1] / "bernini" / "models" / "transformer_wan.py"


class WanMemoryTests(unittest.TestCase):
    def test_single_gpu_transformer_uses_broadcast_timestep_embeddings(self):
        source = TRANSFORMER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "use_per_token_timestep = get_parallel_state().ulysses_enabled or len(batch_image_vae_seqlen) > 1",
            source,
        )
        self.assertIn("else:\n            timestep_proj_indices = None\n            temb = temb[:1]", source)
        self.assertIn(
            "if timestep_proj_indices is not None:\n            timestep_proj = timestep_proj[timestep_proj_indices].unsqueeze(0)\n        else:\n            timestep_proj = timestep_proj[:1]",
            source,
        )


if __name__ == "__main__":
    unittest.main()
