import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_SKILL_PATH_ENV = os.environ.get("BERNINI_TRANSCRIBE_SKILL_PATH")
SKILL_PATH = Path(_SKILL_PATH_ENV) if _SKILL_PATH_ENV else None


def _load_nodes_module():
    spec = importlib.util.spec_from_file_location("rh_bernini_nodes_for_progress_test", PLUGIN_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProgressBar:
    instances = []

    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id
        self.values = []
        FakeProgressBar.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        self.values.append(int(value))


class FakePipeline:
    def __init__(self):
        self.callback_seen = False

    def __call__(self, *args, progress_callback=None, **kwargs):
        self.callback_seen = callable(progress_callback)
        if progress_callback is not None:
            progress_callback("planning", 1, 2)
            progress_callback("planning", 2, 2)
            progress_callback("diffusion", 1, 4)
            progress_callback("diffusion", 4, 4)
            progress_callback("decode", 1, 1)
            progress_callback("save", 1, 1)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        FakeProgressBar.instances.clear()
        comfy_module = types.ModuleType("comfy")
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.ProgressBar = FakeProgressBar
        comfy_module.utils = comfy_utils
        sys.modules["comfy"] = comfy_module
        sys.modules["comfy.utils"] = comfy_utils

        class FakeDevice:
            def __init__(self, name):
                self.name = name
                self.type = str(name).split(":", 1)[0]

            def __str__(self):
                return self.name

        fake_torch = types.ModuleType("torch")
        fake_torch.device = FakeDevice
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            set_device=lambda device: None,
            empty_cache=lambda: None,
        )
        sys.modules["torch"] = fake_torch

    def test_run_bernini_reports_progress_through_comfyui_progress_bar(self):
        nodes = _load_nodes_module()
        fake_pipeline = FakePipeline()
        nodes._resolve_model_dir = lambda model_dir: "/tmp/bernini-model"
        nodes._resolve_acceleration_loras = lambda task_type, acceleration: {}
        nodes._get_pipeline = lambda *args, **kwargs: ("cache-key", fake_pipeline)
        nodes._temp_path = lambda prefix, suffix: f"/tmp/{prefix}{suffix}"

        output_path = nodes._run_bernini(
            task_type="t2i",
            prompt="test prompt",
            num_frames=1,
            height=512,
            width=512,
            num_inference_steps=4,
            seed=1234,
            output_prefix="progress_test",
            keep_model_loaded=True,
        )

        self.assertEqual(output_path, "/tmp/progress_test.png")
        self.assertTrue(fake_pipeline.callback_seen)
        self.assertEqual(len(FakeProgressBar.instances), 1)
        values = FakeProgressBar.instances[0].values
        self.assertGreaterEqual(len(values), 6)
        self.assertEqual(values[-1], 100)
        self.assertEqual(values, sorted(values))

    @unittest.skipUnless(SKILL_PATH and SKILL_PATH.exists(), "transcribe plugin skill is not available in this runtime")
    def test_transcribe_plugin_skill_requires_progress_for_long_comfyui_nodes(self):
        text = SKILL_PATH.read_text(encoding="utf-8")
        self.assertIn("ProgressBar", text)
        self.assertIn("progress_callback", text)
        self.assertIn("长任务", text)


if __name__ == "__main__":
    unittest.main()
