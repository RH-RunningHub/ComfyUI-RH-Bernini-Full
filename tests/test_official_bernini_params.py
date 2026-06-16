import importlib.util
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _load_nodes_module():
    spec = importlib.util.spec_from_file_location("rh_bernini_nodes_params_test", PLUGIN_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialBerniniParamsTests(unittest.TestCase):
    def setUp(self):
        self.nodes = _load_nodes_module()
        self.calls = []

        def fake_run(**kwargs):
            self.calls.append(kwargs)
            return "/tmp/rh_bernini_test.mp4" if int(kwargs["num_frames"]) > 1 else "/tmp/rh_bernini_test.png"

        self.nodes._run_bernini = fake_run
        self.nodes._image_batch_to_paths = lambda image, prefix: [f"/tmp/{prefix}.png"]
        self.nodes._video_to_path = lambda video, prefix: f"/tmp/{prefix}.mp4"
        self.nodes._image_from_path = lambda path: ("IMAGE", path)
        self.nodes._video_from_path = lambda path: ("VIDEO", path)

    def _last(self):
        self.assertTrue(self.calls)
        return self.calls[-1]

    def test_only_current_adapted_lightx2v_lora_is_auto_detected(self):
        self.assertEqual(
            self.nodes._T2V_LORA_CANDIDATES,
            {
                "high": ["Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors"],
                "low": ["Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors"],
            },
        )

    def test_t2i_matches_upstream_run_script_params(self):
        self.nodes.RHBerniniFullTextToImage().generate(
            prompt="prompt",
            negative_prompt="",
            width=512,
            height=512,
            steps=50,
            seed=42,
            quality_preset="standard",
            acceleration="none",
        )
        call = self._last()
        self.assertEqual(call["task_type"], "t2i")
        self.assertEqual(call["max_image_size"], 842)
        self.assertEqual(call["guidance_mode"], "vae_txt_vit_wapg")
        self.assertEqual(call["omega_scale"], 1.0)

    def test_i2i_matches_upstream_run_script_params(self):
        self.nodes.RHBerniniFullImageToImage().generate(
            image=object(),
            prompt="prompt",
            negative_prompt="",
            width=512,
            height=512,
            steps=40,
            seed=42,
            quality_preset="standard",
            acceleration="none",
        )
        call = self._last()
        self.assertEqual(call["task_type"], "i2i")
        self.assertEqual(call["max_image_size"], 842)
        self.assertEqual(call["omega_img"], 1.25)
        self.assertEqual(call["omega_vid"], 1.25)
        self.assertEqual(call["omega_scale"], 0.75)

    def test_video_nodes_match_upstream_run_script_params(self):
        self.nodes.RHBerniniFullTextToVideo().generate(
            prompt="prompt",
            negative_prompt="",
            num_frames=81,
            fps=16,
            width=848,
            height=480,
            steps=50,
            seed=42,
            quality_preset="standard",
            acceleration="none",
        )
        t2v = self._last()
        self.assertEqual(t2v["task_type"], "t2v")
        self.assertEqual(t2v["max_image_size"], 842)
        self.assertEqual(t2v["omega_scale"], 1.0)

        self.nodes.RHBerniniFullVideoToVideo().generate(
            video=object(),
            prompt="prompt",
            negative_prompt="",
            num_frames=81,
            fps=16,
            width=0,
            height=0,
            steps=40,
            seed=42,
            quality_preset="standard",
            acceleration="none",
        )
        v2v = self._last()
        self.assertEqual(v2v["task_type"], "v2v")
        self.assertEqual(v2v["max_image_size"], 848)
        self.assertEqual(v2v["system_prompt"], "You are a helpful assistant specialized in video editing.")
        self.assertEqual(v2v["omega_img"], 1.25)
        self.assertEqual(v2v["omega_vid"], 1.25)
        self.assertEqual(v2v["omega_scale"], 0.75)

    def test_reference_nodes_match_upstream_run_script_params(self):
        self.nodes.RHBerniniFullReferenceToVideo().generate(
            reference_image=object(),
            prompt="prompt",
            negative_prompt="",
            num_frames=81,
            fps=16,
            width=832,
            height=480,
            steps=40,
            seed=42,
            quality_preset="standard",
            acceleration="none",
        )
        r2v = self._last()
        self.assertEqual(r2v["task_type"], "r2v")
        self.assertEqual(r2v["max_image_size"], 842)
        self.assertEqual(r2v["guidance_mode"], "vae_txt_vit_wapg")
        self.assertEqual(r2v["omega_img"], 4.5)
        self.assertEqual(r2v["omega_tgt"], 1.5)
        self.assertEqual(r2v["omega_scale"], 0.8)

        self.nodes.RHBerniniFullReferenceVideoToVideo().generate(
            video=object(),
            reference_image=object(),
            prompt="prompt",
            negative_prompt="",
            num_frames=81,
            fps=16,
            width=0,
            height=0,
            steps=40,
            seed=42,
            quality_preset="standard",
            acceleration="none",
        )
        rv2v = self._last()
        self.assertEqual(rv2v["task_type"], "rv2v")
        self.assertEqual(rv2v["max_image_size"], 848)
        self.assertEqual(rv2v["guidance_mode"], "rv2v_wapg")
        self.assertNotIn("system_prompt", rv2v)
        self.assertEqual(rv2v["omega_img"], 3.0)
        self.assertEqual(rv2v["omega_vid"], 0.75)
        self.assertEqual(rv2v["omega_tgt"], 1.5)
        self.assertEqual(rv2v["omega_scale"], 0.75)


if __name__ == "__main__":
    unittest.main()
