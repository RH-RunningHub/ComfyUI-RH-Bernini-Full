import json
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _load_nodes_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rh_bernini_nodes_contract_test", PLUGIN_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TranscribeSkillContractTests(unittest.TestCase):
    def test_user_visible_inputs_have_tooltips(self):
        nodes = _load_nodes_module()
        for class_name, node_class in nodes.NODE_CLASS_MAPPINGS.items():
            input_types = node_class.INPUT_TYPES()
            for section in ("required", "optional"):
                for input_name, spec in input_types.get(section, {}).items():
                    with self.subTest(node=class_name, input=input_name):
                        self.assertGreaterEqual(len(spec), 2)
                        options = spec[1]
                        self.assertIsInstance(options, dict)
                        self.assertIn("tooltip", options)
                        self.assertTrue(str(options["tooltip"]).strip())

    def test_api_examples_exist_and_use_registered_node_names(self):
        nodes = _load_nodes_module()
        registered = set(nodes.NODE_CLASS_MAPPINGS)
        examples = sorted((PLUGIN_ROOT / "examples").glob("*_api.json"))
        self.assertGreaterEqual(len(examples), 1)

        seen_bernini_nodes = set()
        for example in examples:
            with self.subTest(example=example.name):
                data = json.loads(example.read_text(encoding="utf-8"))
                self.assertIsInstance(data, dict)
                for node in data.values():
                    class_type = node.get("class_type")
                    if class_type in registered:
                        seen_bernini_nodes.add(class_type)

        self.assertEqual(seen_bernini_nodes, registered)


if __name__ == "__main__":
    unittest.main()
