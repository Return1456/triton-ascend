"""Offline tests for the centralized Ascend arch classification table.

Importable without an installed Triton wheel or NPU runtime.
"""
import importlib.util
from pathlib import Path
import unittest


def _load_arch_module():
    backend = Path(__file__).resolve().parents[2] / "backend"
    spec = importlib.util.spec_from_file_location("ascend_arch_test", backend / "ascend_arch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AscendArchTableTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_arch_module()

    def test_register_based_generations(self):
        gen5 = self.mod.AscendArchGen.GEN5
        gen6 = self.mod.AscendArchGen.GEN6
        for arch in ("Ascend910_95", "Ascend950", "Ascend910_958b",
                      "Ascend910_96", "Ascend960", "Ascend960DT_9691"):
            with self.subTest(arch=arch):
                gen = self.mod.arch_generation(arch)
                self.assertIn(gen, (gen5, gen6))
                self.assertTrue(self.mod.is_register_based_arch(arch))

    def test_legacy_not_register_based(self):
        for arch in ("Ascend910A", "Ascend910B1", "Ascend910_93",
                      "Ascend310B", "Ascend910_97", ""):
            with self.subTest(arch=arch):
                self.assertFalse(self.mod.is_register_based_arch(arch))

    def test_ub_and_rf_capacity(self):
        cases = {
            "Ascend960":     (320, 128),
            "Ascend910_96":  (320, 128),
            "Ascend950":     (256, 128),
            "Ascend910_95":  (256, 128),
            "Ascend910B1":   (192, None),
            "Ascend910A":    (192, None),
            "Ascend910_97":  (0, None),
            "":              (0, None),
        }
        for arch, (ub, rf) in cases.items():
            with self.subTest(arch=arch):
                self.assertEqual(self.mod.ub_size_in_kbytes_for_arch(arch), ub)
                self.assertEqual(self.mod.rf_size_in_kbytes_for_arch(arch), rf)

    def test_ffts_policy(self):
        for arch in ("Ascend960", "Ascend910_95", "Ascend950"):
            with self.subTest(arch=arch):
                self.assertTrue(self.mod.is_ffts_disabled_arch(arch))
                self.assertFalse(self.mod.is_ffts_supported(arch))
        # Ascend910A and Ascend310B4 are legacy but have FFTS disabled
        for arch in ("Ascend910A", "Ascend310B4"):
            with self.subTest(arch=arch):
                self.assertTrue(self.mod.is_ffts_disabled_arch(arch))
                self.assertFalse(self.mod.is_ffts_supported(arch))
        # Other legacy arches have FFTS enabled
        for arch in ("Ascend910B1", "Ascend910_93", "Ascend310B", "Ascend910D"):
            with self.subTest(arch=arch):
                self.assertFalse(self.mod.is_ffts_disabled_arch(arch))
                self.assertTrue(self.mod.is_ffts_supported(arch))

    def test_non_string_and_none_inputs(self):
        legacy = self.mod.AscendArchGen.LEGACY
        for bad in (None, 123, [], {}):
            with self.subTest(bad=bad):
                self.assertFalse(self.mod.is_register_based_arch(bad))
                self.assertEqual(self.mod.ub_size_in_kbytes_for_arch(bad), 0)
                self.assertEqual(self.mod.rf_size_in_kbytes_for_arch(bad), None)
                self.assertFalse(self.mod.is_ffts_disabled_arch(bad))
                self.assertEqual(self.mod.arch_generation(bad), legacy)


if __name__ == "__main__":
    unittest.main()
