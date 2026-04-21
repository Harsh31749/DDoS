from __future__ import annotations

import unittest

import pandas as pd

from inference_contract import align_features, validate_bundle


class InferenceContractTests(unittest.TestCase):
    def test_align_features_adds_missing_columns_in_order(self):
        df = pd.DataFrame([{"b": 2.0, "a": 1.0}])
        aligned = align_features(df, ["a", "b", "c"])
        self.assertEqual(list(aligned.columns), ["a", "b", "c"])
        self.assertEqual(float(aligned.iloc[0]["c"]), 0.0)

    def test_validate_bundle_requires_core_keys(self):
        with self.assertRaises(ValueError):
            validate_bundle({"model": object()})


if __name__ == "__main__":
    unittest.main()
