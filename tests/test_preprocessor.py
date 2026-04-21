from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from preprocessor import preprocess_data


class PreprocessorTests(unittest.TestCase):
    def test_preprocess_data_returns_expected_objects(self):
        df = pd.DataFrame(
            {
                "Label": ["BENIGN", "ATTACK", "BENIGN", "ATTACK"],
                "Flow Duration": [1.0, np.inf, 3.0, 4.0],
                "Packet Length Mean": [10.0, 12.0, np.nan, 9.0],
                "Protocol": [6, 6, 6, 6],
            }
        )
        X, y, features, encoder, imputer, report = preprocess_data(df, "Label", "BENIGN")
        self.assertEqual(len(X), len(y))
        self.assertTrue(len(features) > 0)
        self.assertIsNotNone(encoder)
        self.assertIsNotNone(imputer)
        self.assertIn("validation", report)


if __name__ == "__main__":
    unittest.main()
