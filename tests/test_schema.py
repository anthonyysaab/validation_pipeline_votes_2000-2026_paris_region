import unittest

import pandas as pd

from src.get_data.schema import candidate_columns_from_schema, is_missing_like


class SchemaTests(unittest.TestCase):
    def test_candidate_columns_use_explicit_schema_and_numeric_dtype(self) -> None:
        frame = pd.DataFrame(
            {
                "id_bvote": ["1", "2"],
                "nb_exprim": [10, 20],
                "geo_shape": [None, None],
                "alice": [6, 12],
                "bob": [4, 8],
            }
        )

        self.assertEqual(
            candidate_columns_from_schema(frame, ["id_bvote", "nb_exprim"]),
            ["alice", "bob"],
        )

    def test_unknown_text_column_fails_closed(self) -> None:
        frame = pd.DataFrame({"id_bvote": ["1"], "unexpected": ["value"]})

        with self.assertRaisesRegex(ValueError, "unexpected"):
            candidate_columns_from_schema(frame, ["id_bvote"])

    def test_missing_markers_are_consistent(self) -> None:
        values = pd.Series([None, "", "NaN", "none", "<NA>", "null", "candidate"])

        self.assertEqual(is_missing_like(values).tolist(), [True] * 6 + [False])


if __name__ == "__main__":
    unittest.main()
