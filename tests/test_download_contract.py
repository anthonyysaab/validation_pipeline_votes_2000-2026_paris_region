import unittest

import pandas as pd

from src.get_data.download.fetch_non_municipal_data import validate_frame


class DownloadContractTests(unittest.TestCase):
    def test_accepts_canonical_source_schema(self) -> None:
        validate_frame(
            "canonical",
            pd.DataFrame({"scrutin": ["x"], "annee": [2024], "tour": [1]}),
        )

    def test_accepts_documented_legacy_aliases(self) -> None:
        validate_frame(
            "legacy",
            pd.DataFrame(
                {"type_election": ["x"], "annee": [2022], "numero_tour": [2]}
            ),
        )

    def test_rejects_incomplete_source_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "required columns"):
            validate_frame("incomplete", pd.DataFrame({"annee": [2024]}))

    def test_rejects_empty_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "no rows"):
            validate_frame("empty", pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
