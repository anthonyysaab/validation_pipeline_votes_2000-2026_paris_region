import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.get_data.audit.audit_raw_to_clean_cases import prepare_expected_cases
from src.get_data.clean.build_clean_results import build_wide_vote_share_table
from src.get_data.clean.clean_election_results import PROJECT_ROOT, clean_one_file
from src.get_data.clean.clean_municipal_results import clean_one_file as clean_municipal_file
from src.get_data.download.extract_municipales_datagouv import prepare_long_municipal_rows


class PipelineCoreTests(unittest.TestCase):
    def test_wide_source_round_trips_through_clean_and_audit(self) -> None:
        source = pd.DataFrame(
            {
                "id_bvote": ["75001_0001"],
                "scrutin": ["Test"],
                "annee": [2024],
                "tour": [1],
                "num_arrond": [1],
                "num_bureau": [1],
                "nb_inscr": [12],
                "nb_votant": [11],
                "nb_exprim": [10],
                "alice": [6],
                "bob": [4],
            }
        )

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            source_path = Path(temporary) / "elections-test-2024-1ertour.parquet"
            source.to_parquet(source_path, index=False)

            cleaned, report = clean_one_file(source_path)
            expected = prepare_expected_cases(source_path)

        self.assertEqual(report["status"], "ok")
        self.assertIsNotNone(cleaned)
        self.assertIsNotNone(expected)
        assert cleaned is not None and expected is not None

        actual_cases = cleaned[["candidate_source_column", "votes"]].sort_values(
            "candidate_source_column"
        )
        expected_cases = expected[["candidate_source_column", "expected_votes"]].sort_values(
            "candidate_source_column"
        )
        self.assertEqual(actual_cases["candidate_source_column"].tolist(), ["alice", "bob"])
        self.assertEqual(actual_cases["votes"].tolist(), expected_cases["expected_votes"].tolist())

        wide = build_wide_vote_share_table(cleaned)
        self.assertAlmostEqual(float(wide.loc[0, "alice"]), 0.6)
        self.assertAlmostEqual(float(wide.loc[0, "bob"]), 0.4)

    def test_municipal_source_extracts_and_cleans(self) -> None:
        source = pd.DataFrame(
            {
                "id_election": ["2020_muni_t1"],
                "code_departement": ["75"],
                "code_commune": ["75056"],
                "code_bv": [101],
                "voix": [42],
                "ratio_voix_exprimes": [60.0],
                "ratio_voix_inscrits": [50.0],
                "liste": ["Liste test"],
            }
        )

        extracted = prepare_long_municipal_rows(source)
        self.assertEqual(extracted.loc[0, "source_bv_id"], "75056_0101")
        self.assertEqual(extracted.loc[0, "candidate"], "Liste test")

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            source_path = Path(temporary) / "elections-municipales-2020-1ertour.parquet"
            extracted.to_parquet(source_path, index=False)
            cleaned, report = clean_municipal_file(source_path)

        self.assertEqual(report["status"], "ok")
        self.assertIsNotNone(cleaned)
        assert cleaned is not None
        self.assertEqual(cleaned.loc[0, "candidate"], "Liste test")
        self.assertEqual(float(cleaned.loc[0, "votes"]), 42.0)


if __name__ == "__main__":
    unittest.main()
