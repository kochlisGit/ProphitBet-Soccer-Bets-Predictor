"""Synthetic test fixtures only; never used as production training data."""
import unittest

import numpy as np
import pandas as pd

from src.preprocessing.utils.target import TargetType, construct_targets, target_columns, target_labels, parse_target
from src.preprocessing.dataset import DatasetPreprocessor
from backend.app.services.model_service import prepare_target_data, train_model
from backend.app.services.model_artifact import deserialize_model
from backend.app.api.models import TrainRequest, AutoTuneRequest


class MarketTargetTests(unittest.TestCase):
    def test_every_target_accepted_by_api(self):
        for target in TargetType:
            for request in (TrainRequest, AutoTuneRequest):
                self.assertEqual(request(league_id=1, model_type='logistic', target_type=target.value).target_type, target.value)
        self.assertEqual(parse_target('over_under'), TargetType.OVER_UNDER)
        with self.assertRaises(ValueError):
            parse_target('invented-market')

    def test_total_boundaries(self):
        for target in TargetType:
            if target in (TargetType.RESULT, TargetType.BTTS):
                continue
            line = 2.5 if target == TargetType.OVER_UNDER else float(target.value.rsplit('-', 1)[1])
            columns = target_columns(target)
            df = pd.DataFrame({columns[0]: [int(line), int(line) + 1]})
            for column in columns[1:]:
                df[column] = 0
            self.assertEqual(construct_targets(df, target).tolist(), [0, 1], target.value)

    def test_btts_and_result(self):
        self.assertEqual(construct_targets(pd.DataFrame({'HG': [0, 1, 2], 'AG': [1, 0, 3]}), TargetType.BTTS).tolist(), [0, 0, 1])
        self.assertEqual(target_labels('btts'), ['No', 'Yes'])
        self.assertEqual(construct_targets(pd.DataFrame({'Result': ['H', 'D', 'A']}), TargetType.RESULT).tolist(), [0, 1, 2])

    def test_missing_and_invalid_outcomes_not_zero_filled(self):
        for value in (None, -1, 1.2, np.inf, 'unknown'):
            with self.assertRaises(ValueError):
                construct_targets(pd.DataFrame({'HC': [value], 'AC': [2]}), TargetType.CORNERS_95)
        with self.assertRaisesRegex(ValueError, 'recorded columns'):
            prepare_target_data(pd.DataFrame({'HG': [1], 'AG': [2]}), 'corners-9.5')

    def test_target_specific_rows_and_no_outcome_leakage(self):
        df = pd.DataFrame({'HC': [5, None], 'AC': [6, 5], 'HG': [None, None], 'HCF': [4., 3.]})
        df = prepare_target_data(df, 'corners-9.5')
        self.assertEqual(len(df), 1)
        x, y, _ = DatasetPreprocessor().preprocess_dataset(df, TargetType.CORNERS_95)
        self.assertEqual(x.tolist(), [[4.]])
        self.assertEqual(y.tolist(), [1])

    def test_inference_needs_features_not_future_outcomes(self):
        x, y, _ = DatasetPreprocessor().preprocess_dataset(pd.DataFrame({'HCF': [4.]}), TargetType.CORNERS_95, include_targets=False)
        self.assertIsNone(y)
        self.assertEqual(x.tolist(), [[4.]])

    def test_train_safe_roundtrip_and_predict(self):
        # Alternating synthetic counts exercise both classes in time-ordered folds.
        n = 200
        df = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=n),
                           'Home': ['Home'] * n, 'Away': ['Away'] * n,
                           'HG': [1] * n, 'AG': [0] * n, 'Result': ['H'] * n,
                           'HC': [i % 8 for i in range(n)], 'AC': [5] * n,
                           'HCF': [float(i % 5) for i in range(n)]})
        result = train_model(df, 'decision_tree', 'corners-9.5', use_odds=False)
        self.assertEqual(result['metrics']['target_type'], 'corners-9.5')
        model = deserialize_model(result['model_bytes'])
        self.assertEqual(model.target_type, TargetType.CORNERS_95)
        self.assertEqual(model.predict_proba(pd.DataFrame({'HCF': [3.]})).shape, (1, 2))
