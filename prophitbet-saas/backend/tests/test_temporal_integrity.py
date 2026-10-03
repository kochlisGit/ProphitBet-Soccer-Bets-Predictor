import unittest
import numpy as np
import pandas as pd
from src.preprocessing.result_integrity import deduplicate_results
from src.preprocessing.selection import train_test_split
from src.models.temporal_evaluation import evaluate_temporally, probability_scores
from src.preprocessing.utils.target import TargetType


class IntegrityTests(unittest.TestCase):
    def test_duplicate_odds_are_not_selected_arbitrarily(self):
        row = dict(Date='2025-01-01', Home='A', Away='B', HG=2, AG=0, Result='H', Season=2025)
        frame = pd.DataFrame([dict(row, **{'1': 1.5}), dict(row, **{'1': 1.8})])
        clean = deduplicate_results(frame)
        self.assertEqual(len(clean), 1)
        self.assertTrue(pd.isna(clean.iloc[0]['1']))
        frame.loc[1, 'HG'] = 3
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            deduplicate_results(frame)

    def test_date_boundary(self):
        frame = pd.DataFrame({'Date': ['2025-01-03', '2025-01-02', '2025-01-02', '2025-01-01']})
        train, test = train_test_split(frame, 2)
        self.assertLess(train.Date.max(), test.Date.min())
        self.assertEqual(len(test), 3)

    def test_probability_metrics(self):
        result = probability_scores([0, 1], [[1, 0], [0, 1]])
        self.assertEqual(result['brier'], 0)
        self.assertEqual(result['log_loss'], 0)
        for p in ([[.7, .7]], [[float('nan'), 0]], [[-1, 2]]):
            with self.assertRaises(ValueError):
                probability_scores([0], p)

    def test_walk_forward_uses_fresh_models_and_disjoint_dates(self):
        instances = []
        class Dummy:
            def __init__(self):
                self.classifier = type('Estimator', (), {'classes_': np.array([0, 1, 2])})()
                instances.append(self)
            def fit(self, train_df, eval_df=None):
                self.end = train_df.Date.max()
            def predict_proba(self, df):
                assert self.end < df.Date.min()
                return np.full((len(df), 3), 1 / 3)
        frame = pd.DataFrame({'Date': pd.date_range('2024-01-01', periods=25), 'Result': ['H', 'D', 'A', 'H', 'D'] * 5})
        report = evaluate_temporally(frame, Dummy, TargetType.RESULT)
        self.assertEqual(len(instances), 4)
        self.assertEqual(report['folds'][-1]['kind'], 'final_holdout')
        self.assertEqual(sum(f['model']['samples'] for f in report['folds']), 20)


if __name__ == '__main__':
    unittest.main()
