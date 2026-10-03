import unittest
import pandas as pd
from backend.app.services.dataset_validation import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([dict(Date='2025-01-01', Home='A', Away='B', HG=2, AG=1, Result='H')])

    def test_good(self):
        validate_dataset(self.df, 1)

    def test_drop(self):
        with self.assertRaisesRegex(ValueError, '20%'):
            validate_dataset(self.df, 10)

    def test_invalid_records(self):
        for field, value in [('Result','D'), ('HG',-1), ('AG',1.5), ('Home','B'),
                             ('Home',None), ('Date','2099-01-01'), ('Date','unknown')]:
            with self.subTest(field=field, value=value):
                altered = self.df.copy()
                altered[field] = value
                with self.assertRaises(ValueError):
                    validate_dataset(altered)

    def test_duplicate_or_missing(self):
        for frame in [pd.concat([self.df,self.df]), self.df.drop(columns=['Result']), self.df.iloc[:0]]:
            with self.assertRaises(ValueError):
                validate_dataset(frame)
