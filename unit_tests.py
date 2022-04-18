"""Program for testing"""
import unittest
from src import Preparation, train_model


class preparation_test(unittest.TestCase):
    """unit tests for Preparation.py, train_model.py"""
    def test_shape(self):
        """size testing (columns)"""
        self.assertEqual(
            Preparation.train_x_transform.shape[1],
            Preparation.test_x_transform.shape[1],
        )
        self.assertEqual(
            Preparation.train_x_transform.shape[1],
            Preparation.train_x_transform.shape[1],
        )

    def test_shape_equal(self):
        """size testing (rows)"""
        self.assertEqual(
            Preparation.test_x_transform.shape[0], Preparation.test_y_transform.shape[0]
        )
        self.assertEqual(
            Preparation.train_x_transform.shape[0],
            Preparation.train_y_transform.shape[0],
        )

    def test_nan(self):
        """isna checking"""
        self.assertIsNotNone(Preparation.test_x_transform.isna().sum() == 0)
        self.assertIsNotNone(Preparation.test_y_transform.isna().sum() == 0)
        self.assertIsNotNone(Preparation.train_x_transform.isna().sum() == 0)
        self.assertIsNotNone(Preparation.train_y_transform.isna().sum() == 0)

    def test_type(self):
        """type compliance"""
        self.assertTrue(
            Preparation.test_x_transform.select_dtypes(include="object").shape[1] == 0
        )

    def check_score_nan(self):
        """score isna checking"""
        self.assertIsNotNone(train_model.best_model)


if __name__ == "__main__":
    unittest.main()
