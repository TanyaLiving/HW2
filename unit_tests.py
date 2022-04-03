import unittest
from src import Preparation, train_model


class preparation_test(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(
            Preparation.train_X_transform.shape[1],
            Preparation.test_X_transform.shape[1],
        )
        self.assertEqual(
            Preparation.train_X_transform.shape[1],
            Preparation.train_X_transform.shape[1],
        )

    def test_shape_equal(self):
        self.assertEqual(
            Preparation.test_X_transform.shape[0], Preparation.test_y_transform.shape[0]
        )
        self.assertEqual(
            Preparation.train_X_transform.shape[0],
            Preparation.train_y_transform.shape[0],
        )

    def test_nan(self):
        self.assertIsNotNone(Preparation.test_X_transform.isna().sum() == 0)
        self.assertIsNotNone(Preparation.test_y_transform.isna().sum() == 0)
        self.assertIsNotNone(Preparation.train_X_transform.isna().sum() == 0)
        self.assertIsNotNone(Preparation.train_y_transform.isna().sum() == 0)

    def test_type(self):
        self.assertTrue(
            Preparation.test_X_transform.select_dtypes(include="object").shape[1] == 0
        )

    def check_score_nan(self):
        self.assertIsNotNone(train_model.best_model)


if __name__ == "__main__":
    unittest.main()
