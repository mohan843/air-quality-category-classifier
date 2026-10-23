import unittest
from pathlib import Path

from classifier import DecisionTreeClassifier, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "air_quality.csv"


class AirQualityTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 24)
        self.assertEqual(set(labels), {"Good", "Moderate", "Unhealthy", "Very_Unhealthy"})

    def test_gini_purity(self):
        self.assertEqual(DecisionTreeClassifier.gini(["Good", "Good"]), 0.0)
        self.assertAlmostEqual(DecisionTreeClassifier.gini(["Good", "Bad"]), 0.5)

    def test_tree_learns_training_data(self):
        model, score = train(DATA)
        self.assertGreaterEqual(score, 0.95)
        self.assertEqual(model.predict_one([9, 20, 14, 44]), "Good")
        self.assertEqual(model.predict_one([72, 140, 105, 81]), "Very_Unhealthy")

    def test_prediction_count_matches_rows(self):
        rows, _ = load_csv(DATA)
        model, _ = train(DATA)
        self.assertEqual(len(model.predict(rows)), len(rows))

    def test_invalid_training_input_rejected(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy(["a", "b"], ["a", "a"]), 0.5)


if __name__ == "__main__":
    unittest.main()
