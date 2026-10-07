import unittest

from sklearn.model_selection import train_test_split

from penguins_ml.experiment import FEATURES, SEED, load_data, run_experiment


class ExperimentTests(unittest.TestCase):
    def test_dataset_schema_and_split(self):
        frame = load_data()
        self.assertEqual(len(frame), 344)
        self.assertEqual(set(frame["species"]), {"Adelie", "Chinstrap", "Gentoo"})
        self.assertTrue(frame[FEATURES].isna().any().any())
        left, right = train_test_split(frame.index, test_size=0.2, stratify=frame["species"], random_state=SEED)
        self.assertFalse(set(left) & set(right))
        self.assertEqual(len(left) + len(right), len(frame))

    def test_model_selection_uses_cross_validation(self):
        result = run_experiment()
        scores = [model["cv_f1_macro_mean"] for model in result["comparisons"]]
        self.assertEqual(result["selected_model"], result["comparisons"][0]["model"])
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), result["test_rows"])
        for model in result["comparisons"]:
            self.assertGreaterEqual(model["test_f1_macro"], 0)
            self.assertLessEqual(model["test_f1_macro"], 1)


if __name__ == "__main__":
    unittest.main()
