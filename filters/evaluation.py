import numpy as np


class EvaluationFilter:
    def __init__(self):
        self._num_intervals = len(self.odd_intervals)

    @property
    def end_of_interval(self) -> float:
        return 4.00

    @property
    def odd_intervals(self) -> list:
        return [
            (1.00, 1.30), (1.30, 1.60), (1.60, 1.90), (1.90, 2.20),
            (2.20, 2.50), (2.50, 2.80), (2.80, 3.10),
            (3.10, 3.40), (3.40, 3.70), (3.70, self.end_of_interval)
        ]

    @property
    def num_intervals(self) -> int:
        return self._num_intervals

    def compute_prediction_accuracy_per_odd_range(
            self,
            odds: np.ndarray,
            targets: np.ndarray,
            predictions: np.ndarray
    ) -> list:
        accuracies = [0] * (self.num_intervals + 1)
        correct_predictions = [0] * (self.num_intervals + 1)
        wrong_predictions = [0] * (self.num_intervals + 1)


        for i, odd in enumerate(odds):
            if odd >= self.end_of_interval:
                if targets[i] == predictions[i]:
                    correct_predictions[self.num_intervals] += 1
                else:
                    wrong_predictions[self.num_intervals] += 1
            else:
                for j, (left_range, right_range) in enumerate(self.odd_intervals):
                    if left_range <= odd <= right_range:
                        if targets[i] == predictions[i]:
                            correct_predictions[j] += 1
                        else:
                            wrong_predictions[j] += 1

        for i in range(self.num_intervals + 1):
            total_predictions = correct_predictions[i] + wrong_predictions[i]
            accuracies[i] = 0.0 if total_predictions == 0 else correct_predictions[i] / total_predictions
        return accuracies
