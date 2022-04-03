import numpy as np


class EvaluationFilter:
    def __init__(self):
        self.end_of_range = 4.00
        self.odd_ranges = [
            (1.00, 1.30), (1.30, 1.60), (1.60, 1.90), (1.90, 2.20),
            (2.20, 2.50), (2.50, 2.80), (2.80, 3.10),
            (3.10, 3.40), (3.40, 3.70), (3.70, self.end_of_range)
        ]
        self.n_ranges = len(self.odd_ranges)

    def filter_odd_accuracy_per_range(self, inputs, targets, predictions, column):
        correct_predictions = [0] * (self.n_ranges + 1)
        wrong_predictions = [0] * (self.n_ranges + 1)
        accuracies = [0] * (self.n_ranges + 1)
        print(targets)
        print(predictions)
        for i, inp in enumerate(inputs):
            odd = inp[column]

            for j, (left_range, right_range) in enumerate(self.odd_ranges):
                if left_range <= odd <= right_range:
                    if targets[i] == predictions[i]:
                        correct_predictions[j] += 1
                    else:
                        wrong_predictions[j] += 1

                if odd > self.end_of_range:
                    if targets[i] == predictions[i]:
                        correct_predictions[self.n_ranges] += 1
                    else:
                        wrong_predictions[self.n_ranges] += 1

        for i in range(self.n_ranges + 1):
            total_predictions = correct_predictions[i] + wrong_predictions[i]
            if total_predictions == 0:
                accuracies[i] = 0
            else:
                accuracies[i] = correct_predictions[i] / total_predictions
        return accuracies
