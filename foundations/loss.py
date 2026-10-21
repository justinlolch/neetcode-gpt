import numpy as np
from numpy.typing import NDArray
import math


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        total = 0

        for i in range(len(y_true)):
            total += (
                y_true[i] * math.log(y_pred[i]) +
                (1 - y_true[i]) * math.log(1 - y_pred[i])
            )

        return round(-total / len(y_true), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        total = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                y_pred[i] = np.clip(y_pred[i], epsilon, 1 - epsilon)
                if (y_true[i][j] == 1):
                    total += 1 * math.log(y_pred[i][j])
        return round(-total / len(y_true), 4)
