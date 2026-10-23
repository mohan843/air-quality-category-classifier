"""Air-quality category prediction with a small interpretable decision tree."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FEATURES = ("pm25", "pm10", "no2", "humidity")


@dataclass(frozen=True)
class AirObservation:
    pm25: float
    pm10: float
    no2: float
    humidity: float

    def vector(self) -> list[float]:
        return [self.pm25, self.pm10, self.no2, self.humidity]


@dataclass
class Node:
    prediction: str
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 3, min_samples: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples

    @staticmethod
    def gini(labels: Sequence[str]) -> float:
        if not labels:
            return 0.0
        total = len(labels)
        return 1.0 - sum((labels.count(label) / total) ** 2 for label in set(labels))

    @staticmethod
    def majority(labels: Sequence[str]) -> str:
        return max(set(labels), key=labels.count)

    def _best_split(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> tuple[int, float] | None:
        best: tuple[float, int, float] | None = None
        parent = self.gini(labels)
        if parent == 0.0:
            return None
        for feature in range(len(rows[0])):
            values = sorted(set(row[feature] for row in rows))
            for threshold in values[:-1]:
                left = [i for i, row in enumerate(rows) if row[feature] <= threshold]
                right = [i for i, row in enumerate(rows) if row[feature] > threshold]
                if not left or not right:
                    continue
                score = (len(left) * self.gini([labels[i] for i in left]) + len(right) * self.gini([labels[i] for i in right])) / len(labels)
                if best is None or score < best[0]:
                    best = (score, feature, threshold)
        return None if best is None or best[0] >= parent else (best[1], best[2])

    def _grow(self, rows: Sequence[Sequence[float]], labels: Sequence[str], depth: int) -> Node:
        node = Node(self.majority(labels))
        if depth >= self.max_depth or len(rows) < self.min_samples or len(set(labels)) == 1:
            return node
        split = self._best_split(rows, labels)
        if split is None:
            return node
        feature, threshold = split
        left_indices = [i for i, row in enumerate(rows) if row[feature] <= threshold]
        right_indices = [i for i, row in enumerate(rows) if row[feature] > threshold]
        node.feature_index, node.threshold = feature, threshold
        node.left = self._grow([rows[i] for i in left_indices], [labels[i] for i in left_indices], depth + 1)
        node.right = self._grow([rows[i] for i in right_indices], [labels[i] for i in right_indices], depth + 1)
        return node

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> "DecisionTreeClassifier":
        if not rows or len(rows) != len(labels):
            raise ValueError("Rows and labels must be non-empty and equally sized")
        self.root = self._grow(rows, labels, 0)
        return self

    def predict_one(self, row: Sequence[float]) -> str:
        node = self.root
        while node.feature_index is not None:
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return node.prediction

    def predict(self, rows: Sequence[Sequence[float]]) -> list[str]:
        return [self.predict_one(row) for row in rows]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, labels = [], []
        for item in reader:
            rows.append([float(item[name]) for name in FEATURES])
            labels.append(item["category"])
    return rows, labels


def accuracy(actual: Sequence[str], predicted: Sequence[str]) -> float:
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def train(path: str | Path) -> tuple[DecisionTreeClassifier, float]:
    rows, labels = load_csv(path)
    model = DecisionTreeClassifier().fit(rows, labels)
    return model, accuracy(labels, model.predict(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict air-quality category from sensor readings.")
    parser.add_argument("--data", default="data/air_quality.csv")
    parser.add_argument("--pm25", type=float, required=True)
    parser.add_argument("--pm10", type=float, required=True)
    parser.add_argument("--no2", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    args = parser.parse_args()
    model, score = train(args.data)
    prediction = model.predict_one([args.pm25, args.pm10, args.no2, args.humidity])
    print(f"Training accuracy: {score:.1%}")
    print(f"Predicted air-quality category: {prediction}")
