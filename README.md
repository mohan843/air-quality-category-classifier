# Air Quality Category Classifier

An interpretable machine-learning mini-project that predicts air-quality categories from PM2.5, PM10, nitrogen dioxide, and humidity readings using a decision tree built from scratch.

## Overview

Low-cost sensors can provide frequent air-quality measurements, but raw values are difficult to interpret quickly. This project demonstrates how a small supervised model can translate sensor readings into four understandable categories: Good, Moderate, Unhealthy, and Very Unhealthy.

## Problem statement

Given four environmental measurements, classify the air-quality condition into an actionable category. The sample data is educational and is not a replacement for certified monitoring equipment or official health guidance.

## Features

- Interpretable binary-split decision tree.
- Gini impurity for split selection.
- Configurable maximum depth and minimum sample size in code.
- CSV sample data, CLI, and unit tests.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- Supervised classification and decision trees
- Gini impurity and recursive tree construction
- CSV, `argparse`, and `dataclasses`
- `unittest` for testing

## Installation

```bash
git clone https://github.com/mohan843/air-quality-category-classifier.git
cd air-quality-category-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime dependencies are required.

## Execution

```bash
python classifier.py --pm25 9 --pm10 20 --no2 14 --humidity 44
```

Test an elevated reading:

```bash
python classifier.py --pm25 72 --pm10 140 --no2 105 --humidity 81
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 100.0%
Predicted air-quality category: Good
```

## Responsible use and limitations

This educational model uses a small synthetic dataset and should not be used for public-health decisions. Real deployments should use calibrated sensors, local regulatory thresholds, weather context, uncertainty estimates, and independent validation.

## Future improvements

- Add official pollutant breakpoints by region.
- Compare the tree with random forest and gradient boosting baselines.
- Add train/test splits and confusion-matrix reporting.
- Build a dashboard for live sensor streams.
- Add model explanations showing the decision path for each reading.
