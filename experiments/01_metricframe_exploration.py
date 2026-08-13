# import sys

import pandas as pd

data = pd.DataFrame({
    "age": [22, 25, 35, 40, 28, 50, 31, 45],
    "income": [25, 30, 60, 70, 35, 90, 50, 80],
    "gender": ["F", "F", "M", "M", "F", "M", "F", "M"],
    "y_true": [1, 1, 1, 1, 0, 1, 0, 0],
    "y_pred": [1, 0, 1, 1, 0, 1, 1, 0]
})

print(data)

X = data[["age", "income"]]
y = data["y_true"]
sensitive_features = data[["gender", "age"]] #Fairlearn effectively performs intersectional grouping
y_pred = data["y_pred"]

# print("\n--- X ---")
# print(X)
# print(type(X))
# print(X.shape)

# print("\n--- y ---")
# print(y)
# print(type(y))
# print(y.shape)

# print("\n--- sensitive_features ---")
# print(sensitive_features)
# print(type(sensitive_features))
# print(sensitive_features.shape)

# print("\n--- y_pred ---")
# print(y_pred)
# print(type(y_pred))
# print(y_pred.shape)


from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

#That tells us MetricFrame is a Python object, not itself the result like 0.75.
#Think of it like a report, contains the overall metric and the metric by group.
metric_frame = MetricFrame(
    metrics=accuracy_score,
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

# print("\n--- MetricFrame ---")
# print(metric_frame)

# print("\n--- Type ---")
# print(type(metric_frame))

# print("\n--- Overall ---")
# print(metric_frame.overall)
# print(type(metric_frame.overall))

# print("\n--- By Group ---")
# print(metric_frame.by_group)
# print(type(metric_frame.by_group))


# # MetricFrame assessment workflow
# print("\n--- Difference ---") #largest group metric - smallest group metric
# print(metric_frame.difference())

# print("\n--- Ratio ---")
# print(metric_frame.ratio()) #smallest group metric / largest group metric

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)

print("\n--- Demographic Parity ---")

print(
    "Difference:",
    demographic_parity_difference(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
)

print(
    "Ratio:",
    demographic_parity_ratio(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
)

print("\n--- Equalized Odds ---")

print(
    "Difference:",
    equalized_odds_difference(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
)

print(
    "Ratio:",
    equalized_odds_ratio(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
)


#we've established:

# Fairlearn
#    │
#    ├── accepts Pandas
#    │
#    └── accepts NumPy

import numpy as np

y_numpy = np.array(y)
y_pred_numpy = np.array(y_pred)
sensitive_numpy = np.array(sensitive_features)

test_frame = MetricFrame(
    metrics=accuracy_score,
    y_true=y_numpy,
    y_pred=y_pred_numpy,
    sensitive_features=sensitive_numpy
)

print("\n--- NumPy MetricFrame ---")
print(test_frame.by_group)

