"""Chapter 0: a tiny, dependency-light NumPy readiness exercise."""

from __future__ import annotations

import numpy as np


def main() -> None:
    x = np.array(
        [
            [28.0, 5.0, 1.0],
            [45.0, 10.0, 0.0],
            [72.0, 18.0, 1.0],
        ]
    )
    w = np.array([38.0, -6.0, 120.0])
    b = 800.0
    prediction = x @ w + b

    print(f"X.shape={x.shape}  # examples x features")
    print(f"w.shape={w.shape}  # one weight per feature")
    print(f"prediction.shape={prediction.shape}  # one value per example")
    print(f"predictions={prediction}")

    assert x.shape[1] == w.shape[0]
    assert prediction.shape == (x.shape[0],)


if __name__ == "__main__":
    main()
