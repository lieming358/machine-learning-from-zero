"""Chapter 0: a first complete data-to-figure workflow for beginners."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("rent_samples.csv")
OUTPUT = ROOT / "docs" / "assets" / "chapter00-first-workflow.png"


def main() -> None:
    # 1. Load a small table. The column names come from the CSV header.
    table = np.genfromtxt(DATA, delimiter=",", names=True)
    area = table["area"]
    rent = table["rent"]
    print(f"loaded_shape=({len(table)}, 2)")
    print(f"area_range={area.min():.0f}..{area.max():.0f}")

    # 2. Keep the last two rows unseen while fitting a simple baseline.
    train_area, test_area = area[:4], area[4:]
    train_rent, test_rent = rent[:4], rent[4:]
    weight, bias = np.polyfit(train_area, train_rent, deg=1)
    prediction = weight * test_area + bias
    mae = np.mean(np.abs(prediction - test_rent))
    print(f"baseline=rent={weight:.1f}*area+{bias:.1f}")
    print(f"test_mae={mae:.1f}")

    # 3. Visualize observations, the learned line, and the held-out rows.
    grid = np.linspace(area.min() - 3, area.max() + 3, 100)
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.scatter(train_area, train_rent, color="#3478c0", s=70, label="training rows")
    axis.scatter(test_area, test_rent, color="#f28e2b", s=85, marker="s", label="held-out rows")
    axis.plot(grid, weight * grid + bias, color="#19324d", linewidth=2.2, label="simple baseline")
    axis.set(
        xlabel="area",
        ylabel="rent",
        title="First workflow: load, inspect, fit, evaluate, plot",
    )
    axis.grid(alpha=0.15)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    assert table.dtype.names == ("area", "rent")
    assert prediction.shape == test_rent.shape
    assert np.isfinite(mae)
    print(f"figure={OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
