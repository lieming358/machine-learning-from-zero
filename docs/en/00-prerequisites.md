# Chapter 0 — Preparation for True Beginners

[← Home](../../README.en.md) | [中文](../zh-CN/00-prerequisites.md) | [Notation](appendix-a-notation.md) | [Chapter 1 →](01-what-is-ml.md)

CS229 normally assumes Python/NumPy, probability, multivariable calculus, and linear algebra. This
course turns those assumptions into a bridge rather than an entrance barrier.

Create an isolated environment from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements-course.txt
```

Activate it with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on macOS/Linux. A virtual environment prevents this course's package
versions from conflicting with unrelated projects.

Learn only the Python essentials first: values, lists, functions, loops, NumPy arrays, shapes, matrix
multiplication, and reproducible randomness.

```python
import numpy as np

X = np.array([[28., 5., 1.], [45., 10., 0.], [72., 18., 1.]])
w = np.array([38., -6., 120.])
b = 800.
print(X.shape)       # 3 examples by 3 features
print(X @ w + b)     # three predictions at once
```

Build five habits: fix and record random seeds, print shapes/ranges, isolate train/validation/test data,
start with a baseline, and save experiment configuration. Run:

```bash
python code/chapter00/readiness_check.py
python code/chapter00/first_workflow.py
```

The second command is your first complete workflow: load a CSV, inspect its shape and range, keep two rows held out,
fit a simple baseline, compute MAE, draw a figure, and run assertions. You do not need to understand `polyfit` yet;
follow the order **load → inspect → split → fit → evaluate → visualize → assert**.

![A first complete workflow joins data, a baseline, evaluation, and a figure](../assets/chapter00-first-workflow.png)

Change the final area in the CSV, rerun, and observe the MAE. Then temporarily change `deg=1` to `degree=1` and
read the last traceback line before restoring it. For `ModuleNotFoundError`, check the environment and installation;
for `FileNotFoundError`, check the working directory; for shape errors, print every participating array shape.

If you can explain every shape, locate an error line, and modify the data without breaking the calculation, continue.
