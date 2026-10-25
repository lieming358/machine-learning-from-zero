"""Chapter 12: a two-layer NumPy MLP, gradient check, and framework comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


class TwoLayerMLP:
    def __init__(self, input_dim=2, hidden=16, seed=5):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(1/input_dim), (input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, np.sqrt(1/hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        logits = (a1 @ self.w2 + self.b2)[:, 0]
        return z1, a1, logits

    def loss_and_gradients(self, x, y):
        z1, a1, logits = self.forward(x)
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        dlogits = (sigmoid(logits) - y)[:, None] / len(x)
        dw2 = a1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        da1 = dlogits @ self.w2.T
        dz1 = da1 * (1 - np.tanh(z1) ** 2)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return loss, (dw1, db1, dw2, db2)

    def fit(self, x, y, x_val, y_val, epochs=1800, learning_rate=.06):
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        for epoch in range(epochs):
            loss, gradients = self.loss_and_gradients(x, y)
            for parameter, gradient in zip((self.w1,self.b1,self.w2,self.b2), gradients):
                parameter -= learning_rate * gradient
            if epoch % 10 == 0:
                _, _, val_logits = self.forward(x_val)
                val_loss = np.mean(np.logaddexp(0.0, val_logits) - y_val * val_logits)
                history["train_loss"].append(loss); history["val_loss"].append(val_loss)
                history["val_accuracy"].append(np.mean((val_logits >= 0) == y_val))
        return history

    def predict(self, x):
        return (self.forward(x)[2] >= 0).astype(int)


def gradient_check(model, x, y, epsilon=1e-5):
    loss, gradients = model.loss_and_gradients(x, y)
    analytic = gradients[0][0,0]
    original = model.w1[0,0]
    model.w1[0,0] = original + epsilon; plus = model.loss_and_gradients(x,y)[0]
    model.w1[0,0] = original - epsilon; minus = model.loss_and_gradients(x,y)[0]
    model.w1[0,0] = original
    numerical = (plus-minus)/(2*epsilon)
    return abs(analytic-numerical) / max(1e-12, abs(analytic)+abs(numerical))


def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUTPUT / name, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def draw_two_hidden_layer_mlp():
    """Draw a beginner-friendly 2-4-3-1 MLP architecture."""
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    layer_x = [0.7, 3.8, 6.9, 10.0]
    layer_y = [
        [3.5, 1.5],
        [4.1, 3.0, 2.0, 0.9],
        [3.7, 2.5, 1.3],
        [2.5],
    ]
    layer_colors = ["#d8e8f6", "#cfe9df", "#f9dfbf", "#f4c7c3"]
    edge_colors = [BLUE, GREEN, ORANGE, RED]
    node_labels = [
        ["x1", "x2"],
        ["h1_1", "h1_2", "h1_3", "h1_4"],
        ["h2_1", "h2_2", "h2_3"],
        ["p"],
    ]

    # Draw weighted connections first so the neurons stay visually on top.
    for left in range(3):
        for y_start in layer_y[left]:
            for y_end in layer_y[left + 1]:
                ax.plot(
                    [layer_x[left] + 0.23, layer_x[left + 1] - 0.23],
                    [y_start, y_end],
                    color="#9aa9b5",
                    lw=0.9,
                    alpha=0.62,
                    zorder=1,
                )

    for x_pos, y_positions, face, edge, labels in zip(
        layer_x, layer_y, layer_colors, edge_colors, node_labels
    ):
        for y_pos, label in zip(y_positions, labels):
            ax.add_patch(
                Circle(
                    (x_pos, y_pos),
                    0.26,
                    facecolor=face,
                    edgecolor=edge,
                    lw=2.0,
                    zorder=3,
                )
            )
            ax.text(x_pos, y_pos, label, ha="center", va="center", fontsize=10, color=NAVY, zorder=4)

    headings = [
        "Input\n2 features",
        "Hidden layer 1\n4 neurons",
        "Hidden layer 2\n3 neurons",
        "Output\n1 probability",
    ]
    for x_pos, heading in zip(layer_x, headings):
        ax.text(x_pos, 4.72, heading, ha="center", va="center", fontsize=11, weight="bold", color=NAVY)

    operations = [
        ("W1, b1 + activation", 2.25),
        ("W2, b2 + activation", 5.35),
        ("W3, b3 + sigmoid", 8.45),
    ]
    for text_label, x_pos in operations:
        ax.annotate(
            text_label,
            xy=(x_pos + 0.72, 4.35),
            xytext=(x_pos - 0.72, 4.35),
            ha="center",
            va="center",
            fontsize=8.5,
            color="#526473",
            arrowprops=dict(arrowstyle="->", color="#526473", lw=1.25),
        )

    ax.text(
        5.35,
        0.25,
        "Every connection has a learnable weight; biases are omitted from the drawing for clarity.",
        ha="center",
        va="center",
        fontsize=9,
        color="#526473",
    )
    ax.set(xlim=(0, 10.7), ylim=(-0.05, 5.15))
    ax.set_title("A multilayer perceptron with two hidden layers", fontsize=15, weight="bold", color=NAVY, pad=12)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=OUTPUT / "mlp-two-hidden-layers.alignment.json",
            overlay_svg=OUTPUT / "mlp-two-hidden-layers.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    fig.savefig(OUTPUT / "mlp-two-hidden-layers.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "mlp-two-hidden-layers.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "mlp-two-hidden-layers.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    draw_two_hidden_layer_mlp()
    x = np.linspace(-5,5,600)
    sigmoid_y = sigmoid(x); tanh_y = np.tanh(x); relu_y = np.maximum(0,x)
    fig, axes = plt.subplots(1, 2, figsize=(10.5,4))
    axes[0].plot(x, sigmoid_y, color=BLUE, label="Sigmoid"); axes[0].plot(x,tanh_y,color=ORANGE,label="Tanh"); axes[0].plot(x,relu_y,color=GREEN,label="ReLU")
    axes[0].set(title="Activation functions", xlabel="pre-activation z", ylabel="activation"); axes[0].set_ylim(-1.2,3.2)
    axes[1].plot(x, sigmoid_y*(1-sigmoid_y), color=BLUE, label="Sigmoid derivative"); axes[1].plot(x,1-tanh_y**2,color=ORANGE,label="Tanh derivative"); axes[1].plot(x,(x>0).astype(float),color=GREEN,label="ReLU derivative")
    axes[1].set(title="Derivatives control gradient flow", xlabel="pre-activation z", ylabel="derivative")
    for axis in axes: axis.grid(alpha=.15); axis.legend(frameon=False)
    finish(fig, "activation-functions-and-derivatives.png")

    raw_x, y = make_moons(n_samples=600, noise=.2, random_state=9)
    x_fit, x_test, y_fit, y_test = train_test_split(raw_x,y,test_size=.25,stratify=y,random_state=12)
    x_train, x_val, y_train, y_val = train_test_split(x_fit,y_fit,test_size=.22,stratify=y_fit,random_state=4)
    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)
    scratch = TwoLayerMLP(hidden=16, seed=5)
    relative_error = gradient_check(scratch, x_train[:8], y_train[:8])
    history = scratch.fit(x_train,y_train,x_val,y_val)
    test_accuracy = np.mean(scratch.predict(x_test)==y_test)

    xx, yy = np.meshgrid(np.linspace(x_train[:,0].min()-.7,x_train[:,0].max()+.7,300),
                         np.linspace(x_train[:,1].min()-.7,x_train[:,1].max()+.7,300))
    grid = np.c_[xx.ravel(),yy.ravel()]
    probability = sigmoid(scratch.forward(grid)[2]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7,5.2))
    ax.contourf(xx,yy,probability,levels=np.linspace(0,1,15),cmap="RdBu_r",alpha=.55)
    ax.contour(xx,yy,probability,levels=[.5],colors=NAVY,linewidths=2)
    ax.scatter(x_train[:,0],x_train[:,1],c=np.where(y_train==1,ORANGE,BLUE),s=23,edgecolor="white",alpha=.78)
    ax.set(title="A hidden layer bends a linear score into a nonlinear boundary", xlabel="standardized feature 1", ylabel="standardized feature 2")
    finish(fig, "mlp-moons-boundary.png")

    steps = np.arange(len(history["train_loss"]))*10
    fig, axes = plt.subplots(1,2,figsize=(10.5,4))
    axes[0].plot(steps,history["train_loss"],color=BLUE,label="Training BCE"); axes[0].plot(steps,history["val_loss"],color=ORANGE,label="Validation BCE")
    axes[0].set(title="Optimization and generalization are different curves", xlabel="epoch", ylabel="loss")
    axes[1].plot(steps,history["val_accuracy"],color=GREEN); axes[1].set(title="Held-out accuracy", xlabel="epoch", ylabel="accuracy",ylim=(.5,1.0))
    for axis in axes: axis.grid(alpha=.15); axis.legend(frameon=False) if axis is axes[0] else None
    finish(fig, "mlp-training-diagnostics.png")

    rng = np.random.default_rng(22); depth=14; width=128; batch=rng.normal(size=(256,width))
    fig, ax = plt.subplots(figsize=(8,4.3))
    for scale,color,label in [(0.2,BLUE,"too small"),(1.0,GREEN,"Xavier-like"),(2.2,RED,"too large")]:
        activations=batch.copy(); std=[]
        for _ in range(depth):
            weights=rng.normal(0,scale/np.sqrt(width),(width,width)); activations=np.tanh(activations@weights); std.append(activations.std())
        ax.plot(np.arange(1,depth+1),std,"o-",color=color,label=label)
    ax.set(title="Initialization controls whether signals vanish or saturate",xlabel="layer",ylabel="activation standard deviation")
    ax.grid(alpha=.15); ax.legend(frameon=False); finish(fig,"initialization-signal-flow.png")

    framework = MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", solver="lbfgs",
                              max_iter=2500, random_state=5, alpha=1e-4).fit(x_train,y_train)
    framework_accuracy = framework.score(x_test,y_test)
    print(f"gradient_check_relative_error={relative_error:.2e}")
    print(f"scratch_test_accuracy={test_accuracy:.3f}")
    print(f"sklearn_test_accuracy={framework_accuracy:.3f}")
    assert relative_error < 1e-6 and test_accuracy > .90 and framework_accuracy > .90


if __name__ == "__main__":
    main()
