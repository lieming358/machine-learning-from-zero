"""Expanded NumPy demonstrations for Chapter 14: CNN, RNN, LSTM, attention."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY, PURPLE = (
    "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d", "#7a5195"
)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=axis, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def conv1d_valid(x, kernel, bias=0.0):
    """Deep-learning convention: valid cross-correlation, without kernel reversal."""
    output_length = len(x) - len(kernel) + 1
    if output_length <= 0:
        raise ValueError("kernel must not be longer than the input")
    return np.array([x[t:t + len(kernel)] @ kernel + bias for t in range(output_length)])


def conv2d_valid(image, kernel):
    """Single-channel 2-D valid cross-correlation."""
    out_h = image.shape[0] - kernel.shape[0] + 1
    out_w = image.shape[1] - kernel.shape[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel must fit inside the image")
    output = np.empty((out_h, out_w), dtype=float)
    for row in range(out_h):
        for col in range(out_w):
            patch = image[row:row + kernel.shape[0], col:col + kernel.shape[1]]
            output[row, col] = np.sum(patch * kernel)
    return output


def rnn_forward(inputs, wx, wh, bias):
    """Vanilla tanh RNN; returns h_0 through h_T."""
    states = [np.zeros(wh.shape[0])]
    for x_t in inputs:
        states.append(np.tanh(wx @ x_t + wh @ states[-1] + bias))
    return np.asarray(states)


def rnn_final_sum_loss_and_grad_wh(inputs, wx, wh, bias):
    """BPTT for L=sum(h_T), sufficient for a transparent gradient check."""
    states = rnn_forward(inputs, wx, wh, bias)
    grad_wh = np.zeros_like(wh)
    grad_h = np.ones_like(states[-1])
    for time in range(len(inputs), 0, -1):
        grad_pre_activation = grad_h * (1.0 - states[time] ** 2)
        grad_wh += np.outer(grad_pre_activation, states[time - 1])
        grad_h = wh.T @ grad_pre_activation
    return float(states[-1].sum()), grad_wh


def lstm_forward(inputs, weights, biases):
    """A complete single-layer LSTM forward pass."""
    hidden_size = biases["f"].shape[0]
    h_t = np.zeros(hidden_size)
    c_t = np.zeros(hidden_size)
    hidden_states, cell_states = [h_t.copy()], [c_t.copy()]
    gate_history = {name: [] for name in ("f", "i", "o", "g")}
    for x_t in inputs:
        joined = np.concatenate([h_t, x_t])
        f_t = sigmoid(weights["f"] @ joined + biases["f"])
        i_t = sigmoid(weights["i"] @ joined + biases["i"])
        o_t = sigmoid(weights["o"] @ joined + biases["o"])
        g_t = np.tanh(weights["g"] @ joined + biases["g"])
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * np.tanh(c_t)
        hidden_states.append(h_t.copy())
        cell_states.append(c_t.copy())
        for name, value in (("f", f_t), ("i", i_t), ("o", o_t), ("g", g_t)):
            gate_history[name].append(value.copy())
    return np.asarray(hidden_states), np.asarray(cell_states), {
        name: np.asarray(values) for name, values in gate_history.items()
    }


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_cnn_hand_calculation():
    """Explain all three valid windows for x=[0,0,1,1,1], w=[-1,0,1]."""
    signal = np.array([0, 0, 1, 1, 1], dtype=int)
    kernel = np.array([-1, 0, 1], dtype=int)
    output = conv1d_valid(signal, kernel)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.7), sharex=True, sharey=True)

    for step, ax in enumerate(axes):
        window = signal[step:step + 3]
        products = window * kernel
        ax.set_xlim(-0.25, 5.25)
        ax.set_ylim(-0.2, 4.9)
        ax.axis("off")
        ax.text(-0.04, 1.01, chr(ord("a") + step), transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=NAVY, va="bottom")
        ax.set_title(f"Window {step}: positions {step}–{step + 2}", fontsize=11, weight="bold", color=NAVY)

        # Full signal: the orange outline shows which three entries the kernel currently covers.
        for position, value in enumerate(signal):
            face = "#dceaf6" if step <= position < step + 3 else "#f1f3f5"
            edge = BLUE if step <= position < step + 3 else "#c5cbd1"
            ax.add_patch(Rectangle((position, 3.75), .8, .62, facecolor=face, edgecolor=edge, lw=1.5))
            ax.text(position + .4, 4.06, str(value), ha="center", va="center", fontsize=11, color=NAVY)
            ax.text(position + .4, 3.58, str(position), ha="center", va="center", fontsize=8, color="#667784")
        ax.add_patch(Rectangle((step - .08, 3.67), 2.96, .79, fill=False, edgecolor=ORANGE, lw=2.2))
        ax.text(-.12, 4.55, "full input x", ha="left", va="center", fontsize=9, color="#526473")

        row_specs = [
            ("window", window, "#dceaf6", BLUE),
            ("kernel", kernel, "#f9dfbf", ORANGE),
            ("products", products, "#dcefe4", GREEN),
        ]
        for row, (label, values, face, edge) in enumerate(row_specs):
            y_pos = 2.75 - row * .9
            ax.text(.15, y_pos + .3, label, ha="right", va="center", fontsize=9, color="#526473")
            for col, value in enumerate(values):
                x_pos = .45 + col * 1.12
                ax.add_patch(Rectangle((x_pos, y_pos), .92, .62, facecolor=face, edgecolor=edge, lw=1.4))
                ax.text(x_pos + .46, y_pos + .31, str(int(value)), ha="center", va="center",
                        fontsize=11, color=NAVY)

        # Each vertical column is one elementwise multiplication:
        # window value × aligned kernel weight = product.
        for col in range(3):
            center_x = .45 + col * 1.12 + .46
            ax.text(center_x, 2.61, "×", ha="center", va="center", fontsize=10, color="#667784")
            ax.text(center_x, 1.71, "=", ha="center", va="center", fontsize=10, color="#667784")

        expression = " + ".join(str(int(value)) for value in products)
        ax.text(2.25, .35, f"sum: {expression} = {int(output[step])}",
                ha="center", va="center", fontsize=10.5, weight="bold", color=RED)
        ax.annotate("", xy=(2.25, .62), xytext=(2.25, 1.0),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))

    fig.suptitle(
        "The same kernel [-1, 0, 1] slides one position at a time",
        fontsize=15,
        weight="bold",
        color=NAVY,
        y=.995,
    )
    fig.text(
        .5,
        .015,
        "At each position: select three values → multiply element by element → add the three products.",
        ha="center",
        fontsize=9.5,
        color="#526473",
    )
    fig.tight_layout(rect=(0, .06, 1, .94))
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=OUTPUT / "cnn-hand-calculation.alignment.json",
            overlay_svg=OUTPUT / "cnn-hand-calculation.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(OUTPUT / "cnn-hand-calculation.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "cnn-hand-calculation.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "cnn-hand-calculation.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_lstm_flow():
    """Draw the additive cell-state path and its three gates."""
    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, width, height, label, color):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04",
                               facecolor=color, edgecolor=NAVY, linewidth=1.4, alpha=.9)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)

    def arrow(start, end, label="", color=NAVY):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     color=color, linewidth=1.6))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .18,
                    label, ha="center", color=color, fontsize=10)

    box(.3, 3.25, 1.25, .65, "$c_{t-1}$", "#e8f2fb")
    box(9.45, 3.25, 1.25, .65, "$c_t$", "#e8f2fb")
    arrow((1.55, 3.58), (3.2, 3.58), "memory highway")
    box(3.2, 3.23, .72, .72, "$\\times$", "#fdebd0")
    arrow((3.92, 3.58), (6.45, 3.58))
    box(6.45, 3.23, .72, .72, "+", "#e8f5e9")
    arrow((7.17, 3.58), (9.45, 3.58))

    box(2.55, 1.45, 1.55, .7, "forget gate\n$f_t$", "#fdebd0")
    box(4.55, 1.45, 1.55, .7, "input gate\n$i_t$", "#e8f5e9")
    box(6.55, 1.45, 1.55, .7, "candidate\n$\\tilde c_t$", "#e8f5e9")
    box(8.55, 1.45, 1.55, .7, "output gate\n$o_t$", "#eee8f7")
    box(.3, .25, 1.65, .65, "$[h_{t-1},x_t]$", "#f1f3f5")
    arrow((1.95, .58), (3.0, 1.45)); arrow((1.95, .58), (5.0, 1.45))
    arrow((1.95, .58), (7.0, 1.45)); arrow((1.95, .58), (9.0, 1.45))
    arrow((3.33, 2.15), (3.55, 3.23), "$f_t$")
    arrow((5.33, 2.15), (5.45, 2.72), "$i_t$")
    arrow((7.33, 2.15), (5.72, 2.72), "$\\tilde c_t$")
    box(5.1, 2.58, .72, .62, "$\\times$", "#e8f5e9")
    arrow((5.82, 2.9), (6.55, 3.23))
    arrow((9.33, 2.15), (9.85, 2.75), "$o_t$")
    arrow((8.25, 3.58), (9.85, 2.75), "$\\tanh(c_t)$")
    box(9.48, 2.55, .72, .62, "$\\times$", "#eee8f7")
    arrow((10.2, 2.86), (10.7, 2.86), "$h_t$")
    ax.set_title("LSTM gates control an additive cell-state path", fontsize=16, pad=12)
    finish(fig, "lstm-gate-flow.png")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    draw_cnn_hand_calculation()

    signal = np.array([0, 0, 1, 1, 1, 0, 0, -1, -1, 0, 0], dtype=float)
    kernels = {"smooth": np.ones(3) / 3, "edge": np.array([-1, 0, 1.0])}
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True)
    axes[0].stem(np.arange(len(signal)), signal, basefmt=" ", linefmt=BLUE, markerfmt="o")
    axes[0].set(title="Input signal: 0 → 1 → 0 → -1 → 0", ylabel="input value")
    axes[0].text(-.055, 1.02, "a", transform=axes[0].transAxes, fontsize=11,
                 fontweight="bold", color=NAVY, va="bottom")
    axes[0].axvspan(.5, 3.5, color=ORANGE, alpha=.10)
    axes[0].annotate("example window centered at position 2\npositions 1–3: [0, 1, 1]",
                     xy=(2, 1), xytext=(3.9, .62),
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3),
                     fontsize=9, color="#7b5a2b", ha="left")
    for label, kernel, color in [
        ("Local mean [1/3, 1/3, 1/3]", kernels["smooth"], GREEN),
        ("Edge response [-1, 0, 1]", kernels["edge"], RED),
    ]:
        output = conv1d_valid(signal, kernel)
        axes[1].plot(np.arange(1, len(signal) - 1), output, "o-", color=color, label=label)
    axes[1].set(
        title="Each point comes from one 3-value window centered at that position",
        xlabel="window center position",
        ylabel="output value",
    )
    axes[1].text(-.055, 1.02, "b", transform=axes[1].transAxes, fontsize=11,
                 fontweight="bold", color=NAVY, va="bottom")
    axes[1].axhline(0, color="#8d99a6", lw=1.0, ls=":")
    axes[1].axvline(2, color=ORANGE, lw=1.1, ls="--", alpha=.75)
    axes[1].annotate("rising: [0, 1, 1]\nedge = +1", xy=(2, 1), xytext=(1.05, .48),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                     fontsize=8.5, color=RED, ha="left")
    axes[1].annotate("flat: [1, 1, 1]\nedge = 0", xy=(3, 0), xytext=(3.2, .23),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                     fontsize=8.5, color=RED, ha="left")
    axes[1].annotate("falling: [1, 1, 0]\nedge = -1", xy=(4, -1), xytext=(4.55, -.52),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                     fontsize=8.5, color=RED, ha="left")
    axes[1].text(
        9.55,
        .48,
        "green = (left + center + right) / 3\nred = right − left",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#526473",
        bbox=dict(boxstyle="round,pad=.35", facecolor="#f7f9fb", edgecolor="#c7d0d8"),
    )
    axes[1].legend(frameon=False, loc="lower left")
    for axis in axes:
        axis.grid(alpha=.15)
        axis.set_xlim(-.5, 10.5)
    fig.tight_layout()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=OUTPUT / "cnn-shared-kernel.alignment.json",
            overlay_svg=OUTPUT / "cnn-shared-kernel.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(OUTPUT / "cnn-shared-kernel.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "cnn-shared-kernel.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "cnn-shared-kernel.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    image = np.zeros((12, 12)); image[2:10, 6:10] = 1.0; image[5:8, 2:6] = .55
    image += rng.normal(0, .035, image.shape)
    vertical_edge = np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])
    feature = conv2d_valid(image, vertical_edge)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    panels = [(image, "input: 12×12 image", "gray"),
              (vertical_edge, "shared 3×3 kernel", "coolwarm"),
              (feature, "output: 10×10 feature map", "coolwarm")]
    for axis, (values, title, cmap) in zip(axes, panels):
        shown = axis.imshow(values, cmap=cmap)
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
    fig.suptitle("A local kernel turns repeated vertical boundaries into a feature map", fontsize=15)
    finish(fig, "cnn-feature-map.png")

    layers = np.arange(1, 9)
    ordinary = 1 + 2 * layers
    dilated = 1 + 2 * np.cumsum(2 ** np.arange(8))
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    ax.plot(layers, ordinary, "o-", color=BLUE, lw=2.3, label="3-wide kernels, dilation 1")
    ax.plot(layers, dilated, "s--", color=ORANGE, lw=2.3, label="dilation doubles each layer")
    ax.set(title="Stacking or dilation expands the receptive field", xlabel="convolutional layers",
           ylabel="input positions visible to one output")
    ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "cnn-receptive-field.png")

    time = np.arange(30)
    sequence = np.zeros((len(time), 1)); sequence[[3, 16, 23], 0] = [1.0, -.8, .65]
    wx = np.array([[1.5], [-1.0]]); wh = np.array([[.82, .05], [.02, .67]])
    states = rnn_forward(sequence, wx, wh, np.zeros(2))[1:]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    axes[0].stem(time, sequence[:, 0], basefmt=" ", linefmt=BLUE, markerfmt="o")
    axes[0].set(title="Events arrive one time step at a time", ylabel="$x_t$")
    axes[1].plot(time, states[:, 0], color=GREEN, lw=2.2, label="$h_{t,1}$")
    axes[1].plot(time, states[:, 1], color=PURPLE, lw=2.2, ls="--", label="$h_{t,2}$")
    axes[1].set(title="The recurrent state carries a learned summary forward",
                xlabel="time step", ylabel="hidden state")
    axes[1].legend(frameon=False)
    for axis in axes: axis.grid(alpha=.15)
    finish(fig, "rnn-unrolled-state.png")

    steps = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(8.3, 4.3))
    for factor, color, style in [(.75, RED, "-"), (.95, ORANGE, "--"),
                                 (1.0, GREEN, "-"), (1.05, BLUE, ":")]:
        ax.semilogy(steps, np.maximum(factor ** steps, 1e-5), color=color, ls=style,
                    lw=2.2, label=f"recurrent factor={factor}")
    ax.set(title="Repeated Jacobians can erase or amplify long-range influence",
           xlabel="time steps", ylabel="relative gradient magnitude")
    ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "rnn-gradient-memory.png")

    fig, ax = plt.subplots(figsize=(8.3, 4.3))
    for forget, color, style in [(.70, RED, "-"), (.90, ORANGE, "--"), (.99, GREEN, "-")]:
        ax.semilogy(steps, np.maximum(forget ** steps, 1e-5), color=color, ls=style,
                    lw=2.3, label=f"forget gate={forget}")
    ax.set(title="The LSTM forget gate controls the direct memory path",
           xlabel="steps crossed", ylabel="$|\\partial c_T/\\partial c_{T-k}|$")
    ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "lstm-memory-retention.png")
    draw_lstm_flow()

    input_size, hidden_size = 3, 4
    lstm_inputs = rng.normal(size=(7, input_size))
    weights = {name: rng.normal(0, .25, (hidden_size, hidden_size + input_size))
               for name in ("f", "i", "o", "g")}
    biases = {name: np.zeros(hidden_size) for name in ("f", "i", "o", "g")}
    hidden, cell, gates = lstm_forward(lstm_inputs, weights, biases)

    tokens = ["the", "animal", "did", "not", "cross", "because", "it", "was", "tired"]
    token_features = rng.normal(size=(len(tokens), 6))
    token_features[1] = np.array([2, 0, 1, 0, 0, 0])
    token_features[6] = np.array([1.9, .1, 1.1, 0, 0, 0])
    q, k, v = token_features, token_features, rng.normal(size=(len(tokens), 5))
    scores = q @ k.T / np.sqrt(k.shape[1])
    attention_weights = softmax(scores, axis=1)
    context = attention_weights @ v
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    heatmap = ax.imshow(attention_weights, cmap="Blues", vmin=0, vmax=attention_weights.max())
    ax.set(xticks=np.arange(len(tokens)), yticks=np.arange(len(tokens)),
           xticklabels=tokens, yticklabels=tokens, xlabel="key token", ylabel="query token",
           title="Self-attention builds a data-dependent weighted context")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    fig.colorbar(heatmap, ax=ax, label="attention weight")
    finish(fig, "self-attention-weights.png")

    lengths = np.arange(16, 513, 16)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.plot(lengths, lengths, color=BLUE, lw=2.3, label="CNN/RNN sequence work: O(n)")
    ax.plot(lengths, lengths ** 2 / 64, color=RED, lw=2.3, ls="--",
            label="full attention interactions: O(n²)")
    ax.set(title="Sequence length changes the dominant interaction cost",
           xlabel="sequence length n", ylabel="relative work / stored interactions")
    ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "sequence-complexity.png")

    assert np.allclose(conv1d_valid(signal, kernels["edge"]),
                       np.correlate(signal, kernels["edge"], mode="valid"))
    assert feature.shape == (10, 10)

    check_inputs = rng.normal(size=(5, 2)); check_wx = rng.normal(0, .2, size=(3, 2))
    check_wh = rng.normal(0, .2, size=(3, 3)); check_bias = rng.normal(0, .1, size=3)
    _, analytic_grad = rnn_final_sum_loss_and_grad_wh(check_inputs, check_wx, check_wh, check_bias)
    row, col, epsilon = 1, 2, 1e-6
    wh_plus, wh_minus = check_wh.copy(), check_wh.copy()
    wh_plus[row, col] += epsilon; wh_minus[row, col] -= epsilon
    plus = rnn_forward(check_inputs, check_wx, wh_plus, check_bias)[-1].sum()
    minus = rnn_forward(check_inputs, check_wx, wh_minus, check_bias)[-1].sum()
    numerical_grad = (plus - minus) / (2 * epsilon)
    gradient_error = abs(numerical_grad - analytic_grad[row, col])
    assert gradient_error < 1e-6

    assert hidden.shape == (len(lstm_inputs) + 1, hidden_size)
    assert cell.shape == hidden.shape
    assert all(np.all((gates[name] >= 0) & (gates[name] <= 1)) for name in ("f", "i", "o"))
    assert np.allclose(attention_weights.sum(axis=1), 1.0)
    assert context.shape == (len(tokens), 5)

    print(f"conv2d_output_shape={feature.shape}")
    print(f"rnn_bptt_gradient_error={gradient_error:.2e}")
    print(f"lstm_hidden_shape={hidden.shape}")
    print(f"attention_row_sum_error={np.max(np.abs(attention_weights.sum(axis=1)-1)):.2e}")
    print(f"context_shape={context.shape}")


if __name__ == "__main__":
    main()
