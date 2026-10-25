"""A tiny scalar reverse-mode automatic differentiation engine."""

import math


class Value:
    def __init__(self, data, children=(), operation=""):
        self.data = float(data); self.grad = 0.0
        self._previous = set(children); self._operation = operation; self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def backward(): self.grad += out.grad; other.grad += out.grad
        out._backward = backward; return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def backward(): self.grad += other.data*out.grad; other.grad += self.data*out.grad
        out._backward = backward; return out

    __rmul__ = __mul__

    def tanh(self):
        value = math.tanh(self.data); out = Value(value, (self,), "tanh")
        def backward(): self.grad += (1-value**2)*out.grad
        out._backward = backward; return out

    def backward(self):
        order=[]; visited=set()
        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._previous: build(child)
                order.append(node)
        build(self); self.grad=1.0
        for node in reversed(order): node._backward()


if __name__ == "__main__":
    x, w, b = Value(2.0), Value(-3.0), Value(1.0)
    y = (x*w+b).tanh(); y.backward()
    expected = (1-math.tanh(-5.0)**2)*2.0
    print(f"output={y.data:.6f}")
    print(f"autograd_dw={w.grad:.8f}")
    print(f"analytic_dw={expected:.8f}")
    assert abs(w.grad-expected) < 1e-10
