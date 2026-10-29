"""
A minimal reverse-mode autograd engine over scalars.

Each Value holds its data, its gradient, the children that produced it,
and a closure that pushes gradients back to those children.
"""
import math


class Value:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        # unique inputs, in a fixed order (a set's order can change between runs)
        self._prev = tuple(dict.fromkeys(_children))
        self._op = _op
        # False means "frozen": an optimizer should skip updating this value.
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    # ---- arithmetic -------------------------------------------------------

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only int/float exponents"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            if exponent == 0:  # x ** 0 is the constant 1, so its gradient is 0
                return
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    # reflected ops so that 2 + v, 2 * v, etc. work
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    # ---- nonlinearities ---------------------------------------------------

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        # numerically stable form: avoids overflow in exp() for large |x|
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            s = e / (1.0 + e)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ---- backprop ---------------------------------------------------------

    def backward(self):
        # topological order via DFS
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        # set .grad to 0 on this value and every value it was computed from
        visited = set()

        def clear(v):
            if v not in visited:
                visited.add(v)
                v.grad = 0.0
                for child in v._prev:
                    clear(child)

        clear(self)


if __name__ == "__main__":
    # Sanity check: f(a, b) = tanh(a*b + a^2) / b
    a = Value(2.0)
    b = Value(-3.0)
    f = ((a * b + a ** 2).tanh()) / b
    f.backward()
    print("f =", f.data)
    print("df/da =", a.grad)
    print("df/db =", b.grad)

    # Numerical check of the same gradients
    def fn(x, y):
        return math.tanh(x * y + x ** 2) / y

    h = 1e-6
    print("numeric df/da ~", (fn(2.0 + h, -3.0) - fn(2.0, -3.0)) / h)
    print("numeric df/db ~", (fn(2.0, -3.0 + h) - fn(2.0, -3.0)) / h)
