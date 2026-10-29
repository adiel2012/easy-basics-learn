"""
Train a tiny neural network to learn XOR, using the Value autograd engine
from mini_autograd.py.

XOR truth table:
    x1  x2  ->  y
     0   0  ->  0
     0   1  ->  1
     1   0  ->  1
     1   1  ->  0

A single straight line cannot separate these classes, so the network
needs a hidden layer.
"""
import random
from mini_autograd import Value

random.seed(1)


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        # w1*x1 + w2*x2 + ... + b, then tanh to add non-linearity
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.tanh()

    def params(self):
        return self.w + [self.b]


class MLP:
    """2 inputs -> hidden layer of tanh neurons -> 1 sigmoid output."""

    def __init__(self, n_in, n_hidden):
        self.hidden = [Neuron(n_in) for _ in range(n_hidden)]
        self.out_w = [Value(random.uniform(-1, 1)) for _ in range(n_hidden)]
        self.out_b = Value(0.0)

    def __call__(self, x):
        h = [n(x) for n in self.hidden]
        z = sum((wi * hi for wi, hi in zip(self.out_w, h)), self.out_b)
        return z.sigmoid()  # output in (0, 1): read it as a probability

    def params(self):
        ps = []
        for n in self.hidden:
            ps += n.params()
        return ps + self.out_w + [self.out_b]


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

model = MLP(n_in=2, n_hidden=4)
params = model.params()

learning_rate = 0.5
epochs = 2000

# Freezing: a parameter with requires_grad = False keeps its current value.
# Set freeze_hidden = True to train only the output neuron.
freeze_hidden = False
if freeze_hidden:
    for n in model.hidden:
        for p in n.params():
            p.requires_grad = False

trainable = sum(1 for p in params if p.requires_grad)
print(f"trainable parameters: {trainable} of {len(params)}")

for epoch in range(epochs):
    # 1. Forward pass: build the graph and compute the total loss
    loss = Value(0.0)
    for x, y in zip(X, Y):
        pred = model([Value(v) for v in x])
        loss = loss + (pred - y) ** 2  # squared error

    # 2. Backward pass: clear old gradients, then compute new ones
    for p in params:
        p.grad = 0.0
    loss.backward()

    # 3. Update: move each parameter a little against its gradient
    for p in params:
        if p.requires_grad:  # frozen parameters are skipped
            p.data -= learning_rate * p.grad

    if epoch % 400 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:4d}   loss {loss.data:.5f}")

print()
print("x1  x2  target  output  predicted")
for x, y in zip(X, Y):
    out = model([Value(v) for v in x]).data
    print(f"{x[0]}   {x[1]}     {y}      {out:.3f}   {int(out >= 0.5)}")
