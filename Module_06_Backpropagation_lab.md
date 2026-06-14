# 🛠️ PROJECT #1: Train a Neural Network From Scratch

**Difficulty:** 🟡 Medium (Project)
**Time:** 90 minutes
**Prerequisite:** Modules 1-6
**Tools:** Google Colab, Python, NumPy, Matplotlib

---

## 🎯 What We Build

A complete neural network (pure NumPy) that LEARNS to classify our students as pass/fail. We watch the loss drop and reach 100% accuracy!

This combines everything:
- Forward pass (Module 3)
- Loss (Module 3)
- Backpropagation (Module 6)
- Weight updates (Module 5)
- Training loop (Module 4)

---

## 📖 Part 0: The Plan

We will do this in two halves:

**Half 1 (by hand):** Take ONE student, do a complete forward pass, then derive the chain rule for EVERY weight symbolically (dL/dw form), then calculate each gradient number by number, update the weights, and PROVE the loss dropped. This builds deep understanding.

**Half 2 (in code):** Write the full program that trains on ALL 4 students over many epochs, and plot the loss curve.

> **Note on a small difference:** In the by-hand section, we use the raw student numbers and the fixed weights from Module 3, so you can follow every step. In the code, we add **normalization** (scaling data) and **random weight initialization** because those make training more stable in practice. The math logic is identical — only the starting numbers differ.

---

## 📖 Part 1: The Dataset and Network

### Our 4 Students (used the whole course)

| Student | Study (x1) | Sleep (x2) | Pass (1) / Fail (0) |
|---------|:----------:|:----------:|:-------------------:|
| A | 2 | 7 | 0 (Fail) |
| B | 5 | 6 | 1 (Pass) |
| C | 8 | 5 | 1 (Pass) |
| D | 1 | 4 | 0 (Fail) |

### Our Network (2 → 2 → 1)

```
   x1 (study) ─w11─┐
                   ├─▶ h1 (ReLU) ─w31─┐
   x2 (sleep) ─w12─┘                  ├─▶ ŷ (Sigmoid)
                                      │
   x1 (study) ─w21─┐                  │
                   ├─▶ h2 (ReLU) ─w32─┘
   x2 (sleep) ─w22─┘
```

**Activations:** ReLU for hidden, Sigmoid for output.
**Loss:** Binary Cross-Entropy.

### The 9 Parameters (starting values)

```
w11 = 0.5,  w12 = 0.3,  b1 = 0.1     (input → h1)
w21 = 0.4,  w22 = 0.6,  b2 = 0.2     (input → h2)
w31 = 0.7,  w32 = 0.5,  b3 = 0.1     (hidden → output)
```

### The Equations (the "map" of our network)

```
z1 = w11·x1 + w12·x2 + b1        h1 = ReLU(z1)
z2 = w21·x1 + w22·x2 + b2        h2 = ReLU(z2)
z3 = w31·h1 + w32·h2 + b3        ŷ  = sigmoid(z3)
L  = -( y·ln(ŷ) + (1-y)·ln(1-ŷ) )   ← Binary Cross-Entropy
```

Keep this map handy — every chain rule below follows these arrows!

---

## 📖 Part 2: Forward Pass (Student A)

We pick **Student A** (x1=2, x2=7, target y=0) because the network is badly wrong on this one — which makes the gradients big and easy to see!

### Hidden Layer

```
z1 = w11·x1 + w12·x2 + b1
z1 = 0.5·2 + 0.3·7 + 0.1
z1 = 1.0 + 2.1 + 0.1 = 3.2
h1 = ReLU(3.2) = 3.2          (positive, kept)

z2 = w21·x1 + w22·x2 + b2
z2 = 0.4·2 + 0.6·7 + 0.2
z2 = 0.8 + 4.2 + 0.2 = 5.2
h2 = ReLU(5.2) = 5.2          (positive, kept)
```

### Output Layer

```
z3 = w31·h1 + w32·h2 + b3
z3 = 0.7·3.2 + 0.5·5.2 + 0.1
z3 = 2.24 + 2.6 + 0.1 = 4.94
ŷ  = sigmoid(4.94) = 0.9929
```

### Loss

```
L = -( y·ln(ŷ) + (1-y)·ln(1-ŷ) )
L = -( 0·ln(0.9929) + 1·ln(1-0.9929) )
L = -ln(0.0071)
L = 4.948
```

**The problem:** The network confidently says PASS (0.9929), but Student A actually FAILED (target 0). Huge loss = 4.948. Let's fix the weights with backprop!

---

## 📖 Part 3: The Chain Rule for EACH Weight (Symbolic)

Before any numbers, let's write the chain for every weight. Remember: each weight affects the loss through a CHAIN of steps. We multiply the effects along the chain.

### Output Layer Weights

These have the SHORT chain: `weight → z3 → ŷ → L`

```
dL/dw31 = dL/dŷ · dŷ/dz3 · dz3/dw31
dL/dw32 = dL/dŷ · dŷ/dz3 · dz3/dw32
dL/db3  = dL/dŷ · dŷ/dz3 · dz3/db3
```

### Hidden Layer Weights (h1 path)

These have a LONGER chain: `weight → z1 → h1 → z3 → ŷ → L`

```
dL/dw11 = dL/dŷ · dŷ/dz3 · dz3/dh1 · dh1/dz1 · dz1/dw11
dL/dw12 = dL/dŷ · dŷ/dz3 · dz3/dh1 · dh1/dz1 · dz1/dw12
dL/db1  = dL/dŷ · dŷ/dz3 · dz3/dh1 · dh1/dz1 · dz1/db1
```

### Hidden Layer Weights (h2 path)

```
dL/dw21 = dL/dŷ · dŷ/dz3 · dz3/dh2 · dh2/dz2 · dz2/dw21
dL/dw22 = dL/dŷ · dŷ/dz3 · dz3/dh2 · dh2/dz2 · dz2/dw22
dL/db2  = dL/dŷ · dŷ/dz3 · dz3/dh2 · dh2/dz2 · dz2/db2
```

**See the pattern?** Output weights have a 3-link chain. Hidden weights have a 5-link chain (because they're further from the loss). The deeper the weight, the longer its chain!

---

## 📖 Part 4: Calculate Each Small Derivative (Piece by Piece)

Now let's find the value of every single piece. We'll reuse these pieces across many weights.

### Piece 1: dL/dŷ (loss → prediction)

For Binary Cross-Entropy:
```
dL/dŷ = -(y/ŷ) + (1-y)/(1-ŷ)
```
With y=0, ŷ=0.9929:
```
dL/dŷ = -(0/0.9929) + (1)/(1-0.9929)
       = 0 + 1/0.0071
       = 140.85
```

### Piece 2: dŷ/dz3 (sigmoid derivative)

```
dŷ/dz3 = ŷ·(1-ŷ)
       = 0.9929 · (1-0.9929)
       = 0.9929 · 0.0071
       = 0.00705
```

### The Beautiful Simplification!

Multiply Piece 1 × Piece 2:
```
dL/dz3 = dL/dŷ · dŷ/dz3
       = 140.85 · 0.00705
       = 0.9929
```

**Look at this!** `dL/dz3 = 0.9929`, which is exactly `ŷ - y = 0.9929 - 0`. 

> When you pair **Sigmoid + Binary Cross-Entropy**, the messy pieces cancel and `dL/dz3 = ŷ - y`. This clean result is THE reason we always pair sigmoid with cross-entropy!

We'll call this the **output error signal**:
```
dL/dz3 = ŷ - y = 0.9929
```

### Piece 3: dz3/dw31, dz3/dw32, dz3/db3 (linear part of output)

Since `z3 = w31·h1 + w32·h2 + b3`:
```
dz3/dw31 = h1 = 3.2     (the input to this weight)
dz3/dw32 = h2 = 5.2
dz3/db3  = 1            (bias input is always 1)
```

### Piece 4: dz3/dh1 and dz3/dh2 (how z3 depends on hidden outputs)

Since `z3 = w31·h1 + w32·h2 + b3`:
```
dz3/dh1 = w31 = 0.7
dz3/dh2 = w32 = 0.5
```

### Piece 5: dh1/dz1 and dh2/dz2 (ReLU derivative)

ReLU derivative = 1 if input > 0, else 0.
```
dh1/dz1 = 1   (because z1 = 3.2 > 0)
dh2/dz2 = 1   (because z2 = 5.2 > 0)
```

### Piece 6: dz1/dw11 etc. (linear part of hidden)

Since `z1 = w11·x1 + w12·x2 + b1`:
```
dz1/dw11 = x1 = 2
dz1/dw12 = x2 = 7
dz1/db1  = 1

dz2/dw21 = x1 = 2
dz2/dw22 = x2 = 7
dz2/db2  = 1
```

---

## 📖 Part 5: Calculate Each Gradient (Number by Number)

Now we plug the pieces into each chain. No skipping!

### Output Weight Gradients

```
dL/dw31 = dL/dz3 · dz3/dw31 = 0.9929 · 3.2 = 3.177
dL/dw32 = dL/dz3 · dz3/dw32 = 0.9929 · 5.2 = 5.163
dL/db3  = dL/dz3 · dz3/db3  = 0.9929 · 1   = 0.9929
```

### Hidden Weight Gradients (h1 path)

First find `dL/dz1` (reused for w11, w12, b1):
```
dL/dz1 = dL/dz3 · dz3/dh1 · dh1/dz1
       = 0.9929 · 0.7 · 1
       = 0.6950
```
Then:
```
dL/dw11 = dL/dz1 · dz1/dw11 = 0.6950 · 2 = 1.390
dL/dw12 = dL/dz1 · dz1/dw12 = 0.6950 · 7 = 4.865
dL/db1  = dL/dz1 · dz1/db1  = 0.6950 · 1 = 0.695
```

### Hidden Weight Gradients (h2 path)

First find `dL/dz2`:
```
dL/dz2 = dL/dz3 · dz3/dh2 · dh2/dz2
       = 0.9929 · 0.5 · 1
       = 0.4965
```
Then:
```
dL/dw21 = dL/dz2 · dz2/dw21 = 0.4965 · 2 = 0.993
dL/dw22 = dL/dz2 · dz2/dw22 = 0.4965 · 7 = 3.476
dL/db2  = dL/dz2 · dz2/db2  = 0.4965 · 1 = 0.4965
```

### All 9 Gradients Together

| Weight | Gradient |
|--------|:--------:|
| dL/dw11 | 1.390 |
| dL/dw12 | 4.865 |
| dL/db1 | 0.695 |
| dL/dw21 | 0.993 |
| dL/dw22 | 3.476 |
| dL/db2 | 0.4965 |
| dL/dw31 | 3.177 |
| dL/dw32 | 5.163 |
| dL/db3 | 0.9929 |

**Notice:** All gradients are POSITIVE. That makes sense — the prediction (0.9929) is TOO HIGH (target is 0), so we need to DECREASE the weights to lower the prediction. The update rule (subtracting) will do exactly that!

---

## 📖 Part 6: Update Each Weight

Update rule (Module 5): `new = old - (learning_rate × gradient)`. Use learning_rate = 0.1.

```
w11 = 0.5 - 0.1·1.390  = 0.5 - 0.139   = 0.361
w12 = 0.3 - 0.1·4.865  = 0.3 - 0.4865  = -0.1865
b1  = 0.1 - 0.1·0.695  = 0.1 - 0.0695  = 0.0305

w21 = 0.4 - 0.1·0.993  = 0.4 - 0.0993  = 0.3007
w22 = 0.6 - 0.1·3.476  = 0.6 - 0.3476  = 0.2524
b2  = 0.2 - 0.1·0.4965 = 0.2 - 0.0497  = 0.1504

w31 = 0.7 - 0.1·3.177  = 0.7 - 0.3177  = 0.3823
w32 = 0.5 - 0.1·5.163  = 0.5 - 0.5163  = -0.0163
b3  = 0.1 - 0.1·0.9929 = 0.1 - 0.0993  = 0.0007
```

**All weights got smaller** (some even went negative). This will pull the prediction DOWN toward 0 (where it should be for Student A).

---

## 📖 Part 7: PROOF the Loss Dropped! 🎯

Do the forward pass AGAIN with the new weights:

```
z1 = 0.361·2 + (-0.1865)·7 + 0.0305 = 0.722 - 1.3055 + 0.0305 = -0.553
h1 = ReLU(-0.553) = 0           (negative → zeroed!)

z2 = 0.3007·2 + 0.2524·7 + 0.1504 = 0.6014 + 1.7668 + 0.1504 = 2.519
h2 = ReLU(2.519) = 2.519

z3 = 0.3823·0 + (-0.0163)·2.519 + 0.0007 = -0.0404
ŷ  = sigmoid(-0.0404) = 0.4899
```

**Compare:**

| | Before | After |
|---|:------:|:-----:|
| Prediction ŷ | 0.9929 | **0.4899** |
| Target | 0 | 0 |
| Loss | 4.948 | **0.673** |

**The prediction dropped from 0.9929 → 0.4899 (much closer to 0)!**
**The loss crashed from 4.948 → 0.673!** 🎉

In ONE backprop step, the network corrected a confident mistake. Repeat this for all students, many times, and the network becomes perfect!

> **Side note:** After this update, h1 became 0 (its input went negative — a "sleeping" ReLU neuron for this example). That's normal and fine; with more training and other students, it wakes back up. This is exactly the kind of thing you only notice when you build it by hand!

---

## 📖 Part 8: The Complete Code (All 4 Students)

Now the full program. It trains on ALL 4 students over 1000 epochs. We add **normalization** and **random weights** for stability (as explained in Part 0).

### Cell 1: Setup and Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Lock random numbers so results are reproducible
np.random.seed(42)

# Student data: [study_hours, sleep_hours]
X = np.array([
    [2, 7],   # Student A
    [5, 6],   # Student B
    [8, 5],   # Student C
    [1, 4]    # Student D
], dtype=float)

# Targets: 0 = fail, 1 = pass (column shape)
y = np.array([[0], [1], [1], [0]], dtype=float)

names = ['A', 'B', 'C', 'D']
print(f"{X.shape[0]} students, {X.shape[1]} features each")
```

### Cell 2: Normalize the Data

```python
# Standardize: (value - mean) / std. Makes training stable.
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

print("Normalized inputs (centered around 0):")
print(np.round(X_norm, 3))
```

**Why?** Study hours (1-8) and sleep hours (4-7) are different scales. Normalizing makes the network treat them fairly and trains faster.

### Cell 3: Initialize Weights

```python
# 2 inputs → 2 hidden → 1 output
# Small RANDOM weights (NOT zero — to break symmetry)
W_hidden = np.random.randn(2, 2) * 0.5   # (2 neurons, 2 inputs)
b_hidden = np.zeros(2)                     # (2,)
W_output = np.random.randn(1, 2) * 0.5   # (1 neuron, 2 inputs)
b_output = np.zeros(1)                     # (1,)

print("Total parameters:", W_hidden.size + b_hidden.size + W_output.size + b_output.size)
```

### Cell 4: Activation Functions + Derivatives

```python
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu_derivative(z):
    # 1 where z>0, else 0
    return (z > 0).astype(float)
```

### Cell 5: Forward Pass

```python
def forward(X):
    """Forward pass. Returns prediction + cache of intermediate values."""
    # Hidden layer (ReLU)
    z_hidden = X @ W_hidden.T + b_hidden     # (4,2)@(2,2)+(2,) = (4,2)
    h = relu(z_hidden)

    # Output layer (Sigmoid)
    z_output = h @ W_output.T + b_output     # (4,2)@(2,1)+(1,) = (4,1)
    y_hat = sigmoid(z_output)

    cache = {'z_hidden': z_hidden, 'h': h, 'z_output': z_output, 'y_hat': y_hat}
    return y_hat, cache
```

### Cell 6: Loss Function

```python
def compute_loss(y_hat, y):
    """Binary Cross-Entropy, averaged over all students."""
    epsilon = 1e-10
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)   # avoid log(0)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
```

### Cell 7: Backward Pass (Backpropagation)

```python
def backward(X, y, cache):
    """Compute gradients for all weights using backprop."""
    m = X.shape[0]
    h = cache['h']
    y_hat = cache['y_hat']
    z_hidden = cache['z_hidden']

    # Output error signal: for Sigmoid+BCE this is simply (y_hat - y)
    delta_output = (y_hat - y) / m              # (4,1)

    # Output gradients: gradient = delta × input
    dW_output = delta_output.T @ h              # (1,2)
    db_output = np.sum(delta_output, axis=0)    # (1,)

    # Propagate blame to hidden layer, multiply by ReLU derivative
    delta_hidden = (delta_output @ W_output) * relu_derivative(z_hidden)  # (4,2)

    # Hidden gradients
    dW_hidden = delta_hidden.T @ X              # (2,2)
    db_hidden = np.sum(delta_hidden, axis=0)    # (2,)

    return {'dW_hidden': dW_hidden, 'db_hidden': db_hidden,
            'dW_output': dW_output, 'db_output': db_output}
```

### Cell 8: The Training Loop

```python
learning_rate = 0.5
epochs = 1000
loss_history = []

for epoch in range(epochs):
    # 1. Forward
    y_hat, cache = forward(X_norm)
    # 2. Loss
    loss = compute_loss(y_hat, y)
    loss_history.append(loss)
    # 3. Backward
    grads = backward(X_norm, y, cache)
    # 4. Update
    W_hidden -= learning_rate * grads['dW_hidden']
    b_hidden -= learning_rate * grads['db_hidden']
    W_output -= learning_rate * grads['dW_output']
    b_output -= learning_rate * grads['db_output']

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

print(f"\nFinal loss: {loss_history[-1]:.4f}")
```

**Expected:** loss drops from ~0.70 to ~0.01.

### Cell 9: Plot the Loss Curve

```python
plt.figure(figsize=(10, 6))
plt.plot(loss_history, color='blue', linewidth=2)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss (lower = better)")
plt.grid(True, alpha=0.3)
plt.show()
```

You'll see the classic learning curve: steep drop, then gentle level-off.

### Cell 10: Check Final Predictions

```python
final_predictions, _ = forward(X_norm)
correct = 0
print(f"{'Student':<9}{'Prediction':>12}{'Rounded':>9}{'Target':>8}")
for i in range(len(X)):
    pred = final_predictions[i][0]
    rounded = 1 if pred >= 0.5 else 0
    target = int(y[i][0])
    if rounded == target:
        correct += 1
    print(f"Student {names[i]:<3}{pred:>11.4f}{rounded:>9}{target:>8}")
print(f"\nAccuracy: {correct}/{len(X)} = {100*correct/len(X):.0f}%")
```

**Expected:** 100% accuracy! All 4 students classified correctly.

---

## 📋 Key Concepts Recap

### The Chain Rule Pattern

- **Output weights** (close to loss): short chain, 3 links
  - `dL/dw31 = dL/dŷ · dŷ/dz3 · dz3/dw31`
- **Hidden weights** (far from loss): long chain, 5 links
  - `dL/dw11 = dL/dŷ · dŷ/dz3 · dz3/dh1 · dh1/dz1 · dz1/dw11`

### The Reusable Pieces

| Piece | Formula | Meaning |
|-------|---------|---------|
| dL/dŷ | -(y/ŷ)+(1-y)/(1-ŷ) | loss → prediction (BCE) |
| dŷ/dz3 | ŷ·(1-ŷ) | sigmoid derivative |
| **dL/dz3** | **ŷ - y** | the clean output error signal! |
| dz3/dw | the input (h or x) | linear derivative |
| dh/dz | 1 if z>0 else 0 | ReLU derivative |

### The Gradient Formula

> **gradient for any weight = (error signal at its layer) × (its input)**

### The Full Cycle

```
Forward → Loss → Backward (gradients) → Update → Repeat
```

---

## 🤔 Common Doubts

### Q1: Why use Student A for the hand calculation?
Because the network was confidently WRONG on Student A (predicted 0.99 for a fail). This makes the gradients large and easy to see. Big numbers = clear learning!

### Q2: Why do the by-hand weights differ from the code?
The hand section uses fixed weights (0.5, 0.3...) and raw data so you can follow each step. The code uses random weights + normalized data for stable, realistic training. The logic is identical.

### Q3: Why did dL/dz3 simplify to (ŷ - y)?
Because Sigmoid + Binary Cross-Entropy are a "matched pair." The loss derivative and the sigmoid derivative cancel each other's messy parts, leaving the clean `ŷ - y`. This is why they're always used together!

### Q4: Why are all gradients positive in our example?
Because the prediction (0.99) was too HIGH (target 0). Positive gradient + the minus sign in the update rule = weights decrease = prediction drops toward 0. Correct direction!

### Q5: What does dividing by m do in the code?
`m` is the number of students. Dividing averages the gradient across all of them, so the update reflects the whole batch, not just one student.

### Q6: Why did h1 become 0 after the update?
Its input z1 went negative, and ReLU zeroes negatives. This "sleeping neuron" is normal for one example; with full training it usually reactivates.

---

## ✅ Quick Practice

### Question 1
Write the chain rule (symbolically) for dL/dw32.

<details><summary>Answer</summary>

```
dL/dw32 = dL/dŷ · dŷ/dz3 · dz3/dw32
```
(Output weight = short 3-link chain. dz3/dw32 = h2.)
</details>

### Question 2
If the output error signal dL/dz3 = 0.5 and h1 = 4, what is dL/dw31?

<details><summary>Answer</summary>

```
dL/dw31 = dL/dz3 × h1 = 0.5 × 4 = 2.0
```
</details>

### Question 3
Why does a hidden weight have a LONGER chain than an output weight?

<details><summary>Answer</summary>
Because a hidden weight is FURTHER from the loss. Its effect must travel through more steps: its own z, then its activation h, then the output z, then the prediction, then the loss. More steps = more links to multiply.
</details>

---

## 🎉 What You Accomplished

✅ Derived the chain rule for all 9 weights by hand
✅ Calculated every gradient number by number
✅ Proved the loss dropped (4.948 → 0.673 in one step!)
✅ Built a complete training program in NumPy
✅ Trained to 100% accuracy on all students
✅ Plotted the learning curve

**You built a learning machine from absolute scratch — and understand every single line!** Most ML practitioners never do this. When we use PyTorch next, you'll know exactly what it does for you.

---

## 🎬 What's Next: Module 7 — Optimizers + PyTorch

We'll rebuild THIS SAME network in **PyTorch** — and you'll see how ~80 lines becomes ~15 lines! Then we learn **optimizers** (Momentum, Adam) that make training even faster and smarter.

---

*Project #1 Complete! ✅ Your first trained neural network — from scratch! 🚀*
