# 📘 MODULE 4: Batches & Datasets

**Difficulty:** 🟢 Easy
**Time:** 45 minutes
**Prerequisite:** Modules 1, 2, 3

---

## 🎯 What You Will Learn

1. What batch, epoch, and iteration mean
2. The 3 types of gradient descent (Batch, Stochastic, Mini-Batch)
3. Why we split data into Train / Validation / Test sets
4. What overfitting is (preview)
5. How to read training vs validation accuracy

---

## 📖 Why This Module Matters

So far we worked with just 4 students. But real datasets have **thousands or millions** of examples!

Question: When training, do we feed ALL examples at once? One at a time? In groups?

This module answers that, starting with 3 words that confuse every beginner.

---

## 📖 Part 1: Batch, Epoch, Iteration (The 3 Confusing Words)

### The Textbook Analogy 📚

Imagine studying a **1000-page textbook** for an exam. You can't read all 1000 pages at once! So you break it up.

| Word | Meaning | Textbook Example |
|------|---------|------------------|
| **Batch** | How many examples shown before one weight update | Reading 100 pages in one sitting = batch size 100 |
| **Iteration** | One weight update (one batch processed) | One sitting = one iteration |
| **Epoch** | One complete pass through ALL data | Finishing the whole book once = 1 epoch |

### With Our Students

Imagine 1000 students (bigger version of our dataset):

- **Batch** = how many students shown at once before updating weights
- **Iteration** = one weight update (after one batch)
- **Epoch** = going through ALL 1000 students one complete time

### The Simple Formula

```
Iterations per epoch = Total examples ÷ Batch size
```

**Example:** 1000 students, batch size 100:
```
Iterations per epoch = 1000 ÷ 100 = 10 iterations
```

If we train for 5 epochs:
```
Total iterations = 5 epochs × 10 = 50 iterations
```

### Why Multiple Epochs?

Just like you re-read a textbook several times before an exam, the network sees the data many times. Each pass, the weights improve a little more!

More epochs = more revision = better learning (up to a point — too many can cause overfitting!).

---

## 📖 Part 2: The 3 Types of Gradient Descent

The big question: **how many examples before updating weights?** There are 3 answers.

### Type 1: Batch Gradient Descent

**"Read the whole book, then think"**

- Show ALL examples (all 1000 students)
- Calculate the average error
- Update weights ONCE

| Pros | Cons |
|------|------|
| Stable, smooth learning | Very slow |
| | Needs lots of memory for big data |

**Update frequency:** 1 update per epoch

### Type 2: Stochastic Gradient Descent (SGD)

**"Read 1 page, think immediately"**

- Show ONE example
- Update weights
- Next example → update again
- One at a time!

| Pros | Cons |
|------|------|
| Fast updates | Very "jumpy"/noisy learning |
| Less memory | Zig-zags a lot |

**Update frequency:** 1 update per example (1000 updates per epoch!)

### Type 3: Mini-Batch Gradient Descent ⭐ THE WINNER

**"Read a chapter, then think"**

- Show a SMALL group (like 32 or 64 students)
- Update weights
- Next group → repeat

| Pros | Cons |
|------|------|
| Best of both worlds! | (basically none for normal use) |
| Fast AND stable | |
| GPU-friendly | |

**Update frequency:** 1 update per mini-batch

**This is what everyone uses!**

### How Each One "Moves" Toward the Goal

Think of reaching the center of a target (best weights):

- **Batch:** smooth, slow path straight to center
- **Stochastic:** fast but zig-zags wildly
- **Mini-batch:** fast and mostly smooth with small wobbles

All reach the goal, but mini-batch balances speed and smoothness best.

(We'll fully understand "moving toward the goal" in Module 5!)

### Which to Use?

**Mini-Batch Gradient Descent** — almost always!

- Faster than Batch (updates more often)
- More stable than SGD (averaging a group reduces jumpiness)
- GPU-friendly (GPUs process small groups in parallel)

**Common batch sizes:** 16, 32, 64, 128 (powers of 2, because GPUs like those).

### ⚠️ Naming Confusion

When people say "SGD" in modern ML, they USUALLY mean mini-batch! The names got mixed up over the years. When you see `batch_size=32` in code, that's mini-batch gradient descent.

---

## 📖 Part 3: Splitting Data — Train / Validation / Test

### The Exam Analogy 🎓

Imagine preparing for a final exam:

| Split | Analogy | Purpose |
|-------|---------|---------|
| **Training set** | Textbook & practice problems | Network studies and learns from these |
| **Validation set** | Mock tests / sample papers | Check progress, tune settings |
| **Test set** | The REAL final exam | True measure of performance (never seen!) |

**The golden rule:**
> You must NEVER study from the real exam questions! That's cheating — you'd memorize answers instead of truly learning.

### Typical Split Ratios

```
Training:   70%  (network learns from these)
Validation: 15%  (check progress during training)
Test:       15%  (final exam, used once)
```

For 1000 students:
- 700 for training
- 150 for validation
- 150 for test

(Other common splits: 80/10/10, or 60/20/20)

### What Each Set Does

**Training set (70%):**
The network learns from these. It adjusts weights using these examples.

**Validation set (15%):**
During training, we check how the network does on data it DIDN'T learn from. This tells us if it's truly learning OR just memorizing. Used to tune settings (layers, learning rate, etc.).

**Test set (15%):**
The FINAL exam. Touched ONLY ONCE, at the very end. Gives the TRUE measure of performance on brand-new data.

### Why Not Just Test on Training Data?

Because the network has SEEN the training data! Testing on it is like giving students the exam questions beforehand — they'd score 100% but might not have truly learned.

---

## 📖 Part 4: Overfitting (Preview)

> **Overfitting = the network MEMORIZES the training data instead of LEARNING patterns.**

An overfit network is like a student who memorized every practice answer but fails when questions are slightly different.

The validation and test sets help us CATCH overfitting!

(We'll study overfitting deeply in Module 8.)

---

## 📖 Part 5: How the Splits Work During Training

```
DURING TRAINING (repeats each epoch):
  1. Show training batch → forward pass → calculate loss → update weights
  2. After each epoch, check VALIDATION set (no learning, just measuring)
  3. Repeat for many epochs until good

FINAL STEP (once at the very end):
  Use TEST set → get the true score
```

### A Golden Signal to Watch

| Training Accuracy | Validation Accuracy | What It Means |
|:-----------------:|:-------------------:|---------------|
| High (95%) | High (93%) | ✅ Great! Learning well |
| High (98%) | Low (60%) | ❌ Overfitting! Memorizing |
| Low (55%) | Low (53%) | ❌ Underfitting! Too simple or undertrained |

This comparison is one of the most useful tools in all of ML!

---

## 📋 Module 4 Recap

**The 3 words:**
- **Batch** = how many examples shown before one weight update
- **Iteration** = one weight update (one batch processed)
- **Epoch** = one complete pass through ALL data
- Formula: `iterations per epoch = total examples ÷ batch size`

**The 3 types of gradient descent:**
- **Batch GD** = all data at once (stable but slow)
- **Stochastic GD** = one at a time (fast but jumpy)
- **Mini-Batch GD** = small groups (the winner! everyone uses this)

**The 3 data splits:**
- **Training (70%)** = network learns from these
- **Validation (15%)** = check progress during training
- **Test (15%)** = final exam, used once

**Key idea:** Never test on data the network learned from. Splitting helps catch overfitting (memorizing instead of learning).

---

## 🤔 Common Doubts

### Q1: What batch size should I use?
Start with 32 or 64. They work well for most problems. Powers of 2 (16, 32, 64, 128) are GPU-friendly.

### Q2: How many epochs should I train?
Depends! Watch validation accuracy. When it stops improving (or gets worse), stop. This is "early stopping" (Module 8).

### Q3: What if I have very little data?
For small datasets, use 80/10/10, or special techniques like "cross-validation." We'll discuss later.

### Q4: Do I split randomly?
Usually yes — shuffle the data first, then split. But for time-series data (like stock prices), split by time instead.

### Q5: Why is mini-batch GPU-friendly?
GPUs process many things in parallel. A mini-batch of 32 examples can all be computed at the same time — perfect for GPU's parallel power!

### Q6: What's the difference between validation and test set?
- Validation: used MANY times during training to check progress and tune settings
- Test: used ONCE at the very end for the final, honest score

---

## ✅ Quick Practice

### Question 1
You have 800 examples and use a batch size of 50. How many iterations per epoch?

<details><summary>Answer</summary>
800 ÷ 50 = **16 iterations per epoch**
</details>

### Question 2
If you train for 10 epochs with 16 iterations each, how many total weight updates?

<details><summary>Answer</summary>
10 × 16 = **160 total iterations (weight updates)**
</details>

### Question 3
Your network gets 99% accuracy on training data but only 65% on validation data. What's happening?

<details><summary>Answer</summary>
**Overfitting!** The network memorized the training data instead of learning general patterns. It does great on what it has seen but poorly on new data.
</details>

### Question 4
Which gradient descent type does almost everyone use, and why?

<details><summary>Answer</summary>
**Mini-Batch Gradient Descent** — it's fast (updates often), stable (averaging reduces jumpiness), and GPU-friendly (parallel processing of small groups).
</details>

---

## 🎬 What's Next: Module 5 — Gradient Descent

This is where the MAGIC begins! We finally learn HOW the network actually LEARNS — how it turns random weights into smart ones.

We'll learn:
- What a "gradient" is (in simple words)
- How the network follows the slope "downhill" to reduce loss
- What "learning rate" means
- The weight update rule

After Module 5 comes Module 6 (Backpropagation) and then PROJECT #1 — training your first network from scratch!

---

*Module 4 Complete! ✅ Ready for Module 5: Gradient Descent →*
