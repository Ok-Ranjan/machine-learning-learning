# Activation Functions

Activation functions introduce **non-linearity** into neural networks.

Without them, the network behaves like **linear regression**.

---

## Sigmoid

Formula:

σ(x) = 1 / (1 + e^-x)

Range:  0 → 1


Used for binary classification.

---

## Tanh
tanh(x) = (e^x − e^-x) / (e^x + e^-x)

Range: -1 → 1


---

## ReLU

Rectified Linear Unit

f(x) = max(0, x)

Very common in deep learning models.

---

## Softmax

Used for **multi-class classification**.

Converts outputs into probabilities.

---

## Linear

Used in regression problems.