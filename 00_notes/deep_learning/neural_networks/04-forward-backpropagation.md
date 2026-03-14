# Forward & Backpropagation

Neural networks train using two main processes.

---

# Forward Propagation

Steps:

1. Input features are fed into the network
2. Weights are applied
3. Bias is added
4. Weighted sum is calculated
5. Activation function applied
6. Output produced

Input → Weighted Sum → Activation → Output

---

# Loss Function

Prediction error is measured using a loss function.

Example:

Predicted = ŷ
Actual = y

Example loss:

Loss = (y − ŷ)^2

Goal:

Minimize Loss

---

# Backpropagation

Backpropagation updates the weights to reduce loss.

Steps:

1. Compute loss
2. Calculate gradients
3. Update weights using optimizer
4. Repeat training