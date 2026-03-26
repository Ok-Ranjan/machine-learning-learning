# 🐦 Flappy Bird AI using Deep Q-Network (DQN)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![RL](https://img.shields.io/badge/Reinforcement-Learning-green)
![Status](https://img.shields.io/badge/Project-Active-success)

A reinforcement learning project where an AI agent learns to play Flappy Bird using Deep Q-Learning. The agent interacts with the environment, learns from experience replay, and improves its policy over time.

---

## 📌The P Project Overview

This project implements a Deep Q-Network (DQN) to train an agent to play Flappy Bird using the Gymnasium environment.

The agent learns:

* When to **flap**
* When to **stay idle**

by maximizing cumulative reward through trial and error.

---

## 🧠 Core Concepts Used

* Reinforcement Learning (RL)
* Deep Q-Network (DQN)
* Experience Replay
* Target Network
* Epsilon-Greedy Policy
* Batch Training

---

## 🏗️ Project Structure

```
FLAPPYBIRD_RL/
│
├── agent.py                # Main training & testing pipeline
├── dqn.py                  # Neural Network (Q-function approximator)
├── experience_replay.py    # Replay buffer for storing experiences
├── game_flappy_bird.py     # Manual game control (keyboard play)
├── parameters.yaml         # Hyperparameters configuration
├── runs/                   # Saved models & logs
└── README.md               # Project documentation
```

---

## ⚙️ How It Works

### 1. Environment

We use Gymnasium’s Flappy Bird environment:

```python
env = gym.make("FlappyBird-v0")
```

---

### 2. Neural Network (DQN)

Defined in 

* Input: state (environment observation)
* Output: Q-values for actions

```text
state → Neural Network → Q-values → action
```

---

### 3. Experience Replay

Implemented in 

* Stores past experiences:

  ```
  (state, action, next_state, reward, done)
  ```
* Samples random mini-batches for stable learning

---

### 4. Training Loop

From 

Steps:

1. Reset environment
2. Select action (ε-greedy)
3. Take action → get reward & next state
4. Store experience in replay buffer
5. Sample mini-batch
6. Train DQN
7. Update target network periodically

---

### 5. Target Network

* Stabilizes training
* Updated every `network_sync_rate` steps

---

### 6. Epsilon-Greedy Strategy

* Start with exploration (epsilon val = 1)
* Gradually decay epsilon
* Shift towards exploitation

---

## 🧪 Training

Run training:

```bash
python agent.py flappybirdv0 --train
```

---

## 🎮 Testing (Watch AI Play)

```bash
python agent.py flappybirdv0
```

---

## 📊 Hyperparameters

Defined in 

Key parameters:

* Learning rate (`alpha`)
* Discount factor (`gamma`)
* Epsilon decay
* Replay buffer size
* Batch size

---

## 💾 Model Saving

* Best model saved in:

  ```
  runs/flappybirdv0.pt
  ```
* Based on highest reward achieved

---

## ⚠️ Known Issues

* Observation space warning (state normalization applied)
* Reward instability in early training
* Performance depends heavily on hyperparameters

---

## 🚀 Future Improvements

* Training visualization (graphs)

---

## 📌 Key Learnings

* Importance of experience replay
* Stability issues in RL
* Difference between exploration vs exploitation
* Handling tensor shapes in PyTorch

---

## 👨‍💻 Author

Ranjan Kumar
B.Tech CSE | AI & ML Engineer

---
