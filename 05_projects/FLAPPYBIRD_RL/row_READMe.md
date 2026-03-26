

## All steps :
1...


### Implementing - Epsilon-Greedy Policy and Epsilon-Decay:
- here we set the **Epsilon=epsilon_init** value so that our **Agent Explore more** start in training

- **Epsilon_Greedy Policy**:
    - in training more when a `random < epsilon` : so here our Agent **Explore** and take a action
    - othervise: our Agent do **Exploit** by taking optimal action value from policy DQN 

- **Epsilon-Decay:**
    - than after every **episode** we decaying the epsilon value by redefine : **epsilon=max(epsilon*epsilon_decay, epsilon_min)**
    - So that our Agent taking optimal Action and do better performation at end. 
- epsilon only need for training

### Covert values into Tensors
because of we have to pass all params to NN, so we must have to conver all values into tensors

values like: state, action, next_state, reward => these all values we will going to feed NN

- After converting values into tensors:
    - we use **`.item()`** for getting exact value of variable
    ```py
    {
        # insead of just passing - action
        ...(all parameters) = env.step(action)

        # Now we pass - action.item()
        ...(all parameters) = env.step(action.item())
    }
    ```
    - And use **`unsqueeze()`** for converting dim, because tensors value is 1D
    ```py
    {
        # instead of passing - state -> 1D
        action = policy_dqn(state).argmax()

        # we pass state by do unsqueeze(dim=0) - add one more dim in state
        # because: policy_dqn expect that only 2D value
        action = policy_dqn(state.unsqueeze(dim=0)).argmax()

        # After unsqueeze() - we do squeeze() for converting it into same dim
        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
    }
    ```

### Build Target Network & OPtimizers
Target netwoks only need in trainging

- for creating a copy of the **policy_dqn** into the **target_dqn**, we copy the **wt** & **bias** vals from policy => target
```py
{
    target_dqn.load_state_dict(policy_dqn.state_dict())
    # state_dict() -> it stored weight and bias values of a NN
}
```

- **sync the network**:- means we do this same **copy work** for after every - 10 steps (network_sync_rate)

- **Optimizers - Adam** here we pass **`policy_dqn.parameters()`** instead of **nn.parameters()**

- 

### train DQN
after get the smaples, we build a function **optimizer()** for train our DQN:
1. get experience from our experienc replay
2. compute the target Q-value (y-true): by help of target-network
3. calculate **y-pred** -> by policy network
4. compute **loss** - for both network
5. Back Propgations

- For making traing fast: we train DQN in **batch** instead of traing experience one-by-one 

- **traing experience one-by-one**
```py
{
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get experience - by mini_batch
        for state, action, next_state, reward, terminated in mini_batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.gamma * target_dqn(next_state).max()  # y-true
                
            current_q = policy_dqn(state) # y-pred

            # loss
            loss = self.loss_fun(current_q, target_q)

            self.optimizer.zero_grad() # grediant reset to zero
            loss.backward()
            self.optimizer.step() # update wt & bias
}
```

- **train DQN in batch**
```py
{
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences 
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # calculate target Q-values - if termination=true =>> zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]  # y-true

        # claculate y_pred i.e. Q-value from current policy        
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss = self.loss_fun(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad() # grediant reset to zero
        loss.backward()
        self.optimizer.step() # update wt & bias
}
```



### Saving best reward / best performance to a dict
ON which point our Agent do better performance so we saved all parameters values in dict of that point.

**`RUNS_DIR =  "runs"`**  - A folder **runs** which store our best model

And we add some **log dir**
```py
{
    # all logs came inside the RUNS_DIR with .log file
    self.LOG_FILE =  os.path.join(RUNS_DIR, f"{self.param_set}.log")

    # for storing model -> when dq model have best rewards
    self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
}
```
- Tracking the best reward:
    - define a var in run() - training mode: **`best_reward= float(-inf)`**
    - after a episode completed: we update this value with every rewards
