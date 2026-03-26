import gymnasium as gym
import flappy_bird_gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os
import imageio


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RUNS_DIR = "runs"  # A dir(file) which store the best model
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
                
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fun =  nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE_BEST_REWARDS = os.path.join(RUNS_DIR, f"{self.param_set}_best_rewards.log")
        self.LOG_FILE_ALL_REWARDS = os.path.join(RUNS_DIR, f"{self.param_set}_all_rewards.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
        self.GIF_FILE = os.path.join(RUNS_DIR, "best_gameplay.gif")

    def run(self, is_training=True, render=False, record_frames=False):
        if is_training and not render:
            env =  gym.make("FlappyBird-v0") # capture game images 
        else: # testing mood always render= true
            env =  gym.make("FlappyBird-v0", render_mode="rgb_array" if record_frames else "human")

        num_states = env.observation_space.shape[0] # input dim
        num_actions = env.action_space.n # output dim

        policy_dqn = DQN(num_states, num_actions).to(device)
        
        start_episode = 0

        if is_training: # training init block
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt & bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_train_reward = float("-inf")

            # LOAD CHECKPOINT
            if os.path.exists(self.MODEL_FILE):
                print("🔄 Loading previous checkpoint...")

                checkpoint = torch.load(self.MODEL_FILE, map_location=device)

                policy_dqn.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                epsilon = checkpoint["epsilon"]
                steps = checkpoint["steps"]
                best_train_reward = checkpoint["best_reward"]
                start_episode = checkpoint.get("episode", 0)

                print(f"Resumed with epsilon={epsilon}, steps={steps}")
        
        else:
            # best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            # shift to evaluation mode : not training now
            policy_dqn.eval()
            best_test_reward = float("-inf")
            best_frames = None
        
        try:
            # our code running i
            for episode in itertools.count(start=start_episode):
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float, device=device) / 500.0

                episode_reward = 0
                terminated = False
                frames = [] # store all frame as list and reset frames for each episode

                while (not terminated and episode_reward < self.reward_threshold):
                    # Eplementing - Epsilon-greedy Policy
                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample() # explore
                        action = torch.tensor(action, dtype=torch.long, device=device)
                    else:
                        # no as such training happen here, So not compute gradient
                        with torch.no_grad(): # process become fast
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit
                
                    next_state, reward, terminated, _, _ = env.step(action.item())

                    # capture frame
                    if not is_training and record_frames:
                        frame = env.render() # 
                        frames.append(frame)

                    episode_reward += reward 

                    # creating tensors
                    next_state = torch.tensor(next_state, dtype=torch.float, device=device) / 500.0
                    reward = torch.tensor(reward, dtype=torch.float, device=device)


                    if is_training:
                        memory.append((state, action, next_state, reward, terminated))
                        steps += 1
                    
                    state = next_state             

                print(f"episode={episode+1} with total_reward={episode_reward}")
                
                if is_training:
                    # epsilone decay(after evry episode) : we can also do decay with evry step but its do to fast decay
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                    # log All rewards
                    log_msg_all_rewards = f"{episode+1},{episode_reward}"
                    with open(self.LOG_FILE_ALL_REWARDS, "a") as f:
                        f.write(log_msg_all_rewards + "\n")

                    if episode_reward > best_train_reward:
                        log_msg_best_rewards = f"best reward = {episode_reward} for a episode = {episode+1}"

                        with open(self.LOG_FILE_BEST_REWARDS, "a") as f:
                            f.write(log_msg_best_rewards + "\n")
                        
                        # save model
                        torch.save({
                            "model": policy_dqn.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "epsilon": epsilon,
                            "steps": steps,
                            "best_reward": best_train_reward,
                            "episode": episode
                        }, self.MODEL_FILE)

                        best_train_reward = episode_reward

                if is_training and len(memory) > self.mini_batch_size:
                    # get sample
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # sync the network
                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps=0
                
                # Tracking best freames
                if not is_training and episode_reward > best_test_reward:
                    if record_frames:
                        best_frames = frames.copy()
                    best_test_reward = episode_reward

        except KeyboardInterrupt:
            print("\n🛑 Stopped manually")
        
        finally:
            if is_training:
                print(f"BEST REWARDS (train) => {best_train_reward}")
            else:
                print(f"BEST REWARDS (test) => {best_test_reward}")
            
            # Save GIF - Best Frames
            if not is_training and record_frames and best_frames is not None:
                imageio.mimsave(self.GIF_FILE, best_frames, fps=30)
                print(f"Best GIF saved: {self.GIF_FILE}")
        
        # env.close() - because of manual stop
    
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

if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        # dql.run(is_training=False, render=True, record_frames=False)
        dql.run(is_training=False, render=True, record_frames=True)
