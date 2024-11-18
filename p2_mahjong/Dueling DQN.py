import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from wrapper import MJWrapper


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DuelingDQN, self).__init__()

        # Feature extraction layer
        self.feature_fc1 = nn.Linear(input_dim, 512)
        self.feature_bn1 = nn.BatchNorm1d(512)
        self.feature_fc2 = nn.Linear(512, 256)
        self.feature_bn2 = nn.BatchNorm1d(256)

        # Value stream
        self.value_fc1 = nn.Linear(256, 128)
        self.value_bn1 = nn.BatchNorm1d(128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_bn2 = nn.BatchNorm1d(64)
        self.value_out = nn.Linear(64, 1)

        # Advantage stream
        self.advantage_fc1 = nn.Linear(256, 128)
        self.advantage_bn1 = nn.BatchNorm1d(128)
        self.advantage_fc2 = nn.Linear(128, 64)
        self.advantage_bn2 = nn.BatchNorm1d(64)
        self.advantage_out = nn.Linear(64, num_actions)

        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Feature extraction with conditional BatchNorm
        x = self.feature_fc1(x)
        x = self.feature_bn1(x) if batch_size > 1 else x
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.feature_fc2(x)
        x = self.feature_bn2(x) if batch_size > 1 else x
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Value stream
        value = self.value_fc1(x)
        value = self.value_bn1(value) if batch_size > 1 else value
        value = F.relu(value)
        value = self.value_fc2(value)
        value = self.value_bn2(value) if batch_size > 1 else value
        value = F.relu(value)
        value = self.value_out(value)

        # Advantage stream
        advantage = self.advantage_fc1(x)
        advantage = self.advantage_bn1(advantage) if batch_size > 1 else advantage
        advantage = F.relu(advantage)
        advantage = self.advantage_fc2(advantage)
        advantage = self.advantage_bn2(advantage) if batch_size > 1 else advantage
        advantage = F.relu(advantage)
        advantage = self.advantage_out(advantage)

        # Dueling Q-value combination
        temperature = 1.0
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True)) / temperature
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
        )

    def __len__(self):
        return len(self.buffer)


class MahjongEnv:
    def __init__(self):
        self.wrapper = MJWrapper()
        self.state = None

    def reset(self):
        self.state = self.wrapper.reset()
        return self.get_state()

    def step(self, action):
        cards, actions, rewards, is_done, legal_actions = self.wrapper.step([action])

        flat_cards = [item for sublist in cards for item in sublist]
        flat_actions = [item for sublist in actions for item in sublist]
        next_state = np.concatenate([np.array(flat_cards, dtype=np.float32),
                                     np.array(flat_actions, dtype=np.float32)])

        reward = rewards[self.wrapper.get_current_player()]
        return next_state, reward, is_done, legal_actions

    def get_state(self):
        cards, actions = self.wrapper.get_current_obs()
        flat_cards = [item for sublist in cards for item in sublist]
        flat_actions = [item for sublist in actions for item in sublist]
        state = np.concatenate([np.array(flat_cards, dtype=np.float32),
                                np.array(flat_actions, dtype=np.float32)])
        return state

    def get_legal_actions(self):
        return self.wrapper.get_legal_actions()


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = 0.001
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.eps = 1e-5  # Small constant to prevent zero probabilities

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)]
        else:
            probs = self.priorities

        # Calculate sampling probabilities
        probs = probs ** self.alpha
        probs = probs / probs.sum()

        # Sample indices and calculate importance weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Convert lists of numpy arrays to single numpy array for faster tensor creation
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
            indices,
            torch.tensor(weights, dtype=torch.float)
        )

    def update_priorities(self, indices, priorities):
        # Move priorities tensor to CPU before converting to numpy
        priorities = np.abs(priorities.cpu().detach().numpy()) + self.eps
        self.priorities[indices] = priorities

    def __len__(self):
        return len(self.buffer)

def train_dueling_dqn(continue_training=False, model_path="improved_dueling_dqn_mahjong.pth"):
    env = MahjongEnv()
    state_dim = len(env.get_state())
    action_dim = MJWrapper.N_ACTIONS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    policy_net = DuelingDQN(input_dim=state_dim, num_actions=action_dim).to(device)
    target_net = DuelingDQN(input_dim=state_dim, num_actions=action_dim).to(device)

    if continue_training and os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=1e-5)
    replay_buffer = PrioritizedReplayBuffer(capacity=100000)

    # Training parameters
    num_episodes = 1500
    batch_size = 256
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 5000  # Number of frames for epsilon decay
    target_update_freq = 10
    grad_clip = 5.0

    # Training loop improvements
    results = {"dqn_wins": 0, "random_wins": 0, "ties": 0}
    recent_rewards = deque(maxlen=100)
    best_win_rate = 0.0
    frames_done = 0

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        num_steps = 0

        while True:
            # Epsilon decay
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-frames_done / epsilon_decay)

            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            # Action selection with noisy exploration
            if random.random() < epsilon:
                action = random.choice(legal_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    legal_q_values = q_values[0][legal_actions]
                    action = legal_actions[legal_q_values.argmax().item()]

            # Take action and get next state
            next_state, reward, done, next_legal_actions = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            # Training step
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                weights = weights.to(device)

                # Double DQN target calculation
                with torch.no_grad():
                    next_q_values = policy_net(next_states)
                    next_actions = next_q_values.max(1)[1]
                    next_q_values_target = target_net(next_states)
                    next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1))
                    target_q_values = rewards.unsqueeze(1) + gamma * next_q_values * (1 - dones.unsqueeze(1))

                # Current Q-values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))

                # Compute loss with importance sampling weights
                td_errors = torch.abs(target_q_values - current_q_values)
                loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q_values, target_q_values,
                                                                reduction='none')).mean()

                # Update priorities
                replay_buffer.update_priorities(indices, td_errors.squeeze())

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
                optimizer.step()
                episode_loss += loss.item()

            state = next_state
            total_reward += reward
            frames_done += 1
            num_steps += 1

            if done:
                break

            # Random opponent move
            if env.wrapper.get_current_player() == 1:
                legal_actions = env.get_legal_actions()
                if legal_actions:
                    random_action = random.choice(legal_actions)
                    next_state, _, done, _ = env.step(random_action)
                    if done:
                        break

        # Update target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Update learning rate
        scheduler.step()

        # Track results
        is_game_over, winner_id = env.wrapper.get_game_status()
        if winner_id is None:
            results["ties"] += 1
        elif winner_id == 0:
            results["dqn_wins"] += 1
        else:
            results["random_wins"] += 1

        recent_rewards.append(total_reward)
        current_win_rate = results["dqn_wins"] / (episode + 1)

        # Save best model
        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            torch.save(policy_net.state_dict(), f"{model_path}_best")

        # Logging
        if episode % 100 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_loss = episode_loss / num_steps if num_steps > 0 else 0
            print(f"\nEpisode {episode}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Win Rate: {current_win_rate:.2%}")
            print(f"Best Win Rate: {best_win_rate:.2%}")
            print(f"Epsilon: {epsilon:.3f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    print("\nTraining Complete!")
    print(f"DQN Wins: {results['dqn_wins']}")
    print(f"Random Wins: {results['random_wins']}")
    print(f"Ties: {results['ties']}")
    print(f"Final Win Rate: {results['dqn_wins'] / num_episodes:.2%}")
    print(f"Best Win Rate: {best_win_rate:.2%}")

    torch.save(policy_net.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model saved to {model_path}_best")

    return policy_net



if __name__ == "__main__":
    train_dueling_dqn(continue_training=False)