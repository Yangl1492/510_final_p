import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from collections import Counter
import random
import numpy as np
import json
import os
from wrapper import MJWrapper  # Mahjong 游戏的环境封装

# 调整后的 CNN 模型
class CNNMahjongDQN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNMahjongDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * input_shape[1] * input_shape[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
        )

    def __len__(self):
        return len(self.buffer)

# Mahjong 环境封装
class MahjongEnv:
    def __init__(self):
        self.wrapper = MJWrapper()

    def reset(self):
        self.wrapper.reset()
        return self.get_state()

    def step(self, action):
        _, _, rewards, is_done, _ = self.wrapper.step([action])
        next_state = self.get_state()
        reward = rewards[self.wrapper.get_current_player()]
        return next_state, reward, is_done

    def get_state(self):
        """
        获取当前玩家的手牌状态，转换为适合输入到模型的格式。
        """
        def encode_zone(cards):
            """
            将牌区域编码为 [0, 0, ..., 0] 的形式，总具编码16种不同的牌。
            """
            card_counts = {tile: 0 for tile in TILE_MAPPING.keys() if tile != -1}  # 只保留16种牌，忽略 -1 (Empty)
            for card in cards:
                card_value = card.value if hasattr(card, "value") else card
                if card_value in card_counts:
                    card_counts[card_value] += 1
            return list(card_counts.values())  # 返回长度16的列表

        current_obs = self.wrapper.get_current_obs()[0]
        hand_tiles = current_obs[0]
        hand_features = encode_zone(hand_tiles)

        # 转换为模型输入格式
        state = np.array(hand_features, dtype=np.float32).reshape((1, 16, 1))
        return state

# 动作映射
ACTION_MAPPING = {
    0: "Get Card", 1: "Hu",
    12: "Discard Character 1", 13: "Discard Character 2", 14: "Discard Character 3",
    15: "Discard Character 4", 16: "Discard Character 5", 17: "Discard Character 6",
    18: "Discard Character 7", 19: "Discard Character 8", 20: "Discard Character 9",
    30: "Discard Green", 31: "Discard Red", 32: "Discard White",
    33: "Discard East", 34: "Discard West", 35: "Discard North", 36: "Discard South",
    # 其他动作映射略
}

TILE_MAPPING = {
    9: "Character 1", 10: "Character 2", 11: "Character 3",
    12: "Character 4", 13: "Character 5", 14: "Character 6",
    15: "Character 7", 16: "Character 8", 17: "Character 9",
    27: "Green", 28: "Red", 29: "White", 30: "East", 31: "West",
    32: "North", 33: "South", -1: "Empty",
}

def select_action(legal_actions, hand):
    """
    根据优先级选择动作：
    胡 > 杠 > 吃 > 碰 > 打出不容易凑牌的牌。
    """
    # 动作优先级
    priorities = {
        "Hu": 0,
        "Gong": 1,
        "Concealed Gong": 1,
        "Add Gong": 1,
        "Chow": 2,
        "Pong": 3
    }

    # 按优先级选择特殊动作
    for action_name, priority in sorted(priorities.items(), key=lambda x: x[1]):
        for action_id in legal_actions:
            if action_name in ACTION_MAPPING.get(action_id, ""):
                return action_id

    # 如果没有特殊动作，按打牌规则选择
    return discard_least_promising_tile(legal_actions, hand)

def discard_least_promising_tile(legal_actions, hand):
    """
    从手牌中选择最不容易形成顺子或刻子的牌打出。
    """
    tile_count = Counter(hand)
    potential_tiles = set()

    for tile in hand:
        if tile < 0:  # 跳过空位
            continue
        # 考虑顺子可能（仅采取数字牌）
        if 9 <= tile <= 17:
            if (tile - 1 in tile_count) or (tile + 1 in tile_count):
                potential_tiles.add(tile)
            elif (tile - 2 in tile_count) or (tile + 2 in tile_count):
                potential_tiles.add(tile)
        # 考虑刻子可能
        if tile_count[tile] >= 2:
            potential_tiles.add(tile)

    # 从合法动作中找到可以打掉的牌
    for action_id in legal_actions:
        action_name = ACTION_MAPPING.get(action_id, "")
        if "Discard" in action_name:
            # 从描述中提取具体牌名
            target_tile_name = action_name.replace("Discard ", "")
            # 获取目标牌的ID
            target_tile_id = next((k for k, v in TILE_MAPPING.items() if v == target_tile_name), None)
            if target_tile_id is not None and target_tile_id not in potential_tiles:
                return action_id

    # 如果所有牌都可能形成顺子或刻子，则随机打牌
    # 从合法动作中选择一个打牌动作
    discard_actions = [action_id for action_id in legal_actions if "Discard" in ACTION_MAPPING.get(action_id, "")]
    if discard_actions:
        return random.choice(discard_actions)
    else:
        # 没有可供打出的牌，执行其他动作
        return random.choice(legal_actions)

# 加载经验数据

def load_experiences(file_path, replay_buffer):
    with open(file_path, "r") as f:
        data = json.load(f)
        for experience in data:
            hand = experience["state"]["hand"][:16]
            state = np.array(hand, dtype=np.float32).reshape((1, 16, 1))
            action = experience["action"]
            reward = experience.get("reward", 0)
            done = experience.get("done", False)

            if 0 <= action < 238:  # 确保动作编号在范围内
                replay_buffer.push(state, action, reward, done)

# 测试当前模型
def test_model(dqn, env, num_games=100):
    results = {"dqn_wins": 0, "priority_wins": 0, "ties": 0}
    total_rewards = 0
    for _ in range(num_games):
        state = env.reset()
        is_done = False
        steps = 0
        total_reward = 0
        while not is_done and steps < 200:
            steps += 1
            current_player = env.wrapper.get_current_player()
            hand = env.wrapper.get_current_obs()[0][0]
            if current_player == 0:  # DQN 玩家
                legal_actions = env.wrapper.get_legal_actions()
                if not legal_actions:
                    break
                state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float)
                q_values = dqn(state_tensor).detach().numpy().flatten()
                action = max(legal_actions, key=lambda a: q_values[a])
                state, reward, is_done = env.step(action)
                total_reward += reward

            else:  # 对手
                legal_actions = env.wrapper.get_legal_actions()
                action = select_action(legal_actions, hand)
                state, _, is_done = env.step(action)

        if is_done:
            _, winner_id = env.wrapper.get_game_status()
            if winner_id == 0:
                results["dqn_wins"] += 1
            elif winner_id == 1:
                results["priority_wins"] += 1
            else:
                results["ties"] += 1

        total_rewards += total_reward

    avg_reward = total_rewards / num_games
    print(f"Average reward across {num_games} games: {avg_reward}")
    print(f"Results: {results}")
    return avg_reward

# 训练 DQN
def train_dqn(dataset_path="mahjong_priority_dataset.json", model_path="dqn_mahjong_1.pth"):
    env = MahjongEnv()
    input_shape = (1, 16, 1)
    action_dim = 238  # 更新动作空间

    dqn = CNNMahjongDQN(input_shape, action_dim)
    target_dqn = CNNMahjongDQN(input_shape, action_dim)

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        pretrained_dict = torch.load(model_path)
        dqn.load_state_dict(pretrained_dict)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = optim.AdamW(dqn.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=100000)

    print(f"Loading experiences from {dataset_path}...")
    load_experiences(dataset_path, replay_buffer)

    batch_size = 64
    gamma = 0.99
    num_epochs = 1000
    best_avg_reward = -float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        if len(replay_buffer) < batch_size:
            print("Not enough experiences to train. Skipping...")
            break

        states, actions, rewards, dones = replay_buffer.sample(batch_size)
        q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q_values = rewards

        loss = nn.MSELoss()(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
            print(f"Epoch {epoch}: Loss = {loss.item()}")

        # Every 100 episodes, test the model and save the best one
        if (epoch + 1) % 100 == 0:
            avg_reward = test_model(dqn, env)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_model_state = dqn.state_dict()
                print(f"New best model found with average reward: {best_avg_reward}")

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
        print(f"Best model saved to {model_path} with average reward: {best_avg_reward}")

if __name__ == "__main__":
    train_dqn()
