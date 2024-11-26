import random
import json
from p2_mahjong.wrapper import MJWrapper as Wrapper
import os
from collections import Counter

ACTION_MAPPING = {
    0: "Get Card", 1: "Hu",
    12: "Discard Character 1", 13: "Discard Character 2", 14: "Discard Character 3",
    15: "Discard Character 4", 16: "Discard Character 5", 17: "Discard Character 6",
    18: "Discard Character 7", 19: "Discard Character 8", 20: "Discard Character 9",
    30: "Discard Green", 31: "Discard Red", 32: "Discard White",
    33: "Discard East", 34: "Discard West", 35: "Discard North", 36: "Discard South",
    46: "Pong Character 1", 47: "Pong Character 2", 48: "Pong Character 3",
    64: "Pong Green", 65: "Pong Red", 66: "Pong White",
    67: "Pong East", 68: "Pong West", 69: "Pong North", 70: "Pong South",
}

TILE_MAPPING = {
    9: "Character 1", 10: "Character 2", 11: "Character 3",
    12: "Character 4", 13: "Character 5", 14: "Character 6",
    15: "Character 7", 16: "Character 8", 17: "Character 9",
    27: "Green", 28: "Red", 29: "White", 30: "East", 31: "West",
    32: "North", 33: "South", -1: "Empty",
}

ENCODE_MAPPING ={
    9: 0, 10: 1, 11: 2,
    12: 3, 13: 4, 14: 5,
    15: 6, 16: 7, 17: 8,
    27: 9, 28: 10, 29: 11, 30: 12, 31: 13,
    32: 14, 33: 15,
}
def translate_action(action_id):
    return ACTION_MAPPING.get(action_id, f"Unknown Action (ID: {action_id})")

def select_action(legal_actions, hand):
    priorities = {
        "Hu": 0,
        "Gong": 1,
        "Concealed Gong": 1,
        "Add Gong": 1,
        "Chow": 2,
        "Pong": 3
    }
    for action_name, priority in sorted(priorities.items(), key=lambda x: x[1]):
        for action_id in legal_actions:
            if action_name in ACTION_MAPPING.get(action_id, ""):
                return action_id
    return discard_least_promising_tile(legal_actions, hand)

def discard_least_promising_tile(legal_actions, hand):
    tile_count = Counter(hand)
    potential_tiles = set()
    for tile in hand:
        if tile < 0:
            continue
        if 9 <= tile <= 17:
            if (tile - 1 in tile_count) or (tile + 1 in tile_count):
                potential_tiles.add(tile)
            elif (tile - 2 in tile_count) or (tile + 2 in tile_count):
                potential_tiles.add(tile)
        if tile_count[tile] >= 2:
            potential_tiles.add(tile)
    for action_id in legal_actions:
        action_name = ACTION_MAPPING.get(action_id, "")
        if "Discard" in action_name:
            target_tile_name = action_name.replace("Discard ", "")
            target_tile_id = next((k for k, v in TILE_MAPPING.items() if v == target_tile_name), None)
            if target_tile_id is not None and target_tile_id not in potential_tiles:
                return action_id
    discard_actions = [action_id for action_id in legal_actions if "Discard" in ACTION_MAPPING.get(action_id, "")]
    if discard_actions:
        return random.choice(discard_actions)
    return random.choice(legal_actions)

def encode_zone(cards):
    card_counts = [0] * 16  # Initialize counts for 16 possible values
    for card in cards:
        if card != -1:  # Ignore cards with value -1
            # Determine the mapping index from the card value or the card itself
            if hasattr(card, 'value'):
                card_value = card.value
            else:
                card_value = card
            
            if card_value in ENCODE_MAPPING:
                encode_index = ENCODE_MAPPING[card_value]
                card_counts[encode_index] += 1  # Increment the count at the encoded index

    return card_counts  # Return the counts

def save_game_state_data(wrapper, current_player, action, reward, game_data):
    current_obs = wrapper.get_current_obs()[0]
    hand = current_obs[0]
    melded = current_obs[1]
    discarded = current_obs[3]
    state = {
        "hand": encode_zone(hand),
        "melded": encode_zone(melded),
        "discarded": encode_zone(discarded),
    }
    game_data.append({"state": state, "action": int(action), "reward": float(reward)})

def load_existing_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            return []
    return []

if __name__ == "__main__":
    wrapper = Wrapper()
    dataset_filepath = "mahjong_priority_dataset.json"
    dataset = load_existing_data(dataset_filepath)

    for _ in range(10000):
        wrapper.reset()
        game_data = []
        is_game_over = False

        while not is_game_over:
            current_player = wrapper.get_current_player()
            legal_actions = wrapper.get_legal_actions()
            hand = wrapper.get_current_obs()[0][0]
            print(current_player,hand)
            action = select_action(legal_actions, hand)
            _, _, rewards, is_game_over, _ = wrapper.step([action])
            reward = rewards[current_player]
            save_game_state_data(wrapper, current_player, action, reward, game_data)
        dataset.extend(game_data)

    with open(dataset_filepath, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Game data updated and saved to {dataset_filepath}. Total records: {len(dataset)}")
