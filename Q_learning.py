import numpy as np

# 1. Environment Setup
GRID_SIZE = 8
GOAL = (1, 4)
CLIFFS = [
    (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 1), (1, 7),
    (2, 0), (2, 1), (2, 3), (2, 4),
    (3, 3),
    (4, 5), (4, 6),
    (5, 3), (5, 5),
    (6, 1), (6, 3), (6, 5), (6, 7),
    (7, 5), (7, 7)
]

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 2. Q-Learning Parameters
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.1  # Exploration rate
episodes = 5000

# 3. Training Loop
for _ in range(episodes):
    # Random start that isn't a Cliff or Goal
    state = (np.random.randint(0, 8), np.random.randint(0, 8))
    while state in CLIFFS or state == GOAL:
        state = (np.random.randint(0, 8), np.random.randint(0, 8))

    done = False
    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action_idx = np.random.randint(0, 4)
        else:
            action_idx = np.argmax(q_table[state[0], state[1]])

        # Calculate next state
        move = ACTIONS[action_idx]
        next_state = (
            max(0, min(GRID_SIZE - 1, state[0] + move[0])),
            max(0, min(GRID_SIZE - 1, state[1] + move[1]))
        )

        # Assign Reward
        if next_state == GOAL:
            reward = 10
            done = True
        elif next_state in CLIFFS:
            reward = -100
            done = True
        else:
            reward = -1
            done = False

        # Update Q-value
        old_value = q_table[state[0], state[1], action_idx]
        next_max = np.max(q_table[next_state[0], next_state[1]])

        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state[0], state[1], action_idx] = new_value
        state = next_state


# 4. Pathfinding Function
def get_shortest_path(start):
    path = [start]
    curr = start
    steps = 0
    while curr != GOAL and steps < 50:
        action_idx = np.argmax(q_table[curr[0], curr[1]])
        move = ACTIONS[action_idx]
        curr = (max(0, min(7, curr[0] + move[0])), max(0, min(7, curr[1] + move[1])))
        path.append(curr)
        steps += 1
    return path


# Results for your specific coordinates
starts = [(1, 0), (7, 6), (4, 3)]
for s in starts:
    print(f"Shortest Path from {s}: {get_shortest_path(s)}")