import numpy as np

# define the environment
n_states = 16  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_state = 15  # Goal state

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# define parameters
learning_rate = 0.85
discount_factor = 0.96
exploration_prob = 0.2
epochs = 1000

# Q-learning algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)  # Start from a random state

    while current_state != goal_state:
        # Choose action with epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # Simulate the environment (move to the next state)
        # For simplicity, move to the next state
        next_state = (current_state + 1) % n_states

        # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
        reward = 1 if next_state == goal_state else 0

        # Update Q-value using the Q-learning update rule
        Q_table[current_state, action] += learning_rate * \
            (reward + discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action])

        current_state = next_state  # Move to the next state

# After training, the Q-table represents the learned Q-values
print("Learned Q-table:")
print(Q_table)

# --- Prediction Code ---

def predict_optimal_action(state, q_table):
    """Predicts the optimal action for a given state based on the Q-table."""
    if 0 <= state < q_table.shape[0]:
        optimal_action = np.argmax(q_table[state])
        return optimal_action
    else:
        return "Invalid state"

def get_action_name(action_index):
    """Maps action index to a human-readable name."""
    if action_index == 0:
        return "Up"
    elif action_index == 1:
        return "Down"
    elif action_index == 2:
        return "Left"
    elif action_index == 3:
        return "Right"
    else:
        return "Unknown Action"

def predict_path_to_goal(start_state, q_table, max_steps=20):
    """Predicts a possible path from a start state to the goal state using the learned policy."""
    current_state = start_state
    path = [current_state]
    steps = 0
    while current_state != goal_state and steps < max_steps:
        optimal_action_index = predict_optimal_action(current_state, q_table)
        if isinstance(optimal_action_index, str):  # Handle invalid state
            return "Invalid start state"
        current_state = (current_state + 1) % n_states  # Assuming the environment dynamics
        path.append(current_state)
        steps += 1
    if current_state == goal_state:
        return path
    else:
        return "Could not reach goal within max steps"

# Example Predictions:

# Predict the optimal action for a specific state (e.g., state 5)
state_to_predict = 5
optimal_action_index = predict_optimal_action(state_to_predict, Q_table)
optimal_action_name = get_action_name(optimal_action_index)
print(f"\nPredicted optimal action for state {state_to_predict}: {optimal_action_name} (Action Index: {optimal_action_index})")

# Predict the optimal action for another state (e.g., state 12)
state_to_predict = 12
optimal_action_index = predict_optimal_action(state_to_predict, Q_table)
optimal_action_name = get_action_name(optimal_action_index)
print(f"Predicted optimal action for state {state_to_predict}: {optimal_action_name} (Action Index: {optimal_action_index})")

# Predict a possible path from a starting state to the goal state
start_state = 2
predicted_path = predict_path_to_goal(start_state, Q_table)
print(f"\nPredicted path from state {start_state} to goal ({goal_state}): {predicted_path}")

start_state = 10
predicted_path = predict_path_to_goal(start_state, Q_table)
print(f"Predicted path from state {start_state} to goal ({goal_state}): {predicted_path}")