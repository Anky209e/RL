import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt
from tanke_dodge import TankDodge
from tqdm import trange, tqdm

# --- Hyperparameters ---
num_episodes = 20000
learning_rate = 0.0001
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.005
exploration_decay_rate = 0.0001
replay_length = 5000
batch_size = 128
print_freq = 100

print("Hyperparameters:")
print(f"num_episodes = {num_episodes}")
print(f"learning_rate = {learning_rate}")
print(f"discount_factor = {discount_factor}")
print(f"exploration_rate = {exploration_rate}")
print(f"max_exploration_rate = {max_exploration_rate}")
print(f"min_exploration_rate = {min_exploration_rate}")
print(f"exploration_decay_rate = {exploration_decay_rate}")
print(f"replay_length = {replay_length}")
print(f"batch_size = {batch_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Model ---
class VfApproxModel(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.layer_1 = nn.Linear(state_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 4)  # 4 actions
        self.relu = nn.ReLU()
    def forward(self, features):
        out = self.relu(self.layer_1(features))
        out = self.relu(self.layer_2(out))
        out = self.relu(self.layer_3(out))
        out = self.layer_4(out)
        return out

# --- Agent ---
class Agent:
    def __init__(self, replay_length, learning_rate, epsilon, max_epsilon, min_epsilon, epsilon_decay, gamma, action_size, value_function):
        self.replay_memory = deque(maxlen=replay_length)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.action_size = action_size
        self.value_function = value_function
        self.loss_fn = nn.SmoothL1Loss().to(device)
        self.opt  = torch.optim.AdamW(value_function.parameters(), lr=learning_rate, amsgrad=True)

    def add_experience(self, new_state, reward, running, state, action):
        self.replay_memory.append((new_state, reward, running, state, action))
    
    def action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        out = self.value_function(state)
        out = out.cpu().detach().numpy()
        return np.argmax(out)
    
    def replay(self, batch_size):
        batch = random.sample(self.replay_memory, batch_size)
        losses = []
        for new_state, reward, running, state, action in batch:
            q_values = self.value_function(state)
            q_value = q_values[0, action]
            with torch.no_grad():
                next_q_values = self.value_function(new_state)
                max_next_q = torch.max(next_q_values)
                target = reward + self.gamma * max_next_q * float(running)
            loss = self.loss_fn(q_value, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        return np.mean(losses) if losses else 0.0

    def save_weights(self, path):
        torch.save(self.value_function.state_dict(), path)

# --- Environment and Model Setup ---
env = TankDodge(num_enemies=3, headless=True)
state_size = 2 + 4 * env.num_enemies  # 2 (player) + 2*N (enemy pos) + 2*N (relative pos)
value_function = VfApproxModel(state_size).to(device)
agent = Agent(
    replay_length=replay_length,
    learning_rate=learning_rate,
    epsilon=exploration_rate,
    max_epsilon=max_exploration_rate,
    min_epsilon=min_exploration_rate,
    epsilon_decay=exploration_decay_rate,
    gamma=discount_factor,
    action_size=4,
    value_function=value_function
)

# --- Live Plot Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10,5))
rewards_history = []
line1, = ax.plot([], [], label="Reward per Episode")
line2, = ax.plot([], [], label="5-Episode Moving Avg", color='orange')
line3, = ax.plot([], [], label="Cumulative Average", color='red', linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("Reward per Episode")
ax.grid()
ax.legend()

def update_plot(rewards_history):
    line1.set_data(range(len(rewards_history)), rewards_history)
    if len(rewards_history) >= 5:
        avg5 = np.convolve(rewards_history, np.ones(5)/5, mode='valid')
        line2.set_data(range(4, len(rewards_history)), avg5)
    else:
        line2.set_data([], [])
    global_mean = np.cumsum(rewards_history) / (np.arange(len(rewards_history)) + 1)
    line3.set_data(range(len(rewards_history)), global_mean)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

best_reward = float('-inf')
best_weights_path = "saved_weights/best_episode_weights.pth"

# --- Training Loop with Progress Bar ---
from tqdm import trange
losses_per_episode = []
for episode in trange(num_episodes, desc="Training"):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    running = True
    actions_taken_per_episode = 0
    rewards_per_episode = 0
    while running:
        action = agent.action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        agent.add_experience(next_state, reward, not done, state, action)
        state = next_state
        rewards_per_episode += reward
        actions_taken_per_episode += 1
        running = not done

    decay_episodes = int(num_episodes * 0.95)
    if episode < decay_episodes:
        agent.epsilon = agent.max_epsilon - (agent.max_epsilon - agent.min_epsilon) * (episode / decay_episodes)
    else:
        agent.epsilon = agent.min_epsilon

    if len(agent.replay_memory) > batch_size:
        replay_loss = agent.replay(batch_size)
    else:
        replay_loss = 0.0

    losses_per_episode.append(replay_loss)
    rewards_history.append(rewards_per_episode)

    # --- Save best weights logic ---
    if rewards_per_episode > best_reward:
        best_reward = rewards_per_episode
        agent.save_weights(best_weights_path)

    if (episode+1) % print_freq == 0 or episode == 0:
        tqdm.write(
            f"Episode {episode+1}/{num_episodes} | "
            f"Reward: {rewards_per_episode:.2f} | "
            f"Actions: {actions_taken_per_episode} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Replay Loss: {replay_loss:.4f} | "
            f"Best Reward: {best_reward:.2f}"
        )
    # Live plot update
    if (episode+1) % 10 == 0 or episode == 0:
        update_plot(rewards_history)

agent.save_weights("saved_weights/last_episode_weights.pth")

# --- Final Plot ---
plt.ioff()
update_plot(rewards_history)
plt.show()
input("Training complete. Press Enter to exit...")