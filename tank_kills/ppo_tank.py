import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from tanke_dodge import TankDodge
from tqdm import trange, tqdm

# --- Hyperparameters ---
num_episodes = 20000
max_steps = 1000
update_timestep = 2048
gamma = 0.99
lam = 0.95
clip_epsilon = 0.2
ppo_epochs = 4
batch_size = 64
entropy_coef = 0.05
value_coef = 0.5
learning_rate = 0.0003
num_enemies = 3
print_freq = 50

print("Hyperparameters:")
print(f"num_episodes = {num_episodes}")
print(f"max_steps = {max_steps}")
print(f"update_timestep = {update_timestep}")
print(f"gamma = {gamma}")
print(f"lam = {lam}")
print(f"clip_epsilon = {clip_epsilon}")
print(f"ppo_epochs = {ppo_epochs}")
print(f"batch_size = {batch_size}")
print(f"entropy_coef = {entropy_coef}")
print(f"value_coef = {value_coef}")
print(f"learning_rate = {learning_rate}")
print(f"num_enemies = {num_enemies}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- PPO Actor-Critic Model ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# --- PPO Buffer ---
class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

# --- GAE Advantage Calculation ---
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0):
    advantages = []
    gae = 0
    values = values + [last_value]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

# --- PPO Update Step ---
def ppo_update(model, optimizer, buffer, last_value, clip_epsilon=0.2, epochs=4, batch_size=64, gamma=0.99, lam=0.95, entropy_coef=0.01, value_coef=0.5):
    states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=device)
    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32, device=device)
    rewards = buffer.rewards
    dones = buffer.dones
    values = buffer.values

    # Compute advantages and returns
    advantages = compute_gae(rewards, values, dones, gamma, lam, last_value)
    returns = torch.tensor(advantages, dtype=torch.float32, device=device) + torch.tensor(values, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = len(states)
    for _ in range(epochs):
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_logprobs = old_logprobs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            logits, values_pred = model(batch_states)
            dist = torch.distributions.Categorical(logits=logits)
            logprobs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(logprobs - batch_old_logprobs)
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (batch_returns - values_pred.squeeze()).pow(2).mean()
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# --- Environment and Model Setup ---
env = TankDodge(num_enemies=num_enemies, headless=True)
state_dim = 2 + 4 * env.num_enemies  # player pos + enemy pos + relative pos
action_dim = 4

model = ActorCritic(state_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

rewards_history = []
buffer = PPOBuffer()
timestep = 0
best_reward = float('-inf')
best_weights_path = "saved_weights/ppo_best_tank_dodge.pth"

# --- Live Plot Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10,5))
line1, = ax.plot([], [], label="Reward per Episode")
line2, = ax.plot([], [], label="5-Episode Moving Avg", color='orange')
line3, = ax.plot([], [], label="Cumulative Average", color='red', linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("Reward per Episode (PPO)")
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

# --- Training Loop ---
for episode in trange(num_episodes, desc="Training"):
    state = env.reset()
    ep_reward = 0
    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        logprob = dist.log_prob(torch.tensor(action, device=device)).item()

        next_state, reward, done, _ = env.step(action)
        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.logprobs.append(logprob)
        buffer.rewards.append(reward)
        buffer.dones.append(done)
        buffer.values.append(value.item())

        state = next_state
        ep_reward += reward
        timestep += 1

        if timestep % update_timestep == 0:
            with torch.no_grad():
                last_state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                _, last_value = model(last_state_tensor)
            ppo_update(model, optimizer, buffer, last_value=last_value.item(), gamma=gamma, lam=lam,
                       clip_epsilon=clip_epsilon, epochs=ppo_epochs, batch_size=batch_size,
                       entropy_coef=entropy_coef, value_coef=value_coef)
            buffer.clear()

        if done:
            break

    rewards_history.append(ep_reward)

    # Save best weights
    if ep_reward > best_reward:
        best_reward = ep_reward
        torch.save(model.state_dict(), best_weights_path)

    # Print progress
    if (episode+1) % print_freq == 0 or episode == 0:
        tqdm.write(f"Episode {episode+1}/{num_episodes} | Reward: {ep_reward:.2f} | Best: {best_reward:.2f}")

    # Live plot update
    if (episode+1) % 10 == 0 or episode == 0:
        update_plot(rewards_history)

# --- Save Last Model ---
torch.save(model.state_dict(), "saved_weights/ppo_tank_dodge.pth")

# --- Final Plot ---
plt.ioff()
update_plot(rewards_history)
plt.show()
input("Training complete. Press Enter to exit...")