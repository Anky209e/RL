import torch
import pygame
from tanke_dodge import TankDodge
import torch.nn as nn

# --- PPO Actor-Critic Model (must match training) ---
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

# --- Load environment and model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TankDodge(num_enemies=3)
state_dim = 2 + 4 * env.num_enemies  # player pos + enemy pos + relative pos
action_dim = 4

model = ActorCritic(state_dim, action_dim)
model.load_state_dict(torch.load("tank_kills/saved_weights/ppo_best_tank_dodge.pth", map_location=device))
model.eval()
model.to(device)

# --- Run the game with the trained PPO agent ---
state = env.reset()
done = False
clock = pygame.time.Clock()
score = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_tensor)
        action = torch.argmax(logits, dim=1).item()  # Greedy action

    state, reward, done_flag, _ = env.step(action)
    score += reward
    env.render()
    clock.tick(60)
    if done_flag:
        print(f"Game Over! Score: {env.score}")
        done = True

env.close()