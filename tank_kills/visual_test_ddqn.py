import torch
import pygame
from tanke_dodge import TankDodge

# --- Model definition (must match your training code) ---
import torch.nn as nn

class VfApproxModel(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=state_size, out_features=32)
        self.layer_2 = nn.Linear(in_features=32, out_features=32)
        self.layer_3 = nn.Linear(in_features=32, out_features=4)
        self.relu = nn.ReLU()
    def forward(self, features):
        out = self.relu(self.layer_1(features))
        out = self.relu(self.layer_2(out))
        out = self.layer_3(out)
        return out

# --- Load environment and model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TankDodge(num_enemies=3)
state_size = 2 + 2 * env.num_enemies
model = VfApproxModel(state_size)
model.load_state_dict(torch.load("tank_kills/saved_weights/double_dqn_last_episode_weights.pth", map_location=device))
model.eval()
model.to(device)

# --- Run the game with the trained agent ---
state = env.reset()
done = False
clock = pygame.time.Clock()
score = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Agent selects action greedily
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    state, reward, done_flag, _ = env.step(action)
    score += reward
    env.render()
    clock.tick(60)
    if done_flag:
        print(f"Game Over! Score: {env.score}")
        done = True
env.close()