# %%
from tank_kills_v3 import TankKills
import random
from pynput.keyboard import Controller,Key
import time
import pygame
import matplotlib.pyplot as plt
import numpy as np
import json


# %%
ins = [
    Key.up,
    Key.right,
    Key.down,
    Key.left
    ]

# %%
all_actions = ["up","right","down","left"]

# %%
keyboard = Controller()

# %%
q_table = np.zeros((600*600,4))
q_table

# %%
# q_table[300*400]

# %%
num_episodes = 3
# max_steps_per_episode = 5
learning_rate = 0.1 # alpha
discount_factor = 0.90 # gamma

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.007

# %%
rewards_all_episodes = []
all_actions_taken = []
no_of_actions_each_episode = []
for episode in range(num_episodes):
    print(f"-------Game: {episode}/{num_episodes}-------")
    env = TankKills(600,600)
    running = True
    reward_current_episode = 0
    state = 300*400
    time.sleep(0.4)
    reward = 0
    action_taken_in_episode = 0
    while running:
        pygame.display.update()
        reward = 0
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = random.randint(0,3)
        all_actions_taken.append(action)
        action_taken_in_episode += 1
        running,reward,score,pp,ep = env.play(action=all_actions[action])
        pygame.display.update()
        # keyboard.press(ins[action])
        # keyboard.release(ins[action])
        new_state = int(pp[0])*int(pp[1])
        q_table[state,action] = q_table[state,action] * (1-learning_rate) + learning_rate*(reward+discount_factor*np.max(q_table[new_state,:]))
        state = new_state
        reward_current_episode += reward
        if not running:
            pygame.display.quit()
            break
        
        exploration_rate = min_exploration_rate+(max_exploration_rate-min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    
    rewards_all_episodes.append(reward_current_episode)
    no_of_actions_each_episode.append(action_taken_in_episode)
    print(f"-- Score: {score}")
    print(f"-- Total Actions Taken: {action_taken_in_episode}")
    pygame.display.quit()

# %%
pygame.display.quit()

# %%
data = {}
data['q_table'] = q_table.tolist()
with open('q_table_3.json','w') as f:
    json.dump(data,f)


# %% [markdown]
# ### Q Table values of spawn location

# %%
print(q_table[300*400])

# %% [markdown]
# ### Rewards per game

# %%
plt.plot(rewards_all_episodes)
plt.show()

# %% [markdown]
# ### Number of actions each game

# %%
plt.plot(no_of_actions_each_episode)
plt.show()

# %% [markdown]
# ### Distribution of all actions

# %%
plt.hist(all_actions_taken)
plt.show()


