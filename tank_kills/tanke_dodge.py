import pygame
import random
import math

class TankDodge:
    def __init__(self, screen_width=600, screen_height=600, num_enemies=3, headless=False):
        import os
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tank_speed = 2
        self.enemy_speed = 1.2
        self.num_enemies = num_enemies
        self.headless = headless

        if not headless:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Tank Dodge")
        else:
            self.screen = None

        # Colors
        self.bg_color = (50, 168, 82)      # Green background
        self.tank_color = (0, 0, 255)      # Blue tank
        self.enemy_color = (255, 0, 0)     # Red enemy
        self.score_color = (0, 0, 0)       # Black text

        # Shapes
        self.tank_size = 32
        self.enemy_size = 32

        self.font = pygame.font.Font('freesansbold.ttf', 32)
        self.enemy_dirs = []  # <-- Add this line
        self.reset()

    def reset(self):
        self.tank_x = self.screen_width // 2
        self.tank_y = self.screen_height - 80
        self.enemies = []
        self.enemy_dirs = []  # <-- Add this line
        for _ in range(self.num_enemies):
            ex = random.randint(0, self.screen_width - self.enemy_size)
            ey = random.randint(0, self.screen_height // 2)
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle)
            dy = math.sin(angle)
            self.enemies.append([ex, ey])
            self.enemy_dirs.append([dx, dy])
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        # Player position (normalized)
        state = [self.tank_x / self.screen_width, self.tank_y / self.screen_height]
        # Enemy positions (normalized)
        for ex, ey in self.enemies:
            state.append(ex / self.screen_width)
            state.append(ey / self.screen_height)
        # Relative positions (normalized)
        for ex, ey in self.enemies:
            state.append((ex - self.tank_x) / self.screen_width)
            state.append((ey - self.tank_y) / self.screen_height)
        return state

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        base_speed = 1.5
        self.enemy_speed = base_speed + 0.03 * (self.score // 30)

        old_tank_x, old_tank_y = self.tank_x, self.tank_y

        # Move tank
        if action == 0:  # up
            self.tank_y -= self.tank_speed
        elif action == 1:  # right
            self.tank_x += self.tank_speed
        elif action == 2:  # down
            self.tank_y += self.tank_speed
        elif action == 3:  # left
            self.tank_x -= self.tank_speed

        # Clamp tank position
        new_tank_x = max(0, min(self.tank_x, self.screen_width - self.tank_size))
        new_tank_y = max(0, min(self.tank_y, self.screen_height - self.tank_size))
        hit_wall = (self.tank_x != new_tank_x) or (self.tank_y != new_tank_y)
        self.tank_x = new_tank_x
        self.tank_y = new_tank_y

        # Move enemies TOWARD the tank (homing)
        for i, enemy in enumerate(self.enemies):
            dx = self.tank_x - enemy[0]
            dy = self.tank_y - enemy[1]
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                dx /= dist
                dy /= dist
            else:
                dx, dy = 0, 0
            enemy[0] += dx * self.enemy_speed
            enemy[1] += dy * self.enemy_speed
            enemy[0] = max(0, min(enemy[0], self.screen_width - self.enemy_size))
            enemy[1] = max(0, min(enemy[1], self.screen_height - self.enemy_size))

        # --- Updated Reward System ---
        reward = 0.1  # positive reward for surviving a step

        # Encourage movement: penalize standing still
        if (old_tank_x == self.tank_x) and (old_tank_y == self.tank_y):
            reward -= 0.05  # penalty for not moving

        # Penalize hitting the wall
        if hit_wall:
            reward -= 0.2  # penalty for trying to move into wall

        # Encourage dodging: reward for increasing distance from nearest enemy
        min_dist_before = min(
            math.hypot(old_tank_x - ex, old_tank_y - ey) for ex, ey in self.enemies
        )
        min_dist_after = min(
            math.hypot(self.tank_x - ex, self.tank_y - ey) for ex, ey in self.enemies
        )
        if min_dist_after > min_dist_before:
            reward += 0.03  # reward for moving away from nearest enemy
        else:
            reward -= 0.01  # slight penalty for moving closer

        # Check collision
        for ex, ey in self.enemies:
            if self._is_collision(self.tank_x, self.tank_y, ex, ey):
                self.done = True
                reward = -1.0  # moderate penalty for dying
                return self.get_state(), reward, True, {}

        self.score += 1
        return self.get_state(), reward, False, {}

    def render(self):
        if self.headless:
            return  # Do not render in headless mode
        self.screen.fill(self.bg_color)
        # Draw tank as blue square
        pygame.draw.rect(self.screen, self.tank_color, (self.tank_x, self.tank_y, self.tank_size, self.tank_size))
        # Draw enemies as red squares
        for ex, ey in self.enemies:
            pygame.draw.rect(self.screen, self.enemy_color, (ex, ey, self.enemy_size, self.enemy_size))
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.score_color)
        self.screen.blit(score_text, (10, 10))
        pygame.display.update()

    def _is_collision(self, x1, y1, x2, y2):
        # Simple rectangle collision
        return (
            x1 < x2 + self.enemy_size and
            x1 + self.tank_size > x2 and
            y1 < y2 + self.enemy_size and
            y1 + self.tank_size > y2
        )

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = TankDodge()
    clock = pygame.time.Clock()
    state = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_DOWN]:
            action = 2
        elif keys[pygame.K_LEFT]:
            action = 3
        else:
            action = -1  # no move

        if action != -1:
            state, reward, done, _ = env.step(action)
            if done:
                print(f"Game Over! Score: {env.score}")
                state = env.reset()
        env.render()
        clock.tick(60)
    env.close()