import pygame
from pygame.constants import KEYDOWN, K_DOWN, K_LEFT, K_RIGHT, K_UP
from pygame import mixer
import random
import math
from PIL import Image

class TankKills:
    def __init__(self,screen_width,screen_height):
        pygame.init()
        # Display Settings
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.score_value = 0

        self.hit_reward = 0
        # Player Spawn cords
        self.player_x = 300 
        self.player_y = 400
        # Enemy spawn cords
        self.enemy_x = random.randint(0,585)
        self.enemy_y = random.randint(80,100)
        # Enemy speed
        self.enemy_xchange = 0.3
        self.enemy_ychange = 0.2

        # Set tank movement speed similar to enemy
        self.tank_speed = 0.3
        self.playerx_change = 0
        self.playery_change = 0
        # Images and fonts
        self.enemy_image = pygame.image.load("./assets/90_soldier.png")
        self.player_image = pygame.image.load("./assets/90_tank.png")
        self.background_image = pygame.image.load("./assets/grass.jpg")
        self.font = pygame.font.Font('freesansbold.ttf',32)
        self.screen = pygame.display.set_mode((screen_width,screen_height))
        self.running = True
        pygame.display.set_caption("tank_kills_v2")

    # Updates Enemy position on screen
    def enemy(self,x,y):
        self.screen.blit(self.enemy_image,(x,y))
    
    # Updates Player Position on screen
    def player(self,x,y):
        self.screen.blit(self.player_image,(x,y))

    # Checks if player smashed enemy
    def isCollision(self):
        distance = math.sqrt(math.pow(self.enemy_x-self.player_x,2)+math.pow(self.enemy_y-self.player_y,2))
        if distance <=60:
            return True
        else:
            return False
        
    # gameover screen update
    def gameOver(self):
        game_overfnt = self.font.render("GAME OVER",True,(8,8,239))
        self.screen.blit(game_overfnt,(200,150))
        self.score_value = 0
    
    # Updates score on screen
    def ShowScore(self,x,y):
        score = self.font.render("Score: "+str(self.score_value),True,(248,3,3))
        self.screen.blit(score,(x,y))

    # Main play function
    def play(self, action):
        textX = 10
        textY = 10
        distance = round(math.sqrt(math.pow(self.enemy_x-self.player_x,2)+math.pow(self.enemy_y-self.player_y,2))/600)
        self.hit_reward = - distance 
        self.screen.blit(self.background_image,(0,0))
        '''
        up:0
        right:1
        down:2
        left:3
        '''
        # Reset movement before applying new action
        self.playerx_change = 0
        self.playery_change = 0

        if action == 0:  # up
            self.playery_change = -self.tank_speed
        elif action == 1:  # right
            self.playerx_change = self.tank_speed
        elif action == 2:  # down
            self.playery_change = self.tank_speed
        elif action == 3:  # left
            self.playerx_change = -self.tank_speed

        # Update player position
        self.player_x += self.playerx_change
        self.player_y += self.playery_change

        # Optionally, clamp player position to screen bounds
        self.player_x = max(0, min(self.player_x, self.screen_width - self.player_image.get_width()))
        self.player_y = max(0, min(self.player_y, self.screen_height - self.player_image.get_height()))

        if True:
            if action == "left":
                # print("KEY: LEFT")
                self.player_x -= 10
            if action == "right":
                # print("KEY: RIGHT")
                self.player_x += 10
            if action == "up":
                # print("KEY: UP")
                self.player_y -= 10
            if action == "down":
                # print("KEY: DOWN")
                self.player_y += 10
        if True:
            if action == "left":
                self.playerx_change = 0
                
            if action == "right":
                self.playerx_change = 0
            
            if action == "up":
                self.playery_change = 0

            if action == "down":
                self.playery_change = 0

        # Screen Boundries
        self.player_x += self.playerx_change
        self.player_y += self.playery_change

        if self.player_x <=0:
            self.player_x = 1
        elif self.player_x >= self.screen_height - 15:
            self.player_x = self.screen_height - 20
        elif self.player_y <=0:
            self.player_y = 0
        elif self.player_y >=self.screen_width - 15:
            self.player_y = self.screen_width - 20
        
        self.enemy_x += self.enemy_xchange
        self.enemy_y += self.enemy_ychange
        if self.enemy_x <=0:
            self.enemy_xchange = 0.2
        elif self.enemy_x >= self.screen_height - 15:
            self.enemy_xchange = -0.2
        elif self.enemy_y <=0:
            self.enemy_ychange = 0.2
        elif self.enemy_y >=self.screen_height - 15:
            self.enemy_ychange = -0.2
        
        self.screen.blit(self.player_image,(self.player_x,self.player_y))
        self.screen.blit(self.enemy_image,(self.enemy_x,self.enemy_y))

        # self.enemy(self.enemy_x,self.enemy_y)
        # self.player(self.player_x,self.player_y)
        collision = self.isCollision()

        if collision:
            self.score_value += 1
            self.hit_reward = 50
            self.enemy_x = random.randint(0,580)
            self.enemy_y = 100
            self.player_x = 300
            self.player_y = 400
        
        # If enemy crosses border
        if self.enemy_y >=self.screen_height - 100:
            self.hit_reward = - 30
            return False,self.hit_reward,self.score_value,[self.player_x,self.player_y],[self.enemy_x,self.enemy_y]
            # self.gameOver()
    

        self.ShowScore(textX,textY)
        pygame.display.update()
        
        return self.running,self.hit_reward,self.score_value,[self.player_x,self.player_y],[self.enemy_x,self.enemy_y]


if __name__ == "__main__":
    env = TankKills(600,600)
    running = True
    moves = ["up","right","left","down"]
    while running:
        move = random.randint(0,3)
        running,reward,score,pp,ep = env.play(action=moves[move])
        # keyboard.press(ins[move])
        # keyboard.release(ins[move])
        print(running,reward,score,pp,ep)
        # moves.append(move)
    if not running:
        pygame.display.quit()