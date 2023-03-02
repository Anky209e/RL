import pygame
from pygame.constants import KEYDOWN, K_DOWN, K_LEFT, K_RIGHT, K_UP
from pygame import mixer
import random
import math
from PIL import Image


class TankKills:
    def __init__(self,screen_width,screen_height):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.score_value = 0
        self.enemy_image = pygame.image.load("tank_kills/assets/90_soldier.png")
        self.player_image = pygame.image.load("tank_kills/assets/90_tank.png")
        self.background_image = pygame.image.load("tank_kills/assets/grass.jpg")
        self.font = pygame.font.Font('freesansbold.ttf',32)
        self.screen = pygame.display.set_mode((screen_width,screen_height))
        self.running = True

    # Updates Enemy position on screen
    def enemy(self,x,y):
        self.screen.blit(self.enemy_image,(x,y))
    
    # Updates Player Position on screen
    def player(self,x,y):
        self.screen.blit(self.player_image,(x,y))

    # Checks if player smashed enemy
    def isCollision(self,enemy_x,enemy_y,player_x,player_y):
        distance = math.sqrt(math.pow(enemy_x-player_x,2)+math.pow(enemy_y-player_y,2))
        if distance <=random.randint(49,59):
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
    def play(self):
        enemy_x = random.randint(0,585)
        enemy_y = 100
        enemy_xchange = 0.3
        enemy_ychange = 0.2
        
        pygame.display.set_caption("tank_kills")
        player_x = 300
        player_y = 200
        playerx_change = 0
        textX = 10
        textY = 10

        while self.running:
            
            for event in pygame.event.get():
                #--------keybinds------------
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == K_LEFT:
                        playerx_change = -0.5
                    if event.key == K_RIGHT:
                        playerx_change = 0.5
                    if event.key == K_UP:
                        player_y -= random.randint(3,10)
                    if event.key == K_DOWN:
                        player_y += random.randint(3,10)

                if event.type == pygame.KEYUP:
                    if event.key == K_LEFT:
                        playerx_change = 0
                        
                    if event.key == K_RIGHT:
                        playerx_change = 0
                    

            self.screen.blit(self.background_image,(0,0))
            #------boundries------
            player_x += playerx_change
            if player_x <=0:
                player_x = 1
            elif player_x >= self.screen_height - 15:
                player_x = self.screen_height - 20
            elif player_y <=0:
                player_y = 0
            elif player_y >=self.screen_width - 15:
                player_y = self.screen_width - 20
            #-----------------------
            enemy_x += enemy_xchange
            enemy_y += enemy_ychange
            if enemy_x <=0:
                enemy_xchange = 0.2
            elif enemy_x >= self.screen_height - 15:
                enemy_xchange = -0.2
            elif enemy_y <=0:
                enemy_ychange = 0.2
            elif enemy_y >=self.screen_height - 15:
                enemy_ychange = -0.2
            
            
            self.enemy(enemy_x,enemy_y)
            self.player(player_x,player_y)
            collision = self.isCollision(enemy_x,enemy_y,player_x,player_y)
            if collision:

                self.score_value +=1
                enemy_x = random.randint(0,580)
                enemy_y = 100
                player_x = 300
                player_y = 200
            if enemy_y >=self.screen_height - 100:
                self.gameOver()


            self.ShowScore(textX,textY)
            pygame.display.update()






if __name__=="__main__":
    inst1 = TankKills(300,300).play()

