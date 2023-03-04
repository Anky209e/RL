from tank_kills_v2 import TankKills
import random
from pynput.keyboard import Controller,Key
import time
# import keyboard

ins = [
    Key.up,
    Key.right,
    Key.down,
    Key.left
    ]

keyboard = Controller()
game_1 = TankKills(600,600)
running = True
while running:
    move = random.randint(0,3)
    running,score,pp,ep = game_1.play()
    keyboard.press(ins[move])
    keyboard.release(ins[move])
    print(running,score,pp,ep)



