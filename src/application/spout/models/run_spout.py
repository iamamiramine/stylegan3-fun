# load library
import SpoutGL
from OpenGL import GL
from itertools import islice, cycle
from random import randint
import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GLU import *

# def check(receiver):
#     """
#     Check on closed window
#     """
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             for i in range(0,1):
#                 receiver.ReleaseReceiver()
#             pygame.quit()
#             quit()

def randcolor():
    return randint(0, 255)

def main() :

    #window setup
    pygame.init() 
    pygame.display.set_caption( 'Spout For Python' )
    pygame.display.set_mode( (256, 256), DOUBLEBUF|OPENGL )

    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    glOrtho(0, 256, 256, 0, 1, -1 )
    glMatrixMode( GL_MODELVIEW )
    glDisable( GL_DEPTH_TEST )
    glClearColor( 0.0, 0.0, 0.0, 0.0)
    glEnable( GL_TEXTURE_2D )

    # create receiver
    sender = SpoutGL.SpoutSender()
    sender.setSenderName("output")
    # create sender
    # receiver = SpoutGL.SpoutReceiver()
    # receiver.setReceiverName("input")

    while True :
        buffer = None

        # receive data
        # result = receiver.receiveImage(buffer, GL.GL_RGBA, False, 0)
        # send data
        pixels = bytes(islice(cycle([randcolor(), randcolor(), randcolor(), 255]), 256 * 256 * 4))
        sender.sendImage(pixels, 256, 256, GL.GL_RGBA, False, 0)
        sender.setFrameSync("output")
        
        # Wait for next send attempt
        time.sleep(1./1)
    
if __name__ == "__main__":
    main()