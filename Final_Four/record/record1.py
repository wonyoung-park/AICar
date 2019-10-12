import pygame
import maestro
import pygame.camera
from pygame.locals import *
import numpy as np
import time
import csv
import sys
import pygame.display
import pdb

left.joystick

pygame.init()
pygame.joystick.init()
clock = pygame.time.Clock()
done = False
camEnable = 0
num = 1
DEVICE = '/dev/video1'
SIZE = (800, 600)
pygame.camera.init()
camera = pygame.camera.Camera(DEVICE, SIZE)
camera.start()
st_data = [[],[],[]]
f = open("/home/nvidia/AI4AV/Final_Four/DATA/"+sys.argv[1]+"/steering.csv",'a')

while done==False:
    pygame.event.get()
    joystick_count = pygame.joystick.get_count()    

    joystick = pygame.joystick.Joystick(0)
    joystick.init()   
    # Get the name from the OS for the controller/joystick
    name = joystick.get_name()        
    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    axes = joystick.get_numaxes()
    ste = joystick.get_axis(0)  
    ste2 = joystick.get_axis(2)
    
    if ste == 0:
        ste = ste2/5   
    thr = -joystick.get_axis(1)
    st_data[0] = int(ste*2000+6000) # set the angle btwen 4000 - 8000
    st_data[1] = int(thr*2000/2+6000)  #set the speed  

    start_r = joystick.get_button(0)    #A
    stop_r = joystick.get_button(1) #B
    done = joystick.get_button(3)   #X
    command = joystick.get_button(9)
    

    servo = maestro.Controller()
    servo.setAccel(0,35)    #set servo 0 acceleration to 35
    servo.setAccel(1,0)      #set servo 1 acceleration to 0
    servo.setTarget(0, st_data[0]) #set the steering angle
    servo.setTarget(1, st_data[1]) #set the speed
    
    if camEnable == 0:
        if start_r == 1:
            camEnable = 1
    else:
        if stop_r == 1:
            camEnable = 0

    # recording
    if camEnable == 1:
        tic = time.clock()
        cam_img = camera.get_image()
        cam_img = pygame.transform.scale(cam_img,(160,120))
        name = '/home/nvidia/AI4AV/Final_Four/DATA/'+sys.argv[1]+'/img' + str('{:07}'.format(num))+ '.jpg'
        st_data[2] = name 
        pygame.image.save(cam_img,name)
        writer = csv.writer(f)
        writer.writerow(st_data)
        num = num + 1       
        toc = time.clock()
        fps = 1/(toc - tic)
        print("FPS: ", fps)
            
f.close()
servo.setTarget(0, 6000) #set servo to move to center position
servo.setTarget(1, 6000) #set servo to move to center position          
servo.close

