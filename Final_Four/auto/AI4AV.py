import pygame
import maestro
import pygame.camera
from pygame.locals import *
import numpy as np
import time
import csv
import sys
import pygame.display
import torch 
import torch.nn as nn
import pdb
from torch.autograd import Variable
from PIL import Image
import glob
import torch.nn.functional as F
import torch._utils


pygame.init()
pygame.joystick.init()
clock = pygame.time.Clock()
done = False
camEnable = 0
num = 1
DEVICE = '/dev/video1'
SIZE = (800, 600)
#display = pygame.display.set_mode(SIZE)
pygame.camera.init()
camera = pygame.camera.Camera(DEVICE, SIZE)
camera.start()
st_data=[[],[]]

#######copy and paste the work you designed#######

class ConvNet(nn.Module):
  def __init__(self, num_classes):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 9, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(9, 27, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(27, 72, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer4 = nn.Sequential(
      nn.Conv2d(72, 140, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer5 = nn.Sequential(
       nn.Conv2d(140,240, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride=2))


    self.fc1 = nn.Sequential(
        nn.Linear(1*4*240, 500),
        nn.ReLU())
    self.fc7 = nn.Sequential(
        nn.Linear(500, 200),
        nn.ReLU())
    self.fc6 = nn.Sequential(
        nn.Linear(200, 100),
        nn.ReLU())
    self.fc2 = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU())
    self.fc3 = nn.Sequential(
        nn.Linear(50, 10),
        nn.ReLU())
    self.fc4 = nn.Sequential(
        nn.Linear(10, num_classes))


  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc7(out)
    out = self.fc6(out)
    out = self.fc2(out)
    out = self.fc3(out)
    out = self.fc4(out)

    return out
  
  








###Initiate the model object using the class we've already defined####
num_classes = 1
model = ConvNet(num_classes)


#########load the model you trained##################
model.load_state_dict(torch.load("/home/nvidia/AI4AV/Final_Four/auto/model0.ckpt"))

############### Device configuration#################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###########Move the model object to the Device#######

model = model.to(device)
#####################################################
model.eval()

num=0
while done==False:
    pygame.event.get()
    joystick_count = pygame.joystick.get_count()    
    # For each joystick:
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

    st_data[0] = int(ste*2000+6000)
    st_data[1] = int(6300)   #thr*2000/#+6000

    start_r = joystick.get_button(0)
    stop_r = joystick.get_button(1)
    done = joystick.get_button(3)

    if camEnable == 0:
        if start_r == 1:
            camEnable = 1
    else:
        if stop_r == 1:
            camEnable = 0

    # automonous mode
    if camEnable == 1:
        tic = time.clock()
        cam_img = camera.get_image()
        cam_img = pygame.transform.scale(cam_img,(160,120))
        img = pygame.surfarray.pixels3d(cam_img)
        ##### Make proper transformations when you used the cropped images #####


        img = np.array(img)/255.0 #normalize the image
        img = img[:,60:120,:]
        #transform from numpy to Pytorch Tensor
        img = torch.from_numpy(img).float().permute(2,1,0).unsqueeze(0)
        ############ Move tensors to the configured device #################
        img = img.to(device)
       
        ############ Forward Pass #################
        
        ### Transfer the result back to numpy and store in the st_data list ###
        st_data[0] = int(model(img)[0].cpu().data.numpy()[0]*2000+6000)

        #######################################################################
        toc = time.clock()
        fps = 1/(toc - tic)
        num = num+1
        print("Steering: ", st_data[0])

    servo = maestro.Controller()
    servo.setAccel(0,40)      #set servo 0 acceleration to 35
    servo.setAccel(1,0)      #set servo 0 acceleration to 0
    servo.setTarget(0, st_data[0]) #set servo to move to center position
    servo.setTarget(1, st_data[1]) #set servo to move to center position
        

servo.setTarget(0, 6000) #set servo to move to center position
servo.setTarget(1, 6000) #set servo to move to center position          
servo.close

