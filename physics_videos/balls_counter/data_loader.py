import os
from PIL import Image

import torch
from torchvision import datasets, transforms

import abc
import itertools
import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt
import cv2
import random
import itertools
import math
from tqdm import tqdm
from train import write_array_as_video

class Ball:
    def __init__(self, frame_size, ball_size, shape, hue, size_growth=0,max_ball_size=32, no_wall=False):
        self.size_growth = size_growth
        self.max_size = max_ball_size
        self.frame_size = frame_size
        self.size = ball_size
        self.x = random.randrange(0, self.frame_size-self.size)
        self.y = random.randrange(16, self.frame_size-self.size)
        self.speedx = random.randrange(1, 5)
        self.speedy = random.randrange(1, 5)
        self.hue = hue
        self.no_wall = no_wall
        self.shape  = shape

    def update(self):
        if self.size + self.size_growth > self.max_size:
            self.size = 2*self.max_size - self.size - self.size_growth
            self.size_growth *= -1
        elif self.size + self.size_growth < 0 : 
            self.size = -1*(self.size + self.size_growth)
            self.size_growth *= -1
        else:
            self.size += self.size_growth


        pos_x = self.x + self.speedx
        pos_y = self.y + self.speedy
        if not self.no_wall:
          if pos_x + self.size > self.frame_size:
              pos_x =2*self.frame_size - self.x - self.speedx - 2*self.size 
              self.speedx *= -1
          if pos_x < 0 :
              pos_x *= -1
              self.speedx *= -1     

          if pos_y + self.size > self.frame_size:
              pos_y = 2*self.frame_size - self.y - self.speedy - 2*self.size 
              self.speedy *= -1
          if pos_y < 0 :
              pos_y *= -1
              self.speedy *= -1  

        self.x = pos_x
        self.y = pos_y

def stump_ball(frame, ball, color):
    if ball.size <= 0 :
       return
    xA = max(ball.x, 0)
    yA = max(ball.y, 0)
    xB = min(ball.x+ball.size, frame.shape[1])
    yB = min(ball.y+ball.size, frame.shape[0])

    if xA > xB or yA > yB:
        return
    if ball.shape == 0:
      cv2.rectangle(frame, (xA, yA),(xB, yB), color, cv2.FILLED)
    elif ball.shape == 1:
      cv2.circle(frame,(int((xA+xB)/2), int((yA+yB)/2)), int(ball.size/2), color, cv2.FILLED)

    elif ball.shape == 2:
      cv2.rectangle(frame, (int((xA+xB)/2), yA),(int((xA+xB)/2), yB), color, cv2.FILLED)
      triangle_cnt = np.array( [(int((xA+xB)/2), yA), int((xA, (yA+yB)/2)), (xB, int((yA+yB)/2))] )
      cv2.drawContours(frame, [triangle_cnt], 0, color, -1)

def create_video(frame_size, num_frames, max_ball_size):
    num_balls = 1 # np.random.randint(1,max_balls)
    background_hue = np.random.randint(0,128)
    ball_list = []
    change_size = np.random.randint(0,2)
    for i in range(num_balls):
        ball_shape = random.randrange(0, 2)
        ball_hue = np.random.randint(background_hue+72, 255)
        size = random.randrange(3, max_ball_size)
        if change_size:
          ball_list.append(Ball(frame_size, size, ball_shape, ball_hue, size_growth=2,max_ball_size=max_ball_size, no_wall=False))
        else:
          ball_list.append(Ball(frame_size, size, ball_shape, ball_hue, size_growth=0,max_ball_size=max_ball_size, no_wall=False))

    all_frames = []
    # writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (frame_size, frame_size), False)
    for i in range(num_frames):
        frame = np.ones((frame_size, frame_size)).astype('uint8')*background_hue
        for ball in ball_list:
            stump_ball(frame, ball, ball.hue)
            ball.update()
        frame += np.random.uniform(-2, 2, size=frame.shape).astype(np.uint8)
        frame = frame.clip(0,255)
        # writer.write(frame) 
        all_frames += [frame]
    return np.array(all_frames), change_size


class bouncing_balls_dataset(torch.utils.data.Dataset):
  def __init__(self, frame_size=128, video_length=32, num_videos=1000):
    self.frame_size = frame_size
    self.video_length = video_length
    self.num_videos = num_videos
    print("# Creating videos")
    self.labels = []
    self.videos = []
    for i in tqdm(range(self.num_videos)):
      video, label = create_video(self.frame_size, self.video_length, max_ball_size=int(frame_size/2))
      self.videos += [video]
      self.labels += [label]

    for i in range(5):
      write_array_as_video(self.videos[i].astype(np.uint8), os.path.join("train_debug", "DL_series_%d_%d.avi"%(i, self.labels[i])))

    print("\t num videos: %d"%len(self.videos))

  def __repr__(self):
       return "BB_vid"
  def __str__(self):
      return self.__repr__()

  def __len__(self):
    return len(self.videos)

  def __getitem__(self, index):
    return self.videos[index].astype(np.float32), self.labels[index]