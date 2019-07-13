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


class Ball:
    def __init__(self, frame_size, ball_size, hue):
        self.frame_size = frame_size
        self.size = ball_size
        self.x = random.randrange(0, self.frame_size-self.size)
        self.y = random.randrange(16, self.frame_size-self.size)
        self.speedx = random.randrange(1, 5)
        self.speedy = random.randrange(1, 5)
        self.hue = hue

    def update(self):
        pos_x = self.x + self.speedx
        pos_y = self.y + self.speedy
        if pos_x >= self.frame_size:
            pos_x = 2*self.frame_size - pos_x
            self.speedx *= -1
        if pos_x <= 0 :
            pos_x *= -1
            self.speedx *= -1     

        if pos_y >= self.frame_size:
            pos_y = 2*self.frame_size - pos_y
            self.speedy *= -1
        if pos_y <= 0 :
            pos_y *= -1
            self.speedy *= -1  

        self.x = pos_x
        self.y = pos_y

def stump_ball(frame, ball, color):
    xA = max(ball.x, 0)
    yA = max(ball.y, 0)
    xB = min(ball.x+ball.size, frame.shape[1])
    yB = min(ball.y+ball.size, frame.shape[0])

    if xA > xB or yA > yB:
        return
    cv2.rectangle(frame, (xA, yA),(xB, yB), color, cv2.FILLED)

def create_video(frame_size, num_frames, max_ball_size, max_balls):
    num_balls = np.random.randint(1,max_balls)
    background_hue = np.random.randint(0,128)
    ball_list = []
    for i in range(num_balls):
        ball_hue = np.random.randint(background_hue + 64, 255)
        size = random.randrange(2, max_ball_size)
        ball_list.append(Ball(frame_size, size,  ball_hue))

    all_frames = []
    # writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (frame_size, frame_size), False)
    for i in range(num_frames):
        frame = np.ones((frame_size, frame_size)).astype('uint8')*background_hue
        for ball in ball_list:
            stump_ball(frame, ball, ball.hue)
            ball.update()
        frame += np.random.uniform(-5, 5, size=frame.shape).astype(np.uint8)
        frame = frame.clip(0,255)
        # writer.write(frame) 
        all_frames += [frame]
    return np.array(all_frames), num_balls

class bouncing_balls_dataset(torch.utils.data.Dataset):
  def __init__(self, frame_size=128, video_length=32, num_videos=1000, max_balls=20):
    self.frame_size = frame_size
    self.video_length = video_length
    self.num_videos = num_videos
    self.max_balls = max_balls
    print("# Creating videos")
    self.labels = []
    self.videos = []
    for i in range(self.num_videos):
      video, n_balls = create_video(self.frame_size, self.video_length, max_ball_size=int(frame_size/8), max_balls=self.max_balls)
      self.videos += [video]
      self.labels += [n_balls]

    print("\t num videos: %d"%len(self.videos))

  def __repr__(self):
       return "BB_vid"
  def __str__(self):
      return self.__repr__()

  def __len__(self):
    return len(self.videos)

  def __getitem__(self, index):
    return self.videos[index].astype(np.float32), self.labels[index]
      



