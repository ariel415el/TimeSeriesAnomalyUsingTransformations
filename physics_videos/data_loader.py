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
    def __init__(self, frame_size, ball_size, hue, size_growth=0, no_wall=False):
        self.size_growth = size_growth
        self.frame_size = frame_size
        self.size = ball_size
        self.x = random.randrange(0, self.frame_size-self.size)
        self.y = random.randrange(16, self.frame_size-self.size)
        self.speedx = random.randrange(1, 5)
        self.speedy = random.randrange(1, 5)
        self.hue = hue
        self.no_wall = no_wall

    def update(self):
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
    cv2.rectangle(frame, (xA, yA),(xB, yB), color, cv2.FILLED)

def create_video(frame_size, num_frames, max_ball_size, max_balls, create_anomaly=False):
    num_balls = 1# np.random.randint(1,max_balls)
    background_hue = np.random.randint(0,128)
    ball_list = []
    for i in range(num_balls):
        ball_hue = np.random.randint(background_hue+72, 255)
        size = random.randrange(2, max_ball_size)
        if create_anomaly:
          ball_list.append(Ball(frame_size, size,  ball_hue, size_growth=-1, no_wall=False))
        else:
          ball_list.append(Ball(frame_size, size,  ball_hue, size_growth=0, no_wall=False))

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
    return np.array(all_frames)


class balls_dataset(torch.utils.data.Dataset):
  def __init__(self, frame_size=128, video_length=32, num_videos=1000, max_permutaions=1000, train=True, permutations=None):
    self.train = train
    self.frame_size = frame_size
    self.video_length = video_length
    self.num_videos = num_videos
    self.max_permutaions = max_permutaions
    print("# Creating videos")
    self.videos = []
    for i in tqdm(range(self.num_videos)):
      self.videos += [create_video(self.frame_size, self.video_length, max_ball_size=32, max_balls=1, create_anomaly=False)]

    for i in range(5):
      write_array_as_video(self.videos[i].astype(np.uint8), os.path.join("train_debug", "DL_series_%d.avi"%i))

    print("# Creating perms")
    if permutations is not None:
      self._permutations = permutations
    else:
      self._permutations = []
      for i in range(max_permutaions):
        perm = random.sample(range(self.video_length), self.video_length)
        if perm not in self._permutations:
          self._permutations += [perm]

    print("\t num permutations: %d"%len(self._permutations))
    print("\t num videos: %d"%len(self.videos))

  def __repr__(self):
       return "BB_vid"
  def __str__(self):
      return self.__repr__()

  def set_permutations(self, perms):
    self._permutations = perms

  def get_permutations(self):
        return self._permutations

  def get_video_length(self):
        return self.video_length

  def get_num_videos(self):
        return len(self.videos)

  def get_num_permutations(self):
        return len(self._permutations)

  def train(self):
        self.train = True

  def test(self):
        self.train = False

  def __len__(self):
    if self.train:
        return len(self.videos)*len(self._permutations)
    else:
        return len(self.videos)

  def __getitem__(self, index):
    if self.train:
      func_index = index % self.num_videos
      perm_index = int(index / self.num_videos)
      video = self.videos[func_index]

      perm = self._permutations[perm_index]
      permuted_video = video[perm]
      # import pdb;pdb.set_trace()

      return permuted_video.astype(np.float32), perm_index

    else:
      s = np.random.uniform(0,1)
      if s > 0.5:
        video = create_video(self.frame_size, self.video_length, max_ball_size=16, max_balls=10, create_anomaly=False)
        return video.astype(np.float32), 0
      else:
        video = create_video(self.frame_size, self.video_length, max_ball_size=16, max_balls=10, create_anomaly=True)
        return video.astype(np.float32), 1
      



