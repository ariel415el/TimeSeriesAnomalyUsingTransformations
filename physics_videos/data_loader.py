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


def create_fixed_permutations(n):
  perms = []
  base = list(range(n))
  for i in [2, 5, 7, n//4, n//2, n]:
    copy = base.copy()
    tmp = copy[::i]
    tmp.reverse()
    copy[::i] = tmp
    perms += [copy]

    copy = base.copy()
    for j in range(n // i):
      tmp = copy[j*i:(j+1)*i]
      tmp.reverse()
      copy[j*i:(j+1)*i] = tmp

    tmp = copy[(j+1)*i:n]
    tmp.reverse()
    copy[(j+1)*i:n] = tmp

    perms += [copy]

  return perms

def get_perm_applier(perm):
  class f:
    def __init__(self, perm):
      self.perm = perm
    def __call__(self, vid_arr_batch):
      return vid_arr_batch[self.perm]
  return f(perm)

class Ball:
    def __init__(self, frame_w, frame_h, ball_size, shape, hue, size_growth=0, max_ball_size=32, no_wall=False):
        self.size_growth = size_growth
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.max_size = max_ball_size
        self.size = ball_size
        self.x = random.randrange(0, self.frame_w-self.size)
        self.y = random.randrange(16, self.frame_h-self.size)
        self.speedx = random.randrange(1, 5)
        self.speedy = random.randrange(1, 5)
        self.hue = hue
        self.no_wall = no_wall
        self.shape  = shape

    def update(self):
        # update size
        if self.size + self.size_growth > self.max_size:
            self.size = 2*self.max_size - self.size - self.size_growth
            self.size_growth *= -1
        elif self.size + self.size_growth < 0 : 
            self.size = -1*(self.size + self.size_growth)
            self.size_growth *= -1
        else:
            self.size += self.size_growth

        # update position
        pos_x = self.x + self.speedx
        pos_y = self.y + self.speedy
        if not self.no_wall:
          if pos_x + self.size > self.frame_w:
              pos_x =2*self.frame_w - self.x - self.speedx - 2*self.size 
              self.speedx *= -1
          if pos_x < 0 :
              pos_x *= -1
              self.speedx *= -1     

          if pos_y + self.size > self.frame_h:
              pos_y = 2*self.frame_h - self.y - self.speedy - 2*self.size 
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
      # box
      cv2.rectangle(frame, (xA, yA),(xB, yB), color, cv2.FILLED)
    elif ball.shape == 1:
      # ball
      cv2.circle(frame,(int((xA+xB)/2), int((yA+yB)/2)), int(ball.size/2), color, cv2.FILLED)

    elif ball.shape == 2:
      # umbrella
      cv2.rectangle(frame, (int((xA+xB)/2), yA),(int((xA+xB)/2), yB), color, cv2.FILLED)
      triangle_cnt = np.array( [(int((xA+xB)/2), yA), (xA, int((yA+yB)/2)), (xB, int((yA+yB)/2))] )
      cv2.drawContours(frame, [triangle_cnt], 0, color, -1)

    elif ball.shape == 3:
      # daemond
      p1 = (int((xA+xB)/2), yA)
      p2 = (int((xA+xB)/2), yB)
      p3 = (xA, int((yA+yB)/2))
      p4 = (xB, int((yA+yB)/2))
      contour = np.array( [p1,p2,p3,p4] )
      cv2.drawContours(frame, [contour], 0, color, -1)

    elif ball.shape == 4:
      #Cross
      cv2.line(frame, (xA, yA), (xB, yB), color, 2)
      cv2.line(frame, (xA, yB), (xB, yA), color, 2)

def create_video(frame_w,frame_h, num_frames, num_balls, ball_shape=None, background_hue=None):
    # num_balls =  np.random.randint(1,max_balls)
    noise_magnitude = 5
    max_ball_size = int(min(frame_w,frame_h)/2)
    if background_hue is None:
      background_hue = np.random.randint(noise_magnitude, 128)
    ball_list = []


    for i in range(num_balls):
        ball_hue = np.random.randint(background_hue+72, 255)
        ball_size = random.randrange(2, max_ball_size)
        if ball_shape is None:
          ball_shape = random.randrange(0, 5)

        ball_list.append(Ball(frame_w, frame_h, ball_size, ball_shape, ball_hue, size_growth=0, max_ball_size=max_ball_size, no_wall=False))

    all_frames = []
    for i in range(num_frames):
        frame = np.ones((frame_h, frame_w)).astype('uint8')*background_hue
        for ball in ball_list:
            stump_ball(frame, ball, ball.hue)
            ball.update()
        frame += np.random.uniform(-1*noise_magnitude, noise_magnitude, size=frame.shape).astype(np.uint8)
        frame = frame.clip(0,255)
        all_frames += [frame]
    return np.array(all_frames)


class balls_dataset(torch.utils.data.Dataset):
  def __init__(self, frame_w=256,frame_h=128, video_length=32, num_videos=1000, train=True, permutations=None):
    self.train = train
    self.frame_w = frame_w
    self.frame_h = frame_h
    self.video_length = video_length
    self.num_videos = num_videos
    print("# Creating videos")
    self.videos = []
    for i in tqdm(range(self.num_videos)):
      video = create_video(self.frame_w, self.frame_h, self.video_length, num_balls=2)
      self.videos += [video]

    for i in range(5):
      write_array_as_video(self.videos[i].astype(np.uint8), os.path.join("train_debug", "DL_series_%d.avi"%i))

    print("# Creating perms")
    if permutations is not None:
      self._permutations = permutations
    else:
      self._permutations = []
      # for i in range(max_permutaions):
      #   perm = random.sample(range(self.video_length), self.video_length)
      #   if perm not in self._permutations:
      #     self._permutations += [perm]
      perms = create_fixed_permutations(self.video_length)
      for p in perms:
        self._permutations += [get_perm_applier(p)]

    print("\t num permutations: %d"%len(self._permutations))
    print("\t num videos: %d"%len(self.videos))

  def __repr__(self):
       return "BB_vid"
  def __str__(self):
      return self.__repr__()


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
      permuted_video = perm(video)
      # import pdb;pdb.set_trace()

      return permuted_video.astype(np.float32), perm_index

    else:
      s = np.random.uniform(0,1)
      if s > 0.5:
        video = create_video(self.frame_w, self.frame_h, self.video_length,num_balls=2)
        return video.astype(np.float32), 0
      else:
        background_hue = np.random.randint(5, 128)
        video_l  = create_video(int(self.frame_w/2), self.frame_h, self.video_length, num_balls=1,background_hue=background_hue)
        video_r  = create_video(int(self.frame_w/2), self.frame_h, self.video_length, num_balls=1,background_hue=background_hue)
        frames = []
        for i in range(self.video_length):
            frames += [np.hstack([video_l[i],video_r[i]])]
        # import pdb;pdb.set_trace()
        return np.array(frames).astype(np.float32), 1
      


