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
    def __init__(self, frame_w, frame_h,
                 shape,
                 hue,
                 size_growth=0,
                 max_ball_size=32,
                 no_wall=False,
                 incontinous=False):
        self.size_growth = size_growth
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.max_size = max_ball_size
        self.incontinous=incontinous
        self.steps_counter=0
        self.size = random.randrange(2, max_ball_size)
        self.x = random.randrange(0, self.frame_w-self.size)
        self.y = random.randrange(0, self.frame_h-self.size)
        self.speedx = random.randrange(1, 5)
        self.speedy = random.randrange(1, 5)
        self.hue = hue
        self.no_wall = no_wall
        self.shape  = shape

    def update(self):
        # update size
        self.steps_counter += 1

        if self.size + self.size_growth > self.max_size:
            self.size = 2*self.max_size - self.size - self.size_growth
            self.size_growth *= -1
        elif self.size + self.size_growth < 0 : 
            self.size = -1*(self.size + self.size_growth)
            self.size_growth *= -1
        else:
            self.size += self.size_growth

        # update position
        if self.incontinous and  self.steps_counter % 10 == 0:
          self.x = random.randrange(0, self.frame_w-self.size)
          self.y = random.randrange(0, self.frame_h-self.size)
        else:
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
      contour = np.array( [p1,p3,p2,p4] )
      cv2.drawContours(frame, [contour], 0, color, -1)

    elif ball.shape == 4:
      # hourglass
      p1 = (int((xA+xB)/2), yA)
      p2 = (int((xA+xB)/2), yB)
      p3 = (xA, int((yA+yB)/2))
      p4 = (xB, int((yA+yB)/2))
      contour = np.array( [p1,p2,p3,p4] )
      cv2.drawContours(frame, [contour], 0, color, -1)

    elif ball.shape == 5:
      #Cross
      cv2.line(frame, (xA, yA), (xB, yB), color, 2)
      cv2.line(frame, (xA, yB), (xB, yA), color, 2)

class video_params():
  def __init__(self):
    self.background_hue=None
    self.num_balls =1
    self.ball_shape=0
    self.size_growth=0
    self.noise_magnitude = 5
    self.incontinous=False

def create_video(frame_w,frame_h, num_frames, video_params):
    max_ball_size = int(min(frame_w,frame_h)/3)
    if video_params.background_hue is None:
        background_hue = np.random.randint(video_params.noise_magnitude, 128)
    else:
      background_hue = self.background_hue
    ball_list = []

    for i in range(video_params.num_balls):
        ball_hue = np.random.randint(background_hue+72, 255)
        if video_params.ball_shape is None:
          video_params.ball_shape = random.randrange(0, 5)

        ball_list.append(Ball(frame_w,
                              frame_h,
                              video_params.ball_shape,
                              ball_hue,
                              size_growth=video_params.size_growth,
                              max_ball_size=max_ball_size,
                              no_wall=False,
                              incontinous=video_params.incontinous))

    all_frames = []
    for i in range(num_frames):
        frame = np.ones((frame_h, frame_w)).astype('uint8')*background_hue
        for ball in ball_list:
            stump_ball(frame, ball, ball.hue)
            ball.update()
        frame += np.random.uniform(-1*video_params.noise_magnitude, video_params.noise_magnitude, size=frame.shape).astype(np.uint8)
        frame = frame.clip(0,255)
        all_frames += [frame]
    return np.array(all_frames)

class balls_dataset(torch.utils.data.Dataset):
  def __init__(self, frame_w=256,frame_h=128, video_length=32, num_videos=1000, train=True, matrices=[]):
    self.num_data_classes = 2
    self.train = train
    self.frame_w = frame_w
    self.frame_h = frame_h
    self.video_length = video_length
    self.num_videos = num_videos
    print("# Creating videos")
    self.videos = []
    for i in tqdm(range(self.num_videos)):
      video = create_video(self.frame_w, self.frame_h, self.video_length, video_params())
      self.videos += [video]

    # Debug
    for i in range(5):
      write_array_as_video(self.videos[i].astype(np.uint8), os.path.join("train_debug", "DL_series_%d.avi"%i))

    self.matrices = matrices
    if len(self.matrices) == 0 :
      print("No matrices loaded")
      exit()
    print("\t num matrices: %d"%len(self.matrices))
    print("\t num videos: %d"%len(self.videos))

  def __getitem__(self, index):
    if self.train:
      vid_index = index % self.num_videos
      mat_index = int(index / self.num_videos)
      video = self.videos[vid_index]

      mat = self.matrices[mat_index]

      output_video = video.copy()
      for k in range(output_video.shape[0]):
        output_video[k] = cv2.warpAffine(output_video[k], mat, (output_video[k].shape[1], output_video[k].shape[0]), flags=cv2.INTER_LINEAR)
      # output_video = np.matmul(video, mat) # apply the matrix on all frames
      # import pdb;pdb.set_trace()

      return output_video.astype(np.float32), mat_index

    else:
        target =  random.randint(0,1)
        vp = video_params()
        if target == 1:
          vp.incontinous=True

        vid = create_video(int(self.frame_w), self.frame_h, self.video_length, vp)
        return vid, target

  def __repr__(self):
       return "BB_vid"
  def __str__(self):
      return self.__repr__()

  def set_matrices(self, matrices):
        self.matrices = matrices

  def get_matrices(self):
        return self.matrices

  def get_video_length(self):
        return self.video_length

  def get_num_videos(self):
        return len(self.videos)

  def get_num_matrices(self):
        return len(self.matrices)

  def get_number_of_data_classes(self):
    return self.num_data_classes

  def train(self):
        self.train = True

  def test(self):
        self.train = False

  def __len__(self):
    if self.train:
        return len(self.videos)*len(self.matrices)
    else:
        return len(self.videos)





