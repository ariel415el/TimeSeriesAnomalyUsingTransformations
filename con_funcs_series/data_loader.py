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


def create_segment_transforms(num_segments, segment_length):
  transforms = []

  # Noise
  for i in range(1, num_segments):
      def f(segments):
        segments[i] = segments[i-1]
      transforms += [f]
  for i in range(num_segments):
      def f(segments):
        segments[i] = segments[i]*0+np.mean(segments)
      transforms += [f]
  # permutations:
  
  # geometric


def generate_linear(mul_max=10, bias_max=100):
  while True:
    mult = np.random.uniform(-1*mul_max, mul_max, 1)
    bias = 0#np.random.uniform(-1*bias_max, bias_max, 1)
    def f(x):
      return x*mult + bias
    yield f 

def generate_increasing_linear(mul_max=10, bias_max=0):
  while True:
    mult = np.random.uniform(0, mul_max, 1)
    bias = 0#np.random.uniform(0, bias_max, 1)
    def f(x):
      return x*mult + bias
    yield f

def generate_decreasing_linear(mul_max=10, bias_max=0):
  while True:
    mult = np.random.uniform(-1*mul_max,0 , 1)
    bias = 0#np.random.uniform(0, bias_max, 1)
    def f(x):
      return x*mult + bias
    yield f


def generate_sinus(max_amplitude=100, max_freq=2*np.pi):
  while True:
    freq = np.random.uniform(0, max_freq, 1)
    amp = np.random.uniform(-1*max_amplitude, max_amplitude, 1)
    
    def f(x):
      return amp*np.sin(x*freq)
    yield f 


def generate_linear_sinus(mul_max=10, bias_max=100, max_amplitude=100, max_freq=2*np.pi):
  while True:
    mult = np.random.uniform(-1*mul_max, mul_max, 1)
    bias = np.random.uniform(-1*bias_max, bias_max, 1)
    freq = np.random.uniform(0, max_freq, 1)
    amp = np.random.uniform(-1*max_amplitude, max_amplitude, 1)
    
    def f(x):
      return amp*np.sin(x*freq) + x*mult + bias
    yield f 

class con_func_series_dataset(torch.utils.data.Dataset):
  def __init__(self, num_series=10000, max_permutaions=1000, train=True, permutations=None):
    self.train = train
    self.num_series = num_series
    self.max_permutaions = max_permutaions
    self.num_segments = 8
    self.segment_size = 16
    self.serie_length = self.num_segments*self.segment_size
    serie = np.linspace(-100, 100, num=self.serie_length)
    self.segments = [serie[i*self.segment_size:(i+1)*self.segment_size] for i in range(self.num_segments)]
    print("# Creating")
    self.inc_lin_generator = generate_increasing_linear()
    self.dec_lin_generator = generate_decreasing_linear()
    self.sin_generator = generate_sinus()
    # func_generator = generate_linear_sinus()

    self.funcs = [next(self.inc_lin_generator) for i in range(self.num_series)]

    print("\t creating perms")
    # perms = list(itertools.permutations(np.arange(self.serie_length)))
    # if len(perms) > max_permutaions:
    #     perms = random.sample(perms, max_permutaions)
    if permutations is not None:
      self._permutations = permutations
    else:
      self._permutations = []
      for i in range(max_permutaions):
        perm = random.sample(range(self.num_segments), self.num_segments)
        if perm not in self._permutations:
          self._permutations += [perm]

    # self._permutations = generate_permutations(self.serie_length, max_permutaions)
    print("\t serie length: %d"%(self.serie_length))
    print("\t num permutations: %d"%len(self._permutations))
    print("\t num funcs: %d"%len(self.funcs))

  def __repr__(self):
       return "Linear"
  def __str__(self):
      return self.__repr__()

  def set_permutations(self, perms):
    self._permutations = perms

  def get_permutations(self):
        return self._permutations

  def get_num_segments(self):
        return self.num_segments

  def get_num_permutations(self):
        return len(self._permutations)

  def get_serie_length(self):
        return self.serie_length

  def train(self):
        self.train = True

  def test(self):
        self.train = False

  def __len__(self):
    if self.train:
        return self.num_series*len(self._permutations)
    else:
        return self.num_series

  def __getitem__(self, index):
    if self.train:
      func_index = index % self.num_series
      perm_index = int(index / self.num_series)
      func = self.funcs[func_index]
      y_segments = np.array([func(seg) for seg in self.segments])

      perm = self._permutations[perm_index]
      permuted_ys = y_segments[perm].reshape(1,self.serie_length)
      # import pdb;pdb.set_trace()

      max_noise  = (permuted_ys.max() - permuted_ys.min()) / 32
      noise =  np.random.uniform(-1*max_noise, max_noise, len(permuted_ys))
      permuted_ys += noise
      return permuted_ys.astype(np.float32), perm_index

    else:
      s = np.random.uniform(0,1,1)[0]
      if s > 0.8:
        func = next(self.inc_lin_generator)
        return np.array([func(seg) for seg in self.segments]).astype(np.float32), 0
      elif s  > 0.6:
         return np.array([np.random.normal(0, 0.1, self.segment_size) for seg in self.segments]).astype(np.float32), 1
      elif s > 0.4:
        func = next(self.dec_lin_generator)
        return np.array([func(seg) for seg in self.segments]).astype(np.float32), 2
      else:
        func = next(self.sin_generator)
        return np.array([func(seg) for seg in self.segments]).astype(np.float32), 3

