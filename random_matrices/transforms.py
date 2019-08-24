import os
from PIL import Image

import torch
from torchvision import datasets, transforms

import abc
import itertools
import numpy as np
import scipy.ndimage

import cv2
import random
from skimage.transform import AffineTransform,SimilarityTransform, warp


class fixed_affine_transforms(object):
    def __init__(self, num_transforms, max_tx=8, max_ty=8):
        assert(num_transforms <= 72)
        self.transforms = []
        for is_flip, tx, ty, sx, sy, k_rotate, shear in itertools.product((False, True),
                                                       range(-max_tx,max_tx),
                                                       range(-max_ty,max_ty),
                                                       [1,2],
                                                       [1,2],
                                                       [0,30,60,90,120,150,180,210,240,270,300,330],
                                                       [0,45,135,225,315]):
            self.transforms += [AffineTransformation(is_flip, tx ,ty, sx, sy, k_rotate, shear)]
        print("Got %d transforms"% len(self.transforms))
        self.transforms = random.sample(self.transforms, min(len(self.transforms), num_transforms))
        print("Keep only %d transforms"% len(self.transforms))

    def get_transforms(self):
        return self.transforms

    def load_from_file(self,path):
        return

    def save_to_file(self,path):
        return

class AffineTransformation(object):
    def __init__(self, flip, tx, ty,sx, sy, radians, shear):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.sx = sx
        self.sy = sy
        self.shear = shear
        self.radians = radians
    def __call__(self, x):
        x_3d = x.copy()
        x_3d = np.expand_dims(x_3d,3)
        for i in range(x_3d.shape[0]):
            t = SimilarityTransform(scale=None, translation=None, rotation=np.deg2rad(self.radians))
            x_3d[i] = warp(x_3d[i], t, order=1, preserve_range=True, mode='symmetric')
            if self.flip:
                x_3d[i] = np.fliplr(x_3d[i])
        x_3d = x_3d.squeeze(3)
        return x_3d

class perm_applier:
    def __init__(self, perm, segment_size):
        self.perm = perm
        self.segment_size = segment_size
    def __call__(self, vid_arr_batch):
        if self.segment_size == 1:
            return vid_arr_batch[self.perm]
        else:
            tmp = np.array(np.split(vid_arr_batch, len(self.perm)))
            return np.concatenate(tmp[self.perm])


class frame_permuter():
    def __init__(self, num_transforms, serie_length, segment_size):
        self.perms = []
        self.transforms = []
        self.segment_size = segment_size
        while len(self.perms) != num_transforms:
            perm = random.sample(range(serie_length), serie_length)
            if perm not in self.perms:
                self.perms += [perm]

        self.update_transforms()
        print("frame_permuter created")

    def update_transforms(self):
        new_transforms = []
        for perm in self.perms:
            new_transforms += [perm_applier(perm, self.segment_size)]
        self.transforms = new_transforms

    def get_transforms(self):
        return self.transforms

    def load_from_file(self, path):
        lines = open(path,'r').readlines()
        self.perms = [list(line.strip()) for line in lines]
        self.update_transforms()

    def save_to_file(self,path):
        f = open(path,'w')
        for perm in self.perms:
            f.write(str(perm)+"\n")
        f.close()



# def get_random_triangle(im_shape):
#     x = np.zeros((3,2))
#     for i in range(3):
#         x[i][0] = np.random.randint(im_shape[1])
#         x[i][1] = np.random.randint(im_shape[0])
#     return x

# def get_affine_matrices(num_mats):
#     matrices = []

#     for i in range(num_mats):
#         src, dst = get_random_triangle((128,128)), get_random_triangle((128,128))
#         matrices += [cv2.getAffineTransform(src.astype(np.float32),dst.astype(np.float32)) ]
#     return matrices

# class random_affine_matrices():
#     def __init__(self, num_transforms):
#         self.transforms = []
#         for i in range(num_mats):
#             src, dst = get_random_triangle((128,128)), get_random_triangle((128,128))
#             mat =  cv2.getAffineTransform(src.astype(np.float32),dst.astype(np.float32))


#     def get_transforms(self):
#         return self.transforms

#     def load_from_file(self, path):
#          with open(path,"rb") as f:
#                 all_mats = np.load(f)
#                 self.matrices = [all_mats[i] for i in range(all_mats.shape[0])]

#     def save_to_file(self,path):
#         return  

# class apply_3x2_affine_matrix():
#     def __init__(self, matrix):
#         self.matrix = matrix

#     def __call__(self, x):
#         x_3d = x.copy()
#         x_3d = np.expand_dims(x_3d,3)
#         for i in range(x_3d.shape[0]):
            
#         x_3d = x_3d.squeeze(3)
#         return x_3d

