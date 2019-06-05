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

class AffineTransformation(object):
    def __init__(self, flip, tx, ty, k_90_rotate):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate

    def __call__(self, x):
        res_x = x
        if self.flip:
            res_x = np.fliplr(res_x)
        if self.tx != 0 or self.ty != 0:
            # print(res_x.shape, self.ty,self.tx)
            res_x = scipy.ndimage.shift(res_x, np.array([self.ty, self.tx,0]), mode='reflect')
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)

        return res_x


class mnist_png_dataset(torch.utils.data.Dataset):
  def __init__(self, dir_path):
        self._image_paths = [os.path.join(dir_path, fn) for fn in os.listdir(dir_path)]
        self._to_tensor = transforms.ToTensor()
        # self._to_tensor = torch.from_numpy

        # self._normalize =  transforms.Normalize((0.1307,), (0.3081,))
        self.max_tx = 8
        self.max_ty = 8


        print("# Creating dataset")
        transformation_list = []
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)

        self._transformation_list = transformation_list

  def get_num_transformations(self):
        return len(self._transformation_list)

  def __len__(self):
        return len(self._image_paths) * len(self._transformation_list)

  def __getitem__(self, index):
        img_index = index % len(self._image_paths)
        transform_index = int(index / len(self._image_paths))

        im_path = self._image_paths[img_index]
        # img = np.array(Image.open(open(im_path, 'rb'))).reshape(28,28)
        # img = Image.open(open(im_path, 'rb'))
        img = cv2.imread(im_path,  cv2.IMREAD_GRAYSCALE).reshape(28,28,1)
        transformed_image = self._transformation_list[transform_index](img).copy()
        tensor = self._to_tensor(transformed_image)

        return tensor, transform_index


class mnist_png_dataset_test(torch.utils.data.Dataset):
  def __init__(self, root_path, normal_digit, samples_per_class=5):
        self._image_paths=[]
        self._image_labels=[]
        for dn in os.listdir(root_path):
            self._image_paths += [os.path.join(root_path, dn, fn) for fn in random.sample(os.listdir(os.path.join(root_path, dn)), samples_per_class)]
            self._image_labels += [dn]*samples_per_class
        self._to_tensor = transforms.ToTensor()
        # self._to_tensor = torch.from_numpy

        # self._normalize =  transforms.Normalize((0.1307,), (0.3081,))
        self.max_tx = 8
        self.max_ty = 8
        self._transform_labels = []
        print("# Creating dataset")
        transformation_list = []
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)
            self._transform_labels += ["fl_%s_t_%d-%d_rot_%d"%(str(is_flip),tx,ty,k_rotate*90)]

        self._transformation_list = transformation_list

  def get_num_transformations(self):
        return len(self._transformation_list)

  def __len__(self):
        return len(self._image_paths)

  def __getitem__(self, index):
        # img = np.array(Image.open(open(im_path, 'rb'))).reshape(28,28)
        # img = Image.open(open(im_path, 'rb'))
        img = cv2.imread(self._image_paths[index],  cv2.IMREAD_GRAYSCALE).reshape(28,28,1)
        image_tranforms = []
        tranform_labels = []
        for i in range(len(self._transformation_list)):
            transformed_image = self._transformation_list[i](img).copy()
            image_tranforms += [transformed_image.transpose(2,0,1).astype(np.float32)]
            tranform_labels += [self._transform_labels[i]]

        return img, np.array(image_tranforms), tranform_labels, self._image_labels[index]