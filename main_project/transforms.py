import itertools
import numpy as np
from math import factorial
import random
from skimage.transform import AffineTransform,SimilarityTransform, warp
import ast

class matrix_applier(object):
    def __init__(self, mat):
        self.mat = mat
    def __call__(self, serie):
        return np.dot(self.mat, serie)

class fixed_affine_1d_transformer(object):
    def __init__(self, num_transforms, serie_length):
        self.random_mats = []
        for i in range(num_transforms):
            self.random_mats += [np.random.rand(serie_length, serie_length)]
        self.update_transforms()

    def get_transforms(self):
        return self.transforms

    def set_mats(self, mats):
        self.random_mats = mats

    def update_transforms(self):
        self.transforms = []
        for mat in self.random_mats:
            self.transforms += [matrix_applier(mat)]
        print("Got %d transforms"% len(self.transforms))

    def load_from_file(self,path):
        with open(path, "rb") as f:
            all_mats = np.load(f)
            matrices = [all_mats[i] for i in range(all_mats.shape[0])]
            self.set_mats(matrices)
            self.update_transforms()

    def save_to_file(self,path):
        with open(path,"wb") as f:
            np.save(f ,np.stack(self.random_mats, axis=0))

class fixed_affine_image_transformer(object):
    def __init__(self, num_transforms, max_tx=8, max_ty=8):
        self.transforms = []
        for is_flip, tx, ty, sx, sy, k_rotate, shear in itertools.product((False, True),
                                                                          range(-max_tx,max_tx),
                                                                          range(-max_ty,max_ty),
                                                                          [1,2],
                                                                          [1,2],
                                                                          [0,30,60,120,150,210,240,300,330],
                                                                          [0,45,135,225,315]):
            self.transforms += [AffineTransformation(is_flip, tx ,ty, sx, sy, k_rotate, shear)]
        print("Got %d transforms"% len(self.transforms))
        if num_transforms > len(self.transforms):
            print("Can't generate this much transforms")
            exit(1)
        if num_transforms < len(self.transforms):
            self.transforms = self.transforms[:num_transforms]
        # self.transforms = random.sample(self.transforms, min(len(self.transforms), num_transforms))
        print("Keep only %d transforms"% len(self.transforms))

    def get_transforms(self):
        return self.transforms

    def load_from_file(self,path):
        return # Fixed transforms, no need to save

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
        assert (serie_length % segment_size == 0)
        self.perms = []
        self.transforms = []
        self.segment_size = segment_size
        perm_length = int(serie_length/segment_size)
        while len(self.perms) != min(num_transforms, factorial(perm_length)):
            perm = random.sample(range(perm_length), perm_length)
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
        self.perms = []
        for line in lines:
            strings_list = ast.literal_eval(line.strip())
            l = [int(x) for x in strings_list]
            self.perms += [l]
        self.update_transforms()

    def save_to_file(self,path):
        f = open(path,'w')
        for perm in self.perms:
            f.write(str(perm)+"\n")
        f.close()

