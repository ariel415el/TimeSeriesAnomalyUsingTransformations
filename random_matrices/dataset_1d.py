import torch.utils.data

import numpy as np
import random
import os
from debug_utils import plot_1d_serie

def generate_linear(mul_max=1000, bias_max=10000):
    class f:
        def __init__(self, m, b):
            self.m = m
            self.b = b

        def __call__(self, x):
            return x * self.m + self.b

    while True:
        mult = np.random.uniform(-1 * mul_max, mul_max, 1)
        bias = np.random.uniform(-1 * bias_max, bias_max, 1)
        yield f(mult, bias)

def generate_increasing_linear(mul_max=1000, bias_max=10000):
    class f:
        def __init__(self, m, b):
            self.m = m
            self.b = b

        def __call__(self, x):
            return x * self.m + self.b

    while True:
        mult = np.random.uniform(0, mul_max, 1)
        bias = np.random.uniform(0, bias_max, 1)
        yield f(mult, bias)

def generate_decreasing_linear(mul_max=1000, bias_max=10000):
    class f:
        def __init__(self, m, b):
            self.m = m
            self.b = b

        def __call__(self, x):
            return x * self.m + self.b

    while True:
        mult = np.random.uniform(-1 * mul_max, 0, 1)
        bias = np.random.uniform(0, bias_max, 1)
        yield f(mult, bias)

def generate_sinus(max_amplitude=2, max_freq=np.pi / 8):
    class f:
        def __init__(self, freq, amp):
            self.freq = freq
            self.amp = amp

        def __call__(self, x):
            return self.amp * np.sin(x * self.freq)

    while True:
        freq = np.random.uniform(max_freq / 2, max_freq, 1)
        amp = np.random.uniform(-1 * max_amplitude, max_amplitude, 1)
        yield f(freq, amp)

def generate_linear_sinus(mul_max=10, bias_max=100, max_amplitude=50, max_freq=np.pi / 2):
    class f:
        def __init__(self, mult, bias, freq, amp):
            self.mult = mult
            self.bias = bias
            self.freq = freq
            self.amp = amp

        def __call__(self, x):
            return self.amp * np.sin(x * self.freq) + x * self.mult + self.bias

    while True:
        mult = np.random.uniform(-1 * mul_max, mul_max, 1)
        bias = np.random.uniform(-1 * bias_max, bias_max, 1)
        freq = np.random.uniform(max_freq / 2, max_freq, 1)
        amp = np.random.uniform(-1 * max_amplitude, max_amplitude, 1)
        yield f(mult, bias, freq, amp)

def generate_power_sinus(*coef_maxs, max_amplitude=50, max_freq=np.pi / 2):
    class f:
        def __init__(self, freq, amp, *coef_maxs):
            self.coef_maxs = coef_maxs
            self.freq = freq
            self.amp = amp

        def __call__(self, x):
            line = 0
            for k in range(len(self.coef_maxs)):
                line += self.coef_maxs[k] * (x ** k)
            return self.amp * np.sin(x * self.freq) + line

    while True:
        # import pdb; pdb.set_trace()
        coefs = [np.random.uniform(-1 * coef_maxs[i], coef_maxs[i]) for i in range(len(coef_maxs))]
        freq = np.random.uniform(max_freq / 2, max_freq)
        amp = np.random.uniform(-1 * max_amplitude, max_amplitude)
        yield f(freq, amp, *coefs)


class con_func_series_dataset(torch.utils.data.Dataset):
    def __init__(self, serie_length=64, num_series=1000, train=True, transforms=[], func_type='Linear'):
        self.train = train
        self.num_series = num_series
        self.serie_length = serie_length
        self.series_xs = np.linspace(-100, 100, num=self.serie_length)
        print("# Creating funcs")
        if func_type == 'Linear':
            self.func_generator = generate_linear()
        elif func_type == 'Linear-Sinus':
            self.func_generator = generate_linear_sinus()
        elif func_type == 'Power-Sinus':
            self.func_generator = generate_power_sinus(128, 16, 2, max_amplitude=256, max_freq=np.pi / 2)
        else:
            self.func_generator = generate_increasing_linear()
        # self.func_generator = generate_decreasing_linear()

        # self.funcs = [next(self.func_generator) for i in range(self.num_series)]
        self.series = []
        for i in range(self.num_series):
            func = next(self.func_generator)
            self.series += [func(self.series_xs)]

        self.transforms = transforms

        print("\t serie length: %d" % (self.serie_length))
        print("\t num permutations: %d" % len(self.transforms))
        print("\t num series: %d" % len(self.series))

    def __len__(self):
        if self.train:
            return len(self.series) * len(self.transforms)
        else:
            return len(self.series)

    def __getitem__(self, index):
        if self.train:
            serie_index = index % self.num_series
            transform_index = int(index / self.num_series)
            serie = self.series[serie_index]
            transform = self.transforms[transform_index]

            transformed_serie = transform(serie)


            # max_noise = (transformed_serie.max() - transformed_serie.min()) / 32
            # noise = np.random.uniform(-1 * max_noise, max_noise, len(transformed_serie))
            # transformed_serie += noise
            return transformed_serie.astype(np.float32), transform_index

        else:
            s = np.random.uniform(0, 1, 1)[0]
            serie = self.series[index]
            if s < 0.5:
                # func = next(self.sin_generator)
                return serie.astype(np.float32), 0
            elif s >= 0.5:
                x_radius = 10
                start_idx = np.random.randint(0, self.serie_length - x_radius)
                # mean = serie[start_idx: start_idx + x_radius].mean()
                y_radius = serie[start_idx: start_idx + x_radius].max() - serie[start_idx: start_idx + x_radius].min()
                serie[start_idx: start_idx + x_radius] += np.random.uniform(-1, 1, len(serie[start_idx: start_idx + x_radius])) * y_radius * 3
                return serie.astype(np.float32), 1

    def get_number_of_test_classes(self):
        return 2

    def __repr__(self):
        return "Linear"

    def __str__(self):
        return self.__repr__()

    def get_transforms(self):
        return self.transforms

    def get_num_transforms(self):
        return len(self.transforms)

    def get_serie_length(self):
        return self.serie_length

    def train(self):
        self.train = True

    def test(self):
        self.train = False

    def dump_debug_images(self, path):
          os.makedirs(path, exist_ok=True)
          debug_transforms_idxs = random.sample(range(len(self.transforms)), 5)
          debug_serie_idxs = random.sample(range(len(self.series)), 5)
          for s_idx in debug_serie_idxs:
              serie = self.series[s_idx]
              plot_1d_serie(self.series_xs, serie , os.path.join(path, "serie_%d.png" % (s_idx)))
              for t_idx in debug_transforms_idxs:
                  t = self.transforms[t_idx]
                  t_serie = t(serie)
                  plot_1d_serie(self.series_xs, t_serie, os.path.join(path, "serie_%d_trasform_%d.png" % (s_idx, t_idx)))
              self.test()
              serie, label = self.__getitem__(s_idx)
              plot_1d_serie(self.series_xs, serie, os.path.join(path, "serie_%d_test_label-%d.png" % (s_idx, label)))

