import cv2
from matplotlib import pyplot as plt

def write_array_as_video(arr, path):
    # print("writing array as video ", arr.shape)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'PIM1'), 22, (arr.shape[-2], arr.shape[-1]), False)
    for i in range(arr.shape[0]):
        writer.write(arr[i])

def scatter_1d_serie(xs, ys, path):
    plt.scatter(xs, ys)
    plt.savefig(path)
    plt.clf()

def plot_1d_serie(xs, ys, path):
    plt.plot(xs, ys)
    plt.savefig(path)
    plt.clf()