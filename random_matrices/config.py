import argparse

parser = argparse.ArgumentParser(description='PyTorch time series Anomaly detection')

# general
parser.add_argument('--debug', action='store_true',default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--data_type', type=str, default="2d")
parser.add_argument('--trained_model', type=str, default=R"W:\AnomalyDetection\TimeSeriesAnomalyUsingTransformations\random_matrices\train_dir_2d\BB_vid_0.79570_ckp.pt")
parser.add_argument('--train_dir', type=str, default="train_dir_2d")

# common parameters
parser.add_argument('--num_train_series', type=int, default=1000)
parser.add_argument('--num_val_series', type=int, default=100)
parser.add_argument('--num_transforms', type=int, default=100)
parser.add_argument('--transformations_type', type=str, default="Permutations", help='Permutations/Affine')

# 1d dataset config
parser.add_argument('--func_type', type=str, default="Linear", help='Linear/Sinus-Linear/Power-Sinus')

# 2d dataset config
parser.add_argument('--anomaly_type', type=str, default="Shapes", help='Shapes/Incontinous')
parser.add_argument('--frame_h', type=int, default=128)
parser.add_argument('--frame_w', type=int, default=128)

args = parser.parse_args()

if args.data_type == "2d":
    args.num_frames = 64
    args.segment_size = 1
    args.batch_size = 80
else:
    args.batch_size = 256
    args.num_frames = 128
    args.segment_size = 16

# Pytorch parameters
args.num_workers = 0
args.pin_memory = True
args.seed = 1

# train hyper parameters
args.epochs = 10
args.lr = 0.0001
args.lr_decay = 0.5
args.decay_epochs = 1
args.num_workers =0
args.num_workers =0