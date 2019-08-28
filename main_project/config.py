import argparse

parser = argparse.ArgumentParser(description='PyTorch time series Anomaly detection')

# general
parser.add_argument('--debug', action='store_true',default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--data_type', type=str, default="2d")
parser.add_argument('--trained_model', type=str, default=R"")
parser.add_argument('--train_dir', type=str, default="")

# common parameters
parser.add_argument('--num_train_series', type=int, default=5000)
parser.add_argument('--num_val_series', type=int, default=100)
parser.add_argument('--num_transforms', type=int, default=100)
parser.add_argument('--transformations_type', type=str, default="Affine", help='Permutations/Affine')
# 1d dataset config
parser.add_argument('--func_type', type=str, default="Power-Sinus", help='Linear/Sinus-Linear/Power-Sinus')

# 2d dataset config
parser.add_argument('--anomaly_type', type=str, default="Shapes", help='Shapes/Incontinous')
parser.add_argument('--frame_h', type=int, default=128)
parser.add_argument('--frame_w', type=int, default=128)

# Pytorch parameters
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)


args = parser.parse_args()

if args.data_type == "2d":
    args.num_frames = 64
    args.segment_size = 1
    args.batch_size = 80
    if args.train_dir == "":
        args.train_dir = "training-2d-%s-%s"%(args.transformations_type, args.anomaly_type)
else:
    args.batch_size = 256
    args.num_frames = 128
    args.segment_size = 16
    if args.train_dir == "":
        args.train_dir = "training-1d-%s-%s"%(args.transformations_type, args.func_type)

# Pytorch parameters
args.pin_memory = False
args.seed = 1

# train hyper parameters
args.epochs = 40
args.lr = 0.0001
args.lr_decay = 0.5
args.decay_epochs = 10
