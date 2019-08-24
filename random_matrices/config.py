import argparse

parser = argparse.ArgumentParser(description='PyTorch time series Anomaly detection')
# general
parser.add_argument('--data_type', type=str, default="2d")
parser.add_argument('--trained_model', type=str, default="")
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--train_dir', type=str, default="train_dir")

# train hyper parameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay', type=float, default=0.4)
parser.add_argument('--decay_epochs', type=int, default=10)

# 2d dataset config
parser.add_argument('--frame_h', type=int, default=128)
parser.add_argument('--frame_w', type=int, default=128)
parser.add_argument('--num_train_videos', type=int, default=500)
parser.add_argument('--num_val_videos', type=int, default=50)
parser.add_argument('--num_frames', type=int, default=64)
parser.add_argument('--num_transforms', type=int, default=100)

# Pytorch parameters
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--pin_memory', action='store_false', default=True)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

