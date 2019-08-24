import os
import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from transforms import *
from utils import  write_array_as_video
from config import args
import dataset_1d
import dataset_2d
import models

def get_val_datasets(args):
    if args.data_type == "2d":
        val_dataset = dataset_2d.balls_dataset(frame_w=args.frame_w,
                                                frame_h=args.frame_h,
                                                video_length=args.num_frames,
                                                num_videos=args.num_val_videos,
                                                transforms=transformer.get_transforms())


    return val_dataset

def get_train_datasets(args):
    if args.data_type == "2d":
        train_dataset = dataset_2d.balls_dataset(frame_w=args.frame_w,
                                                  frame_h=args.frame_h,
                                                  video_length=args.num_frames,
                                                  num_videos=args.num_train_videos,
                                                  transforms=transformer.get_transforms())
    return train_dataset

def decrease_learning_rate(optimizer, lr_decay):
    for g in optimizer.param_groups:
        g['lr'] *= lr_decay

def train(args, model, train_loader, optimizer, epoch, tb_writer):
    model.train()
    start = time.time()
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(data)

        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = F.softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        mini_batch_correct = pred.eq(target.view_as(pred)).sum().item()
        mini_batch_examples = len(pred)

        gs = (epoch - 1) * len(train_loader) + batch_idx
        tb_writer.add_scalar('train_loss', torch.tensor(loss.item()), global_step=gs)
        num_processed_imgs = (batch_idx + 1) * int(args.batch_size)

        if batch_idx % 50 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}/{}\texamples/sec: {:.5f}\tLR: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    int(mini_batch_correct),
                    mini_batch_examples,
                    float(num_processed_imgs / (time.time() - start)),
                    optimizer.param_groups[0]['lr']))

def validate(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    num_batchs = len(val_loader)
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            # Inference
            output = model(data)

            batch_loss = nn.CrossEntropyLoss()(output, target).item()
            # print("validation batch loss: %f "%batch_loss)
            val_loss += batch_loss  # sum up batch loss

            output = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    b_val_loss = val_loss / num_batchs
    print('Validation: Average loss: {:.6f}({}/{}), correct {}/{}'.format(b_val_loss, val_loss, num_batchs, correct,
                                                                          len(val_loader.dataset)))

    return b_val_loss

def test(args, model, test_loader, transforms, debug_dir=None):
    model.eval()
    result_dict = {i: [] for i in range(test_loader.dataset.get_number_of_data_classes())}
    os.makedirs("test_debug", exist_ok=True)
    with torch.no_grad():
        for i, (serie, label) in enumerate(test_loader):
            serie = serie.cuda()
            transformed_series = []
            for j, transform in enumerate(transforms):
                np_serie = serie.detach().cpu().numpy()[0].astype(np.float32)
                np_serie = transform(np_serie)
                transformed_series += [torch.tensor(np_serie).cpu()]
            transformed_series = torch.stack(transformed_series)
            output = model(transformed_series.cuda())
            if debug_dir is not None:
                os.makedirs(debug_dir,exist_ok=True)
                if i < 5:
                    write_array_as_video(serie.cpu().numpy()[0].astype(np.uint8),
                                         os.path.join(debug_dir, "series_idx-%d_lable-%d.avi" % (i, label.item())))
                    for j in range(30, 35):
                        write_array_as_video(transformed_series[j].cpu().numpy().astype(np.uint8),
                                             os.path.join(debug_dir, "series_%d_%d_%d.avi" % (i, j, label)))

            output = -1 * F.log_softmax(output, dim=1)
            score = torch.diagonal(output, 0, 0, 1).cpu().data.numpy().sum()

            result_dict[label.item()] += [score]

    for k in result_dict:
        print("%d: avg_score (%d): %.8f" % (k, len(result_dict[k]), np.mean(result_dict[k])))
    all_fake_scores = []
    for i in range(1, len(result_dict)):
        all_fake_scores += result_dict[i]

    th = min(all_fake_scores)
    tp = np.sum(np.array(all_fake_scores) >= th)
    fp = np.sum(np.array(result_dict[0]) >= th)
    print("th", th)
    print("tpr: %f" % (tp / len(all_fake_scores)))
    print("fpr: %f" % (fp / len(result_dict[0])))

    th = np.array(all_fake_scores).mean()
    tp = np.sum(np.array(all_fake_scores) >= th)
    fp = np.sum(np.array(result_dict[0]) >= th)
    print("th", th)
    print("tpr: %f" % (tp / len(all_fake_scores)))
    print("fpr: %f" % (fp / len(result_dict[0])))


if __name__ == '__main__':
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}
    torch.manual_seed(args.seed)
    # model_calss = models.conv_FC
    # model_calss = models.multi_head_FC
    # model = model_calss(args.num_transforms, args.num_frames, args.frame_w, args.frame_h).to(device)

    model_calss = models.CNN3D
    model = model_calss(args.num_transforms).cuda()

    transformer = frame_permuter(args.num_transforms, args.num_frames, 1)

    transforms_file = os.path.join(args.train_dir, "transforms_def.npy")
    if args.trained_model != "":
        print("Loading model")
        model.load_state_dict(torch.load(args.trained_model))
        transformer.load_from_file(transforms_file)
    else:
        os.makedirs(args.train_dir, exist_ok=True)
        transformer.save_to_file(transforms_file)

    if args.test:
        val_dataset = get_val_datasets(args)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        # val_loss = validate(args, model, device, test_loader)
        # print("val_loss: ", val_loss)

        val_dataset.test()
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

        test(args, model, device, test_loader, transformer.get_transforms())

    else:
        val_dataset = get_val_datasets(args)
        train_dataset =  get_train_datasets(args)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        # assert(train_dataset.get_permutations()==val_dataset.get_permutations())
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        tb_writer = SummaryWriter(log_dir=args.train_dir)
        best_val = 99999
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Parametes in model: ~%.1fk" % (pytorch_total_params / 1000))
        print("Started training")
        for epoch in range(1, args.epochs + 1):

            train(args, model, train_loader, optimizer, epoch, tb_writer)

            val_loss = validate(args, model, val_loader)

            gs = (epoch - 1) * len(train_loader)
            tb_writer.add_scalar('val_loss', torch.tensor(val_loss), global_step=gs)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           os.path.join(args.train_dir, str(train_dataset) + "_%.5f_ckp.pt" % best_val))
            if epoch > 0 and epoch % args.decay_epochs == 0:
                decrease_learning_rate(optimizer, args.lr_decay)