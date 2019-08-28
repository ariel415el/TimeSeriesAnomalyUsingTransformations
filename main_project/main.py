import os
import time
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from transforms import *
from config import args
import dataset_1d
import dataset_2d
import models_1d
import models_2d

def log_and_print(f, s):
    print(s)
    if f is not None:
        f.write(s)
        f.write("\n")

def get_model(args):
    if args.data_type == "2d":
        model = models_2d.CNN3D(args.num_transforms, t_dim=args.num_frames, img_x=args.frame_w, img_y=args.frame_w).to(torch.device("cuda:%d"%args.gpu_id))
    else: # args.data_type == "1d"
        model = models_1d.conv_FC(serie_size=args.num_frames, num_classes=args.num_transforms).to(torch.device("cuda:%d"%args.gpu_id))

    return model

def get_transformer(args):
    if args.transformations_type == "Permutations":
        transformer = frame_permuter(args.num_transforms, args.num_frames, args.segment_size)

    else: # Affine
        if args.data_type == "2d":
            transformer = fixed_affine_image_transformer(args.num_transforms)
        else: # args.data_type == "1d"
            transformer = fixed_affine_1d_transformer(args.num_transforms, args.num_frames)

    return transformer

def get_val_datasets(args, transforms):
    if args.data_type == "2d":
        val_dataset = dataset_2d.balls_dataset(frame_w=args.frame_w,
                                                frame_h=args.frame_h,
                                                video_length=args.num_frames,
                                                num_videos=args.num_val_series,
                                                transforms=transforms,
                                                anomaly_type=args.anomaly_type)

    else: # args.data_type == "1d"
        val_dataset = dataset_1d.con_func_series_dataset(serie_length=args.num_frames,
                                                         num_series=args.num_val_series,
                                                         train=True,
                                                         transforms=transforms,
                                                         func_type=args.func_type)
    return val_dataset

def get_train_datasets(args, transforms):
    if args.data_type == "2d":
        train_dataset = dataset_2d.balls_dataset(frame_w=args.frame_w,
                                                  frame_h=args.frame_h,
                                                  video_length=args.num_frames,
                                                  num_videos=args.num_train_series,
                                                  transforms=transforms,
                                                  anomaly_type=args.anomaly_type)
    else: # args.data_type == "1d"
        train_dataset = dataset_1d.con_func_series_dataset(serie_length=args.num_frames,
                                                         num_series=args.num_train_series,
                                                         train=True,
                                                         transforms=transforms,
                                                         func_type=args.func_type)
    return train_dataset

def decrease_learning_rate(optimizer, lr_decay):
    for g in optimizer.param_groups:
        g['lr'] *= lr_decay

def train(args, model, train_loader, optimizer, epoch, tb_writer):
    model.train()
    start = time.time()
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(torch.device("cuda:%d"%args.gpu_id),non_blocking=True)
        target = target.to(torch.device("cuda:%d"%args.gpu_id), non_blocking=True)

        output = model(data)

        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = F.softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        mini_batch_correct = pred.eq(target.view_as(pred)).sum().item()
        mini_batch_examples = len(pred)

        gs = (epoch - 1) * len(train_loader) + batch_idx
        tb_writer.add_scalar('loss', torch.tensor(loss.item()), global_step=gs)
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
            data, target = data.to(torch.device("cuda:%d"%args.gpu_id)), target.to(torch.device("cuda:%d"%args.gpu_id))
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
    accuracy = correct / len(val_loader.dataset)
    return b_val_loss, accuracy

def test(model, test_loader, gpu_idx, log_file=None):
    model.eval()
    transforms = test_loader.dataset.get_transforms()
    result_dict = {i: [] for i in range(test_loader.dataset.get_number_of_test_classes())}
    with torch.no_grad():
        for i, (serie, label) in enumerate(test_loader):
            serie = serie.to(torch.device("cuda:%d"%gpu_idx))
            transformed_series = []
            for j, transform in enumerate(transforms):
                np_serie = serie.detach().cpu().numpy()[0].astype(np.float32)
                np_serie = transform(np_serie).astype(np.float32)
                transformed_series += [torch.tensor(np_serie).cpu()]
            transformed_series = torch.stack(transformed_series)
            output = model(transformed_series.to(torch.device("cuda:%d"%gpu_idx)))

            output = -1 * F.log_softmax(output, dim=1)
            score = torch.diagonal(output, 0, 0, 1).cpu().data.numpy().sum()

            result_dict[label.item()] += [score]

    for k in result_dict:
        log_and_print(log_file,"%d: avg_score (%d): %.8f" % (k, len(result_dict[k]), np.mean(result_dict[k])))
    all_fake_scores = []
    for i in range(1, len(result_dict)):
        all_fake_scores += result_dict[i]

    auc_scores = np.concatenate((result_dict[0], all_fake_scores))
    auc_labels = np.concatenate([np.zeros(len(result_dict[0])), np.ones(len(all_fake_scores))])
    auc = metrics.roc_auc_score(auc_labels, auc_scores)
    log_and_print(log_file, "AUC: %f"% auc)
    log_and_print(log_file,"Results fixing TPR to 100%")
    th = min(all_fake_scores)
    tp = np.sum(np.array(all_fake_scores) >= th)
    fp = np.sum(np.array(result_dict[0]) >= th)
    log_and_print(log_file,"th %f"%th)
    log_and_print(log_file,"tpr: %f" % (tp / len(all_fake_scores)))
    log_and_print(log_file,"fpr: %f" % (fp / len(result_dict[0])))

    log_and_print(log_file,"Results fixing TPR to 50%")
    th = np.median(np.array(all_fake_scores))
    tp = np.sum(np.array(all_fake_scores) >= th)
    fp = np.sum(np.array(result_dict[0]) >= th)
    log_and_print(log_file,"th %f"%th)
    log_and_print(log_file,"tpr: %f" % (tp / len(all_fake_scores)))
    log_and_print(log_file,"fpr: %f" % (fp / len(result_dict[0])))


if __name__ == '__main__':

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}
    torch.manual_seed(args.seed)

    model = get_model(args)
    transformer = get_transformer(args)

    transforms_file = os.path.join(args.train_dir, "transforms_def.npy")
    if args.trained_model != "":
        print("Loading model")
        model.load_state_dict(torch.load(args.trained_model))
        transformer.load_from_file(transforms_file)
    else:
        os.makedirs(args.train_dir, exist_ok=True)
        transformer.save_to_file(transforms_file)

    if args.test:
        test_log = open(os.path.join(args.train_dir, "test-log.txt"), "w")
        val_dataset = get_val_datasets(args, transformer.get_transforms())
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        val_loss, accuracy = validate(model, test_loader)
        log_and_print(test_log, "val_loss: %f; accuracy %f"%(val_loss,accuracy))

        val_dataset.test()
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

        test(model, test_loader, args.gpu_id, test_log)
        test_log.close()
        exit()

    elif args.debug:
        print("Debuging train and debug images")
        dataset = get_val_datasets(args, transformer.get_transforms())
        dataset.dump_debug_images("debug_%s"%args.data_type)
        exit()

    else:
        val_dataset = get_val_datasets(args, transformer.get_transforms())
        train_dataset =  get_train_datasets(args, transformer.get_transforms())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        # assert(train_dataset.get_permutations()==val_dataset.get_permutations())
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_writer = SummaryWriter(log_dir=os.path.join(args.train_dir,"train_log"))
        val_writer = SummaryWriter(log_dir=os.path.join(args.train_dir,"val_log"))

        best_val = 99999
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Parametes in model: ~%.1fk" % (pytorch_total_params / 1000))
        print("Started training")
        for epoch in range(1, args.epochs + 1):

            train(args, model, train_loader, optimizer, epoch, train_writer)

            val_loss, accuracy = validate(model, val_loader)

            gs = epoch * len(train_loader)
            val_writer.add_scalar('loss', torch.tensor(val_loss), global_step=gs)
            val_writer.add_scalar('accuracy', torch.tensor(accuracy), global_step=gs)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           os.path.join(args.train_dir, str(train_dataset) + "_%.5f_ckp.pt" % best_val))
            if epoch > 0 and epoch % args.decay_epochs == 0:
                decrease_learning_rate(optimizer, args.lr_decay)