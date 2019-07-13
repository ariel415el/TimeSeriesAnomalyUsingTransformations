import argparse
import os
import sys
import time
import numpy as np
import pickle
import tqdm

import data_loader
import models

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

def write_array_as_video(arr, path):
    print("writing array as video ", arr.shape)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'PIM1'), 25, (arr.shape[-2], arr.shape[-1]), isoColor=False)
    for i in range(arr.shape[0]):
        writer.write(arr[i]) 


def decrease_learning_rate(optimizer, lr_decay):
    for g in optimizer.param_groups:
        g['lr'] *= lr_decay


def train(args, encoder, decoder, device, train_loader, optimizer, epoch, tb_writer):
    encoder.train()
    decoder.train()
    device = torch.device("cuda")
    start = time.time()
    num_batchs_per_epoch = len(train_loader)

    loss_mini_batch = 0
    mini_batch_examples = 0
    mini_batch_correct = 0
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx > 1000000: 
        #     break
        #     
        # import pdb;pdb.set_trace()
        # if batch_idx ==0:
        #     for j in range(5):
        #         np_serie = data[j].cpu().numpy().reshape(data.shape[2])
        #         plt.plot(range(len(np_serie)), np_serie)
        #         # import pdb; pdb.set_trace()
        #         plt.savefig(os.path.join("train_debug","series_%d_%d"%(j,target[j].item())))
        #         plt.clf()
        data = data.to(device)
        target = target.to(device)

        output  = decoder(encoder(data))
        # loss = F.nll_loss(output, target)
        # import pdb;pdb.set_trace()
        loss = nn.CrossEntropyLoss()(output, target)
        
        loss_mini_batch += loss.item()
        loss.backward()

        pred = F.softmax(output,dim=1).argmax(dim=1, keepdim=True) # get the index of the max log-probability
        mini_batch_correct += pred.eq(target.view_as(pred)).sum().item()
        mini_batch_examples += len(pred)

        # import pdb;pdb.set_trace()
        if batch_idx % args.virtual_batch == 0 : 
            optimizer.step()
            optimizer.zero_grad()
            loss_mini_batch /= args.virtual_batch
            if batch_idx > 0 and batch_idx% (args.virtual_batch*50) == 0:
                gs = (epoch-1)*num_batchs_per_epoch+batch_idx
                tb_writer.add_scalar('train_loss', torch.tensor(loss_mini_batch), global_step=gs)
                num_processed_imgs = (batch_idx+1)*int(args.batch_size)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}/{}\texamples/sec: {:.5f}\tLR: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_mini_batch,
                    int(mini_batch_correct), mini_batch_examples,
                    float(num_processed_imgs / (time.time() - start) ),
                    optimizer.param_groups[0]['lr']))

            mini_batch_examples = 0
            mini_batch_correct = 0
            loss_mini_batch = 0

def validate(args, encoder, decoder, device, val_loader):
    encoder.eval()
    decoder.eval()
    val_loss = 0
    correct = 0
    num_batchs = len(val_loader)
    with torch.no_grad():

        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # if i ==0:
            #     for j in range(5):
            #         np_serie = data[j].cpu().numpy().reshape(data.shape[2])
            #         plt.plot(range(len(np_serie)), np_serie)
            #         # import pdb; pdb.set_trace()
            #         plt.savefig(os.path.join("val_debug","series_%d_%d"%(j,target[j].item())))
            #         plt.clf()

            # Inference
            output  = decoder(encoder(data))

            batch_loss = nn.CrossEntropyLoss()(output, target).item()
            # print("validation batch loss: %f "%batch_loss)
            val_loss += batch_loss # sum up batch loss

            output = F.softmax(output,dim=1)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # import pdb;pdb.set_trace()


    b_val_loss = val_loss /  num_batchs
    print('Validation: Average loss: {:.6f}({}/{}), correct {}/{}'.format(b_val_loss, val_loss, num_batchs, correct,len(val_loader.dataset)))

    return b_val_loss

def test(args, encoder,decoder, device, test_loader, all_permutations, debug_dir=None):
    encoder.eval()
    decoder.eval()

    result_dict ={i:{"score":0,"#":0} for i in range(2)}
    if not os.path.exists("test_debug"):
        os.makedirs("test_debug")
    with torch.no_grad():
        for i, (serie, label) in enumerate(test_loader):
            serie = serie.to(device)
            permed_series = []
            for j,perm in enumerate(all_permutations):
                # import pdb  ;pdb.set_trace()
                permed_series += [serie[:, perm, :, :].squeeze(0)]
  
            permed_series = torch.stack(permed_series)
            output = decoder(encoder(permed_series.cuda()))
            output = F.softmax(output,dim=1)

            pred = output.argmax(dim=1, keepdim=True)
            if i  < 10:
                write_array_as_video(serie.cpu().numpy()[0].astype(np.uint8), os.path.join("test_debug", "series_%d.avi"%i))
                for j in range(5):
                    write_array_as_video(permed_series[j].cpu().numpy().astype(np.uint8), os.path.join("test_debug", "series_%d_%d_%d.avi"%(i,j,label)))

            target = torch.tensor(range(len(all_permutations))).view(len(all_permutations),1).cuda()
            
            correct = pred.eq(target.view_as(pred)).sum().item()

            score = correct / float(len(all_permutations))
            result_dict[label.item()]["score"] += score
            result_dict[label.item()]["#"] += 1

    for k in result_dict:
        print("%d: avg_score (%d): %f"%(k,result_dict[k]["#"], result_dict[k]["score"]/float(result_dict[k]["#"]) ))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch time series Anomaly detection')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--virtual_batch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.4)

    parser.add_argument('--trained_encoder', type=str, default="")
    parser.add_argument('--trained_decoder', type=str, default="")
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default="train_dir")

    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    args = parser.parse_args()
 
    torch.manual_seed(args.seed)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}

    device = torch.device("cuda")

    train_dataset = data_loader.balls_dataset(frame_size=128, video_length=64, num_videos=100, max_permutaions=100)
    permutations = train_dataset.get_permutations()
    val_dataset = data_loader.balls_dataset(frame_size=128, video_length=200, num_videos=100, permutations=permutations)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)  

    if args.trained_decoder != "":
        print("Loading model")
        perms = np.loadtxt(os.path.join(args.train_dir, str(train_dataset)+"_perms.txt"), dtype=int)
        train_dataset.set_permutations(perms)
        val_dataset.set_permutations(perms)
        cnn_encoder = models.ResCNNEncoder().to(device)
        rnn_decoder = models.DecoderRNN(len(permutations)).to(device)
        cnn_encoder.load_state_dict(torch.load(args.trained_encoder))
        rnn_decoder.load_state_dict(torch.load(args.trained_decoder))
    else:
        np.savetxt(os.path.join(args.train_dir, str(train_dataset)+"_perms.txt"), permutations, fmt='%d')
        cnn_encoder = models.ResCNNEncoder().to(device)
        rnn_decoder = models.DecoderRNN(len(permutations)).to(device)

    if args.test:
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        # val_loss = validate(args, model, device, test_loader)
        # print("val_loss: ", val_loss)

        val_dataset.test()
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, **kwargs)

        test(args, cnn_encoder, rnn_decoder, device, test_loader, val_dataset.get_permutations())
        exit()


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    assert(train_dataset.get_permutations()==val_dataset.get_permutations())
    
    learnable_paramas = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                              list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                              list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())
    optimizer = optim.Adam(learnable_paramas, lr=args.lr)


    tb_writer = SummaryWriter(log_dir=args.train_dir)
    best_val = 99999
    all_params = [x for x in rnn_decoder.parameters()] + [x for x in cnn_encoder.parameters()]
    pytorch_total_params = sum(p.numel() for p in learnable_paramas if p.requires_grad)
    pytorch_total_params_2 = sum(p.numel() for p in all_params if p.requires_grad)
    all_params = sum(p.numel() for p in all_params)
    print("Learnable in model: ~%.1fk"%(pytorch_total_params/1000))
    print("Learnable in model: ~%.1fk"%(pytorch_total_params_2/1000))
    print("All in model: ~%.1fk"%(all_params/1000))
    print("Started training")
    for epoch in range(1, args.epochs + 1):

        train(args, cnn_encoder, rnn_decoder,  device, train_loader, optimizer, epoch, tb_writer)

        val_loss = validate(args, cnn_encoder, rnn_decoder, device, val_loader)

        # viz.line(X=[epoch*len(train_loader)],Y=[val_loss],win="train", update='append',name='val')
        gs = (epoch-1)*len(train_loader)
        tb_writer.add_scalar('val_loss', torch.tensor(val_loss), global_step=gs)
        if val_loss < best_val:
            best_val = val_loss                
            torch.save(cnn_encoder.state_dict(), os.path.join(args.train_dir, str(train_dataset) + "_encoder_%.5f_ckp.pt"%best_val))
            torch.save(rnn_decoder.state_dict(), os.path.join(args.train_dir, str(train_dataset) + "_decoder_%n.5f_ckp.pt"%best_val))

        decrease_learning_rate(optimizer, args.lr_decay)