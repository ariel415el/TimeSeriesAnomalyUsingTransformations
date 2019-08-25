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

import matplotlib.pyplot as plt
def decrease_learning_rate(optimizer, lr_decay):
    for g in optimizer.param_groups:
        g['lr'] *= lr_decay


def train(args, model, device, train_loader, optimizer, epoch, tb_writer):
    model.train()
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
        if batch_idx ==0:
            for j in range(5):
                np_serie = data[j].cpu().numpy().reshape(data.shape[2])
                plt.plot(range(len(np_serie)), np_serie)
                # import pdb; pdb.set_trace()
                plt.savefig(os.path.join("train_debug","series_%d_%d"%(j,target[j].item())))
                plt.clf()
            
        data = data.to(device)
        target = target.to(device)

        output  = model(data)
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

def validate(args, model, device, val_loader):
    model.eval()
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
            output = model(data)

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

def test(args, model, device, test_loader, all_permutations, debug_dir=None):
    model.eval()
    result_dict ={i:{"score":0,"#":0} for i in range(2)}
    if not os.path.exists("test_debug"):
        os.makedirs("test_debug")
    with torch.no_grad():
        for i, (serie, noise_type) in enumerate(test_loader):
            serie = serie.to(device)

            permed_series = []
            for j,perm in enumerate(all_permutations):
                # import pdb  ;pdb.set_trace()
                permed_series += [serie[:, perm].reshape(1,-1)]
  
            permed_series = torch.stack(permed_series)
            output = model(permed_series.cuda())
            output = F.softmax(output,dim=1)

            pred = output.argmax(dim=1, keepdim=True)
            if i  < 10:
                plt.plot(range(len(permed_series[0][0].cpu().numpy())), serie.cpu().numpy().reshape(-1))
                plt.savefig(os.path.join("test_debug","series_%d.png"%i))
                plt.clf()
                for j in range(5):
                    np_serie = permed_series[j][0].cpu().numpy()
                    plt.plot(range(len(np_serie)), np_serie)
                    plt.savefig(os.path.join("test_debug","series_%d_%d_%d.png"%(i,j,noise_type)))
                    plt.clf()

            target = torch.tensor(range(len(all_permutations))).view(len(all_permutations),1).cuda()
            
            correct = pred.eq(target.view_as(pred)).sum().item()

            score = correct / float(len(all_permutations))
            result_dict[noise_type.item()]["score"] += score
            result_dict[noise_type.item()]["#"] += 1

    for k in result_dict:
        print("%d: avg_score (%d): %f"%(k,result_dict[k]["#"], result_dict[k]["score"]/float(result_dict[k]["#"]) ))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch time series Anomaly detection')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--virtual_batch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.4)

    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default="train_dir")

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    args = parser.parse_args()
 
    torch.manual_seed(args.seed)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}

    device = torch.device("cuda")

    model_calss = models.conv_FC

    train_dataset = data_loader.con_func_series_dataset(num_series=10000, max_permutaions=100)
    permutations = train_dataset.get_permutations()
    val_dataset = data_loader.con_func_series_dataset(num_series=1000, permutations=permutations)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)  

    if args.trained_model != "":
        print("Loading model")
        perms = np.loadtxt(os.path.join(args.train_dir, str(train_dataset)+"_perms.txt"), dtype=int)
        train_dataset.set_permutations(perms)
        val_dataset.set_permutations(perms)
        model = model_calss(train_dataset.get_serie_length(), len(perms)).to(device)
        model.load_state_dict(torch.load(args.trained_model))
    else:
        np.savetxt(os.path.join(args.train_dir, str(train_dataset)+"_perms.txt"), permutations, fmt='%d')
        model = model_calss(train_dataset.get_serie_length(), len(permutations)).to(device)

    if args.test:
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        val_loss = validate(args, model, device, test_loader)

        val_dataset.test()
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, **kwargs)

        print("val_loss: ", val_loss)
        test(args, model, device, test_loader, val_dataset.get_permutations())
        exit()


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    assert(train_dataset.get_permutations()==val_dataset.get_permutations())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    tb_writer = SummaryWriter(log_dir=args.train_dir)
    best_val = 99999
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parametes in model: ~%.1fk"%(pytorch_total_params/1000))
    print("Started training")
    for epoch in range(1, args.epochs + 1):

        train(args, model,  device, train_loader, optimizer, epoch, tb_writer)

        val_loss = validate(args, model, device, val_loader)

        # viz.line(X=[epoch*len(train_loader)],Y=[val_loss],win="train", update='append',name='val')
        gs = (epoch-1)*len(train_loader)
        tb_writer.add_scalar('val_loss', torch.tensor(val_loss), global_step=gs)
        if val_loss < best_val:
            best_val = val_loss                
            torch.save(model.state_dict(), os.path.join(args.train_dir, str(train_dataset) + "_%.5f_ckp.pt"%best_val))

        decrease_learning_rate(optimizer, args.lr_decay)