import argparse
import os
import sys
import time

import data_loader
import models

import torch
import torch.optim as optim
import visdom
import torch.nn.functional as F


def decrease_learning_rate(optimizer, lr_decay):
    for g in optimizer.param_groups:
        g['lr'] *= lr_decay


def train(args, model, device, train_loader, optimizer, epoch, vis):
    model.train()
    device = torch.device("cuda")
    start = time.time()
    num_batchs_per_epoch = len(train_loader)

    loss_mini_batch = 0
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > args.limit_train:
            break
        data = data.to(device)
        assert(len(data) == args.batch_size)
        target = target.to(device)

        # Inference
        output  = model(data)
        loss = F.nll_loss(output, target)
        
        loss_mini_batch += loss.item()
        loss.backward()

        if batch_idx % args.virtual_batch == 0 : 
            optimizer.step()
            optimizer.zero_grad()
            loss_mini_batch /= args.virtual_batch
            if batch_idx > 0 and batch_idx% (args.virtual_batch*50) == 0:
                train_plot = viz.line(X=[epoch*num_batchs_per_epoch+batch_idx], Y=[loss_mini_batch], win="train", update='append',name='train')
                num_processed_imgs = (batch_idx+1)*int(args.batch_size)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\texamples/sec: {:.5f}\tLR: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_mini_batch,
                    float(num_processed_imgs / (time.time() - start) ),
                    optimizer.param_groups[0]['lr']))

            loss_mini_batch = 0

def validate(args, model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    print("# Validation")
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if i > args.limit_val:
                break
            data, target = data.to(device), target.to(device)

            # Inference
            output = model(data)
            
            val_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # if debug_dir is not None:
            #     if not os.path.exists(debug_dir):
            #         os.makedirs(debug_dir)
            #     input_im= data.cpu().detach().numpy()[0].transpose(1,2,0)
            #     output= output[0].cpu().detach().numpy()
            #     cv2.imwrite(os.path.join(debug_dir, "%d_in.png"%i) ,(255*input_im).astype(int))
            #     cv2.imwrite(os.path.join(debug_dir, "%d_out.png"%i) ,(255*np.amax(output,axis=0)).astype(int))

    val_loss /= len(val_loader.dataset)
    print('Validation: Average loss: {:.4f}, correctL {}'.format(val_loss, correct))

    return val_loss


def test(args, model, device, test_loader, debug_dir=None):
    model.eval()
    result_dict ={str(i):{"score":0,"#":0} for i in range(10)}
    with torch.no_grad():
        for i, (org_img, transformed_tensors, transform_labels, image_label) in enumerate(test_loader):
            assert(transformed_tensors.shape[0] == 1)
            transformed_tensors = transformed_tensors.to(device)[0]
            output = model(transformed_tensors)
            pred = output.argmax(dim=1, keepdim=True)
            target = torch.tensor(range(72)).view(72,1).cuda()
            correct = pred.eq(target.view_as(pred)).sum().item()
            score = correct / float(len(transformed_tensors))
            result_dict[image_label[0]]["score"] += score
            result_dict[image_label[0]]["#"] += 1
    for k in result_dict:
        print("%s: avg_score (%d): %f"%(k,result_dict[k]["#"], result_dict[k]["score"]/float(result_dict[k]["#"]) ))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch time series Anomaly detection')
    parser.add_argument('--train_root', type=str)
    parser.add_argument('--val_root', type=str)
    parser.add_argument('--normal_digit', type=str, default="1")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--virtual_batch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.4)

    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--save-model', action='store_true', default=False)
    # parser.add_argument('--test', type=str, default="", help='path to root')
    parser.add_argument('--test', action='store_true', default=False)


    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--limit_train', type=int, default=1000)
    parser.add_argument('--limit_val', type=int, default=50)
    args = parser.parse_args()
 
    torch.manual_seed(args.seed)



    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}



    device = torch.device("cuda")

    train_dataset = data_loader.mnist_png_dataset(os.path.join(args.train_root, args.normal_digit))
    val_dataset = data_loader.mnist_png_dataset(os.path.join(args.val_root, args.normal_digit))

    model = models.mnist_arch(train_dataset.get_num_transformations()).to(device)

    if args.trained_model != "":
        print("Loading model")
        model.load_state_dict(torch.load(args.trained_model))

    if args.test:
        test_dataset = data_loader.mnist_png_dataset_test(args.val_root, args.normal_digit, samples_per_class=30)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

        test(args, model, device, test_loader)
        exit()


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    viz = visdom.Visdom()
    best_val = 99999

    print("Started training")
    for epoch in range(1, args.epochs + 1):

        train(args, model,  device, train_loader, optimizer, epoch, viz)

        val_loss = validate(args, model, device, val_loader)

        viz.line(X=[epoch*len(train_loader)],Y=[val_loss],win="train", update='append',name='val')
        if val_loss < best_val:
            best_val = val_loss
            if not os.path.exists("trained_models"):
                os.makedirs("trained_models")
            torch.save(model.state_dict(), os.path.join("trained_models", "normal_" + args.normal_digit + "_%.3f_ckp.pt"%best_val))

        decrease_learning_rate(optimizer, args.lr_decay)