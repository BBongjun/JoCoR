# -*- coding:utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.ssd import *
import argparse, sys
import datetime
from algorithm.jocor import JoCoR
import random





parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=0.02)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='pairflip')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--co_lambda', type=float, default=0.1)
parser.add_argument('--adjust_lr', type=int, default=0)
parser.add_argument('--model_type', type=str, help='[mlp,cnn]', default='TCN')
parser.add_argument('--save_model', type=str, help='save model?', default="False")
parser.add_argument('--save_result', type=str, help='save result?', default="True")
parser.add_argument('--data_path', default='', type=str, help='path to dataset')

parser.add_argument('--channels', default=40, type=int)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--in_channels', default=27, type=int)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--out_channels', default=80, type=int)
parser.add_argument('--reduced_size', default=40, type=int)
parser.add_argument('--clf_hidden_node', default=80, type=int)
parser.add_argument('--clf_dropout_rate', default=0.2, type=int)

parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--gpuid', default=0, type=int)

args = parser.parse_args()

# # Seed
# torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    #torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

def set_env(args):
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

set_env(args)

# Hyper Parameters
batch_size = 256
learning_rate = args.lr


if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
        
    if args.dataset == 'ssd':
        init_epoch = 5
        # args.epoch_decay_start = 100
        # args.n_epoch = 200
        filter_outlier = False
        args.model_type = "TCN"
        loader = ssd_dataloader(args.dataset, r=args.noise_rate, batch_size=args.batch_size,
                            num_workers=5,
                            root_dir=args.data_path,
                            args=args,
                            noise_file='%s/%.1f.json' % (args.data_path, args.noise_rate))

        train_dataset, train_loader = loader.run('train')

        test_dataset, test_loader = loader.run('test')

    # Define models
    print('building model...')

    model = JoCoR(args, train_dataset, device)

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0

    # evaluate models with random weights
    test_acc1, test_acc2 = model.evaluate(test_loader)

    print(
        'Epoch [%d/%d] Test Accuracy on the %s test sequences: Model1 %.4f %% Model2 %.4f ' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))


    acc_list = []
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = model.train(train_loader, epoch)

        # evaluate models
        test_acc1, test_acc2 = model.evaluate(test_loader)

        # save results
        if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test sequences: Model1 %.4f %% Model2 %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        else:
            # save results
            mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
            mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test sequences: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1,
                    mean_pure_ratio2))


        if epoch >= args.n_epoch-5:
            acc_list.extend([test_acc1, test_acc2])

    #avg_acc = sum(acc_list)/len(acc_list)
    #print(len(acc_list))
    #print("the average acc in last 5 epochs: {}".format(str(avg_acc)))

    print('\n')
    print('========== Test per Disk ==========')
    accuracy_1, macro_f1_1, weighted_f1_1, FDR_1, FAR_1, accuracy_2, macro_f1_2, weighted_f1_2, FDR_2, FAR_2 = model.evaluate(test_loader, mode='final_test')



if __name__ == '__main__':
    main()
