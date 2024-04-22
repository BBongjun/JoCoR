# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import MLPNet,CNN
from model import causal_cnn
import numpy as np
import pandas as pd
from common.utils import accuracy
from datetime import datetime
from tqdm import tqdm
from algorithm.loss import loss_jocor
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, accuracy_score, recall_score, precision_score


class JoCoR:
    def __init__(self, args, train_dataset, device):

        # Hyper Parameters
        self.batch_size = 128
        learning_rate = args.lr

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        # if args.model_type == "cnn":
        #     self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        #     self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        # elif args.model_type == "mlp":
        #     self.model1 = MLPNet()
        #     self.model2 = MLPNet()

        if args.model_type == "TCN":
            self.model1 = causal_cnn.TCNClassifier(in_channels=args.in_channels,
                    channels=args.channels,
                    depth=args.depth,
                    reduced_size=args.reduced_size,
                    out_channels=args.out_channels,
                    kernel_size=args.kernel_size,
                    clf_hidden_node=args.clf_hidden_node,
                    clf_dropout_rate=args.clf_dropout_rate,
                    num_class=args.num_classes)
            
            self.model2 = causal_cnn.TCNClassifier(in_channels=args.in_channels,
                    channels=args.channels,
                    depth=args.depth,
                    reduced_size=args.reduced_size,
                    out_channels=args.out_channels,
                    kernel_size=args.kernel_size,
                    clf_hidden_node=args.clf_hidden_node,
                    clf_dropout_rate=args.clf_dropout_rate,
                    num_class=args.num_classes)
            
        self.model1.to(device)
        #print(self.model1.parameters)

        self.model2.to(device)
        #print(self.model2.parameters)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        self.loss_fn = loss_jocor


        self.adjust_lr = args.adjust_lr

    # Evaluate the Model
    def evaluate(self, test_loader, mode='test'):
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        if mode == "test":
            print('Evaluating ...')
            with torch.no_grad():        
                correct1 = 0
                total1 = 0
                for inputs, target, _ in test_loader:
                    inputs = inputs[:,1:,:].to(self.device)
                    logits1 = self.model1(inputs)
                    outputs1 = F.softmax(logits1, dim=1)
                    _, pred1 = torch.max(outputs1.data, 1)
                    total1 += target.size(0)
                    correct1 += (pred1.cpu() == target).sum()

                correct2 = 0
                total2 = 0
                for inputs, target, _ in test_loader:
                    inputs = inputs[:,1:,:].to(self.device)
                    logits2 = self.model2(inputs)
                    outputs2 = F.softmax(logits2, dim=1)
                    _, pred2 = torch.max(outputs2.data, 1)
                    total2 += target.size(0)
                    correct2 += (pred2.cpu() == target).sum()

                acc1 = 100 * float(correct1) / float(total1)
                acc2 = 100 * float(correct2) / float(total2)

                
            return acc1, acc2
                
        elif mode =='final_test':
            print('Evaluating per disk...')
            disk_id_list1, disk_id_list2 = [], []
            predictions1, labels1 = [], []
            predictions2, labels2 = [], []
            with torch.no_grad():
                correct1 = 0
                for inputs, target, _ in test_loader:
                    disk_id_list1.extend(inputs[:,0,0].cpu().numpy().flatten().tolist())
                    inputs = inputs[:,1:,:].to(self.device)
                    logits1 = self.model1(inputs)
                    outputs1 = F.softmax(logits1, dim=1)
                    _, pred1 = torch.max(outputs1.data, 1)
                    predictions1.extend(pred1.cpu().tolist())
                    labels1.extend(target.cpu().tolist())

                correct2 = 0
                for inputs, target, _ in test_loader:
                    disk_id_list2.extend(inputs[:,0,0].cpu().numpy().flatten().tolist())
                    inputs = inputs[:,1:,:].to(self.device)
                    logits2 = self.model2(inputs)
                    outputs2 = F.softmax(logits2, dim=1)
                    _, pred2 = torch.max(outputs2.data, 1)
                    predictions2.extend(pred2.cpu().numpy().tolist())
                    labels2.extend(target.cpu().numpy().tolist())
            df1 = pd.DataFrame({'disk_id':disk_id_list1,
                'label':labels1,
                'pred':predictions1})
            
            df2 = pd.DataFrame({'disk_id':disk_id_list2,
                'label':labels2,
                'pred':predictions2})
            
            # 디스크별로 라벨과 예측 결과를 집계 (모델 1)
            df1 = pd.DataFrame({'disk_id': disk_id_list1, 'label': labels1, 'pred': predictions1})
            disk_results1 = df1.groupby('disk_id').agg({'label': 'max', 'pred': 'max'})

            # 디스크별로 라벨과 예측 결과를 집계 (모델 2)
            df2 = pd.DataFrame({'disk_id': disk_id_list2, 'label': labels2, 'pred': predictions2})
            disk_results2 = df2.groupby('disk_id').agg({'label': 'max', 'pred': 'max'})

            # 성능 지표 계산 및 출력 (모델 1)
            accuracy_1, macro_f1_1, weighted_f1_1, FDR_1, FAR_1 = self.print_metrics(disk_results1, "Model 1")

            # 성능 지표 계산 및 출력 (모델 2)
            accuracy_2, macro_f1_2, weighted_f1_2, FDR_2, FAR_2 = self.print_metrics(disk_results2, "Model 2")

            # 결과를 파일로 저장
            now = datetime.now()
            disk_results1.to_csv(f'./test_result/test_result_model1_{now.month}_{now.day}_{now.hour}.csv', index=True)
            disk_results2.to_csv(f'./test_result/test_result_model2_{now.month}_{now.day}_{now.hour}.csv', index=True)

            return accuracy_1, macro_f1_1, weighted_f1_1, FDR_1, FAR_1, accuracy_2, macro_f1_2, weighted_f1_2, FDR_2, FAR_2

    def print_metrics(self, disk_results, model_name):
        accuracy = accuracy_score(disk_results['label'], disk_results['pred'])
        macro_f1 = f1_score(disk_results['label'], disk_results['pred'], average='macro')
        weighted_f1 = f1_score(disk_results['label'], disk_results['pred'], average='weighted')
        FDR = recall_score(disk_results['label'], disk_results['pred']) * 100
        FAR = (1 - recall_score(disk_results['label'], disk_results['pred'], pos_label=0)) * 100

        print(f"Results for {model_name}:")
        print(classification_report(disk_results['label'], disk_results['pred'], target_names=['healthy', 'failed'], digits=4))
        print('\n')
        print(confusion_matrix(disk_results['label'], disk_results['pred']))
        print('\n')
        print(f"Final test result : Acc : {accuracy:.4f}, Macro_f1 : {macro_f1:.4f}, Weighted_f1 : {weighted_f1:.4f}, FDR : {FDR:.4f}, FAR : {FAR:.4f}")

        return accuracy, macro_f1, weighted_f1, FDR, FAR

    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (inputs, target, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()

            inputs = inputs[:,1:,:].to(self.device)
            target = Variable(target).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(inputs)
            prec1 = accuracy(logits1, target, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(inputs)
            prec2 = accuracy(logits2, target, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, target, self.rate_schedule[epoch],
                                                                 ind, self.noise_or_not, self.co_lambda)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.item(), loss_2.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
