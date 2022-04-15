import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from capsnet import CapsuleNet
import numpy as np
import pickle
from sklearn.model_selection import KFold
import time
import math
import torch.utils.data as Data
from single_label import *
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing
# from bce import binary_cross_entropy_with_logits
learning_rate = 0.0001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def Normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def seqreduction(seqs):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    origin_seq = []
    for i in range(len(seqs)):
        elemt, st = [], seqs[i]
        for j in range(len(st)):
            if st[j] == 0:
                continue
            elemt.append(amino_acids[int(st[j])])
        origin_seq.append(elemt)
    return origin_seq


def read_raw_data(rawdata_dir):
    rawdata_dir = rawdata_dir.rstrip("/")
    # gii = open(rawdata_dir + '/' + 'ch_one.pckl', 'rb')

    gii = open(rawdata_dir + '/' + 'seqfeature.pckl', 'rb')
    seqs = pickle.load(gii)
    gii.close()
    # drug_feature_one = Normalize(drug_feature_one)
    drug_feature_one = preprocessing.scale(seqs)

    gii = open(rawdata_dir + '/' + 'profeature.pckl', 'rb')
    drug_feature_two = pickle.load(gii)
    gii.close()
    # drug_feature_two = Normalize(drug_feature_two)
    drug_feature_two = preprocessing.scale(drug_feature_two)


    gii = open(rawdata_dir + '/' + 'IUfeature.pckl', 'rb')
    drug_feature_three = pickle.load(gii)
    gii.close()
    # drug_feature_three = Normalize(drug_feature_three)
    drug_feature_three = preprocessing.scale(drug_feature_three)


    gii = open(rawdata_dir + '/' + 'ss3feature.pckl', 'rb')
    drug_feature_four = pickle.load(gii)
    gii.close()
    # drug_feature_four = Normalize(drug_feature_four)
    drug_feature_four = preprocessing.scale(drug_feature_four)

    gii = open(rawdata_dir + '/' + 'glbfeature.pckl', 'rb')
    drug_feature_five = pickle.load(gii)
    gii.close()
    # drug_feature_five = Normalize(drug_feature_five)
    drug_feature_five = preprocessing.scale(drug_feature_five)

    gii = open(rawdata_dir + '/' + 'kmerfeature.pckl', 'rb')
    drug_feature_six = pickle.load(gii)
    gii.close()
    # drug_feature_five = Normalize(drug_feature_five)
    drug_feature_six = preprocessing.scale(drug_feature_six)

    gii = open(rawdata_dir + '/' + 'tngfeature.pckl', 'rb')
    drug_feature_seven = pickle.load(gii)
    gii.close()
    # drug_feature_five = Normalize(drug_feature_five)
    drug_feature_seven = preprocessing.scale(drug_feature_seven)


    gii = open(rawdata_dir + '/' + 'label.pckl', 'rb')
    drug_label = pickle.load(gii)
    gii.close()

    # gii = open(rawdata_dir + '/' + 'glove_wordEmbedding.pkl', 'rb')
    # ATC_feature = pickle.load(gii)
    # gii.close()
    Drug_feature = np.concatenate((drug_feature_one, drug_feature_two,drug_feature_three,drug_feature_four,drug_feature_five,drug_feature_six,drug_feature_seven), axis=1)
    return Drug_feature, drug_label,seqs

def ten_fold_files(Drug_feature, Drug_ATC_label,seqs, args):
    X = Drug_feature
    # index = np.zeros((len(X), 1))
    # for i in range(len(X)):
    #     index[i,0] = i
    # X_index = np.hstack((index, X))
    y = Drug_ATC_label
    fold = 1
    total_Precision, total_Coverages, total_Abs_True_Rates, total_Abs_False_Rates, total_Accuracys, total_Hamming_losses = [], [], [], [], [], [], [], [],[],[],[],[],[],[],[],[],[],[],[],[]
    kk = 0
    kfold = KFold(10, random_state=1, shuffle=True)
    zeros = np.zeros((1,5))
    temp = []
    P,R,FPR,TPR = [],[],[],[]
    for k, (train, test) in enumerate(kfold.split(X, y)):
        kk = kk + 1
        # train_drug_label = Drug_ATC_label.copy()
        # for i in range(test.shape[0]):
        #     train_drug_label[i] = zeros
        print("==================================fold {} start".format(fold))
        print("Train Index:", train, ",Test Index:", test)
        total_Precision, total_Coverage, total_Abs_True_Rate, total_Abs_False_Rate, total_Accuracy, total_Hamming_loss ,labels = train_test(X[train], y[train], X[test], y[test], args, test,train,seqs[test])
        total_Precision.append(total_Precision)
        total_Coverages.append(total_Coverage)
        total_Abs_True_Rates.append(total_Abs_True_Rate)
        total_Abs_False_Rates.append(total_Abs_False_Rate)
        total_Accuracys.append(total_Accuracy)
        total_Hamming_losses.append(total_Hamming_loss)

        print("==================================fold {} end".format(fold))
        fold += 1
        print("=====Ten fold cross validation result:\n\t\t "
              'Precision: {:.4f}, Coverage: {:.4f}, Abs_True_Rate: {:.4f}, '
                      'Abs_False_Rate: {:.4f}, Accuracy: {:.4f}, Hamming_loss: {:.4f}'.format(np.mean(total_Precision), np.mean(total_Coverages),
                                                   np.mean(total_Abs_True_Rates), np.mean(total_Abs_False_Rates), np.mean(total_Accuracys), np.mean(total_Hamming_losses)))


        sys.stdout.flush()


def train_test(X_index_train, y_train, x_index_test, y_test, args, test,train,seqs):
    train_dataset = Data.TensorDataset(torch.from_numpy(X_index_train), torch.from_numpy(y_train),torch.from_numpy(train))
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_dataset = Data.TensorDataset(torch.from_numpy(x_index_test), torch.from_numpy(y_test),torch.from_numpy(test),torch.from_numpy(seqs))
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model_dir = args.model_dir.rstrip("/")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model = CapsuleNet().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start = time.time()
    total_step = len(train_loader)
    print("train_step: {}".format(total_step))
    acc_best = 0
    weight = []
    for epoch in range(args.num_epochs):
        model.train()
        accs_epoch,temp = [],[]
        alltrain_index = []
        for step, (batch_x, batch_y,train_index) in enumerate(train_loader):
            # batch_x = batch_x_index.numpy()[:,1::]
            # index = batch_x_index.numpy()[:, 0]
            # batch_x = torch.from_numpy(batch_x)
            feature1 = batch_x[:, 0:517].to(device)
            feature2 = batch_x[:, 517:1034].to(device)
            feature3 = batch_x[:, 1034:1551].to(device)
            feature4 = batch_x[:, 1551:2068].to(device)
            feature5 = batch_x[:, 2068:2073].to(device)
            feature6 = batch_x[:, 2073:2093].to(device)
            feature7 = batch_x[:, 2093::].to(device)
            batch_y = batch_y.to(device)
            outputs = model(feature1.float(),feature2.float(),feature3.float(),feature4.float(),feature5.float(),feature6.float(),feature7.float(),len(feature1))
            loss = criterion(outputs.float(), batch_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            alltrain_index.append(train_index.numpy())
            if (step + 1) % args.step == 0:
                if device.type == 'cuda':
                    batch_y = batch_y.cpu()
                    predicted = outputs.cpu()
                predicted = predicted.detach().numpy()
                predicted[predicted >= 0.5] = 1
                predicted[predicted < 0.5] = 0
                predicted = predicted.astype(int)
                D = batch_y.numpy().shape[0]
                i_Aiming = Precision(batch_y.numpy(), predicted, D)
                i_Coverage = Coverage(batch_y.numpy(), predicted, D)
                i_Abs_True_Rate = Abs_True_Rate(batch_y.numpy(), predicted, D)
                i_Abs_False_Rate = Abs_False_Rate(batch_y.numpy(), predicted, D)
                i_Accuracy = Accuracy(batch_y.numpy(), predicted, D)
                i_Hamming_loss = Hamming_loss(batch_y.numpy(), predicted, D)
                zhc = i_Accuracy
                accs_epoch.append(zhc)
                time_cost = time.time() - start
                print('TRAIN Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
                      'Aiming: {:.4f}, Coverage: {:.4f}, Abs_True_Rate: {:.4f}, '
                      'Abs_False_Rate: {:.4f}, Accuracy: {:.4f}, Hamming_loss: {:.4f}, Time: {:.2f}s'

                      .format(epoch + 1, args.num_epochs, step + 1, total_step, loss.item(),
                              i_Aiming, i_Coverage, i_Abs_True_Rate, i_Abs_False_Rate, i_Accuracy, i_Hamming_loss, time_cost))
                sys.stdout.flush()
                start = time.time()
        if np.mean(accs_epoch) > acc_best:
            torch.save(model.state_dict(), model_dir + '/model3.ckpt')
            acc_best = np.mean(accs_epoch)
            print("TRAIN Epoch [{}/{}], best accuracy: {:.4f}, saving model".format(epoch + 1,
                                                                                    args.num_epochs, acc_best))
        # scheduler.step()
    test_step = len(test_loader)
    print("test_step: {}".format(test_step))
    start = time.time()
    model = CapsuleNet().to(device)
    para_dict = torch.load(model_dir + '/model3.ckpt')
    model_dict = model.state_dict()
    model_dict.update(para_dict)
    model.load_state_dict(model_dict)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    temp = []
    labels_all = []
    predicted_all = []
    seqs_all = []
    index_all = []
    score_all = []
    total_test = 0
    with torch.no_grad():
        for step, (batch_x, batch_y,test_index,batch_seqs) in enumerate(test_loader):
            # batch_x = batch_x_index.numpy()[:, 1::]
            # index = batch_x_index.numpy()[:, 0]
            # batch_x = torch.from_numpy(batch_x)
            # batch_x = batch_x.to(device)
            feature1 = batch_x[:, 0:517].to(device)
            feature2 = batch_x[:, 517:1034].to(device)
            feature3 = batch_x[:, 1034:1551].to(device)
            feature4 = batch_x[:, 1551:2068].to(device)
            feature5 = batch_x[:, 2068:2073].to(device)
            feature6 = batch_x[:, 2073:2093].to(device)
            feature7 = batch_x[:, 2093::].to(device)
            batch_y = batch_y.to(device)
            outputs = model(feature1.float(),feature2.float(),feature3.float(),feature4.float(),feature5.float(),feature6.float(),feature7.float(),len(feature1))
            loss = criterion(outputs.float(), batch_y.float())

            if device.type == 'cuda':
                batch_y = batch_y.cpu()
                predicted = outputs.cpu()
                score = outputs.cpu()
            labels_all.append(batch_y.numpy())
            predicted = predicted.numpy()
            score = score.numpy()
            score_all.append(score)
            predicted_all.append(predicted)
            seqs_all.append(batch_seqs.numpy())
            index_all.append(test_index.numpy())
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predicted = predicted.astype(int)
            D = batch_y.numpy().shape[0]
            total_test = total_test + D
            time_cost = time.time() - start
            if (step + 1) % args.step == 0:
                i_Precision = Precision(batch_y.numpy(), predicted, D)
                i_Coverage = Coverage(batch_y.numpy(), predicted, D)
                i_Abs_True_Rate = Abs_True_Rate(batch_y.numpy(), predicted, D)
                i_Abs_False_Rate = Abs_False_Rate(batch_y.numpy(), predicted, D)
                i_Accuracy = Accuracy(batch_y.numpy(), predicted, D)
                i_Hamming_loss = Hamming_loss(batch_y.numpy(), predicted, D)

                print('TEST, Step [{}/{}], Loss: {:.4f}, '
                      'Precision: {:.4f}, Coverage: {:.4f}, Abs_True_Rate: {:.4f}, '
                      'Abs_False_Rate: {:.4f}, Accuracy: {:.4f}, Hamming_loss: {:.4f}, Time: {:.2f}s'
                      .format(step + 1, test_step, loss.item(),
                              i_Precision, i_Coverage, i_Abs_True_Rate, i_Abs_False_Rate, i_Accuracy, i_Hamming_loss,
                              time_cost))
                sys.stdout.flush()
            start = time.time()

    total_labels = np.concatenate(labels_all)
    total_preds = np.concatenate(predicted_all)

    total_Precision = Precision(total_labels, total_preds, total_test)
    total_Coverage = Coverage(total_labels, total_preds, total_test)
    total_Abs_True_Rate = Abs_True_Rate(total_labels, total_preds, total_test)
    total_Abs_False_Rate = Abs_False_Rate(total_labels, total_preds, total_test)
    total_Accuracy = Accuracy(total_labels, total_preds, total_test)
    total_Hamming_loss = Hamming_loss(total_labels, total_preds, total_test)

    print('===TEST, total test Precision: {:.4f}, Coverage: {:.4f}, Abs_True_Rate: {:.4f}, Abs_False_Rate: {:.4f}, '
          'Accuracy: {:.4f}, Hamming_loss: {:.4f}'.format(total_Precision,
                                  total_Coverage, total_Abs_True_Rate, total_Abs_False_Rate,
                                  total_Accuracy, total_Hamming_loss,))
    sys.stdout.flush()
    return total_Precision, total_Coverage, total_Abs_True_Rate, total_Abs_False_Rate, total_Accuracy, total_Hamming_loss,total_labels

def train_test_five_fold(args):
    print("[main]reading data===============")
    Drug_feature, Drug_ATC_label,seqs = read_raw_data(args.rawdata_dir)
    print("[main]cross validation===============")
    ten_fold_files(Drug_feature, Drug_ATC_label,seqs, args)


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rawdata_dir', type=str, default='./data1', required=False)
    parser.add_argument('--model_dir', type=str, default='./model', required=False)

    parser.add_argument('--hin', type=int, default=7, required=False)
    parser.add_argument('--win', type=int, default=50, required=False)
    parser.add_argument('--num_epochs', type=int, default=100, required=False)
    parser.add_argument('--batch_size', type=int, default=128, required=False)
    parser.add_argument('--step', type=int, default=10, required=False)

    args = parser.parse_args()
    display_args(args)
    train_test_five_fold(args)


if __name__ == '__main__':
    main()

