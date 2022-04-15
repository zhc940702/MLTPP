# -*- coding: utf-8 -*-

import pandas as pd
import scipy.io as sio
import numpy as np
from sklearn import metrics
from sklearn.metrics import hamming_loss
from sklearn.metrics import multilabel_confusion_matrix,roc_auc_score

# hamming_loss（y_true，y_pred，*，sample_weight = None ）

def Hamming_loss(test_targets, preLabels,  D):
    return hamming_loss(test_targets, preLabels)

def Intersect_set(a, b):
    countL = 0
    for i in range(len(a)):
        if a[i] == 1 and b[i] == 1:
            countL += 1
        else:
            continue
    return countL


def unionset(line1, line2):
    sum2 = 0
    for i in range(len(line1)):
        if (line1[i] == 0 and line2[i] == 1) or (line1[i] == 1 and line2[i] == 0) or (line1[i] == 1 and line2[i] == 1):
            sum2 += 1
    return sum2


def Precision(test_targets,preLabels, D):
    # molecular
    sumsum1 = 0
    for i in range(D):
        line1, line2 = preLabels[i], test_targets[i]
        # denominator
        line1_count = 0
        for i in range(len(line1)):
            if line1[i] == 1:
                line1_count += 1
        sumsum1 += Intersect_set(line1, line2) / (line1_count + 1e-6)
    return sumsum1 / D


def Coverage(test_targets,preLabels,  D):
    # molecular
    sumsum1 = 0
    for i in range(D):
        line1, line2 = preLabels[i], test_targets[i]
        # denominator
        line2_count = 0
        for i in range(len(line2)):
            if line2[i] == 1:
                line2_count += 1
        sumsum1 += Intersect_set(line1, line2) / (line2_count + 1e-6)
    return sumsum1 / D


def Abs_True_Rate(test_targets,preLabels, D):
    correct_pairs = 0
    for i in range(len(preLabels)):
        if np.all(preLabels[i] == test_targets[i]):
            correct_pairs += 1
    abs_true = correct_pairs / D
    return abs_true


def Total_Abs_False_Rate( test_targets,preLabels, D,index,total_seqs):
    correct_pairs = 0.0
    ABF,or_ABF,findex,fseqs=[],[],[],[]
    for i in range(len(preLabels)):
        line1, line2,line3,line4 = preLabels[i], test_targets[i],index[i],total_seqs[i]
        abf = (unionset(line1, line2) - Intersect_set(line1, line2)) / 5
        correct_pairs += abf
        if abf > 0.5:
            ABF.append(line2)
            or_ABF.append(line1)
            findex.append(line3)
            fseqs.append(line4)
    abs_false = correct_pairs / D
    return abs_false,ABF,or_ABF,findex,fseqs

def Abs_False_Rate(test_targets,preLabels,  D):
    correct_pairs = 0.0
    for i in range(len(preLabels)):
        line1, line2 = preLabels[i], test_targets[i]
        correct_pairs += (unionset(line1, line2) - Intersect_set(line1, line2)) / 5
    abs_false = correct_pairs / D
    return abs_false

def Accuracy(test_targets,preLabels,  D):
    acc_score = 0
    for i in range(len(preLabels)):
        item_inter = Intersect_set(preLabels[i], test_targets[i])
        item_union = unionset(preLabels[i], test_targets[i])
        acc_score += item_inter / (item_union + 1e-6)
    accuracy = acc_score / D
    return accuracy

def Singlelabel( test_target,prelabels):
    # cm = multilabel_confusion_matrix(test_target,prelabels)
    FPR,TPR,P,R = [],[],[],[]
    SEN,SPE,ACC,AUC,AUPR = np.array([0.0,0.0,0.0,0.0,0.0]),np.array([0.0,0.0,0.0,0.0,0.0]),np.array([0.0,0.0,0.0,0.0,0.0]),np.array([0.0,0.0,0.0,0.0,0.0]),np.array([0.0,0.0,0.0,0.0,0.0])
    for i in range(AUC.size):
         # SEN[i]= cm[i][1][1]/(cm[i][1][1]+cm[i][1][0])
         # SPE[i]= cm[i][0][0]/(cm[i][0][0]+cm[i][0][1])
         # ACC[i]= (cm[i][0][0]+cm[i][1][1])/(cm[i][0][0]+cm[i][0][1]+cm[i][1][1]+cm[i][1][0])
         p,r,t = metrics.precision_recall_curve(test_target.T[i], prelabels.T[i])
         P.append(p)
         R.append(r)
         fpr,tpr,th = metrics.roc_curve(test_target.T[i],prelabels.T[i])
         FPR.append(fpr)
         TPR.append(tpr)
         AUPR[i] = metrics.auc(r,p)
         AUC[i] = roc_auc_score(test_target.T[i], prelabels.T[i])
    return P,R,FPR,TPR,AUC,AUPR

def falsepositive(testtargets, prelabels,total_seqs):
    AMPFP,ACPFP,ADPFP,AHPFP,AIPFP = [],[],[],[],[]
    for i in range(len(prelabels)):
        line1, line2, line3 = prelabels[i], testtargets[i],  total_seqs[i]
        if line1[0] == 1 and line2[0] == 0:
            AMPFP.append(line3)
        if line1[1] == 1 and line2[1] == 0:
            ACPFP.append(line3)
        if line1[2] == 1 and line2[2] == 0:
            ADPFP.append(line3)
        if line1[3] == 1 and line2[3] == 0:
            AHPFP.append(line3)
        if line1[4] == 1 and line2[4] == 0:
            AIPFP.append(line3)
    return AMPFP,ACPFP,ADPFP,AHPFP,AIPFP