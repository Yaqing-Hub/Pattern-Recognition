#coding :utf-8

'''任务二：利用马氏距离作为KNN算法的度量函数，对测试集进行分类。'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
np.seterr(invalid='ignore')

import data_process

#利用梯度更新方法更新马氏距离的可学习参数A
class UPA():
    def __init__(self, learning_rate=0.001, epoch=500):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def train(self, X, Y):
        (n, d) = X.shape
        self.n_samples = n  # 样本数
        self.dim = d  # 样本点的维度

        # A:d*e
        e = 2  # A的维度，在距离度量时实际进行了降维
        A_shape = (d, e)
        # 初始化特征矩阵A
        self.A = 0.1*np.random.standard_normal(size = A_shape)
        # 利用梯度下降训练参数A
        count = 0  # 迭代的计数器
        f = 0  # 优化目标f(A)=sum_i(p_i)
        f_list = []
        while count < self.epoch:
            if count % 5 == 0 and count > 1:
                print('Step{},f(a)={}...'.format(count, f))
            # 计算AX:e
            XA = np.dot(X, self.A)
            # 距离度量
            sum_row = np.sum(XA ** 2, axis=1)
            XAATXT = np.dot(XA, XA.transpose())
            dist_mat = sum_row + np.reshape(sum_row, (-1, 1)) - 2 * XAATXT
            # 将距离转变为概率
            exp_neg_dist = np.exp(-dist_mat)
            exp_neg_dist = exp_neg_dist - np.diag(np.diag(exp_neg_dist))
            prob_mat = exp_neg_dist / np.sum(exp_neg_dist, axis=1).reshape((-1, 1))
            # pi = sum_{j  in C_i}p_{ij} pi构成的矩阵
            prob_row = np.array([np.sum(prob_mat[i][Y == Y[i]]) for i in range(self.n_samples)])
            f = np.sum(prob_row)
            if np.isnan(f):
                break
            f_list.append(f)
            # A的梯度，即f对A的偏导
            gradients = np.zeros((self.dim, self.dim))
            # 对i的循环作为外层循环
            for i in range(self.n_samples):
                # 梯度中的第一项
                first_item = np.zeros((self.dim, self.dim))
                # 梯度中的第二项
                second_item = np.zeros((self.dim, self.dim))
                # 对j的循环作为内层循环
                for k in range(self.n_samples):
                    out_prob = np.outer(X[i] - X[k], X[i] - X[k])
                    first_item += prob_mat[i][k] * out_prob
                    if Y[k] == Y[i]:
                        second_item += prob_mat[i][k] * out_prob
                gradients += prob_row[i] * first_item - second_item
            gradients = 2 * np.dot(gradients, self.A)
            # 利用梯度更新A
            self.A += self.learning_rate * gradients
            # 循环计数器次数+1
            count += 1
        return self.A
NCA_model = UPA()
a=np.array(data_process.train_data)
X=a[:,0:4]
Y=a[:,4]
A=NCA_model.train(X,Y)
#print(A)

def Mahalanobis_distance(train_data,data):
    '''计算单个样本与训练集各个样本间的马氏距离，并按升序排列'''
    A1=np.array(A)
    a1 = np.array(train_data)
    a2 = np.array(data)
    td = a1[:, 0:4]
    d = a2[0:4]
    label = a1[:, 4]
    C_distance = []
    dis_lab = []  # 存储该样本与训练样本的马氏距离和训练样本对应的类别
    for sample in td:
        temp=np.dot(d-sample,A1)
        temp_tr=np.transpose(temp)
        mul=np.dot(temp,temp_tr)
        norm = math.sqrt(mul)
        C_distance.append(norm)
    for i in range(data_process.count_total(train_data)):
        dis_lab.append([C_distance[i], label[i]])
    sort = sorted(dis_lab, key=(lambda x: x[0]))  # 将马氏距离按升序排列
    return sort


def decide_label(train_data,val_data):
    '''用验证集计算各个K下的正确率'''
    total=data_process.count_total(val_data)
    a=np.array(val_data)
    data=a[:,0:4]
    label=a[:,4]
    corr = []#各个K下的正确率
    for K in range(1,30,2):
        predict = []  # 对验证集各样本进行分类
        for sample in data:
            sort = Mahalanobis_distance(train_data, sample)
            lab_0=0
            lab_1=0
            for i in range(K):
                if sort[i][1]==0:
                    lab_0+=1
                else:
                    lab_1+=1
            if lab_0>lab_1:
                predict.append(0)
            else:
                predict.append(1)
        correct=0
        for i , lab in enumerate(label):
            if predict[i]==lab:
                correct+=1
        corr.append(correct/total)
    #绘制K对精度影响的曲线图
    K=list(range(1,30,2))
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('K of Mahalanobis', fontsize = 14, fontweight='bold')
    ax.set_xlabel("K")
    ax.set_ylabel("correction rate")
    plt.plot(K, corr)
    plt.show()
    K2=corr.index(max(corr))*2+1#记录准确率最高的K
    return K2
print(decide_label(data_process.train_data,data_process.val_data))

def Mahalanobis_classify(train,test):
    '''用Mahalanobis对测试集分类'''
    total=data_process.count_total(test)
    #K=Manhattan.decide_label(train,val_data)
    K=1
    a=np.array(test)
    t_data=a[:,0:4]
    predict = [] # 对测试集各样本进行分类
    for sample in t_data:
        sort = Mahalanobis_distance(train, sample)
        lab_0 = 0
        lab_1 = 0
        for i in range(K):
            if sort[i][1] == 0:
                lab_0 += 1
            else:
                lab_1 += 1
        if lab_0 > lab_1:
            predict.append(0)
        else:
            predict.append(1)
    Mahalanobis_result=list(test)
    for i in range(total):
        Mahalanobis_result[i].append(predict[i])
        #Mahalanobis_result[i][4]=predict[i]
    title=['Recency (months)','Frequency (times)','Monetary (c.c. blood)','Time (months)','My prediction']
    Mahalanobis_result.insert(0, title)
    submit2 = pd.DataFrame(data=Mahalanobis_result)
    #print(submit)
    submit2.to_csv('./task2_test_prediction.csv', encoding='gbk', header=None, index=None)  

Mahalanobis_classify(data_process.train_data,data_process.test_data)


