#coding :utf-8

'''任务一：利用欧式距离、切比雪夫距离、曼哈顿距离作为KNN算法的度量函数对测试集进行分类'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import data_process

def Euclidean_distance(train_data,data):
    '''计算单个样本与训练集各个样本间的Euclidean_distance，并按升序排列'''
    a1=np.array(train_data)
    a2=np.array(data)
    td=a1[:,0:4]
    d=a2[0:4]
    label=a1[:,4]
    E_distance=[]
    dis_lab=[] # 存储该样本与训练样本的欧式距离和训练样本对应的类别
    for sample in td:
        norm=np.linalg.norm(sample-d)
        E_distance.append(norm)
    for i in range(data_process.count_total(train_data)):
        dis_lab.append([E_distance[i],label[i]])
    sort=sorted(dis_lab,key=(lambda x:x[0])) # 将欧氏距离按升序排列
    return sort
#print(Euclidean_distance(data_process.train_data,data_process.val_data[0]))

def Manhattan_distance(train_data,data):
    '''计算单个样本与训练集各个样本间的Manhattan_distance，并按升序排列'''
    a1=np.array(train_data)
    a2=np.array(data)
    td=a1[:,0:4]
    d=a2[0:4]
    label=a1[:,4]
    C_distance=[]
    dis_lab=[]#存储该样本与训练样本的Manhattan_distance和训练样本对应的类别
    for sample in td:
        temp=np.array([feature for feature in sample-d])
        norm=sum(abs(temp))
        C_distance.append(norm)
    for i in range(data_process.count_total(train_data)):
        dis_lab.append([C_distance[i],label[i]])
    sort=sorted(dis_lab,key=(lambda x:x[0]))#将Manhattan_distance按升序排列
    return sort
#print(Manhattan_distance(data_process.train_data,data_process.val_data[0]))

def Chebyshev_distance(train_data,data):
    '''计算单个样本与训练集各个样本间的Chebyshev_distance，并按升序排列'''
    a1=np.array(train_data)
    a2=np.array(data)
    td=a1[:,0:4]
    d=a2[0:4]
    label=a1[:,4]
    C_distance=[]
    dis_lab=[]#存储该样本与训练样本的Chebyshev_distance和训练样本对应的类别
    for sample in td:
        norm=max(abs(sample-d))
        C_distance.append(norm)
    for i in range(data_process.count_total(train_data)):
        dis_lab.append([C_distance[i],label[i]])
    sort=sorted(dis_lab,key=(lambda x:x[0]))#将Chebyshev_distance按升序排列
    return sort
#print(Chebyshev_distance(data_process.train_data,data_process.val_data[0]))

def decide_label_Euclidean(train_data,val_data):
    '''用验证集计算各个K下的正确率'''
    total=data_process.count_total(val_data)
    a=np.array(val_data)
    data=a[:,0:4]
    label=a[:,4]
    corr = []#各个K下的正确率
    for K in range(1,30,2):
        predict = []  # 对验证集各样本进行分类
        for sample in data:
            sort = Euclidean_distance(train_data, sample)
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
    fig.suptitle('K of Euclidean_distance', fontsize = 14, fontweight='bold')
    ax.set_xlabel("K")
    ax.set_ylabel("correction rate")
    plt.plot(K, corr)
    plt.show()
    K1=corr.index(max(corr))*2+1#记录准确率最高的K
    return K1

def decide_label_Chebyshev(train_data,val_data):
    '''用验证集计算各个K下的正确率'''
    total=data_process.count_total(val_data)
    a=np.array(val_data)
    data=a[:,0:4]
    label=a[:,4]
    corr = []#各个K下的正确率
    for K in range(1,30,2):
        predict = []  # 对验证集各样本进行分类
        for sample in data:
            sort = Chebyshev_distance(train_data, sample)
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
    fig.suptitle('K of Chebyshev', fontsize = 14, fontweight='bold')
    ax.set_xlabel("K")
    ax.set_ylabel("correction rate")
    plt.plot(K, corr)
    plt.show()
    K1=corr.index(max(corr))*2+1#记录准确率最高的K
    return K1

def decide_label_Manhattan(train_data,val_data):
    '''用验证集计算各个K下的正确率'''
    total=data_process.count_total(val_data)
    a=np.array(val_data)
    data=a[:,0:4]
    label=a[:,4]
    corr = []#各个K下的正确率
    for K in range(1,30,2):
        predict = []  # 对验证集各样本进行分类
        for sample in data:
            sort = Manhattan_distance(train_data, sample)
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
    fig.suptitle('K of Manhattan', fontsize = 14, fontweight='bold')
    ax.set_xlabel("K")
    ax.set_ylabel("correction rate")
    plt.plot(K, corr)
    plt.show()
    K1=corr.index(max(corr))*2+1#记录准确率最高的K
    return K1

#print(decide_label_Euclidean(data_process.train_data,data_process.val_data))
#print(decide_label_Chebyshev(data_process.train_data,data_process.val_data))
#print(decide_label_Manhattan(data_process.train_data,data_process.val_data))


def Euclidean_classify(train,test):
    '''用Euclidean_distance对测试集分类'''
    total=data_process.count_total(test)
    K=1
    a=np.array(test)
    t_data=a[:,0:4]
    predict = [] # 对测试集各样本进行分类
    for sample in t_data:
        sort = Euclidean_distance(train, sample)
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
    Euclidean_result=list(test)
    for i in range(total):
        Euclidean_result[i].append(predict[i])
    title=['Recency (months)','Frequency (times)','Monetary (c.c. blood)','Time (months)','My prediction']
    Euclidean_result.insert(0,title)
    submit = pd.DataFrame(data=Euclidean_result)
    #print(submit)
    submit.to_csv('./task1_test_Euclidean.csv',encoding='gbk', header=None, index=None) 
Euclidean_classify(data_process.train_data,data_process.test_data)

def Chebyshev_classify(train,test1):
    '''用Chebyshev_distance对测试集分类'''
    total=data_process.count_total(test1)
  
    K=1
    a=np.array(test1)
    t_data=a[:,0:4]
    predict = [] # 对测试集各样本进行分类
    for sample in t_data:
        sort = Chebyshev_distance(train, sample)
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
    Chebyshev_result=list(test1)
    for i in range(total):
        Chebyshev_result[i][4]=predict[i]
    title=['Recency (months)','Frequency (times)','Monetary (c.c. blood)','Time (months)','My prediction']
    Chebyshev_result.insert(0, title)
    submit1 = pd.DataFrame(data=Chebyshev_result)
    #print(submit)
    submit1.to_csv('./task1_test_Chebyshev.csv',encoding='gbk', header=None, index=None)  
Chebyshev_classify(data_process.train_data,data_process.test_data)

def Manhattan_classify(train,test2):
    '''用Manhattan对测试集分类'''
    total=data_process.count_total(test2)
    K=1
    a=np.array(test2)
    t_data=a[:,0:4]
    predict = [] # 对测试集各样本进行分类
    for sample in t_data:
        sort = Manhattan_distance(train, sample)
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
    Manhattan_result=list(test2)
    for i in range(total):
        Manhattan_result[i][4]=predict[i]
    title=['Recency (months)','Frequency (times)','Monetary (c.c. blood)','Time (months)','My prediction']
    Manhattan_result.insert(0, title)
    submit2 = pd.DataFrame(data=Manhattan_result)
    #print(submit)
    submit2.to_csv('./task1_test_Manhattan.csv',encoding='gbk', header=None, index=None)  
Manhattan_classify(data_process.train_data,data_process.test_data)