# coding:utf-8


import numpy as np
import pandas as pd
import math

train_data = pd.read_csv(r"train_data.csv",header=None)
# print(train_data)

# 将dataframe转换为list
train_datasets=train_data.values.tolist()
# print("list",train_data)

# 计算各个类的总数以及训练样本总数
def count_total(data):
    count = [0,0,0]
    total = 0
    training_data=np.array(data)
    classes=training_data[:,0]
    for num in classes:
        # 统计各个类的总数
        if num == 1: count[0] += 1
        if num == 2: count[1] += 1
        if num == 3: count[2] += 1
    total = count[0]+count[1]+count[2]
    return count, total
# print("total",count_total(train_datasets))


# 计算各类红酒的先验概率
def cal_prior_rate(data):
    classes, total = count_total(data)
    cal_prior_rate = [0,0,0]
    for label in range(3):
        prior_prob = classes[label] / total
        cal_prior_rate[label] = prior_prob
    return cal_prior_rate
# print("prior",cal_prior_rate(train_datasets))

# 利用高斯分布估计均值和方差
# 本实验规定采用高斯分布估计类条件概率。其中，均值和方差分别用训练集的样本均值和样本方差估计。
def cal_Gaussian(data):
    count, total = count_total(data)
    new_data=np.array(data)
    all_model = []
    for i in range(1,14,1):
        # 遍历十三个特征值
        feature=new_data[:,i]
        para=[]
        model = []
        sum1 = 0
        sum2 = 0
        for k in range(count[0]):
            # 遍历第一个类中各个样本的第i个特征
            sum1 += feature[k]
        mean=sum1/count[0]
        model.append(mean)
        for k in range(count[0]):
            sum2 += (feature[k]-mean)**2
        variance=sum2/count[0]
        model.append(variance)
        para.append(model)
        model = []
        sum1 = 0
        sum2 = 0
        for k in range(count[0],count[0]+count[1]):
            # 遍历第二个类中各个样本的第i个特征
            sum1 += feature[k]
        mean=sum1/count[1]
        model.append(mean)
        for k in range(count[0], count[0] + count[1]):
            sum2 += (feature[k]-mean)**2
        variance=sum2/count[1]
        model.append(variance)
        para.append(model)
        model = []
        sum1 = 0
        sum2 = 0
        for k in range(count[0]+count[1],total):
            # 遍历第三个类中各个样本的第i个特征
            sum1 += feature[k]
        mean=sum1/count[2]
        model.append(mean)
        for k in range(count[0] + count[1], total):
            sum2 += (feature[k]-mean)**2
        variance=sum2/count[2]
        model.append(variance)
        para.append(model)
        all_model.append(para)
    return all_model
# print("Gaussian",cal_Gaussian(train_datasets))


def cal_likelihood(data):
    '''给定高斯分布采用朴素贝叶斯分类计算单个样本的类条件概率'''
    model=cal_Gaussian(train_datasets)
    a=np.array(model)
    likelihood=[]
    multi=1
    feature_model = a[:, 0, :]#取13个特征值的第一类高斯分布模型的参数
    for i in range(13):#遍历每个特征，计算该样本的类条件概率
        part_model=feature_model[i]
        prob=1/(math.sqrt(2*math.pi*part_model[1]))*\
             math.exp((-(data[i+1]-part_model[0])**2)/(2*part_model[1]))
        multi *= prob
    likelihood.append(multi)
    multi=1
    feature_model = a[:, 1, :] # 取13个特征值的第二类高斯分布模型的参数
    for i in range(13): # 遍历每个特征，计算该样本的类条件概率
        part_model=feature_model[i]
        prob=1/(math.sqrt(2*math.pi*part_model[1]))*\
            math.exp((-(data[i+1]-part_model[0])**2)/(2*part_model[1]))
        multi *= prob
    likelihood.append(multi)
    multi=1
    feature_model = a[:, 2, :]
    # 取13个特征值的第三类高斯分布模型的参数
    for i in range(13):
        # 遍历每个特征，计算该样本的类条件概率
        part_model=feature_model[i]
        prob = 1/(math.sqrt(2*math.pi*part_model[1]))*\
            math.exp((-(data[i+1]-part_model[0])**2)/(2*part_model[1]))
        multi *= prob
    likelihood.append(multi)
    return likelihood
# print(train_datasets[0])
# print("likelihood",cal_likelihood(train_datasets[0]))

class Naive_bayesian_classifier:
    def __init__(self, data):
        self._data = data # 初始化数据
        self._model=cal_Gaussian(train_datasets) # 用训练数据初始化高斯分布的模型
        a=np.array(self._data)
        self._initial_labels = a[:,0]  # 存放最初的类别标签
        self._final_labels=[] # 初始化最终类别标签[,...,]
        self._prior_prob = cal_prior_rate(train_datasets) # 计算各个类的先验概率
        self._likelihood_prob = [] # 计算各个样本的在三个类的类条件概率[[,,]...[,,]]
        self._post_prob = [] # 存放各个样本在三个类的后验概率[[,,]...[,,]]
        self._correct_rate=0 # 正确率

    # 下面的函数可以直接调用上面类中定义的变量
    def get_likelihood_prob(self):
        # 计算各个样本在三个类的类条件概率
        for sample in self._data:
            self._likelihood_prob.append(cal_likelihood(sample))

    def get_post_prob(self):
        # 计算各个样本在三个类中的后验概率
        for sample in self._likelihood_prob:
            post=[]
            for i in range(3):
                post.append(sample[i]*self._prior_prob[i])
            self._post_prob.append(post)
        return self._post_prob

    def get_final_labels(self):
        # 判断各个样本最终类别
        for sample in self._post_prob:
            max=sample[0]
            label=1
            if sample[1]>max:
                max=sample[1]
                label=2
            if sample[2]>max:
                max=sample[2]
                label=3
            self._final_labels.append(label)
        return self._final_labels

    def get_correct_rate(self):
        # 计算准确率
        count, total = count_total(self._data)
        sum = 0
        for i in range(total):
            if self._initial_labels[i]==self._final_labels[i]:
                sum += 1
        self._correct_rate=float(sum/total)
        return self._correct_rate