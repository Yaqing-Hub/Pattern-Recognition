# 分类
# coding:utf-8
import pandas as pd
import Bayesian_Classifier


def main():
    test_data=pd.read_csv(r"test_data.csv",header=None)
    test_datasets=test_data.values.tolist()
    classfier = Bayesian_Classifier.Naive_bayesian_classifier(test_datasets)
    count,total = Bayesian_Classifier.count_total(test_datasets)
    classfier.get_likelihood_prob() # 计算各个样本在三个类的类条件概率
    post=classfier.get_post_prob() # 计算各个样本在三个类中的后验概率
    #print(post)
    labels=classfier.get_final_labels() # 判断各个样本最终类别
    result=[]
    for i in range(total):
        # 遍历样本
        sample=[]
        sample.append(labels[i])
        for j in post[i]:
            sample.append(j)
        result.append(sample)
    result.append(['预测正确率',classfier.get_correct_rate()])
    test = pd.DataFrame(data=result)
    print(test)
    test.to_csv('D:/Files/A2020-2021-3/模式识别-薛晖/实验5选3/我的实验/58119104-蔡雅清-贝叶斯分类任务实验/test_prediction.csv',
                encoding='gbk', header=None , index=None)


if __name__ == '__main__':
    main()