#coding :utf-8

import csv

# 训练集导入
with open('train_data.csv',encoding='utf-8') as train:
    train_csv=csv.reader(train)
    tmp=list(train_csv)
    tmp.remove(tmp[0])
    train_data =[]
    for row in tmp:
        trow=[]
        for element in row:
            trow.append(float(element))
        train_data.append(trow)
#print(train_data)

# 验证集导入
with open('val_data.csv',encoding='utf-8') as valid:
    valid_csv=csv.reader(valid)
    temp=list(valid_csv)
    temp.remove(temp[0])
    val_data =[]
    for row in temp:
        trow=[]
        for element in row:
            trow.append(float(element))
        val_data.append(trow)
#print(val_data)

#导入测试集
with open('test_data.csv',encoding='utf-8') as f:
    f_csv=csv.reader(f)
    temp=list(f_csv)
    temp.remove(temp[0])
    test_data =[]
    for row in temp:
        trow=[]
        for element in row:
            trow.append(float(element))
        test_data.append(trow)
#print(test_data)


def count_total(dataset):
    return len(dataset)


print(count_total(train_data))