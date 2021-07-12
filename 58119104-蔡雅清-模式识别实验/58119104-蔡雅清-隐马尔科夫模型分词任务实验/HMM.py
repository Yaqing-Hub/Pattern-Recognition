#coding:utf-8

import pandas as pd
import codecs
from numpy import *
import numpy as np
import sys
import re

STATES = ['B', 'M', 'E', 'S']
array_A = {}    #状态转移概率矩阵
array_B = {}    #发射概率矩阵
array_E = {}    #测试集存在的字符，但在训练集中不存在，发射概率矩阵
array_Pi = {}   #初始状态分布
word_set = set()    #训练数据集中所有字的集合
count_dic = {}  #‘B,M,E,S’每个状态在训练集中出现的次数
line_num = 0    #训练集语句数量
 
#初始化所有概率矩阵
def Init_Array():
    for state0 in STATES:
        array_A[state0] = {}
        for state1 in STATES:
            array_A[state0][state1] = 0.0
    for state in STATES:
        array_Pi[state] = 0.0
        array_B[state] = {}
        array_E = {}
        count_dic[state] = 0
 
#对训练集获取状态标签
def get_tag(word):
    tag = []
    if len(word) == 1:
        tag = ['S']
    elif len(word) == 2:
        tag = ['B', 'E']
    else:
        num = len(word) - 2
        tag.append('B')
        tag.extend(['M'] * num)
        tag.append('E')
    return tag
 
 
#将参数估计的概率取对数，对概率0取无穷小-3.14e+100
def Prob_Array():
    for key in array_Pi:
        if array_Pi[key] == 0:
            array_Pi[key] = -3.14e+100
        else:
            array_Pi[key] = log(array_Pi[key] / line_num)
    for key0 in array_A:
        for key1 in array_A[key0]:
            if array_A[key0][key1] == 0.0:
                array_A[key0][key1] = -3.14e+100
            else:
                array_A[key0][key1] = log(array_A[key0][key1] / count_dic[key0])
    # print(array_A)
    for key in array_B:
        for word in array_B[key]:
            if array_B[key][word] == 0.0:
                array_B[key][word] = -3.14e+100
            else:
                array_B[key][word] = log(array_B[key][word] /count_dic[key])
 
#将字典转换成数组
def Dic_Array(array_b):
    tmp = np.empty((4,len(array_b['B'])))
    for i in range(4):
        for j in range(len(array_b['B'])):
            tmp[i][j] = array_b[STATES[i]][list(word_set)[j]]
    return tmp
 
#判断一个字最大发射概率的状态
def dist_tag():
    array_E['B']['begin'] = 0
    array_E['M']['begin'] = -3.14e+100
    array_E['E']['begin'] = -3.14e+100
    array_E['S']['begin'] = -3.14e+100
    array_E['B']['end'] = -3.14e+100
    array_E['M']['end'] = -3.14e+100
    array_E['E']['end'] = 0
    array_E['S']['end'] = -3.14e+100
 
def dist_word(word0,word1,word2,array_b):
    if dist_tag(word0,array_b) == 'S':
        array_E['B'][word1] = 0
        array_E['M'][word1] = -3.14e+100
        array_E['E'][word1] = -3.14e+100
        array_E['S'][word1] = -3.14e+100
    return
 

