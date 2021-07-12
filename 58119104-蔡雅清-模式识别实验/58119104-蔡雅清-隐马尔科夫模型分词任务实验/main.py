#coding: utf:8

import HMM
import Viterbi
import wordSplit


if __name__ == '__main__':
    trainingSet = open('RenMinData.txt_utf8', encoding='utf-8')     #读取训练集
    testSet = open('testSet.txt', encoding='utf-8')       #读取测试集
    # trainlist = []
 
    HMM.Init_Array()
 
    for line in trainingSet:
        line = line.strip()
        # trainlist.append(line)
        HMM.line_num += 1
 
        word_list = []
        for k in range(len(line)):
            if line[k] == ' ':continue
            word_list.append(line[k])
        # print(word_list)
        word_set = HMM.word_set | set(word_list)    #训练集所有字的集合
 
        line = line.split(' ')
        # print(line)
        line_state = []     #这句话的状态序列
 
        for i in line:
            line_state.extend(HMM.get_tag(i))
        # print(line_state)
        HMM.array_Pi[line_state[0]] += 1  # array_Pi用于计算初始状态分布概率
 
        for j in range(len(line_state)-1):
            # count_dic[line_state[j]] += 1   #记录每一个状态的出现次数
            HMM.array_A[line_state[j]][line_state[j+1]] += 1  #array_A计算状态转移概率
 
        for p in range(len(line_state)):
            HMM.count_dic[line_state[p]] += 1  # 记录每一个状态的出现次数
            for state in HMM.STATES:
                if word_list[p] not in HMM.array_B[state]:
                    HMM.array_B[state][word_list[p]] = 0.0  #保证每个字都在STATES的字典中
            # if word_list[p] not in array_B[line_state[p]]:
            #     # print(word_list[p])
            #     array_B[line_state[p]][word_list[p]] = 0
            # else:
            HMM.array_B[line_state[p]][word_list[p]] += 1  # array_B用于计算发射概率
 
    HMM.Prob_Array()    #对概率取对数保证精度
 
    output = ''
 
    for line in testSet:
        line = line.strip()
        tag = Viterbi.Viterbi(line, HMM.array_Pi, HMM.array_A, HMM.array_B)
        # print(tag)
        seg = wordSplit.tag_seg(line, tag)
        # print(seg)
        list = ''
        for i in range(len(seg)):
            list = list + seg[i] + ' '
        # print(list)
        output = output + list + '\n'
    print(output)
    outputfile = open('output.txt', mode='w', encoding='utf-8')
    outputfile.write(output)