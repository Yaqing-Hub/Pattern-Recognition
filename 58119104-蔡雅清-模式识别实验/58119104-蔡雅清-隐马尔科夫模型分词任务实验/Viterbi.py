#coding: utf:8

import HMM
#from HMM import STATES

#Viterbi算法求测试集最优状态序列
def Viterbi(sentence,array_pi,array_a,array_b):
    tab = [{}]  #动态规划表
    path = {}
 
    if sentence[0] not in array_b['B']:
        for state in HMM.STATES:
            if state == 'S':
                array_b[state][sentence[0]] = 0
            else:
                array_b[state][sentence[0]] = -3.14e+100
 
    for state in HMM.STATES:
        tab[0][state] = array_pi[state] + array_b[state][sentence[0]]
        # print(tab[0][state])
        #tab[t][state]表示时刻t到达state状态的所有路径中，概率最大路径的概率值
        path[state] = [state]
    for i in range(1,len(sentence)):
        tab.append({})
        new_path = {}
        # if sentence[i] not in array_b['B']:
        #     print(sentence[i-1],sentence[i])
        for state in HMM.STATES:
            if state == 'B':
                array_b[state]['begin'] = 0
            else:
                array_b[state]['begin'] = -3.14e+100
        for state in HMM.STATES:
            if state == 'E':
                array_b[state]['end'] = 0
            else:
                array_b[state]['end'] = -3.14e+100
        for state0 in HMM.STATES:
            items = []
            # if sentence[i] not in word_set:
            #     array_b[state0][sentence[i]] = -3.14e+100
            # if sentence[i] not in array_b[state0]:
            #     array_b[state0][sentence[i]] = -3.14e+100
            # print(sentence[i] + state0)
            # print(array_b[state0][sentence[i]])
            for state1 in HMM.STATES:
                # if tab[i-1][state1] == -3.14e+100:
                #     continue
                # else:
                if sentence[i] not in array_b[state0]:  #所有在测试集出现但没有在训练集中出现的字符
                    if sentence[i-1] not in array_b[state0]:
                        prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0]['end']
                    else:
                        prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0]['begin']
                    # print(sentence[i])
                    # prob = tab[i-1][state1] + array_a[state1][state0] + array_b[state0]['other']
                else:
                    prob = tab[i-1][state1] + array_a[state1][state0] + array_b[state0][sentence[i]]    #计算每个字符对应STATES的概率
#                     print(prob)
                items.append((prob,state1))
            # print(sentence[i] + state0)
            # print(array_b[state0][sentence[i]])
            # print(sentence[i])
            # print(items)
            best = max(items)   #bset:(prob,state)
            # print(best)
            tab[i][state0] = best[0]
            # print(tab[i][state0])
            new_path[state0] = path[best[1]] + [state0]
        path = new_path
 
    prob, state = max([(tab[len(sentence) - 1][state], state) for state in HMM.STATES])
    return path[state]