#coding: utf:8
import HMM
import Viterbi


#根据状态序列进行分词
def tag_seg(sentence,tag):
    word_list = []
    start = -1
    started = False
 
    if len(tag) != len(sentence):
        return None
 
    if len(tag) == 1:
        word_list.append(sentence[0])   #语句只有一个字，直接输出
 
    else:
        if tag[-1] == 'B' or tag[-1] == 'M':    #最后一个字状态不是'S'或'E'则修改
            if tag[-2] == 'B' or tag[-2] == 'M':
                tag[-1] = 'E'
            else:
                tag[-1] = 'S'
 
 
        for i in range(len(tag)):
            if tag[i] == 'S':
                if started:
                    started = False
                    word_list.append(sentence[start:i])
                word_list.append(sentence[i])
            elif tag[i] == 'B':
                if started:
                    word_list.append(sentence[start:i])
                start = i
                started = True
            elif tag[i] == 'E':
                started = False
                word = sentence[start:i + 1]
                word_list.append(word)
            elif tag[i] == 'M':
                continue
 
    return word_list