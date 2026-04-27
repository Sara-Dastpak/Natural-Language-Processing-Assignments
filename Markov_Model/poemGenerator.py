import numpy as np
import string
import random
import math

RF = []
line_num = 0
word_num = 0
file_path = r"C:\Users\Zhivar\Desktop\programs\NLP\markovmodels\poetclassifier\t\robert_frost.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.lower().translate(str.maketrans('', '', string.punctuation))
        clean_line = line.strip()
        word_num += len(clean_line.split())
        if clean_line:
            RF.append(clean_line)
            line_num += 1
            word_num += len(clean_line.split())
num_word_perline = int(word_num/line_num)
def pi_A1(document):
    pi_counts = {}
    A1_counts = {}
    # 1. Gather all counts first
    for line in document:
        words = line.split()
        if len(words) < 2: 
            continue  
        first_word = words[0]
        second_word = words[1]  
        pi_counts[first_word] = pi_counts.get(first_word, 0) + 1
        if first_word not in A1_counts:
            A1_counts[first_word] = {}
        A1_counts[first_word][second_word] = A1_counts[first_word].get(second_word, 0) + 1
    # 2. Convert counts to probabilities
    pi_dict = {}
    total_pi = sum(pi_counts.values())
    for word, count in pi_counts.items():
        pi_dict[word] = (count / total_pi)
    A1_dict = {}
    for first_word, next_words_dict in A1_counts.items():
        A1_dict[first_word] = {}
        total_A1 = sum(next_words_dict.values())       
        for second_word, count in next_words_dict.items():
            A1_dict[first_word][second_word] = (count / total_A1)        
    return pi_dict, A1_dict
def A2(document):
    A2_counts = {}
    for lines in document:
        words = lines.split()
        for i in range(len(words) - 2):
            w1 = words[i]
            w2 = words[i+1]
            w3 = words[i+2]
            if w1 not in A2_counts:
                A2_counts[w1] = {}
            if w2 not in A2_counts[w1]:
                A2_counts[w1][w2] = {}
            A2_counts[w1][w2][w3] = A2_counts[w1][w2].get(w3, 0) + 1
    A2_dict = {}
    for first_word, next_words_dict in A2_counts.items():
        A2_dict[first_word] = {}
        for second_word, third_words_dict in A2_counts[first_word].items():
            A2_dict[first_word][second_word] = {}
            total_A2 = sum(third_words_dict.values())       
            for third_word, count in third_words_dict.items():
                A2_dict[first_word][second_word][third_word] = (count / total_A2)
    return A2_dict
pi_dict, A1_dict = pi_A1(RF)
A2_dict = A2(RF)
def Markov_Model(sequence, pi_dict, A1_dict, A2_dict):
    words = sequence.split()
    if len(words) < 3:
        return -9999 
    w1 = words[0]
    w2 = words[1]
    pi_prob = np.log(pi_dict.get(w1))
    if w1 in A1_dict:
        A1_prob = np.log(A1_dict[w1].get(w2))
    A2_prob = 0
    for i in range(len(words)-2):
        i1 = words[i]
        i2 = words[i+1]
        i3 = words[i+2]
        if i1 in A2_dict and i2 in A2_dict[i1]:
            A2_prob += np.log(A2_dict[i1][i2].get(i3))
    final_prob =  pi_prob + A1_prob + A2_prob
    return math.exp(final_prob)


# Generating text
for i in range(5):
    r1 = random.random() 
    sum_pi = 0.0
    generated = []
    for word, prob in pi_dict.items():
        sum_pi += prob
        if r1 < sum_pi:
            generated.append(word)
            break
    r2 = random.random() 
    sum_A1 = 0.0
    for word, prob in A1_dict[generated[0]].items():
        sum_A1 += prob
        if r2 < sum_A1:
            generated.append(word)
            break
    for i in range(num_word_perline-2):
        r3 = random.random()
        sum_A3 = 0.0
        w1, w2 = generated[-2], generated[-1]
        if w1 in A2_dict and w2 in A2_dict[w1]:
            for word, prob in A2_dict[generated[-2]][generated[-1]].items():
                sum_A3 += prob
                if r3 < sum_A3:
                    generated.append(word)
                    break
        else:
            continue
    sentence = " ".join(generated)
    score = Markov_Model(sentence, pi_dict, A1_dict, A2_dict)
    print(sentence , score)