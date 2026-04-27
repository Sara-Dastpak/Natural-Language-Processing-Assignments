from sklearn.model_selection import train_test_split
import numpy as np
import string
import os

#Robert Frost poems converting to lists
RF = []
file_path = os.path.join(os.path.dirname(__file__), "robert_frost.txt")
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.lower().translate(str.maketrans('', '', string.punctuation))
        clean_line = line.strip()
        if clean_line:
            RF.append(clean_line)

# Train Data & Test Data of RF
train_data_RF, test_data_RF = train_test_split(RF, test_size=0.2)

# Building Markov model NO.1
vocab1 = set()
fullist1 = []
for lines in train_data_RF:
    for words in lines.split():
        vocab1.add(words)
        fullist1.append(words)
dict_RF = {}
for index, word in enumerate(vocab1):
    dict_RF[word] = index
vocab_size1 = len(vocab1)
epsilon = 0.002
pi_array_RF = np.full(vocab_size1, epsilon)
A_matrix_RF = np.full((vocab_size1, vocab_size1), epsilon)
# filling pi array NO.1
for lines in train_data_RF:
    words = lines.split()
    temp_word = words[0]
    index = dict_RF[temp_word]
    pi_array_RF[index] += 1
pi_array_RF = pi_array_RF / pi_array_RF.sum()
# filling A matrix NO.1
for lines in train_data_RF:
    words = lines.split()
    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i+1]
        i1 = dict_RF[w1]
        i2 = dict_RF[w2]
        A_matrix_RF[i1, i2] += 1
A_matrix_RF = A_matrix_RF / A_matrix_RF.sum(axis=1, keepdims=True)
# final Markov model NO.1
def markov_model_RF(sequence):
    log_prob = 0
    words = sequence.split()
    w0 = words[0]
    if w0 in dict_RF:
        i0 = dict_RF[w0]
        log_prob  = np.log(pi_array_RF[i0])
    else:
        log_prob += np.log(epsilon / vocab_size1)
    for index,word in enumerate(words):
        if index == 0:
            continue
        else:
            wi = words[index-1]
            wj = words[index]
            if wi in dict_RF and wj in dict_RF:
                i1 = dict_RF[wi]
                i2 = dict_RF[wj]
                log_prob += np.log(A_matrix_RF[i1,i2])
            else:
                log_prob += np.log(0.0001)
    return log_prob / len(words)

#Edgar Allan Poe poems converting to lists            
EAP = []
file_path = os.path.join(os.path.dirname(__file__), "edgar_allan_poe.txt")
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.lower().translate(str.maketrans('', '', string.punctuation))
        clean_line = line.strip() 
        if clean_line:
            EAP.append(clean_line)

# Train Data & Test Data of RF
train_data_EAP, test_data_EAP = train_test_split(EAP, test_size=0.2)

# Building Markov model NO.2
vocab2 = set()
fullist2 = []
for lines in train_data_EAP:
    for word in lines.split():
        vocab2.add(word)
        fullist2.append(word)
dict_EAP = {}
for index, word in enumerate(vocab2):
    dict_EAP[word] = index
vocab_size2 = len(vocab2)
pi_array_EAP = np.full(vocab_size2, epsilon)
A_matrix_EAP = np.full((vocab_size2, vocab_size2), epsilon)
# filling pi array NO.2
for lines in train_data_EAP:
    words = lines.split()
    temp_word = words[0]
    index = dict_EAP[temp_word]
    pi_array_EAP[index] += 1
pi_array_EAP = pi_array_EAP / pi_array_EAP.sum()
# filling A matrix NO.2
for lines in train_data_EAP:
    words = lines.split()
    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i+1]
        i1 = dict_EAP[w1]
        i2 = dict_EAP[w2]
        A_matrix_EAP[i1, i2] += 1
A_matrix_EAP = A_matrix_EAP / A_matrix_EAP.sum(axis=1, keepdims=True)
# final Markov model NO.2
def markov_model_EAP(sequence):
    log_prob = 0
    words = sequence.split()
    w0 = words[0]
    if w0 in dict_EAP:
        i0 = dict_EAP[w0]
        log_prob = np.log(pi_array_EAP[i0])
    else:
        log_prob += np.log(epsilon / vocab_size2)
    for index,word in enumerate(words):
        if index == 0:
            continue
        else:
            wi = words[index-1]
            wj = words[index]
            if wi in dict_EAP and wj in dict_EAP:
                i1 = dict_EAP[wi]
                i2 = dict_EAP[wj]
                log_prob += np.log(A_matrix_EAP[i1,i2])
            else:
                log_prob += np.log(0.0001)
    return log_prob / len(words)
# probability of each author
author_prob = [0,0]
sum = len(fullist1) + len(fullist2)
author_prob[0] = np.log(len(fullist1)/sum)
author_prob[1] = np.log(len(fullist2)/sum)
# Test list
test_dict = {}
for lines in test_data_RF:
    test_dict[lines] = "Robert Frost"
for lines in test_data_EAP:
    test_dict[lines] = "Edgar Allan Poe"
# Efficiency
eff = 0
length = 0
for sentence, author in test_dict.items():
    length += 1
    markov_RF = markov_model_RF(sentence)
    prob_score_RF = markov_RF + author_prob[0]
    markov_EAP = markov_model_EAP(sentence)
    prob_score_EAP = markov_EAP + author_prob[1]
    if prob_score_RF > prob_score_EAP:
        temp = "Robert Frost"
    elif prob_score_RF < prob_score_EAP:
        temp = "Edgar Allan Poe"
    if temp == author:
        eff +=1
efficiency = eff*100/length
print(efficiency)

train_dict = {}
for line in train_data_RF: 
    train_dict[line] = "Robert Frost"
for line in train_data_EAP: 
    train_dict[line] = "Edgar Allan Poe"
acc = 0
total_train = 0
for sentence, author in train_dict.items():
    total_train += 1
    markov_RF = markov_model_RF(sentence)
    prob_score_RF = markov_RF + author_prob[0]
    markov_EAP = markov_model_EAP(sentence)
    prob_score_EAP = markov_EAP + author_prob[1]
    if prob_score_RF > prob_score_EAP:
        temp = "Robert Frost"
    elif prob_score_RF < prob_score_EAP:
        temp = "Edgar Allan Poe"
    if temp == author:
        acc +=1
accuracy = acc*100/total_train
print(accuracy)