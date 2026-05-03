import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer

bbc_text = []
word_set = set()
file_path = os.path.join(os.path.dirname(__file__), "bbc_text.csv")
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.lower()
        clean_line = line.strip()
        if clean_line:
            bbc_text.append(clean_line)
        for words in word_tokenize(clean_line):
            word_set.add(words)
stop_words = set(stopwords.words('english'))
# Markov_model:
count_dict = {}
A_dict = {}
for line in bbc_text:
    temp_list = []
    for words in word_tokenize(line):
        temp_list.append(words)
    for words in temp_list:
        count_dict[words] = count_dict.get(words, 0) + 1
    n = len(temp_list)
    if len(temp_list) < 3:
        continue
    for i in range(n-2):
        wpre = temp_list[i]
        wmid = temp_list[i+1]
        wnex = temp_list[i+2]
        if wpre not in A_dict:
            A_dict[wpre] = {}
        if wnex not in A_dict[wpre]:
            A_dict[wpre][wnex] = {}
        A_dict[wpre][wnex][wmid] = A_dict[wpre][wnex].get(wmid, 0) + 1
for wpre, wnexs in A_dict.items():
    for wnex, wmids in A_dict[wpre].items():
        total_wnex_sum = sum(wmids.values())
        for wmid, count in wmids.items():
            A_dict[wpre][wnex][wmid] = count / total_wnex_sum 
def chosen_word(wp, wn):
    if wp in A_dict and wn in A_dict[wp]:
        mid_words = A_dict[wp][wn]
        best_word = max(mid_words, key=mid_words.get)
        return best_word
    else:
        return None
# choosing which word to change
threshold = 4
b = 0
bbc_text_new = []
for index, line in enumerate(bbc_text):
    b = 0
    word_list = word_tokenize(line)
    n = len(word_list)
    num = max(1, int(n * 0.5))
    for i, word in enumerate(word_list):
        if i > 0 and i + 1 < n and b < num:
            if (word not in stop_words
                and word not in string.punctuation
                and count_dict.get(word, 0) > threshold):
                w = chosen_word(word_list[i-1], word_list[i+1])
                if w is not None and w != word:
                    word_list[i] = w
                    b += 1
    new_line = TreebankWordDetokenizer().detokenize(word_list)
    bbc_text_new.append(new_line)
    """
for i, line in enumerate(bbc_text[:10], 1):
    print(f"{i}: {line}")
    """
for i in range(10):  # compare the first 10 lines
    print("ORIGINAL:", bbc_text[i])
    print("SPUN     :", bbc_text_new[i])
    print("-" * 80)
changed = 0

for i, (old, new) in enumerate(zip(bbc_text, bbc_text_new)):
    if old != new:
        changed += 1

print("Changed lines:", changed)
print("Total lines  :", len(bbc_text))