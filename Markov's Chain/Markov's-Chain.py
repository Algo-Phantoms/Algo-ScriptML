#!/usr/bin/env python
# coding: utf-8

# Here, taken a text file having speech of US president, Biden. This model helps to predict next few lines by taking single word as input.

# In[1]:


import numpy as np


# In[12]:


#Read the data set
corpus=open('Trump-Speech.txt',encoding='utf8').read()


# In[13]:


corpus


# In[16]:


#Split the data set into individual words
corpus = corpus.split()


# In[17]:


corpus


# In[4]:


'''
Next, creating a function that generates the different pairs of words in the speeches. To save up space, 
we’ll use a generator object.Here, we are creating a pair of every adjacent words to form a tuple which will 
be used to make prediction in later stages .
'''
def make_pairs(corpus):
    for i in range(len(corpus) - 1):
        yield (corpus[i], corpus[i + 1])
pairs = make_pairs(corpus)

#As shown below are few pairs of adjacent words


# In[5]:


'''
Here,  initializing an empty dictionary to store the pairs of words.

In case the first word in the pair is already a key in the dictionary, just append the next potential 
word to the list of words that follow the word. But if the word is not a key, then create a new entry in the dictionary 
and assign the key equal to the first word in the pair.

'''
word_dict = {}
for word_1, word_2 in pairs:
    if word_1 in word_dict.keys():
        word_dict[word_1].append(word_2)
    else:
        word_dict[word_1] = [word_2]


# In[6]:


'''
Random choice will give any word which doesnt lead to good prediction so instead of taking any word 
,creating  a function which will give only starting word of new sentence (assuming starting word of new sentence starts
with capital letter in the transcipt) .
'''


first_word = np.random.choice(corpus) #Initally ,randomly pick the first word
chain = [first_word]
 
#Pick the first word as a capitalized word so that the picked word is not taken from in between a sentence
def capitalized_word(first_word):
    while first_word.islower():
        first_word = np.random.choice(corpus)
 
# Start the chain from the picked word
        chain = [first_word]
    return chain
    


chain=capitalized_word(first_word) #here, it seen that capital word is optained


chain


# In[9]:


'''
Following the first word, each word in the chain is randomly sampled from the list of words which have followed 
that specific word in biden’s live speeches and its appended to list chain

'''
n_words = 40  #Initialize the number of stimulated words
for i in range(n_words):
    chain.append(np.random.choice(word_dict[chain[-1]]))


# In[10]:


#The simulated words are displayed
print(' '.join(chain))

