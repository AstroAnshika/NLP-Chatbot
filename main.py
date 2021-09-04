import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

import pickle


data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
labels = data['labels']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]

    sentence_words = clean_up_sentence(sentence)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def classify(sentence, model):
        #ERROR_THRESHOLD = 0.25
        # generate probabilities from the model
        results = model.predict([bag_of_words(sentence, words)])
        # filter out predictions below a threshold
        results = [[i, r] for i, r in enumerate(results)]# if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
        return return_list

def getResponse(sentence):
    result = classify(sentence)
    if result:
        while result:
            for i in intents['intents']:
                if i['tag'] == result[0][0]:
                    res = random.choice(i['responses'])
                    return res

#def chat(text):
 #   ints = classify(text, model)
  #  res = getResponse(ints, intents)
   # return res

while(1):
    ques = input('You:')
    classify(ques, model)
    getResponse(ques)
    print(ques)
    if ques == 'close':
        break



