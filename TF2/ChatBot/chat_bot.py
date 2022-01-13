import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')
unclear_resonse = 'Sorry. I do not understand. Could you please describe it in details?'

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentenc_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentenc_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence, ERROR_THRESHOLD):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    unclear = 0
    if len(intents_list) == 0:
        result = unclear_resonse
    elif len(intents_list) == 1:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = "Are you asking about "
        for ints in intents_list:
            tag = ints['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = result + tag + "?"
                    unclear = 1

    return result, unclear

def get_accurate_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

def self_learning(message, tag):
    pass

if __name__ == "__main__":
    counter = 0
    while True:
        message = input()
        ints = []
        ERROR_THRESHOLD = 0.8
        while len(ints) == 0 and ERROR_THRESHOLD >= 0.5:
            ints = predict_class(message, ERROR_THRESHOLD)
            ERROR_THRESHOLD = ERROR_THRESHOLD - 0.05
        res, unclear = get_response(ints, intents)
        counter = counter + 1 if res == unclear_resonse else 0
        if unclear:
            print(res)
            message = input()
            self_learning(res, message)
            res = get_accurate_response(message, intents)
        if counter >= 3:
            print('I will connect you to one of our representatives.')
            break
        print(res)
        if len(ints) > 0 and ints[0]['intent'] == 'goodbye':
            break
