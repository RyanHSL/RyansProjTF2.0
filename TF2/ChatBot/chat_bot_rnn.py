import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

# load trained model
model = keras.models.load_model('chatbot_rnn.h5')

unclear_resonse = 'Sorry. I do not understand. Could you please describe it in details?'
# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

def predict_class(message, max_len,ERROR_THRESHOLD):
    res = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([message]),
                                                                      truncating='post', maxlen=max_len))
    res = res[0,:]
    results = [[np.where(res == r), r] for r in res if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    # tag = lbl_encoder.inverse_transform([np.argmax(res)])
    for r in results:
        return_list.append({'intent': lbl_encoder.classes_[r[0]], 'probability': str(r[1])})

    return return_list
    # for i in data['intents']:
    #     if i['tag'] == tag:
    #         print(np.random.choice(i['responses']))
    #         break
    # if tag == "goodbye":
    #     break

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
                    result = result + i['tag'] + "?"
                    unclear = 1

    return result, unclear

def get_accurate_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = unclear_resonse
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

def self_learning(message, tag):
    pass

def chat(ERROR_THRESHOLD):
    # parameters
    max_len = 20
    counter = 0
    while True:
        message = input('')
        ints = []
        while len(ints) == 0 and ERROR_THRESHOLD >= 0.5:
            ints = predict_class(message, max_len, ERROR_THRESHOLD)
            ERROR_THRESHOLD = ERROR_THRESHOLD - 0.1
        res, unclear = get_response(ints, data)
        if unclear:
            print(res)
            message = input()
            # self_learning(res, message)
            res = get_accurate_response(message, data)
        counter = counter + 1 if res == unclear_resonse else 0
        if counter >= 3:
            print('I will connect you to one of our representatives.')
            break
        print(res)
        if len(ints) > 0 and ints[0]['intent'] == 'goodbye':
            break


if __name__ == "__main__":
    print("Hello there! I am Rui Mao's chatbot. Nice to meet you!")
    chat(0.8)