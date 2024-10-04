import json
import random
import pickle
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import process, fuzz

lemmatizer = WordNetLemmatizer()

with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Extracting all words from patterns in intents
all_words = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern)
        all_words.extend(words)

# Lemmatize and remove duplicates
all_words = list(set([lemmatizer.lemmatize(word.lower()) for word in all_words]))

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
words.extend(all_words)
words = list(set(words))

classes = pickle.load(open('classes.pkl', 'rb'))
classes.extend(intent['tag'] for intent in intents['intents'])
classes = list(set(classes))

model = tf.keras.models.load_model('chatbot_model.h5')

ERRORTHRESHOLD = 0.25
MAX_SIMILARITY = 70

# Context tracking
context = {}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERRORTHRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{'intent': 'unknown', 'probability': '1.0'}]

    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(patterns, responses, message):
    patterns.append({'patterns': ['خروج'], 'tag': 'end'})

    responses_list = []

    for word in nltk.word_tokenize(message):
        for pattern in patterns:
            if word in pattern['patterns']:
                responses_list.extend(pattern['responses'])

    responses_list = list(set(responses_list))  # Remove duplicates

    if responses_list:
        return ' '.join(responses_list)  # Combine responses into one string

    return f"أنا آسف، لم أفهم تماماً. هل يمكنك توضيح أكثر؟ ({message})"

print("☺ مرحبا , كيف يمكن لسليم مساعدتك ؟ ,اذا كنت لا تحتاج المساعدة اكتب 'خروج' ")

while True:
    user_input = input("أنت:")

    # Handle user exit
    if 'خروج' in user_input:
        print("سليم: وداعا!")
        break

    # Get response for each word
    full_response = []
    for word in nltk.word_tokenize(user_input):
        # Predict intent and get response for each word
        classification_result = predict_class(word)
        intent = classification_result[0]['intent']
        response = get_response(intents['intents'], intents, word)

        # Collect responses for each word
        full_response.append(response)

        # Check for follow-up actions based on context
        if intent in context:
            full_response.append(context[intent])
            context.pop(intent)  # Clear context after responding

    # Provide feedback to the user for the entire message
    full_response_text = ' '.join(full_response)
    print("سليم:", full_response_text)
