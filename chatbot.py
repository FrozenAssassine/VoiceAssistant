import json 
import numpy as np
from tensorflow import keras
import pickle
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import wikipediaapi

#download stopwords and sentence splitter:
nltk.download('stopwords')
nltk.download('punkt')

wiki_wiki = wikipediaapi.Wikipedia('de')
stop_words = set(stopwords.words('german'))
max_len = 100

stop_words.remove("wie")

with open("intents.json") as file:
    data = json.load(file)

# load trained model
model = keras.models.load_model('chat_model')

with open('./chat_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('./chat_model/label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

def search_wikipedia(topic):
    page_py = wiki_wiki.page(topic)
    summary =  page_py.summary
    return summary if len(summary) < 200 else '. '.join(nltk.sent_tokenize(summary)[:3])

def hasKeyword(sequence):
    return "<" in sequence and ">" in sequence

def executeSequence(input_sequence, question):
    if "<CURRENT_TIME>" in input_sequence:
        time = datetime.now()
        return input_sequence.replace("<CURRENT_TIME>", time.strftime("%H:%M"))
    elif "<CURRENT_DATE>" in input_sequence:
        date = datetime.now()
        return input_sequence.replace("<CURRENT_DATE>", date.strftime("%d.%m.%Y"))
    elif "<WEATHER>" in input_sequence:
        return input_sequence.replace("<WEATHER>", "Not implemented yet...")
    elif "<GOOGLE>" in input_sequence:
        query = question.lower().replace('google', '')
        if len(query) > 0:
            return f"Ich habe folgendes gefunden: {search_wikipedia(query)}"
        return None
    return input_sequence

def preprocess_question(question):
    words = word_tokenize(question)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    print(f"Question: {question} -> {filtered_text}")
    return filtered_text

def chat(question):    
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([preprocess_question(question)]),
                                            truncating='post', maxlen=max_len))
    
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    print(f"Tag: {tag}")

    for i in data['intents']:
        if i['tag'].lower() == tag:
            sequence = np.random.choice(i['responses'])
            if hasKeyword(sequence):
                return executeSequence(sequence, question)
            return sequence
    return "NULL"