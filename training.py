import json 
import numpy as np 
import pickle
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import keras

vocab_size = 5000
embedding_dim = 1048
max_len = 100
epochs = 160
batch_size = 64
hidden_size = 2048


def show_result(callback):

	x = [i for i in range(1, len(callback.accuracy) + 1)]
	
	plt.plot(x, callback.accuracy, label="Accuracy")
	plt.legend(loc='best')

	# naming the x axis
	plt.xlabel('Epochs')
	# naming the y axis
	plt.ylabel('Losses')

	plt.show()

class Callback(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracy.append(logs.get('accuracy'))


with open('intents.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern.lower())
        training_labels.append(intent['tag'].lower())

    responses.append(res.lower() for res in intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'].lower())

num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)


#build the model:
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()

callback = Callback()

#train the model:
history = model.fit(padded_sequences, np.array(training_labels), batch_size=batch_size, epochs=epochs, callbacks=callback)

show_result(callback)

# to save the trained model
model.save("chat_model")

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

print("Done")
exit()