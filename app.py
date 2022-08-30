import sys
import os
import glob
import re
import numpy as np

import pandas as pd
from keras.preprocessing import sequence
import re 
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
data = pd.read_csv('Tweets.csv')
data = data[['text','airline_sentiment']]
data = data[data.airline_sentiment != "neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X)
data['airline_sentiment'].replace(['positive','negative'],[1,0],inplace=True)
Y = data['airline_sentiment'].values.astype('float32')
embed_dim = 128
lstm_out = 196
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
batch_size = 32
model.fit(X, Y, epochs = 3, batch_size=batch_size, verbose = 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tokenizer.texts_to_sequences(data)
        vect = pad_sequences(vect, maxlen=32)
        result = model.predict(vect)
        if result < 0.5:
            return render_template('result.html',prediction = 0)
        elif result > 0.5:
            return render_template('result.html',prediction = 1)
        else:
            return render_template('result.html',prediction = 2)
      

if __name__ == '__main__':
    app.run(debug=True)